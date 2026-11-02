# Flight Delay Prediction

> **Question:** Which flights will arrive 15+ minutes late, using only information available before the plane pushes back — and is acting on that prediction actually worth the cost?

**🔗 [Live demo](https://flight-delay-risk.streamlit.app/)** · Trained on 3.4M US domestic flights (US DOT BTS, Jan–Jun 2025)

## Key findings

**1. Delay propagation is the dominant signal.** Airlines reuse the same physical aircraft across several flights a day, so a morning delay cascades through every later leg. Tracking this with window functions over tail number produced four features that contribute **+0.146 ROC-AUC** and nearly double PR-AUC (0.389 → 0.725).

**2. The model is only as good as its inputs allow.** Where an inbound aircraft exists, ROC-AUC is **0.839**. For first flights of the day — 24% of the dataset, with no delay to inherit — it drops to **0.678**, almost exactly matching a model trained without propagation features at all (0.670). The model is strong where the signal is and honest where it isn't.

**3. Precision matters more than recall when intervention costs money.** Flagging every flight *loses* $44,395 per 1,000 flights. Flagging every afternoon flight captures $9,624. The model's cost-optimized threshold captures **$45,371** — 4.7x the best heuristic.

**4. A model trained on a fixed window goes stale.** Trained on a 19.7% delay rate, tested on a 25.9% period, the model systematically under-predicted by 3.5 points. Isotonic recalibration on two weeks of recent data closed ~60% of the gap without changing ROC-AUC.

## The leakage trap

The BTS dataset includes `DepDelay`, `TaxiOut`, `AirTime`, and five delay-cause columns. **All of them are recorded after the flight operates.** Including `DepDelay` produces a model with excellent metrics and zero practical value — a flight that pushed back 90 minutes late will obviously land late, and by the time you know that, the prediction is worthless.

Many public versions of this problem report very high accuracy for exactly this reason.

This project uses only features knowable at scheduled departure:

| Category | Features |
|---|---|
| Published schedule | departure hour/minute, day of week, month, carrier, origin, destination, distance, scheduled duration |
| Historical aggregates | route delay rate, carrier delay rate, origin/destination delay rates, airport-hour congestion |
| Aircraft status | inbound flight's arrival delay, scheduled turnaround, leg number, first-leg flag |

Post-departure columns were still downloaded — they're needed to construct the target and the historical aggregates — but never used as features for the flight being predicted.

**A subtler leak:** historical aggregates like "this route's average delay" must be computed from training months only. Calculating them across the full dataset bakes test-period information into training features, inflating scores in a way that can't hold up in production.

## Delay propagation

Delay rate for a flight, grouped by how late the same aircraft's previous leg arrived that day:

| Previous flight | Flights | Delayed |
|---|---|---|
| Arrived early | 1,610,076 | **12.8%** |
| First leg of day | 854,910 | 14.6% |
| 0–15 min late | 430,844 | 21.3% |
| 15–45 min late | 269,151 | 50.1% |
| 45–90 min late | 121,253 | **83.9%** |
| 90+ min late | 99,588 | 83.1% |

A 6.5x spread from one feature. The sharp jump between 0–15 and 15–45 minutes is where the delay exceeds the scheduled turnaround buffer and the aircraft can no longer recover. The effect plateaus past 45 minutes — once a plane is badly late, more lateness doesn't increase the certainty.

Feature importance (by gain) confirms it: `prev_arr_delay` contributes 3.6x more than the next feature, and three of the top six are propagation features.

## Results

Train: Jan–Apr 2025 (2,188,776 flights, 19.7% delayed) · Test: May–Jun 2025 (1,197,046 flights, 25.9% delayed)

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Baseline: base rate | 0.500 | 0.259 | 0.196 |
| Baseline: route history | 0.576 | 0.309 | 0.194 |
| Logistic regression | 0.763 | 0.599 | 0.154 |
| **LightGBM** | **0.816** | **0.725** | **0.125** |

The route-history baseline at 0.576 answers the "could a lookup table do this?" question: it can't. Delay is driven by same-day conditions, not route identity.

### Ablation

| | ROC-AUC | PR-AUC |
|---|---|---|
| With propagation features | 0.816 | 0.725 |
| Without | 0.670 | 0.389 |

## Business impact

Translating probabilities into a decision requires knowing what action costs. Using illustrative figures — $1,500 per unanticipated delay, $200 to act on a flagged flight, 40% of delays prevented when you act — the expected-cost optimum is a **0.30 threshold**, flagging 20% of flights at 71% precision and 55% recall.

| Strategy | Flagged | Precision | Recall | Savings/1k flights |
|---|---|---|---|---|
| **Model (threshold 0.30)** | 20.0% | **71.2%** | 54.8% | **$45,371** |
| Flag afternoon/evening | 40.5% | 37.3% | 58.3% | $9,624 |
| Flag all non-first legs | 75.6% | 29.5% | 86.1% | −$17,157 |
| Flag everything | 100% | 25.9% | 100% | −$44,395 |

The threshold lands below 0.5 because of cost asymmetry: a missed delay costs $1,500, a false alarm costs $200.

**Sensitivity:** the optimal threshold ranges 0.11–0.93 across a grid of cost and mitigation assumptions, so the *cutoff* is assumption-dependent — an airline would set it with its own figures. The *ranking* is not. The approach only loses money in one corner of the grid: cheap delays ($750) with poor mitigation (20%), at −$3,695 per 1,000.

## Data

- **Source:** US DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance
- **Period:** January–June 2025, downloaded month by month (tail number is only available through the bulk download, not the summary query tools)
- **Rows:** 3,385,822 after excluding cancelled and diverted flights
- **Target:** `ArrDel15` — arrival 15+ minutes late, the industry-standard definition

Cancelled and diverted flights have no arrival delay, so they're excluded. The model therefore predicts delay *given the flight operates*.

## Methods

1. **Ingest (DuckDB):** six monthly CSVs read directly off disk with explicit type casting — BTS stores times as text to preserve leading zeros, which silently breaks arithmetic downstream.
2. **Features (SQL):** window functions partitioned by `(tail_number, flight_date)` and ordered by scheduled departure build the propagation features. Historical aggregates computed from training months only, with a 50-flight minimum before a route gets its own rate.
3. **Models:** LightGBM against logistic regression and two baselines. Categorical features and NULLs handled natively — `turnaround_min` is NULL for first legs, and that absence is itself informative.
4. **Validation:** time-based split. A random split would scatter flights from the same day and aircraft across both sets, letting the model look up conditions rather than predict them.
5. **Calibration:** isotonic regression fitted on a held-out slice of the test period, simulating periodic production recalibration.
6. **Deployment:** Streamlit dashboard, containerized with Docker.

## Limitations

- **No weather data.** This is almost certainly why first legs predict poorly — with no inbound aircraft to inherit from, weather and airport conditions are the main remaining signal.
- **First legs are under-served.** 14.7% actual delay rate but only 7.4% flagged. The model is appropriately conservative where it's weak, but those delays go largely unaddressed.
- **Historical rates are frozen** at the training period. Production would refresh them on a rolling basis.
- **Calibration can't fully keep up with a trending base rate.** Delays kept climbing through the test window, so even the recalibrated model stayed 1.5 points low.
- **Cost assumptions are illustrative**, not sourced from an airline.
- **Six months, one year.** Results may not generalize across years or to international operations.

## Next steps

- Join NOAA weather data by airport and hour, targeting the first-leg gap specifically
- Rolling recalibration on a weekly window rather than a one-time fit
- Model cancellation separately, since excluding cancelled flights conditions the whole analysis
- Extend propagation beyond one leg: does a delay two legs back still carry signal?

## Repository

```
├── sql/
│   └── 01_features.sql          # feature engineering with window functions
├── notebooks/
│   ├── 01_explore.ipynb         # SQL exploration, propagation confirmed
│   ├── 02_features.ipynb        # feature table construction
│   ├── 03_models.ipynb          # baselines, LightGBM, ablation, calibration
│   └── 04_business.ipynb        # threshold selection, sensitivity
├── app/
│   └── app.py                   # Streamlit dashboard
├── results/                     # model, lookups, comparison tables
├── figures/
├── notes.md                     # full analysis log and decisions
├── Dockerfile
└── requirements.txt
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Download six months of [Reporting Carrier On-Time Performance](https://transtats.bts.gov/Tables.asp?QO_VQ=EFD) data into `data/raw/`, then run notebooks 01–04 in order.

To run the dashboard locally:

```bash
streamlit run app/app.py
```

Or with Docker:

```bash
docker build -t flight-delay .
docker run -p 8501:8501 flight-delay
```

## Author

[Rayyan Siddiqui] · [https://www.linkedin.com/in/rayyan-siddiqui12/] · [rayyansiddiquizz060@gmail.com]