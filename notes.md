# Project notes and decisions

## Step 1: Background concepts

**1. Why can't you use DepDelay as a feature?**
Because you don't know it yet when you're making the prediction. The whole point is to predict
before the plane leaves, and DepDelay only gets recorded after it leaves. If a flight pushed
back 90 minutes late, of course it's going to land late — the model wouldn't be predicting
anything, just restating what already happened. A lot of Kaggle notebooks on this dataset
report really high accuracy for exactly this reason.

**2. Name three columns that ARE safe to use, and explain why.**
CRSDepTime, Origin/Dest, and Distance. All of these come from the published schedule, so they
exist months before the plane ever moves. Same goes for carrier, day of week, and scheduled
flight time.

**3. What is delay propagation, and why does tail number matter for it?**
Airlines fly the same plane on multiple routes in one day. So if the morning flight runs late,
that plane shows up late for its next leg, and the delay keeps rolling forward. Tail number is
basically the plane's license plate — it's the only way to connect a flight to what that same
aircraft did earlier. Flight numbers won't work since those belong to routes, not planes. BTS
actually tracks this as its own delay category (LateAircraftDelay), and it's one of the biggest.

**4. For a feature like "this route's average delay," what time period should it cover, and
why not the whole dataset?**
Only the training months. If I calculate route averages across the whole dataset, those numbers
have test-month data baked into them, so the model ends up using the future to predict the
future. That would look great on the test set but wouldn't hold up in production, where you
only ever have past data. This one's sneaky because it doesn't involve any obviously
off-limits column.

**5. Why split by time rather than randomly?**
That's just how it would actually get used — train on old flights, run on new ones. A random
split puts flights from the same day and the same plane on both sides, so the model can pick up
"March 14th was a disaster at O'Hare" from training and apply it to test rows from that exact
day. That's looking things up, not predicting. Splitting by time means it has to handle periods
it's never seen. Worse score, but it's the real one.

## Decisions log

### Data source
BTS Reporting Carrier On-Time Performance (1987-present). Pulled [N] months, [month/year range].
Had to download month by month since tail number is only available through the bulk download
page, not the summary queries.

### Leakage policy
Features are limited to things you'd know before scheduled departure. The post-departure
columns (DepTime, DepDelay, TaxiOut, AirTime, ArrTime, and the five delay-cause fields) still
get downloaded, but only to build the target and the historical averages — never as features
for the flight being predicted.

### Cancelled and diverted flights
Dropped these since they have no arrival delay, so there's no target to predict. Worth noting
in the README that the model predicts delay *given the flight actually operates*.


## Step 2: Data exploration

### Dataset
3,385,822 flights after dropping cancelled and diverted, across January-June 2025.
Columns cast explicitly on load — CRSDepTime and CRSArrTime came in as VARCHAR because
the CSVs store times with leading zeros (0800), so DuckDB preserved them as text.
Used TRY_CAST so bad rows become NULL instead of failing the whole query.

### Base rate
21.9% of flights arrive 15+ minutes late. Mean arrival delay is 7.9 minutes but the median
is -6.0 — most flights actually arrive early, and a minority of very late flights pull the
average up. That skew is why a binary target makes more sense here than regressing on minutes.

### Delay propagation confirmed

| Previous flight (same aircraft, same day) | Flights | Delayed |
|---|---|---|
| Arrived early | 1,610,076 | 12.8% |
| First leg of day | 854,910 | 14.6% |
| 0-15 min late | 430,844 | 21.3% |
| 15-45 min late | 269,151 | 50.1% |
| 45-90 min late | 121,253 | 83.9% |
| 90+ min late | 99,588 | 83.1% |

A 6.5x spread from a single feature. The sharp jump between 0-15 and 15-45 minutes is where
the delay exceeds the scheduled turnaround buffer and the aircraft can't recover. The effect
plateaus past 45 minutes (83.9% then 83.1%), so the relationship is non-linear — a tree model
should handle this better than a linear one.

### Delay by scheduled departure hour
Delay rate more than triples across the day: 9.6% at 5am, rising steadily to 31.4% at 7pm,
then falling back to 19.5% by 11pm. Morning hours carry the most volume (233k flights at 6am)
and the lowest delay rates — the network hasn't backed up yet. The evening decline likely
reflects last-leg flights with no downstream pressure. Midnight-4am has very low volume
(231-5,450 flights/hour) and noisy rates.

Hour and propagation capture different things: hour is network-wide congestion, propagation
is this specific aircraft's situation.

### Legs per aircraft per day
Some aircraft fly up to 13 legs in a single day (1,442 flights on leg 10), so there's plenty
of multi-leg data for propagation features to work with.

## Decisions log

### Integer division gotcha
DuckDB does true division on integers, so CRSDepTime / 100 returns 23.59 instead of 23.
Used integer division (//) to extract the hour. Easy to miss — the first version produced
1,440 "hours" instead of 24.

### Column type casting
Explicitly cast all numeric columns on load rather than relying on auto-detection. BTS CSVs
store times as text to preserve leading zeros, which breaks arithmetic downstream.

## Step 3: Feature engineering

### Historical aggregates (training months only)
- route_hist: 4,914 routes with 50+ historical flights
- carrier_hist: 14 carriers
- airport_hist: 333 airports
- airport_hour_hist: 3,894 airport-hour combinations

All computed from flights before the cutoff date, then joined to both train and test.
Computing them over the full dataset would leak test-period information into training
features — a subtle leak that involves no obviously forbidden column.

Routes with fewer than 50 historical flights fall back to the overall base rate (0.22)
via COALESCE rather than being left NULL.

### Feature table
3,385,822 rows, 21 features. Verified every feature is knowable before departure —
no DepDelay, TaxiOut, AirTime, ArrTime, or delay-cause columns.

Propagation features: prev_arr_delay, is_first_leg, leg_number, turnaround_min.
These come from window functions partitioned by (tail, flight_date) and ordered by
scheduled departure, so each flight sees what the same physical aircraft did earlier
that day.

Spot-checking the output confirmed the cascade is captured correctly: aircraft N8885 on
April 10 shows leg 1 as first-of-day, leg 2 inheriting a 0-minute delay and arriving 19
minutes late, then leg 3 inheriting that 19 and recovering.

### turnaround_min data quality
42,195 flights (1.7%) had negative turnaround from midnight crossings — previous flight
scheduled to arrive 23:30, this one departs 06:15, so raw subtraction gives a large
negative (minimum was -625). Nulled those plus ~10,000 values over 720 minutes (overnight
gaps rather than real turnarounds), for 52,284 clipped total.

Combined with 854,910 first legs, 907,194 rows have NULL turnaround. LightGBM handles
NULLs natively and "no previous leg" is itself informative, so no imputation.

### Feature separation check (before modeling)

| Feature | Delayed (n=741,313) | On-time (n=2,644,509) |
|---|---|---|
| prev_arr_delay | +29.2 min | -3.9 min |
| dep_hour | 14.5 | 12.6 |
| leg_number | 3.25 | 2.75 |
| route_delay_rate | 0.210 | 0.193 |
| turnaround_min | 79.2 | 74.3 |

prev_arr_delay separates the classes sharply — delayed flights follow a previous leg that
was ~30 min late, on-time flights follow one that arrived ~4 min early. Expect this to
dominate feature importance.

turnaround_min runs opposite to intuition (delayed flights have *longer* average
turnarounds). Likely because it's NULL for first legs, which are the least-delayed group
and therefore excluded from the average, and because longer scheduled turnarounds may
correlate with larger hubs and longer routes. Something the model will need to untangle.

route_delay_rate separates only weakly (0.210 vs 0.193), which fits — route identity is
static while delay is driven mostly by same-day conditions.

### Historical features computed from training months only
Route, carrier, airport, and airport-hour delay rates are calculated using only pre-cutoff
flights, then applied to both train and test. Using the full dataset would mean the model
sees future information when predicting the future — inflating test scores in a way that
can't hold up in production.

### Minimum sample size for historical aggregates
Routes need 50+ historical flights to get their own delay rate. Below that the average is
noise, and the model would treat it as signal.

### turnaround_min clipping
Nulled negative values (midnight crossings) and anything over 720 minutes (overnight gaps).
Left as NULL rather than imputed, since LightGBM handles missing values natively and the
absence of a previous leg carries real information.

## Step 4: Models and validation

### Split
Train: Jan 1 – Apr 30 (2,188,776 flights, 19.7% delayed)
Test:  May 1 – Jun 30 (1,197,046 flights, 25.9% delayed)

The test period has a notably higher base rate — May/June is peak summer travel and
thunderstorm season. The model trained on a calmer period, so results are arguably
conservative.

### Results

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Baseline: base rate | 0.500 | 0.259 | 0.196 |
| Baseline: route history | 0.576 | 0.309 | 0.194 |
| Logistic regression | 0.763 | 0.599 | 0.154 |
| LightGBM | 0.816 | 0.725 | 0.125 |

The route-history baseline at 0.576 shows a lookup table gets you almost nothing here —
delay is driven by same-day conditions, not route identity. PR-AUC of 0.725 against a
25.9% base rate is the more meaningful number given the class imbalance.

### Ablation: propagation features
With propagation:    ROC-AUC 0.8163, PR-AUC 0.7250
Without propagation: ROC-AUC 0.6699, PR-AUC 0.3888
**Improvement: +0.146 ROC-AUC, +0.336 PR-AUC**

Four features derived from one window function nearly double PR-AUC. Without them the model
barely beats logistic regression *with* them.

### Where the model works and where it doesn't

| | Flights | Actual delay rate | ROC-AUC |
|---|---|---|---|
| Has previous leg | 904,691 | 29.6% | 0.839 |
| First leg of day | 292,355 | 14.8% | 0.678 |

The model is strong where propagation signal exists and mediocre where it doesn't. The
first-leg ROC-AUC (0.678) nearly matches the no-propagation ablation model (0.670) — a clean
internal consistency check. Practical implication: this works well for the 76% of flights
that aren't first-of-day. Improving first-leg prediction would require weather or
airport-condition data.

### Feature importance (gain, not split count)

| Feature | Gain |
|---|---|
| prev_arr_delay | 3,289,904 |
| turnaround_min | 908,149 |
| Dest | 417,551 |
| Origin | 387,861 |
| hour_delay_rate | 233,674 |
| is_first_leg | 223,148 |
| route_delay_rate | 185,230 |
| carrier | 111,315 |
| leg_number | 99,698 |
| dep_hour | 96,458 |

prev_arr_delay contributes 3.6x more gain than the next feature. Three of the top six are
propagation features.

Note: LightGBM's default feature_importances_ counts split *frequency*, which ranked Dest and
Origin first — high-cardinality categoricals (333 airports) get split on constantly without
contributing proportional gain. Used importance_type="gain" instead, which measures how much
each split actually improved the model.

### Calibration
Mean predicted probability 0.224 vs actual 0.259 — the model systematically under-predicts by
~3.5 points because it learned a 19.7% base rate and was tested in a 25.9% period. It has no
way to know summer arrived.

Applied isotonic regression fitted on the first two weeks of the test period (simulating a
production practice of periodic recalibration), then evaluated on the remainder:

| | Before | After | Actual |
|---|---|---|---|
| Mean prediction | 0.228 | 0.252 | 0.267 |
| Brier | 0.1266 | 0.1254 | — |
| ROC-AUC | 0.819 | 0.819 | — |

Closed about 60% of the bias. ROC-AUC unchanged, confirming isotonic regression preserves
ranking while adjusting probabilities. The residual gap exists because delays kept climbing
through the test window — the calibration period (May 1–14) was itself calmer than what
followed. In production this would need a rolling recalibration window, not a one-time fit.



## Step 5: Threshold selection and business impact

### Cost assumptions (illustrative)
- Unanticipated 15+ min delay: $1,500
- Cost to act on a flagged flight: $200
- Fraction of delays prevented by acting: 40%

### Optimal operating point
Threshold **0.30** — flags 238,952 flights (20.0%) at 71.2% precision and 54.8% recall.
Estimated savings $54.3M across 1.2M flights, or **$45,371 per 1,000 flights**.

The threshold lands below 0.5 because of cost asymmetry: a missed delay costs $1,500, a
false alarm costs $200.

### Model vs simple rules

| Strategy | Flagged | Precision | Recall | Savings/1k |
|---|---|---|---|---|
| Model (threshold 0.30) | 20.0% | 71.2% | 54.8% | $45,371 |
| Flag afternoon/evening (hour ≥ 15) | 40.5% | 37.3% | 58.3% | $9,624 |
| Flag all non-first legs | 75.6% | 29.5% | 86.1% | -$17,157 |
| Flag everything | 100% | 25.9% | 100% | -$44,395 |

The model captures **4.7x more value than the best heuristic**. Two intuitive rules actively
destroy value — flagging everything loses $44k per 1,000 flights because you pay $200 each on
741 flights that would have been fine. This is the clearest demonstration that precision
matters more than recall when intervention isn't free.

### Sensitivity
Optimal threshold ranges 0.11–0.93 across the assumption grid, so the cutoff is highly
sensitive to inputs. The structure is coherent though: higher delay cost or better mitigation
both push the threshold down, since intervention becomes more worthwhile.

The approach loses money in exactly one cell — cheap delays ($750) with poor mitigation (20%),
at -$3,695/1k. That's the honest boundary condition: if acting barely helps and delays aren't
expensive, don't bother.

Important framing: the model's *ranking* is robust; only the *cutoff* moves. An airline would
set the threshold with its own cost figures, and the value doesn't depend on getting those
exactly right.

### Who gets flagged
Flagged rates track actual delay rates sensibly — F9 (34.5% actual, 30.5% flagged) and OH
(33.7%, 29.2%) highest; HA (15.2%, 13.4%) lowest. The model responds to real differences
rather than carrier identity.

Several carriers are under-flagged relative to their actual rates (AS 27.8% actual vs 18.5%
flagged; B6 27.0% vs 17.9%), likely because they fly more first legs or longer-turnaround
routes where the model has less signal.

First legs: 14.7% actual delay rate but only 7.4% flagged. The model is appropriately
conservative where it's weak, but this means first-leg delays go largely unaddressed — a real
operational gap, and the clearest case for adding weather data.









## Decisions log

### Data source
BTS Reporting Carrier On-Time Performance (1987-present). Pulled 6 months, Jan–Jun 2025.
Had to download month by month since tail number is only available through the bulk download
page, not the summary queries.

### Leakage policy
Features are limited to things you'd know before scheduled departure. The post-departure
columns (DepTime, DepDelay, TaxiOut, AirTime, ArrTime, and the five delay-cause fields) still
get downloaded, but only to build the target and the historical averages — never as features
for the flight being predicted.

### Cancelled and diverted flights
Dropped these since they have no arrival delay, so there's no target to predict. Worth noting
in the README that the model predicts delay *given the flight actually operates*.

### Column type casting
Explicitly cast all numeric columns on load rather than relying on auto-detection. BTS CSVs
store times as text to preserve leading zeros, which breaks arithmetic downstream.

### Integer division gotcha
DuckDB does true division on integers, so CRSDepTime / 100 returns 23.59 instead of 23. Used
integer division (//) to extract the hour. Easy to miss — the first version produced 1,440
"hours" instead of 24.

### Historical features computed from training months only
Route, carrier, airport, and airport-hour delay rates are calculated using only pre-cutoff
flights, then applied to both train and test. Using the full dataset would mean the model sees
future information when predicting the future — inflating test scores in a way that can't hold
up in production.

### Minimum sample size for historical aggregates
Routes need 50+ historical flights to get their own delay rate. Below that the average is
noise, and the model would treat it as signal. Routes below the threshold fall back to the
overall base rate (0.22) via COALESCE.

### turnaround_min clipping
Nulled negative values (midnight crossings) and anything over 720 minutes (overnight gaps),
52,284 rows total. Left as NULL rather than imputed, since LightGBM handles missing values
natively and the absence of a previous leg carries real information.

### Model choice
Compared LightGBM against two baselines (base rate, route-history lookup) and logistic
regression. The route-history baseline is the key control — it answers whether a lookup table
could replace the model. At 0.576 ROC-AUC, it can't.

### Imputation computed from training only
Medians used to fill NULLs for logistic regression come from the training set, then applied to
both splits. Computing them across all data would be another small leak. LightGBM needed no
imputation since it handles NULLs natively.

### Calibration approach
Chose isotonic regression over Platt scaling since the miscalibration is a level shift rather
than a sigmoid distortion. Fitted on a held-out slice of the test period rather than on
training data, because the whole problem is that the training period's base rate no longer
applies.


