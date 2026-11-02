import json, pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Flight Delay Risk", page_icon="✈️", layout="wide")

THRESHOLD = 0.30
BASE_RATE = 0.259

@st.cache_resource
def load_assets():
    with open("results/model.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open("results/lookups.json") as f:
        lk = json.load(f)
    try:
        with open("results/calibrator.pkl", "rb") as f:
            cal = pickle.load(f)
    except Exception:
        cal = None
    return bundle, lk, cal

bundle, LK, CAL = load_assets()
model = bundle["model"]
feat_cols = bundle["feat_cols"]
cat_cols = bundle["cat_cols"]

st.title("Flight Delay Risk Predictor")
st.caption(
    "Predicts the probability of a 15+ minute arrival delay using only information "
    "available before departure. Trained on 3.4M US domestic flights (Jan–Jun 2025, "
    "US DOT BTS). Test ROC-AUC 0.82."
)

left, right = st.columns(2)

with left:
    st.subheader("Flight")
    carriers = LK["carriers"]
    airports = LK["airports"]
    carrier = st.selectbox("Carrier", carriers,
                           index=carriers.index("WN") if "WN" in carriers else 0)
    origin = st.selectbox("Origin", airports,
                          index=airports.index("ATL") if "ATL" in airports else 0)
    dest = st.selectbox("Destination", airports,
                        index=airports.index("ORD") if "ORD" in airports else 1)
    hour = st.slider("Scheduled departure hour", 0, 23, 15)
    dow = st.selectbox("Day of week", list(range(7)),
                       format_func=lambda d: ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][d])
    distance = st.number_input("Distance (miles)", 50, 3000, 600, step=50)

with right:
    st.subheader("Aircraft status")
    st.caption("Most of the signal lives here — a late inbound aircraft is the single "
               "strongest predictor in the model.")
    first_leg = st.checkbox("First flight of the day for this aircraft", value=False)
    if first_leg:
        prev_delay, leg, turnaround = 0, 1, None
        st.info("No inbound flight to inherit delay from. The model is notably weaker "
                "for first legs (ROC-AUC 0.68 vs 0.84).")
    else:
        prev_delay = st.slider("Inbound flight's arrival delay (min)", -30, 240, 0)
        leg = st.slider("Leg number today", 2, 8, 3)
        turnaround = st.slider("Scheduled turnaround (min)", 20, 300, 60)

# Look up historical rates, same values used in training
r  = LK["route"].get(f"{origin}-{dest}", [0.22, 8.0])
c  = LK["carrier"].get(carrier, 0.22)
ao = LK["airport"].get(origin, [0.22, 50])
ad = LK["airport"].get(dest, [0.22, 50])
ah = LK["airport_hour"].get(f"{origin}-{hour}", [5, 0.22])

row = pd.DataFrame([{
    "carrier": carrier, "Origin": origin, "Dest": dest,
    "dep_hour": hour, "dep_minute": 0, "day_of_week": dow, "month": 6,
    "Distance": float(distance), "CRSElapsedTime": distance / 7.5 + 30,
    "prev_arr_delay": float(prev_delay), "is_first_leg": int(first_leg),
    "leg_number": leg, "turnaround_min": turnaround,
    "route_delay_rate": r[0], "route_mean_delay": r[1],
    "carrier_delay_rate": c,
    "origin_delay_rate": ao[0], "origin_daily_departures": ao[1],
    "dest_delay_rate": ad[0],
    "departures_this_hour": ah[0], "hour_delay_rate": ah[1],
}])[feat_cols]

for col in cat_cols:
    row[col] = row[col].astype("category")

prob = float(model.predict_proba(row)[0, 1])
if CAL is not None:
    prob = float(CAL.predict([prob])[0])
prob = min(max(prob, 0.01), 0.97)

st.divider()
m1, m2, m3 = st.columns([1, 1, 2])
m1.metric("Delay probability", f"{prob:.0%}")
m2.metric("vs. average flight", f"{prob - BASE_RATE:+.0%}")
m3.progress(min(prob, 1.0))

if prob >= THRESHOLD:
    st.error(f"**Flagged.** Above the {THRESHOLD:.0%} intervention threshold — worth "
             "proactive rebooking, crew adjustment, or added schedule buffer.")
else:
    st.success(f"**Not flagged.** Below the {THRESHOLD:.0%} threshold.")

with st.expander("How this threshold was chosen"):
    st.markdown("""
The 30% cutoff comes from expected cost, not an arbitrary 0.5. Assuming an unanticipated
delay costs \\$1,500, acting costs \\$200, and intervention prevents 40% of delays, this
threshold maximizes savings: it flags 20% of flights at 71% precision, capturing an
estimated **\\$45,371 per 1,000 flights**.

For comparison, flagging every afternoon flight captures only \\$9,624 per 1,000, and
flagging every flight *loses* \\$44,395 — paying to act on flights that would have been
fine anyway.
""")

with st.expander("Model details and limitations"):
    st.markdown("""
**Model:** LightGBM on 21 features. Trained Jan–Apr 2025, tested May–Jun 2025.
**Test performance:** ROC-AUC 0.816, PR-AUC 0.725.

**Key finding:** four aircraft-propagation features (inbound delay, turnaround buffer, leg
number, first-leg flag) contribute **+0.146 ROC-AUC**. Without them the model scores 0.670.

**Limitations**
- No weather data, likely why first legs predict poorly
- Cancelled and diverted flights excluded — predicts delay *given the flight operates*
- Historical rates fixed from the training period; production would refresh them
- Cost assumptions are illustrative, not airline-sourced
""")