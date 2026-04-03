import streamlit as st
import plotly.graph_objects as go
import numpy as np
from joblib import load

# Load models
fitbit_model = load("fitbit_model.pkl")
sleep_model = load("sleep_model.pkl")
wesad_model = load("wesad_model.pkl")

# Load scalers
fitbit_scaler = load("fitbit_scaler.pkl")
sleep_scaler = load("sleep_scaler.pkl")
wesad_scaler = load("wesad_scaler.pkl")

st.set_page_config(layout="wide")
st.title("🧠 AI Multimodal Health Dashboard")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Enter Your Health Data")

steps = st.sidebar.number_input("Steps (Daily)", 0, 20000, 5000)
sleep_hours = st.sidebar.number_input("Sleep Hours", 0.0, 12.0, 7.0)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", 40, 150, 80)

predict_btn = st.sidebar.button("Predict Health")

# ---------------- DEFAULT VALUES ---------------- #
overall_index = 0
overall_risk = 0
stress_risk = 0
sleep_risk = 0
activity_risk = 0

# ---------------- PREDICTION ---------------- #
if predict_btn:

    # -------- ACTIVITY -------- #
    very_active = steps * 0.1
    fairly_active = steps * 0.05
    lightly_active = steps * 0.2
    sedentary = max(0, 1440 - (very_active + fairly_active + lightly_active))

    # -------- SLEEP -------- #
    sleep_quality = sleep_hours / 8 if sleep_hours > 0 else 0
    mean = sleep_hours * 4
    std = max(0.5, 3 - sleep_hours * 0.2)

    delta = min(1.0, sleep_quality * 0.5)
    theta = min(1.0, sleep_quality * 0.3)
    alpha = max(0.1, 1 - sleep_quality * 0.5)
    beta = max(0.1, 1 - sleep_quality * 0.6)

    # -------- STRESS -------- #
    eda = heart_rate / 40
    ecg = heart_rate
    resp = heart_rate / 5
    acc_x = acc_y = acc_z = 0

    # -------- FITBIT MODEL -------- #
    fitbit_raw = np.array([[steps, very_active, fairly_active, lightly_active, sedentary]])
    fitbit_scaled = fitbit_scaler.transform(fitbit_raw)

    fitbit_features = []
    for s in fitbit_scaled[0]:
        fitbit_features.extend([s, 0.0])

    while len(fitbit_features) < 12:
        fitbit_features.append(0.0)

    fitbit_input = np.array([fitbit_features])
    activity_value = fitbit_model.predict(fitbit_input)[0]

    activity_risk = min(max(activity_value / 3000 * 100, 0), 100)

    # -------- SLEEP MODEL -------- #
    sleep_input = np.array([[mean, std, delta, theta, alpha, beta]])
    sleep_input = sleep_scaler.transform(sleep_input)

    sleep_risk = sleep_model.predict_proba(sleep_input)[0][1] * 100

    if sleep_hours >= 7:
        sleep_risk *= 0.5
    elif sleep_hours < 5:
        sleep_risk *= 1.2

    sleep_risk = min(sleep_risk, 100)

    # -------- WESAD MODEL -------- #
    wesad_signals = [eda, ecg, resp, acc_x, acc_y, acc_z]

    wesad_features = []
    for s in wesad_signals:
        wesad_features.extend([s, 0.0, s, s])

    wesad_input = np.array([wesad_features])
    wesad_input = wesad_scaler.transform(wesad_input)

    stress_risk = wesad_model.predict_proba(wesad_input)[0][1] * 100

    # ===============================
    # 🔥 RULE-BASED CORRECTIONS
    # ===============================

    if steps < 1000:
        activity_risk = max(activity_risk, 80)

    if sleep_hours < 4:
        sleep_risk = max(sleep_risk, 85)

    if heart_rate < 45 or heart_rate > 110:
        stress_risk = max(stress_risk, 75)

    # ===============================
    # 🔥 WEIGHTED FUSION
    # ===============================

    overall_risk = (
        0.4 * sleep_risk +
        0.35 * stress_risk +
        0.25 * activity_risk
    )

    overall_index = 100 - overall_risk

    # ===============================
    # 🔥 CRITICAL OVERRIDE
    # ===============================

    if steps == 0 and sleep_hours == 0:
        overall_index = 10

# ---------------- CLASSIFICATION ---------------- #
if overall_index >= 70:
    classification = "GOOD HEALTH CONDITION"
    color = "green"
elif overall_index >= 40:
    classification = "MODERATE HEALTH CONDITION"
    color = "orange"
else:
    classification = "HIGH RISK CONDITION"
    color = "red"

# ---------------- DASHBOARD ---------------- #

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Health Index", f"{round(overall_index,2)}%")
col2.metric("Stress Risk", f"{round(stress_risk,2)}%")
col3.metric("Sleep Risk", f"{round(sleep_risk,2)}%")
col4.metric("Activity Risk", f"{round(activity_risk,2)}%")

st.markdown("---")

# Gauge
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=overall_index,
    title={'text': "Overall Health Index"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': color},
        'steps': [
            {'range': [0,40], 'color': "red"},
            {'range': [40,70], 'color': "orange"},
            {'range': [70,100], 'color': "green"}
        ],
    }
))

col1, col2 = st.columns(2)
col1.plotly_chart(fig, use_container_width=True)

# Donut
donut = go.Figure(data=[go.Pie(
    labels=['Stress','Sleep','Activity'],
    values=[stress_risk, sleep_risk, activity_risk],
    hole=.6
)])
donut.update_layout(title_text="Risk Distribution")

col2.plotly_chart(donut, use_container_width=True)

st.markdown("---")

# Classification
st.subheader("🔎 Classification")
st.markdown(f"### {classification}")

# ---------------- RECOMMENDATIONS ---------------- #
st.subheader("💡 Personalized Recommendations")

recommendations = []

if steps < 3000:
    recommendations.append("🚨 Very low activity. Aim for 7000–10000 steps.")
elif steps < 7000:
    recommendations.append("🚶 Moderate activity. Increase movement.")
else:
    recommendations.append("✅ Good activity level.")

if sleep_hours < 5:
    recommendations.append("😴 Severe sleep deprivation.")
elif sleep_hours < 7:
    recommendations.append("⚠️ Slightly low sleep.")
elif sleep_hours <= 9:
    recommendations.append("✅ Healthy sleep duration.")
else:
    recommendations.append("🛌 Oversleeping detected.")

if heart_rate > 100:
    recommendations.append("⚠️ High heart rate.")
elif heart_rate > 85:
    recommendations.append("😐 Slightly elevated HR.")
elif heart_rate < 50:
    recommendations.append("💙 Low heart rate.")
else:
    recommendations.append("✅ Normal heart rate.")

if overall_index < 40:
    recommendations.append("🚨 High health risk. Take action immediately.")
elif overall_index < 70:
    recommendations.append("⚠️ Moderate condition. Improve lifestyle.")
else:
    recommendations.append("🎉 You are in good health!")

for rec in recommendations:
    st.write(rec)