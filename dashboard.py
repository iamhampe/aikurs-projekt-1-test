import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="AI Security Dashboard",
    page_icon="🛡️",
    layout="wide",
)

logs = pd.read_csv("output/scored_logs.csv")
metrics = pd.read_csv("output/model_metrics.csv")
suspicious = pd.read_csv("output/suspicious_events.csv")

st.title("🛡️ AI Security Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Events", len(logs))

with col2:
    st.metric("Detected Threats", len(logs[logs["Prediction"] == 1]))

with col3:
    st.metric("Average Threat Score", f"{logs['ThreatProbability'].mean():.2%}")

with col4:
    best_model = metrics.sort_values("f1", ascending=False).iloc[0]
    st.metric("Best Model F1", f"{best_model['f1']:.3f}")

st.divider()

st.subheader("Threat Activity by Hour")

threats = logs[logs["Prediction"] == 1]

fig_hour = px.histogram(
    threats,
    x="Hour",
    title="Detected Threats by Hour",
)

st.plotly_chart(fig_hour, use_container_width=True)

st.subheader("Threats by Event ID")

fig_event = px.histogram(
    threats,
    x="Id",
    title="Detected Threats by Sysmon Event ID",
)

st.plotly_chart(fig_event, use_container_width=True)

st.subheader("Model Performance Comparison")

fig_model = px.bar(
    metrics,
    x="model",
    y="f1",
    title="F1 Score by Model",
)

st.plotly_chart(fig_model, use_container_width=True)

st.dataframe(metrics)

st.subheader("Top Suspicious Events")

if len(suspicious) > 0:
    columns = [
        col
        for col in [
            "TimeCreated",
            "Id",
            "ThreatProbability",
            "RuleRiskScore",
            "Message",
        ]
        if col in suspicious.columns
    ]

    st.dataframe(
        suspicious.sort_values("ThreatProbability", ascending=False)[columns].head(50),
        use_container_width=True,
    )
else:
    st.success("No suspicious events found.")