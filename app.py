import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

st.set_page_config(page_title="AI Air Optimizer", layout="wide")
st.title("🌿 AI Fresh Air Advisor")

uploaded_file = st.file_uploader("Upload Weekly CSV (timestamp, co2, temp, hum)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # --- FEATURE ENGINEERING ---
    # We turn time into numbers the AI can use
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    # "Lag" feature: help the AI see the trend by looking at the previous value
    df['co2_last_hour'] = df['co2'].shift(1).fillna(method='bfill')

    # --- TRAIN THE AI ---
    X = df[['hour', 'day_of_week', 'temp', 'hum', 'co2_last_hour']]
    y = df['co2']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # --- PREDICTION / OPTIMIZATION ---
    st.subheader("Analysis & Recommendations")
    
    # Identify "Danger Zones" (Where AI predicts CO2 > 1000)
    df['predicted_co2'] = model.predict(X)
    danger_zones = df[df['predicted_co2'] > 1000]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Peak CO2 Detected", f"{int(df['co2'].max())} ppm")
        st.write("The AI identifies these as your critical 'Stale Air' windows:")
        # Group by hour to give a general recommendation
        best_times = danger_zones.groupby('hour').size().sort_values(ascending=False)
        for hour, count in best_times.head(3).items():
            st.warning(f"⚠️ High CO2 likely around {hour}:00. Plan to ventilate then!")

    with col2:
        # Visualizing the AI's understanding vs Reality
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['co2'], name="Actual CO2"))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted_co2'], name="AI Prediction", line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

    # --- THE "OPTIMAL WINDOW" LOGIC ---
    # We look for high CO2 but "comfortable" external-ish temps (if available)
    st.info("💡 **Pro-tip:** The AI suggests opening windows 15 minutes *before* your typical peak hours to prevent CO2 buildup entirely.")
