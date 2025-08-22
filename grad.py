# SkySense: Weather Forecasting Web App (LSTM & BiLSTM)
# ----------------------------------------------------
# Run with:  streamlit run skysense_streamlit_app.py
# 
# Notes:
# - Uses WeatherAPI (current, forecast, history) with your key.
# - Web UI lets you choose LSTM or BiLSTM, tweak epochs/sequence length, etc.
# - Charts use matplotlib only (no seaborn), one chart per figure (per platform rules).
# - Free WeatherAPI plans typically allow recent history (e.g., last 7 days). Adjust history_days accordingly.

pip install matplotlib
import os
import time
from datetime import datetime, timedelta, date
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import requests

import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="SkySense: Weather Forecasting", page_icon="🌤️", layout="wide")
st.title("🌤️ SkySense – Weather Forecasting (LSTM & BiLSTM)")

# -----------------------------
# Sidebar Controls
# -----------------------------
DEFAULT_KEY = "9facd06e231b4b66b06142645252208"  # Provided by user
api_key = st.sidebar.text_input("WeatherAPI Key", value=os.getenv("WEATHERAPI_KEY", DEFAULT_KEY), type="password")
city = st.sidebar.text_input("City", value="Amman")

col1, col2 = st.sidebar.columns(2)
with col1:
    history_days = st.number_input("History days", min_value=0, max_value=30, value=7, step=1, help="Number of past days to pull via history.json (free plans ~7 days)")
with col2:
    forecast_days = st.number_input("Forecast days", min_value=1, max_value=14, value=7, step=1, help="Number of future days to forecast with the model")

model_type = st.sidebar.selectbox("Model Type", ["LSTM", "BiLSTM"], index=0)
sequence_length = st.sidebar.slider("Sequence length (timesteps)", min_value=3, max_value=30, value=7)
epochs = st.sidebar.slider("Training epochs", min_value=5, max_value=200, value=40)
learning_rate = st.sidebar.select_slider("Learning rate", options=[0.0005, 0.001, 0.002, 0.005], value=0.001)
batch_size = st.sidebar.select_slider("Batch size", options=[8, 16, 32, 64], value=32)

run_btn = st.sidebar.button("🚀 Run Forecast")

# -----------------------------
# WeatherAPI Helpers
# -----------------------------
BASE_URL = "http://api.weatherapi.com/v1/"

@st.cache_data(show_spinner=False)
def fetch_current(api_key: str, city: str) -> Optional[dict]:
    try:
        resp = requests.get(
            f"{BASE_URL}current.json",
            params={"key": api_key, "q": city, "aqi": "no"}, timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Error fetching current weather: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_forecast(api_key: str, city: str, days: int) -> Optional[dict]:
    try:
        resp = requests.get(
            f"{BASE_URL}forecast.json",
            params={"key": api_key, "q": city, "days": days, "aqi": "no", "alerts": "no"}, timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Error fetching forecast: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_history_day(api_key: str, city: str, day_str: str) -> Optional[dict]:
    """Fetch history for a single day (YYYY-MM-DD)."""
    try:
        resp = requests.get(
            f"{BASE_URL}history.json",
            params={"key": api_key, "q": city, "dt": day_str, "aqi": "no"}, timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.info(f"History not available for {day_str}: {e}")
        return None

# -----------------------------
# Data Assembly
# -----------------------------
def assemble_dataframe(current: dict, forecast: dict, history_days: int) -> Tuple[pd.DataFrame, Optional[float], Optional[float]]:
    rows: List[dict] = []

    # Current
    if current:
        rows.append({
            "date": pd.to_datetime(current["location"]["localtime"][:10]),
            "temperature": current["current"]["temp_c"],
            "humidity": float(current["current"]["humidity"]),
            "pressure": float(current["current"]["pressure_mb"]),
            "wind_speed": float(current["current"]["wind_kph"]) / 3.6,  # kph -> m/s
            "precipitation": float(current["current"].get("precip_mm", 0.0)),
            "weather_main": current["current"]["condition"]["text"],
            "lat": current["location"]["lat"],
            "lon": current["location"]["lon"],
        })

    # Forecast (daily aggregates provided)
    if forecast and "forecast" in forecast:
        for d in forecast["forecast"]["forecastday"]:
            rows.append({
                "date": pd.to_datetime(d["date"]),
                "temperature": d["day"]["avgtemp_c"],
                "humidity": d["day"]["avghumidity"],
                "pressure": float(d["hour"][0]["pressure_mb"]) if d.get("hour") else np.nan,
                "wind_speed": float(d["day"]["maxwind_kph"]) / 3.6,
                "precipitation": float(d["day"].get("totalprecip_mm", 0.0)),
                "weather_main": d["day"]["condition"]["text"],
            })

    # History (loop past N days up to yesterday)
    coords = (None, None)
    if history_days > 0 and current:
        for i in range(1, history_days + 1):
            day = (datetime.now() - timedelta(days=i)).date().strftime("%Y-%m-%d")
            h = fetch_history_day(api_key, city, day)
            if h and "forecast" in h:
                for d in h["forecast"]["forecastday"]:
                    rows.append({
                        "date": pd.to_datetime(d["date"]),
                        "temperature": d["day"]["avgtemp_c"],
                        "humidity": d["day"]["avghumidity"],
                        "pressure": float(d["hour"][0]["pressure_mb"]) if d.get("hour") else np.nan,
                        "wind_speed": float(d["day"]["maxwind_kph"]) / 3.6,
                        "precipitation": float(d["day"].get("totalprecip_mm", 0.0)),
                        "weather_main": d["day"]["condition"]["text"],
                    })
        coords = (current["location"]["lat"], current["location"]["lon"]) if current else (None, None)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, None, None

    # Daily aggregation and cleaning
    df = df.groupby("date").agg({
        "temperature": "mean",
        "humidity": "mean",
        "pressure": "mean",
        "wind_speed": "mean",
        "precipitation": "sum"
    }).reset_index().sort_values("date")

    return df, coords[0], coords[1]

# -----------------------------
# Feature Engineering
# -----------------------------
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])    
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek

    # Cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Moving stats
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        df[f"{col}_ma7"] = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_std7"] = df[col].rolling(7, min_periods=1).std().fillna(0)

    # Lags
    for lag in [1, 2, 3, 7]:
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)

    return df.dropna().reset_index(drop=True)

# -----------------------------
# Sequence Prep
# -----------------------------
def make_sequences(df: pd.DataFrame, target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [c for c in df.columns if c not in ["date", target_col]]
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(df[feature_cols].iloc[i-seq_len:i].values)
        y.append(df[target_col].iloc[i])
    return np.array(X), np.array(y), feature_cols

# -----------------------------
# Model Builders
# -----------------------------
def build_lstm(input_shape: Tuple[int, int], lr: float) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def build_bilstm(input_shape: Tuple[int, int], lr: float) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# -----------------------------
# Forecast Generation (Iterative)
# -----------------------------
def iterative_forecast(model: Sequential, scaler_X: MinMaxScaler, scaler_y: MinMaxScaler,
                       last_seq: np.ndarray, feature_cols: List[str], start_date: pd.Timestamp,
                       days: int) -> pd.DataFrame:
    preds = []
    dates = []
    seq = last_seq.copy()  # shape (1, seq_len, n_features)

    # index mapping for temporal cyclical features if present
    name_to_idx = {name: i for i, name in enumerate(feature_cols)}

    for i in range(days):
        # Predict next target
        seq_scaled = scaler_X.transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape)
        yhat_scaled = model.predict(seq_scaled, verbose=0)
        yhat = scaler_y.inverse_transform(yhat_scaled)[0, 0]

        forecast_date = (start_date + timedelta(days=i+1)).date()
        dates.append(pd.to_datetime(forecast_date))
        preds.append(float(yhat))

        # Prepare next step features by copying last timestep features
        next_features = seq[0, -1, :].copy()

        # Update cyclical temporal features to the new date if they exist
        if "month_sin" in name_to_idx:
            m = pd.to_datetime(forecast_date).month
            doy = pd.to_datetime(forecast_date).dayofyear
            # Safely set values if keys exist
            for nm, val in {
                "month_sin": np.sin(2 * np.pi * m / 12),
                "month_cos": np.cos(2 * np.pi * m / 12),
                "doy_sin": np.sin(2 * np.pi * doy / 365),
                "doy_cos": np.cos(2 * np.pi * doy / 365)
            }.items():
                if nm in name_to_idx:
                    next_features[name_to_idx[nm]] = val

        # Roll sequence window forward
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, :] = next_features

    return pd.DataFrame({
        "date": dates,
        "predicted_temperature": preds
    })

# -----------------------------
# Main Run
# -----------------------------
if run_btn:
    if not api_key:
        st.error("Please provide a WeatherAPI key.")
        st.stop()

    with st.spinner("Fetching weather data..."):
        cur = fetch_current(api_key, city)
        fc = fetch_forecast(api_key, city, max(int(forecast_days), 1))
        df_raw, lat, lon = assemble_dataframe(cur, fc, int(history_days))

    if df_raw.empty:
        st.error("No data returned. Try a different city or increase history days.")
        st.stop()

    st.success(f"Fetched {len(df_raw)} daily records for {city}.")

    # Show raw table
    st.subheader("Raw Daily Weather Data")
    st.dataframe(df_raw.tail(20), use_container_width=True)

    # Feature Engineering
    df = featurize(df_raw)

    # Target and scaling
    target = "temperature"
    feature_cols_for_scaling = [c for c in df.columns if c not in ["date", target]]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_all, y_all, feature_cols = make_sequences(df, target, sequence_length)

    if len(X_all) < 20:
        st.warning("Not enough sequence samples to train a model. Try reducing sequence length or increasing history days.")
        st.stop()

    # Split train/val/test (80/10/10)
    n = len(X_all)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train:n_train+n_val]
    y_val = y_all[n_train:n_train+n_val]
    X_test = X_all[n_train+n_val:]
    y_test = y_all[n_train+n_val:]

    # Scale features
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    if model_type == "BiLSTM":
        model = build_bilstm(input_shape, learning_rate)
    else:
        model = build_lstm(input_shape, learning_rate)

    # Train
    st.subheader("Training")
    with st.spinner(f"Training {model_type} for {epochs} epochs..."):
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=int(epochs), batch_size=int(batch_size), verbose=0,
            callbacks=[es]
        )
    st.success("Training complete.")

    # Evaluate
    preds_scaled = model.predict(X_test_scaled, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    st.subheader("Evaluation on Test Set")
    st.write({"MAE": round(mae, 3), "RMSE": round(rmse, 3), "Samples": len(y_test)})

    # Plot 1: Historical temperature
    st.subheader("Historical Temperature")
    fig1 = plt.figure(figsize=(10, 3))
    plt.plot(df_raw["date"], df_raw["temperature"])
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.tight_layout()
    st.pyplot(fig1)

    # Plot 2: Predictions vs Actual (scatter on test)
    st.subheader("Predictions vs Actual (Test Set)")
    fig2 = plt.figure(figsize=(5, 5))
    plt.scatter(y_test, preds, alpha=0.6)
    minv, maxv = float(min(y_test.min(), preds.min())), float(max(y_test.max(), preds.max()))
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    st.pyplot(fig2)

    # Future Forecast (iterative)
    st.subheader("Future Forecast")
    last_seq = df[feature_cols].tail(sequence_length).values.reshape(1, sequence_length, -1)
    future_df = iterative_forecast(model, scaler_X, scaler_y, last_seq, feature_cols, df["date"].iloc[-1], int(forecast_days))

    st.dataframe(future_df, use_container_width=True)

    # Plot 3: Next N days forecast line
    fig3 = plt.figure(figsize=(10, 3))
    plt.plot(future_df["date"], future_df["predicted_temperature"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Predicted Temp (°C)")
    plt.tight_layout()
    st.pyplot(fig3)

    # Footer info
    if lat is not None and lon is not None:
        st.caption(f"Coordinates: {lat:.4f}, {lon:.4f}")
    st.caption("Data source: WeatherAPI.com | Models: LSTM & BiLSTM (Keras)")

else:
    st.info("Fill the sidebar, then click **Run Forecast**.")

