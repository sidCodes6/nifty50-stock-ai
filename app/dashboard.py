"""
Stock-AI: Professional NSE Stock Price Prediction & Live Intelligence Dashboard
Multivariate Log-Return LSTM Architecture + Real-Time Intraday Streaming Engine
"""

import os
import sys
import math
import time
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Ensure root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow.keras.models import load_model
from src.models.lstm import (
    download_market_data,
    engineer_multivariate_features,
    predict_next_day,
    train_and_save,
    compute_rsi,
    FEATURE_COLS,
    TARGET_COL,
    ARTIFACT_VERSION
)
from src.advisory import evaluate_stock_position, send_position_advisory_email, calculate_rsi

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stock-AI | Multivariate Log-Return Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Executive Dark Theme CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Top Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 13.5px;
        margin-top: 4px;
    }

    /* Prediction Hero Showcase Banner */
    .prediction-hero-card {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1.5px solid rgba(96, 165, 250, 0.3);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 15px 35px -10px rgba(37, 99, 235, 0.2);
        position: relative;
        overflow: hidden;
    }

    .hero-subcard {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .hero-subcard:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    .subcard-label {
        font-size: 11.5px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .subcard-val {
        font-size: 28px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 6px 0 2px 0;
    }
    .subcard-move {
        font-size: 13px;
        font-weight: 600;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Signal Badges */
    .badge-bullish {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    .badge-bearish {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.5px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Demo Investor Login Gate
# ---------------------------------------------------------
if "logged_in" not in st.session_state and st.query_params.get("ticket"):
    dashboard_ticket = st.query_params.get("ticket")
    st.query_params.clear()
    try:
        handoff_response = requests.post(
            os.getenv("AUTH_BACKEND_URL", "http://127.0.0.1:5000") + "/api/session-ticket/consume",
            json={"ticket": dashboard_ticket}, timeout=5,
        )
        handoff_response.raise_for_status()
        handoff_user = handoff_response.json()["user"]
        st.session_state.logged_in = True
        st.session_state.user_name = handoff_user.get("email", "Investor").split("@")[0]
        st.session_state.user_email = handoff_user["email"]
    except (requests.RequestException, KeyError, ValueError):
        st.error("Your sign-in link expired. Please sign in again.")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_url = os.getenv("LOGIN_URL", "http://localhost:5000/")
    st.markdown(f'<meta http-equiv="refresh" content="0;url={login_url}">', unsafe_allow_html=True)
    st.info("Redirecting to secure sign-in…")
    st.stop()

user_name = st.session_state.get("user_name", "Investor")
user_email = st.session_state.get("user_email", "investor@example.com")

# ---------------------------------------------------------
# Supported Assets
# ---------------------------------------------------------
STOCKS = {
    "NIFTY 50 (^NSEI)": {"ticker": "^NSEI", "name": "NIFTY 50 Index", "currency": "₹"},
    "Reliance Industries (RELIANCE.NS)": {"ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd", "currency": "₹"},
    "Tata Consultancy Services (TCS.NS)": {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "currency": "₹"},
    "HDFC Bank (HDFCBANK.NS)": {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "currency": "₹"},
}

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Secrets fallback for SMTP
email_secrets = {}
try:
    if hasattr(st, "secrets") and "email" in st.secrets:
        email_secrets = st.secrets["email"]
except Exception:
    email_secrets = {}
default_smtp_sender = email_secrets.get("sender_email", "") if isinstance(email_secrets, dict) else ""
default_smtp_password = email_secrets.get("app_password", "") if isinstance(email_secrets, dict) else ""


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_multivariate_artifacts(ticker: str):
    """
    Loads saved multivariate model (.keras), feature scaler (.pkl), and target scaler (.pkl).
    """
    safe_ticker = ticker.replace("^", "").replace(".", "_").lower()
    
    specific_model = os.path.join(MODELS_DIR, f"lstm_model_{safe_ticker}.keras")
    default_model = os.path.join(MODELS_DIR, "lstm_model.keras")
    # Only the NIFTY index may use the legacy default filenames. Never apply
    # an index model/scaler to an unrelated stock.
    model_path = specific_model if os.path.exists(specific_model) else (
        default_model if ticker == "^NSEI" else None
    )

    specific_f_scaler = os.path.join(MODELS_DIR, f"feature_scaler_{safe_ticker}.pkl")
    default_f_scaler = os.path.join(MODELS_DIR, "feature_scaler.pkl")
    feat_scaler_path = specific_f_scaler if os.path.exists(specific_f_scaler) else (
        default_f_scaler if ticker == "^NSEI" else None
    )

    specific_t_scaler = os.path.join(MODELS_DIR, f"target_scaler_{safe_ticker}.pkl")
    default_t_scaler = os.path.join(MODELS_DIR, "target_scaler.pkl")
    targ_scaler_path = specific_t_scaler if os.path.exists(specific_t_scaler) else (
        default_t_scaler if ticker == "^NSEI" else None
    )

    metrics_path = os.path.join(MODELS_DIR, f"{safe_ticker}_metrics.pkl")
    if (
        not model_path or not feat_scaler_path or not targ_scaler_path
        or not os.path.exists(model_path)
        or not os.path.exists(feat_scaler_path)
        or not os.path.exists(targ_scaler_path)
        or not os.path.exists(metrics_path)
    ):
        return None, None, None

    # Existing models were trained with misaligned sequences and must be rebuilt.
    try:
        metadata = joblib.load(metrics_path)
    except Exception:
        return None, None, None
    if metadata.get("artifact_version") != ARTIFACT_VERSION or metadata.get("ticker") != ticker:
        return None, None, None

    model = load_model(model_path)
    feature_scaler = joblib.load(feat_scaler_path)
    target_scaler = joblib.load(targ_scaler_path)
    return model, feature_scaler, target_scaler


@st.cache_data(ttl=300, show_spinner=False)
def fetch_cached_market_data(ticker: str) -> pd.DataFrame:
    """Fetches and cleans daily market data with a 5-minute TTL cache."""
    return download_market_data(ticker=ticker, start_date="2020-01-01", save_raw=False)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    # 1. Logged-in Investor Profile Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 16px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; border-radius: 50%; width: 38px; height: 38px; 
                        display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px;">
                {user_name[:1].upper()}
            </div>
            <div style="flex: 1; min-width: 0;">
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    Welcome, {user_name}
                </div>
                <div style="font-size: 12px; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {user_email}
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px;">
            <span style="color: #64748b;">Access: <strong style="color: #60a5fa;">Demo Investor</strong></span>
            <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 2px 8px; border-radius: 10px; font-weight: 600;">ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Log Out", use_container_width=True, key="sidebar_logout_btn"):
        st.session_state.logged_in = False
        st.session_state.pop("user_name", None)
        st.session_state.pop("user_email", None)
        st.rerun()

    st.markdown("### 📊 Active Asset & Model Scope")
    
    selected_stock_label = st.selectbox(
        "Select Stock / Index:",
        options=list(STOCKS.keys()),
        index=0  # NIFTY 50 by default
    )
    selected_info = STOCKS[selected_stock_label]
    ticker = selected_info["ticker"]
    stock_name = selected_info["name"]

    history_window = st.select_slider(
        "Historical Chart Lookback:",
        options=[60, 90, 120, 250],
        value=90,
        help="Number of past daily sessions to display on forecast chart"
    )

    st.markdown("---")
    # 2. Live Streaming Inference Engine
    st.markdown("### ⚡ Live AI Streaming Engine")
    live_tracker_enabled = st.toggle(
        "🔴 Start Live AI Tracker",
        value=False,
        help="Continuously poll live 1-minute intraday candle every 10s, reconstruct 4-feature sequence, and stream new predictions in real-time."
    )
    if live_tracker_enabled:
        st.caption("🟢 **Streaming Status:** Active polling every 10s")
    else:
        st.caption("⚪ **Streaming Status:** Inactive (Click toggle to begin)")

    st.markdown("---")
    # 3. Email Price Alert Settings
    st.markdown("### 🔔 Email Price Alert Settings")
    st.caption("Receive automated email alerts directly to your inbox when key price triggers occur.")
    
    alert_target_price = st.number_input(
        "Alert Target Price (₹):",
        min_value=0.0,
        value=0.0,
        step=50.0,
        format="%.2f",
        help="Set your price alert target threshold"
    )

    alert_condition = st.selectbox(
        "Alert Trigger Rule:",
        [
            "When Price Crosses Target (Breakout)",
            "When Sell Signal or Stop-Loss Triggered",
            "Send Instant Market Price Briefing"
        ]
    )

    with st.expander("⚙️ Email SMTP Configuration"):
        smtp_sender = st.text_input(
            "Sender Gmail:",
            value=default_smtp_sender,
            placeholder="yourname@gmail.com",
            help="Your Gmail address for sending SMTP alerts"
        )
        smtp_password = st.text_input(
            "Gmail App Password:",
            value=default_smtp_password,
            type="password",
            placeholder="16-character App Password",
            help="Generate at: https://myaccount.google.com/apppasswords"
        )
        receiver_email = st.text_input(
            "Alert Destination Email:",
            value=user_email,
            help="Linked to your login account"
        )

    send_test_alert = st.button("🚀 Send Test Price Alert to Email", use_container_width=True)

    st.markdown("---")
    with st.expander("🛠️ Retrain Model / Advanced"):
        st.write("Train or retrain Multivariate LSTM directly from dashboard:")
        train_epochs = st.number_input("Training Epochs:", min_value=10, max_value=100, value=30, step=5)
        if st.button(f"⚡ Retrain Model for {ticker}", use_container_width=True):
            with st.spinner(f"Training 4-feature LSTM for {ticker} ({train_epochs} epochs)..."):
                try:
                    train_and_save(ticker=ticker, epochs=int(train_epochs), batch_size=32, patience=5)
                    st.cache_resource.clear()
                    st.success(f"Model trained and saved for {ticker}!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Training failed: {ex}")


# ---------------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------------
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <h1 class="header-title">STOCK-AI MULTIVARIATE INTELLIGENCE PLATFORM</h1>
        <div class="header-subtitle">
            Stationary Log-Return LSTM Neural Network &bull; <strong>{stock_name} ({ticker})</strong> &bull;
            Logged in as <strong>{user_name}</strong>
        </div>
    </div>
    <div style="text-align: right;">
        <span style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #93c5fd; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">
            📡 REAL-TIME INTRADAY FEED (1m)
        </span>
        <div style="color: #64748b; font-size: 12px; margin-top: 6px;">
            Updated: {datetime.now().strftime("%d %b %Y, %H:%M:%S IST")}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Pipeline & Model Artifact Loading
# ---------------------------------------------------------
with st.spinner(f"Fetching real-time market data & preparing multivariate features for {ticker}..."):
    try:
        df_hist = fetch_cached_market_data(ticker)
    except Exception as e:
        st.error(f"Failed to fetch market data: {e}")
        st.stop()

if df_hist is None or len(df_hist) < 100:
    st.error("Insufficient market data available for prediction.")
    st.stop()

# 1. Fetch real-time live price via 1-minute intraday candle (bypassing fast_info cache)
try:
    intraday_data = yf.download(ticker, period="1d", interval="1m", progress=False)
    if isinstance(intraday_data.columns, pd.MultiIndex):
        intraday_data.columns = intraday_data.columns.get_level_values(0)
    live_price = float(intraday_data["Close"].iloc[-1])
except Exception:
    live_price = float(df_hist["Close"].iloc[-1])

prev_close = float(df_hist["Close"].iloc[-2])
day_change = live_price - prev_close
day_change_pct = (day_change / prev_close) * 100
latest_date = df_hist.index[-1]

# Load cached multivariate artifacts
model, feature_scaler, target_scaler = load_multivariate_artifacts(ticker)

if model is None or feature_scaler is None or target_scaler is None:
    st.warning(f"⚠️ Model artifacts not found. Automatically initializing baseline multivariate training for **{ticker}**...")
    with st.spinner("Training baseline multivariate model (this happens only once)..."):
        train_and_save(ticker=ticker, epochs=25, batch_size=32, patience=5)
        st.cache_resource.clear()
        st.rerun()

# 2. Run initial Multivariate Inference
try:
    inference_result = predict_next_day(
        live_price=live_price,
        recent_df=df_hist,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        time_step=60
    )
except Exception as infer_err:
    st.error(f"Inference error: {infer_err}")
    st.stop()

tomorrow_predicted_price = inference_result["tomorrow_predicted_price"]
predicted_log_return = inference_result["predicted_log_return"]
expected_change = inference_result["expected_change"]
expected_pct = inference_result["expected_pct"]
current_rsi = inference_result["rsi_14"]
current_sma_ratio = inference_result["sma_20_ratio"]
current_volatility = inference_result["volatility_10"]

is_bullish = predicted_log_return >= 0
trend_label = "🟢 BULLISH (T+1 LONG)" if is_bullish else "🔴 BEARISH (T+1 SHORT)"
pred_move_color = "#34d399" if expected_change >= 0 else "#f87171"
live_change_color = "#34d399" if day_change >= 0 else "#f87171"

# Target date for next trading session (skips weekends)
today_dt = datetime.now()
days_to_add = 3 if today_dt.weekday() == 4 else (2 if today_dt.weekday() == 5 else 1)
next_session_date = today_dt + timedelta(days=days_to_add)
next_day_date_str = next_session_date.strftime("%A, %d %B %Y")

# Position Advisory Snapshot
default_buy_price = round(live_price * 0.98, 2)
mock_forecast_df = pd.DataFrame({
    "Date": [next_session_date],
    "Predicted_Close": [tomorrow_predicted_price]
})
advisory = evaluate_stock_position(
    ticker=ticker,
    stock_name=stock_name,
    live_price=live_price,
    buy_price=default_buy_price,
    quantity=10,
    stop_loss_pct=-2.5,
    take_profit_pct=4.0,
    bearish_threshold_pct=-1.5,
    forecast_df=mock_forecast_df,
    hist_close_series=df_hist["Close"]
)


# ---------------------------------------------------------
# Dynamic UI Helper Functions
# ---------------------------------------------------------
def render_hero_html(stock_name, ticker, live_price, day_change, day_change_pct, live_change_color,
                     tomorrow_predicted_price, expected_change, expected_pct, pred_move_color,
                     predicted_log_return, next_day_date_str, is_bullish, trend_label,
                     current_rsi, current_sma_ratio, current_volatility, advisory, is_live=False):
    live_badge = """<span style="background: rgba(16, 185, 129, 0.25); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; padding: 4px 12px; border-radius: 20px; font-size: 11.5px; font-weight: 700; letter-spacing: 0.5px;">🔴 LIVE STREAMING ACTIVE (10s Cycle)</span>""" if is_live else """<span style="background: rgba(37, 99, 235, 0.25); border: 1px solid rgba(96, 165, 250, 0.4); color: #93c5fd; padding: 4px 12px; border-radius: 20px; font-size: 11.5px; font-weight: 700; letter-spacing: 0.5px;">⚡ MULTIVARIATE LOG-RETURN INFERENCE</span>"""

    return f"""<div class="prediction-hero-card">
<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 20px;">
<div>
{live_badge}
<h2 style="margin: 8px 0 2px 0; font-size: 26px; font-weight: 800; color: #ffffff;">
{stock_name} ({ticker}) AI Price Forecast
</h2>
<div style="font-size: 13px; color: #94a3b8;">
Statistically stationary return modeling: <code>P(t+1) = P(t) &times; exp(&rcirc;)</code> evaluated on 60-day feature window
</div>
</div>
<div style="text-align: right; background: rgba(0,0,0,0.3); padding: 10px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06);">
<div style="font-size: 11.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Real-Time 1-Minute Intraday Spot</div>
<div style="font-size: 24px; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace;">
₹{live_price:,.2f}
</div>
<div style="font-size: 12.5px; font-weight: 600; color: {live_change_color};">
{day_change:+,.2f} ({day_change_pct:+.2f}%) vs Previous Close
</div>
</div>
</div>
<!-- Key Prediction Figures Grid -->
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 20px;">
<div class="hero-subcard" style="border-left: 4px solid #60a5fa;">
<div class="subcard-label">TOMORROW'S PREDICTED PRICE ({next_day_date_str})</div>
<div class="subcard-val" style="color: #60a5fa;">₹{tomorrow_predicted_price:,.2f}</div>
<div class="subcard-move" style="color: {pred_move_color};">
Expected Move: {expected_change:+,.2f} ({expected_pct:+.2f}%)
</div>
</div>
<div class="hero-subcard" style="border-left: 4px solid #a78bfa;">
<div class="subcard-label">PREDICTED LOG RETURN (T+1 HORIZON)</div>
<div class="subcard-val" style="color: #a78bfa;">{predicted_log_return:+.4f}</div>
<div class="subcard-move" style="color: #cbd5e1;">
Continuously Compounded: {predicted_log_return * 100:+.2f}%
</div>
</div>
<div class="hero-subcard" style="border-left: 4px solid {'#10b981' if is_bullish else '#ef4444'};">
<div class="subcard-label">DIRECTIONAL MOMENTUM SIGNAL</div>
<div style="margin-top: 10px; margin-bottom: 6px;">
<span class="{'badge-bullish' if is_bullish else 'badge-bearish'}" style="font-size: 18px; padding: 8px 18px;">
{trend_label}
</span>
</div>
<div style="font-size: 12px; color: #94a3b8;">
Status: <strong style="color: {advisory['theme_color']};">{advisory['status']}</strong> &bull; Advisory Confidence: <strong>HIGH</strong>
</div>
</div>
</div>
<!-- Technical Features Card Snapshot -->
<div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px;">
<div style="font-size: 11.5px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">
🔬 Active Multivariate Inputs (Current Model Features)
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; text-align: center;">
<div style="font-size: 11px; color: #94a3b8; font-weight: 700;">14-DAY WILDER RSI</div>
<div style="font-size: 18px; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{current_rsi:.1f}</div>
<div style="font-size: 11px; color: {'#34d399' if current_rsi < 70 and current_rsi > 30 else '#f87171'};">{advisory['rsi_status']}</div>
</div>
<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; text-align: center;">
<div style="font-size: 11px; color: #94a3b8; font-weight: 700;">20-DAY SMA RATIO</div>
<div style="font-size: 18px; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{current_sma_ratio:.4f}</div>
<div style="font-size: 11px; color: {'#34d399' if current_sma_ratio >= 1.0 else '#f87171'};">{'Above 20-SMA' if current_sma_ratio >= 1.0 else 'Below 20-SMA'}</div>
</div>
<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; text-align: center;">
<div style="font-size: 11px; color: #94a3b8; font-weight: 700;">10-DAY VOLATILITY</div>
<div style="font-size: 18px; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{current_volatility * 100:.2f}%</div>
<div style="font-size: 11px; color: #94a3b8;">Standard Deviation</div>
</div>
<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 10px 14px; text-align: center;">
<div style="font-size: 11px; color: #94a3b8; font-weight: 700;">CURRENT LOG RETURN</div>
<div style="font-size: 18px; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin: 4px 0;">{day_change_pct:+.2f}%</div>
<div style="font-size: 11px; color: {live_change_color};">Intraday Delta</div>
</div>
</div>
</div>
</div>"""

def render_kpi_metrics_to_container(container, spot_label, live_price, day_change, day_change_pct,
                                    pred_label, tomorrow_price, expected_change, expected_pct,
                                    pred_return, rsi_val):
    with container.container():
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                label=f"Live Intraday Spot ({spot_label})",
                value=f"₹{live_price:,.2f}",
                delta=f"{day_change:+,.2f} ({day_change_pct:+.2f}%)",
                delta_color="normal"
            )
        with k2:
            st.metric(
                label=f"Tomorrow's Predicted Price ({pred_label})",
                value=f"₹{tomorrow_price:,.2f}",
                delta=f"{expected_change:+,.2f} ({expected_pct:+.2f}%)",
                delta_color="normal"
            )
        with k3:
            st.metric(
                label="Predicted T+1 Log-Return",
                value=f"{pred_return:+.4f}",
                delta=f"{pred_return * 100:+.2f}% return",
                delta_color="normal"
            )
        with k4:
            st.metric(
                label="14-Day RSI Momentum Indicator",
                value=f"{rsi_val:.1f}",
                delta="Wilder EMA Smoothing",
                delta_color="off"
            )

def create_next_day_forecast_figure(hist_data, live_p, pred_p, pred_ret, current_vol):
    """Creates focused Next-Day (T+1) target chart."""
    recent = hist_data.tail(90).copy()
    
    fig = go.Figure()

    # Historical Close Line
    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent["Close"],
        mode="lines",
        name="Historical Daily Close",
        line=dict(color="#38bdf8", width=2.5),
        hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Close:</b> ₹%{y:,.2f}<extra></extra>"
    ))

    # Live Spot Tick Marker
    fig.add_trace(go.Scatter(
        x=[recent.index[-1]],
        y=[live_p],
        mode="markers",
        name="Current Spot Tick (1m)",
        marker=dict(size=10, color="#fbbf24", symbol="diamond"),
        hovertemplate="<b>Live Spot:</b> ₹%{y:,.2f}<extra></extra>"
    ))

    # Target Tomorrow T+1 Marker & Bridge
    tomorrow_dt = recent.index[-1] + timedelta(days=1 if recent.index[-1].weekday() < 4 else 3)
    bridge_x = [recent.index[-1], tomorrow_dt]
    bridge_y = [live_p, pred_p]

    fig.add_trace(go.Scatter(
        x=bridge_x,
        y=bridge_y,
        mode="lines+markers",
        name="Tomorrow T+1 AI Prediction",
        line=dict(color="#f43f5e", width=2.5, dash="dash"),
        marker=dict(size=9, color="#fb7185"),
        hovertemplate="<b>Tomorrow Target:</b> ₹%{y:,.2f}<br><b>Date:</b> %{x|%d %b %Y}<extra></extra>"
    ))

    # Volatility band around predicted target
    upper_target = pred_p * (1.0 + current_vol)
    lower_target = pred_p * (1.0 - current_vol)
    fig.add_trace(go.Scatter(
        x=[tomorrow_dt, tomorrow_dt],
        y=[upper_target, lower_target],
        mode="lines",
        name="1-Sigma Volatility Envelope",
        line=dict(color="rgba(244, 63, 94, 0.4)", width=4),
        hoverinfo="skip"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(11, 15, 25, 0.8)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=30, b=40),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", type="date"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickprefix="₹", tickformat=",.0f")
    )
    return fig


# ---------------------------------------------------------
# Dynamic Placeholders for In-Place Updates
# ---------------------------------------------------------
hero_placeholder = st.empty()
metrics_placeholder = st.empty()

# Initial render
hero_placeholder.markdown(
    render_hero_html(
        stock_name=stock_name,
        ticker=ticker,
        live_price=live_price,
        day_change=day_change,
        day_change_pct=day_change_pct,
        live_change_color=live_change_color,
        tomorrow_predicted_price=tomorrow_predicted_price,
        expected_change=expected_change,
        expected_pct=expected_pct,
        pred_move_color=pred_move_color,
        predicted_log_return=predicted_log_return,
        next_day_date_str=next_day_date_str,
        is_bullish=is_bullish,
        trend_label=trend_label,
        current_rsi=current_rsi,
        current_sma_ratio=current_sma_ratio,
        current_volatility=current_volatility,
        advisory=advisory,
        is_live=False
    ),
    unsafe_allow_html=True
)

render_kpi_metrics_to_container(
    container=metrics_placeholder,
    spot_label=latest_date.strftime("%d %b %Y"),
    live_price=live_price,
    day_change=day_change,
    day_change_pct=day_change_pct,
    pred_label=next_session_date.strftime("%d %b"),
    tomorrow_price=tomorrow_predicted_price,
    expected_change=expected_change,
    expected_pct=expected_pct,
    pred_return=predicted_log_return,
    rsi_val=current_rsi
)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Handle Test Email Alert Dispatch
# ---------------------------------------------------------
if send_test_alert:
    with st.spinner(f"Sending test price alert to {receiver_email}..."):
        success, msg = send_position_advisory_email(
            sender_email=smtp_sender,
            app_password=smtp_password,
            receiver_email=receiver_email,
            advisory=advisory,
            user_name=user_name
        )
        if success:
            st.toast("✅ Price Alert dispatched to email!", icon="📧")
            st.success(f"Alert sent to {receiver_email}!")
        else:
            st.toast(f"❌ {msg}", icon="⚠️")
            st.error(msg)


# ---------------------------------------------------------
# Focused Interactive Plotly Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Next-Day Price Target Chart",
    "🧪 Model Backtest & Metrics",
    "🛡️ Position & Risk Advisory",
    "⚙️ Multivariate Model Architecture"
])

# ---------------------------------------------------------
# TAB 1: Next-Day Price Target Chart
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Next-Day (T+1) Target Projection — {stock_name}")
    st.caption("Mathematically converted using logarithmic return: Tomorrow_Predicted_Price = Live_Spot * exp(Predicted_Log_Return).")

    chart_placeholder = st.empty()
    fig_next_day = create_next_day_forecast_figure(
        hist_data=df_hist,
        live_p=live_price,
        pred_p=tomorrow_predicted_price,
        pred_ret=predicted_log_return,
        current_vol=current_volatility
    )
    chart_placeholder.plotly_chart(
        fig_next_day,
        use_container_width=True,
        key="next_day_forecast_chart"
    )

# ---------------------------------------------------------
# TAB 2: Model Backtest & Metrics
# ---------------------------------------------------------
with tab2:
    st.subheader(f"Model Backtest: Out-of-Sample Evaluation — {stock_name}")
    st.caption("Rigorous verification on the 20% holdout test partition evaluating reconstructed prices from predicted returns.")

    safe_ticker = ticker.replace("^", "").replace(".", "_").lower()
    backtest_file = os.path.join(DATA_PROCESSED_DIR, f"{safe_ticker}_backtest.csv")
    metrics_file = os.path.join(MODELS_DIR, f"{safe_ticker}_metrics.pkl")

    has_precomputed = False
    metrics_dict = {}
    if os.path.exists(backtest_file) and os.path.exists(metrics_file):
        try:
            metrics_dict = joblib.load(metrics_file)
            has_precomputed = (
                metrics_dict.get("artifact_version") == ARTIFACT_VERSION
                and metrics_dict.get("ticker") == ticker
            )
        except Exception:
            has_precomputed = False

    if has_precomputed:
        backtest_df = pd.read_csv(backtest_file)
        backtest_df["Date"] = pd.to_datetime(backtest_df["Date"])
        test_metrics = metrics_dict["test_metrics"]
    else:
        st.info("Backtest metrics will refresh after this asset is retrained with the corrected prediction pipeline.")
        backtest_df = pd.DataFrame()
        test_metrics = {}

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Test Directional Accuracy", f"{test_metrics['directional_accuracy']:.2f}%" if "directional_accuracy" in test_metrics else "N/A", help="% of days trend direction (Up/Down) was correctly predicted")
    with m2:
        st.metric("Price RMSE", f"₹{test_metrics['rmse']:,.2f}" if "rmse" in test_metrics else "N/A")
    with m3:
        st.metric("Price MAE", f"₹{test_metrics['mae']:,.2f}" if "mae" in test_metrics else "N/A")
    with m4:
        st.metric("Test Accuracy (Within 1%)", f"{test_metrics['accuracy_1pct']:.2f}%" if "accuracy_1pct" in test_metrics else "N/A")

    if not backtest_df.empty:
        fig_backtest = go.Figure()
        fig_backtest.add_trace(go.Scatter(
            x=backtest_df["Date"],
            y=backtest_df["Actual_Close"],
            mode="lines",
            name="Actual Market Close",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Actual:</b> ₹%{y:,.2f}<extra></extra>"
        ))
        fig_backtest.add_trace(go.Scatter(
            x=backtest_df["Date"],
            y=backtest_df["Predicted_Close"],
            mode="lines",
            name="LSTM Predicted Close (from Returns)",
            line=dict(color="#f97316", width=2),
            hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Predicted:</b> ₹%{y:,.2f}<extra></extra>"
        ))
        fig_backtest.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(11, 15, 25, 0.8)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(family="Plus Jakarta Sans", color="#94a3b8"),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=30, b=40),
            height=460,
            legend=dict(orientation="h", y=1.02, x=1, bgcolor="rgba(15, 23, 42, 0.8)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", type="date"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)", tickprefix="₹", tickformat=",.0f")
        )
        st.plotly_chart(fig_backtest, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Position & Risk Advisory
# ---------------------------------------------------------
with tab3:
    st.subheader(f"Portfolio Risk & Advisory Status — {stock_name}")
    st.caption("Live position analysis combining Stop-Loss, Take-Profit targets, LSTM forecast, and RSI-14 momentum.")

    st.markdown(f"""
    <div style="background: {advisory['bg_color']}; border: 1.5px solid {advisory['border_color']}; border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="font-size: 18px; font-weight: 800; color: {advisory['theme_color']};">
            {advisory['badge_label']}
        </div>
        <div style="font-size: 14px; color: #cbd5e1; margin-top: 4px;">
            {advisory['summary_verdict']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    adv_c1, adv_c2 = st.columns(2)
    with adv_c1:
        st.markdown("#### 💼 Position Health Snapshot")
        st.write(f"• **Asset:** `{stock_name}` (`{ticker}`)")
        st.write(f"• **Live Market Price:** `₹{advisory['live_price']:,.2f}`")
        st.write(f"• **Calculated Entry Benchmark:** `₹{advisory['buy_price']:,.2f}`")
        st.write(f"• **Unrealized Gain / Return:** `₹{advisory['unrealized_pnl']:+,.2f}` (`{advisory['unrealized_pct']:+.2f}%`)")
        st.write(f"• **14-Period RSI:** `{current_rsi:.1f}` ({advisory['rsi_status']})")

    with adv_c2:
        st.markdown("#### 🎯 Execution Boundaries")
        st.write(f"• **Stop-Loss Price:** `₹{advisory['stop_loss_price']:,.2f}` (`{advisory['stop_loss_pct']:+.1f}%`)")
        st.write(f"• **Take-Profit Target:** `₹{advisory['take_profit_price']:,.2f}` (`{advisory['take_profit_pct']:+.1f}%`)")
        st.write(f"• **Tomorrow Predicted Move:** `{expected_pct:+.2f}%` (`₹{expected_change:+,.2f}`)")
        st.write(f"• **Predicted Log Return:** `{predicted_log_return:+.4f}`")
        st.write(f"• **Overall Health:** **{advisory['status']}**")

    st.markdown("---")
    st.markdown("#### 📋 Diagnostic Justification Reasons")
    for r in advisory["reasons"]:
        st.markdown(f"• {r}", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4: Architecture & Model Specs
# ---------------------------------------------------------
with tab4:
    st.subheader("Multivariate Log-Return LSTM Architecture")
    st.markdown("""
    #### 🛡️ Why Multivariate Log-Returns Outperform Raw Prices
    - **Stationarity:** Raw stock prices exhibit non-stationary random-walk trends, leading models to predict yesterday's close. Logarithmic returns:
      `Log_Return = ln(P_t / P_{t-1})` are mathematically stationary with stable mean and variance.
    - **4 Predictive Feature Channels:**
      1. `Log_Return`: Normalized price velocity.
      2. `RSI_14`: 14-period Wilder Relative Strength Index.
      3. `SMA_20_Ratio`: Distance from 20-day moving average (`Close / SMA_20`).
      4. `Volatility_10`: 10-day rolling standard deviation of returns.
    - **Target Variable (Zero Lookahead Leakage):**
      `Target_Next_Return = Log_Return.shift(-1)` captures the return of day $T+1$ without lookahead bias.
    - **Exact Price Reconstruction Formula:**
      $$\\text{Tomorrow\\_Predicted\\_Price} = \\text{Live\\_Price} \\times \\exp(\\hat{r}_{T+1})$$

    #### 🧠 Deep Stacked LSTM Architecture
    ```
    Input(shape=(60, 4))  # 60 timesteps x 4 features
      │
      ├──> LSTM Layer 1: 64 Units (return_sequences=True)
      ├──> Dropout (0.20)
      │
      ├──> LSTM Layer 2: 32 Units (return_sequences=False)
      ├──> Dropout (0.20)
      │
      ├──> Dense Layer 1: 16 Units (ReLU Activation)
      └──> Dense Output Layer: 1 Unit (Predicted Log-Return)
    ```
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #475569; font-size: 13px; margin-top: 50px; padding: 20px 0; border-top: 1px solid rgba(255,255,255,0.05);">
    Stock-AI Enterprise Suite &bull; Multivariate Log-Return Architecture &bull; Built with Python, Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Real-Time Live Streaming Inference Loop
# ---------------------------------------------------------
if live_tracker_enabled:
    st.markdown("---")
    live_banner_placeholder = st.empty()
    cycle_index = 0

    while True:
        cycle_index += 1
        try:
            # 1. Fetch real-time live market price via 1-minute intraday candle (bypasses fast_info cache)
            try:
                intraday_data = yf.download(ticker, period="1d", interval="1m", progress=False)
                if isinstance(intraday_data.columns, pd.MultiIndex):
                    intraday_data.columns = intraday_data.columns.get_level_values(0)
                live_price = float(intraday_data["Close"].iloc[-1])
            except Exception:
                live_price = float(df_hist["Close"].iloc[-1])

            # 2. Live Feature Engineering & 3D Sequence Construction
            df_work = df_hist.copy()
            if isinstance(df_work.columns, pd.MultiIndex):
                df_work.columns = df_work.columns.get_level_values(0)

            # Update or append current live price
            if not df_work.empty and df_work.index[-1].date() == datetime.now().date():
                df_work.iloc[-1, df_work.columns.get_loc("Close")] = live_price
            else:
                new_row = pd.DataFrame({"Close": [live_price]}, index=[pd.Timestamp.now()])
                df_work = pd.concat([df_work, new_row])

            df_work["Log_Return"] = np.log(df_work["Close"] / df_work["Close"].shift(1))
            df_work["RSI_14"] = compute_rsi(df_work["Close"], period=14)
            df_work["SMA_20_Ratio"] = df_work["Close"] / df_work["Close"].rolling(20).mean()
            df_work["Volatility_10"] = df_work["Log_Return"].rolling(10).std()

            feat_slice = df_work[FEATURE_COLS].dropna()
            last_60 = feat_slice.iloc[-60:].copy()
            live_features_scaled = feature_scaler.transform(last_60)

            # Strictly 3D Input: (1, 60, 4)
            X_input = live_features_scaled.reshape(1, 60, 4)

            # Run model prediction to output predicted log return
            pred_scaled = model.predict(X_input, verbose=0)
            live_pred_return = float(target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0])

            # Convert predicted return back to price: live_price * np.exp(predicted_log_return)
            live_tomorrow_price = float(live_price * np.exp(live_pred_return))
            live_expected_change = float(live_tomorrow_price - live_price)
            live_expected_pct = float((live_expected_change / live_price) * 100.0)

            live_rsi = float(last_60["RSI_14"].iloc[-1])
            live_sma_ratio = float(last_60["SMA_20_Ratio"].iloc[-1])
            live_vol = float(last_60["Volatility_10"].iloc[-1])

            live_day_change = live_price - prev_close
            live_day_change_pct = (live_day_change / prev_close) * 100
            live_is_bullish = live_pred_return >= 0
            live_trend_label = "🟢 BULLISH (T+1 LONG)" if live_is_bullish else "🔴 BEARISH (T+1 SHORT)"
            live_pred_move_col = "#34d399" if live_expected_change >= 0 else "#f87171"
            live_spot_change_col = "#34d399" if live_day_change >= 0 else "#f87171"

            # 3. Dynamic UI updates via placeholders
            updated_hero_html = render_hero_html(
                stock_name=stock_name,
                ticker=ticker,
                live_price=live_price,
                day_change=live_day_change,
                day_change_pct=live_day_change_pct,
                live_change_color=live_spot_change_col,
                tomorrow_predicted_price=live_tomorrow_price,
                expected_change=live_expected_change,
                expected_pct=live_expected_pct,
                pred_move_color=live_pred_move_col,
                predicted_log_return=live_pred_return,
                next_day_date_str=next_day_date_str,
                is_bullish=live_is_bullish,
                trend_label=live_trend_label,
                current_rsi=live_rsi,
                current_sma_ratio=live_sma_ratio,
                current_volatility=live_vol,
                advisory=advisory,
                is_live=True
            )
            hero_placeholder.markdown(updated_hero_html, unsafe_allow_html=True)

            render_kpi_metrics_to_container(
                container=metrics_placeholder,
                spot_label=f"Live {datetime.now().strftime('%H:%M:%S')}",
                live_price=live_price,
                day_change=live_day_change,
                day_change_pct=live_day_change_pct,
                pred_label=next_session_date.strftime("%d %b"),
                tomorrow_price=live_tomorrow_price,
                expected_change=live_expected_change,
                expected_pct=live_expected_pct,
                pred_return=live_pred_return,
                rsi_val=live_rsi
            )

            # Update Plotly Chart container in place
            updated_chart = create_next_day_forecast_figure(
                hist_data=df_hist,
                live_p=live_price,
                pred_p=live_tomorrow_price,
                pred_ret=live_pred_return,
                current_vol=live_vol
            )
            chart_placeholder.plotly_chart(
                updated_chart,
                use_container_width=True,
                key="next_day_forecast_chart"
            )

            # Live banner display
            current_time_str = datetime.now().strftime("%H:%M:%S IST")
            live_banner_placeholder.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
                        border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 12px 20px;
                        display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="display: inline-block; width: 10px; height: 10px; background: #34d399; border-radius: 50%; box-shadow: 0 0 10px #34d399;"></span>
                    <strong style="color: #34d399; font-size: 14px;">MULTIVARIATE STREAMING INFERENCE ACTIVE (Cycle #{cycle_index})</strong>
                </div>
                <div style="font-size: 13px; color: #cbd5e1;">
                    Polled: <strong>{current_time_str}</strong> &bull; Spot: <strong>₹{live_price:,.2f}</strong> &bull; Tomorrow T+1 AI: <strong>₹{live_tomorrow_price:,.2f}</strong> ({live_pred_return * 100:+.2f}%) &bull; Next cycle in 10s
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as loop_err:
            st.warning(f"⚠️ Live inference stream warning: {loop_err}")

        # Rate Limiting: 10 seconds interval
        time.sleep(10)
