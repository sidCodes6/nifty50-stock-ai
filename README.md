# Stock-AI: Professional NSE Stock Price Prediction & Live Alert Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, production-ready Deep Learning forecasting suite, authentication gateway, and executive portfolio advisory dashboard for Indian National Stock Exchange (NSE) equities and indices (NIFTY 50, Reliance, TCS, HDFC Bank). 

Engineered with a decoupled 3-layer stacked LSTM architecture, strict data leakage elimination, calendar-aware business trading day projections, Google/Microsoft OAuth authentication, and an intelligent **"Safe to Hold" vs. "Sell Now"** email advisory system.

---

## 🌟 Key Upgrades & Features

### 1. 🔐 Enterprise Authentication Gateway (`src/auth.py`)
* **Google & Microsoft OAuth (OIDC):** Native Streamlit authentication via `Authlib>=1.3.2` connecting to Google Cloud OAuth 2.0 and Microsoft Entra ID (Azure AD).
* **Smart Local / Demo Fallback:** Immediate access via quick sign-in with your name, registered email address (Gmail / Outlook), and provider choice. Stores session state securely without requiring external cloud setup.
* **User Profile Header & Logout:** Displays active user badge, email address, provider indicator, and a one-click session logout button in the sidebar.

### 2. 🛡️ Smart Stock Advisory Engine ("SAFE TO HOLD" vs. "SELL NOW") (`src/advisory.py`)
* **Live Portfolio Risk Monitoring:** Links your logged-in email and tracks your entry price, shares held, Stop-Loss threshold (%), Take-Profit target (%), and AI Bearish Alert threshold (%).
* **Multi-Factor Decision Engine:**
  - **🟢 STATUS: SAFE TO HOLD (OR ACCUMULATE):** Live market price is above Stop-Loss, LSTM multi-step forecast trajectory is stable/bullish, and 14-day RSI is balanced (< 75).
  - **🔴 STATUS: ACTION REQUIRED — SELL / BOOK PROFITS / CUT LOSS:** Triggered when Stop-Loss is breached, OR the LSTM model predicts an impending bearish reversal beyond the user threshold, OR Take-Profit target is hit with RSI overbought exhaustion.
* **Automated & On-Demand HTML Emailing (`smtplib`):** Color-coded executive reports dispatched directly to the user's email via Gmail SMTP, with an auto-send trigger whenever a SELL condition is detected.

### 3. 🧠 Machine Learning & Data Leakage Prevention (`src/models/lstm.py`)
* **Leakage-Free Preprocessing:** Data is split 80% Train / 20% Test prior to scaling. `MinMaxScaler` is fitted strictly on the training partition. The last 60 days of training data are prepended to the test set for seamless lookback continuity without data gaps.
* **Resilient MultiIndex Handling:** Robust flattening of `yfinance` MultiIndex headers so `df['Close']` always resolves to a clean 1D Series.
* **Zero Dropped Timestamps:** Indexing uses `range(time_step, len(dataset))` so the most recent market close is accurately retained.
* **Calendar Business Days:** Multi-step projections use `pd.bdate_range()`, ensuring non-trading weekend sessions (Saturdays & Sundays) are skipped.

---

## 📂 Project Structure

```
stock-ai/
├── .streamlit/
│   └── secrets.toml.example # OAuth & SMTP configuration template
├── data/
│   ├── raw/                 # Downloaded raw market CSV snapshots
│   └── processed/           # Processed datasets and backtest holdout evaluations
├── models/                  # Serialized models (.keras) and fitted scalers (.pkl)
│   ├── lstm_model.keras     # Default trained LSTM network (^NSEI)
│   ├── scaler.pkl           # Fitted MinMaxScaler
│   └── *_metrics.pkl        # Precomputed performance evaluations
├── src/
│   ├── __init__.py
│   ├── auth.py              # Google & Microsoft OAuth with instant demo fallback
│   ├── advisory.py          # Portfolio decision engine (Safe vs Sell) & HTML emailer
│   └── models/
│       ├── __init__.py
│       └── lstm.py          # Refactored LSTM backend training & evaluation pipeline
├── app/
│   └── dashboard.py         # Executive Streamlit frontend & Alert subsystem
├── .gitignore               # Ignores sensitive secrets.toml and build files
├── requirements.txt         # Core dependencies
└── README.md                # Comprehensive documentation
```

---

## ⚡ Quickstart & Installation

### 1. Environment Setup
```bash
git clone https://github.com/your-username/stock-ai.git
cd stock-ai
pip install -r requirements.txt
```

### 2. Configure Credentials (Optional)
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
```toml
[auth.google]
client_id = "YOUR_GOOGLE_CLIENT_ID"
client_secret = "YOUR_GOOGLE_CLIENT_SECRET"

[auth.microsoft]
client_id = "YOUR_MICROSOFT_CLIENT_ID"
client_secret = "YOUR_MICROSOFT_CLIENT_SECRET"

[email]
sender_email = "your-email@gmail.com"
app_password = "your-16-char-gmail-app-password"
```
*(Note: If cloud OAuth credentials are not entered, the application automatically enables the smart Quick Sign-In fallback).*

### 3. Backend Model Training
Train the default NIFTY 50 (`^NSEI`) model and generate evaluation metrics:
```bash
python src/models/lstm.py
```

### 4. Launch the Executive Dashboard
```bash
streamlit run app/dashboard.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📊 Dashboard Tabs & Capabilities

1. **Login Landing Page:** Modern glassmorphic gateway with Google & Microsoft identity integration.
2. **Executive KPI Row:** Live market price (₹ and % delta), Next-Day AI forecast, N-Day target price, and Bullish/Bearish signal badge.
3. **Live Advisory Banner:** Dynamic color-coded banner displaying **🟢 SAFE TO HOLD** vs. **🔴 TIME TO SELL**, net unrealized P&L, diagnostic reasons, and immediate email dispatch buttons.
4. **Tab 1 (Future Forecast):** Historical price curve with 30-day dotted LSTM projection, dynamic volatility envelope, and interactive range slider.
5. **Tab 2 (Model Backtest):** 20% holdout test evaluation with live RMSE, MAE, R², 1% tolerance accuracy, and residual error bars.
6. **Tab 3 (Forecast Table):** Day-by-day projected return breakdown with one-click CSV export.
7. **Tab 4 (Portfolio Health & Advisory):** Complete balance sheet of user holdings, capital invested, liquidation value, and decision rules.
8. **Tab 5 (Architecture & Specs):** Full engineering overview of neural layers and data pipelines.

---

## 🛡️ License
Distributed under the MIT License.
