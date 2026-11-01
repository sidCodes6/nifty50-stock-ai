"""
Portfolio Position & Email Advisory Engine
Evaluates: Live Price vs Buy Price, Stop-Loss/Take-Profit, LSTM Projections & 14-Day RSI.
Classifies stock into: 'SAFE TO HOLD' vs 'SELL NOW' and dispatches rich HTML alerts.
"""

import math
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """
    Computes standard 14-period Relative Strength Index (RSI).
    """
    if len(series) < period + 1:
        return 50.0

    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

    if avg_loss == 0 or math.isnan(avg_loss):
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(round(rsi, 2))


def evaluate_stock_position(
    ticker: str,
    stock_name: str,
    live_price: float,
    buy_price: float,
    quantity: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    bearish_threshold_pct: float,
    forecast_df: pd.DataFrame,
    hist_close_series: pd.Series
) -> Dict[str, Any]:
    """
    Comprehensive Decision Engine evaluating position health:
    Combines P&L, Stop-Loss/Take-Profit boundaries, LSTM forecasts, and 14-Day RSI.
    """
    # 1. P&L & Position Calculations
    invested_capital = buy_price * quantity
    current_value = live_price * quantity
    unrealized_pnl = current_value - invested_capital
    unrealized_pct = ((live_price - buy_price) / buy_price) * 100.0 if buy_price > 0 else 0.0

    stop_loss_price = buy_price * (1.0 + (stop_loss_pct / 100.0))
    take_profit_price = buy_price * (1.0 + (take_profit_pct / 100.0))

    # 2. LSTM Trajectory Analysis
    next_day_forecast = float(forecast_df["Predicted_Close"].iloc[0])
    next_day_move_pct = ((next_day_forecast - live_price) / live_price) * 100.0

    n_day_target = float(forecast_df["Predicted_Close"].iloc[-1])
    n_day_move_pct = ((n_day_target - live_price) / live_price) * 100.0
    forecast_days = len(forecast_df)

    # 3. Technical Indicator (RSI-14)
    rsi_14 = calculate_rsi(hist_close_series, period=14)
    rsi_status = "Oversold" if rsi_14 < 30 else ("Overbought" if rsi_14 > 70 else "Neutral Momentum")

    # 4. Intelligent Classification: SAFE TO HOLD vs SELL NOW
    sell_triggers: List[str] = []
    safe_reasons: List[str] = []

    # Rule 1: Stop-Loss Breach Check
    if live_price <= stop_loss_price or unrealized_pct <= stop_loss_pct:
        sell_triggers.append(
            f"🚨 <strong>Stop-Loss Breached:</strong> Current price (₹{live_price:,.2f}) has breached your configured Stop-Loss threshold of {stop_loss_pct:+.1f}% (₹{stop_loss_price:,.2f}). Current position loss is {unrealized_pct:+.2f}% (-₹{abs(unrealized_pnl):,.2f}). Capital preservation dictates an immediate exit."
        )

    # Rule 2: AI Neural Bearish Reversal Warning
    # If the multi-step horizon forecast drops worse than user threshold (e.g. drop worse than -2%)
    if n_day_move_pct <= bearish_threshold_pct or next_day_move_pct <= bearish_threshold_pct:
        sell_triggers.append(
            f"📉 <strong>AI Bearish Reversal Warning:</strong> The LSTM neural forecast projects a decline of {n_day_move_pct:+.2f}% over the next {forecast_days} sessions (Target: ₹{n_day_target:,.2f}). This exceeds your risk tolerance of {bearish_threshold_pct:+.1f}%."
        )

    # Rule 3: Take-Profit Achieved with Technical Exhaustion / Impending Pullback
    if unrealized_pct >= take_profit_pct:
        if rsi_14 >= 68 or next_day_move_pct < 0:
            sell_triggers.append(
                f"🎯 <strong>Take-Profit Target Achieved + Overbought:</strong> Unrealized gain is {unrealized_pct:+.2f}% (+₹{unrealized_pnl:,.2f}), exceeding your target of {take_profit_pct:+.1f}%. RSI-14 is elevated at {rsi_14:.1f} ({rsi_status}) with AI signaling a potential near-term consolidation. Strongly recommend locking in profits."
            )

    # Decision Verdict
    if len(sell_triggers) > 0:
        status = "SELL NOW"
        badge_label = "🔴 ACTION REQUIRED — SELL / BOOK PROFITS"
        theme_color = "#ef4444"
        bg_color = "rgba(239, 68, 68, 0.12)"
        border_color = "rgba(239, 68, 68, 0.35)"
        summary_verdict = f"Sell or trim position in {stock_name} ({ticker}) immediately based on {len(sell_triggers)} risk trigger(s)."
        reasons = sell_triggers
    else:
        status = "SAFE TO HOLD"
        badge_label = "🟢 STATUS: SAFE TO HOLD (OR ACCUMULATE)"
        theme_color = "#10b981"
        bg_color = "rgba(16, 185, 129, 0.12)"
        border_color = "rgba(16, 185, 129, 0.35)"
        summary_verdict = f"Your position in {stock_name} ({ticker}) remains in a secure, bullish/stable corridor. No exit triggers active."
        
        safe_reasons.append(
            f"🛡️ <strong>Risk Boundary Intact:</strong> Current market price (₹{live_price:,.2f}) is trading comfortably above your Stop-Loss level of ₹{stop_loss_price:,.2f} ({stop_loss_pct:+.1f}%)."
        )
        safe_reasons.append(
            f"🧠 <strong>Neural Trend Stability:</strong> LSTM predicts {next_day_move_pct:+.2f}% next trading day and {n_day_move_pct:+.2f}% over {forecast_days} sessions (Target: ₹{n_day_target:,.2f})."
        )
        safe_reasons.append(
            f"📊 <strong>Balanced Momentum:</strong> 14-day RSI is at {rsi_14:.1f} ({rsi_status}), showing solid price structure without extreme overbought exhaustion."
        )
        reasons = safe_reasons

    return {
        "status": status,
        "badge_label": badge_label,
        "theme_color": theme_color,
        "bg_color": bg_color,
        "border_color": border_color,
        "summary_verdict": summary_verdict,
        "reasons": reasons,
        "ticker": ticker,
        "stock_name": stock_name,
        "live_price": live_price,
        "buy_price": buy_price,
        "quantity": quantity,
        "invested_capital": invested_capital,
        "current_value": current_value,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pct": unrealized_pct,
        "stop_loss_price": stop_loss_price,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_price": take_profit_price,
        "take_profit_pct": take_profit_pct,
        "next_day_forecast": next_day_forecast,
        "next_day_move_pct": next_day_move_pct,
        "n_day_target": n_day_target,
        "n_day_move_pct": n_day_move_pct,
        "forecast_days": forecast_days,
        "rsi_14": rsi_14,
        "rsi_status": rsi_status,
        "evaluated_at": datetime.now().strftime("%d %b %Y, %H:%M:%S IST")
    }


def send_position_advisory_email(
    sender_email: str,
    app_password: str,
    receiver_email: str,
    advisory: Dict[str, Any],
    user_name: str = "Investor"
) -> Tuple[bool, str]:
    """
    Sends a formatted color-coded HTML stock advisory report via Gmail SMTP.
    """
    if not sender_email or not app_password or not receiver_email:
        return False, "Sender email, App Password, and Receiver email are required."

    status = advisory["status"]
    is_safe = (status == "SAFE TO HOLD")
    banner_color = "#10b981" if is_safe else "#ef4444"
    banner_title = "🟢 STOCK POSITION IS SAFE TO HOLD" if is_safe else "🔴 URGENT ADVISORY: ACTION REQUIRED — SELL STOCK"
    pnl_color = "#34d399" if advisory["unrealized_pnl"] >= 0 else "#f87171"

    subject = f"[{'SAFE' if is_safe else 'URGENT SELL'}] Stock-AI Position Advisory: {advisory['stock_name']} ({advisory['ticker']})"

    reasons_html = "".join([f"<li style='margin-bottom: 8px; line-height: 1.5;'>{r}</li>" for r in advisory["reasons"]])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{subject}</title>
    </head>
    <body style="margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b0f19; color: #f8fafc;">
      <div style="max-width: 650px; margin: 0 auto; background: #111827; border: 1px solid #1f2937; border-radius: 14px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
        
        <!-- Header Banner -->
        <div style="background-color: {banner_color}; color: #ffffff; padding: 20px 24px; text-align: center;">
          <h2 style="margin: 0; font-size: 20px; font-weight: 800; letter-spacing: 0.5px;">
            {banner_title}
          </h2>
          <div style="font-size: 13px; opacity: 0.95; margin-top: 4px;">
            Stock-AI Neural Intelligence System &bull; Portfolio Health Check
          </div>
        </div>

        <div style="padding: 24px;">
          <!-- Greeting -->
          <p style="font-size: 15px; color: #cbd5e1; margin-top: 0;">
            Hello <strong>{user_name}</strong>,
          </p>
          <p style="font-size: 14px; color: #94a3b8; line-height: 1.5;">
            Here is your live position health evaluation for <strong>{advisory['stock_name']} ({advisory['ticker']})</strong> as of <strong>{advisory['evaluated_at']}</strong>.
          </p>

          <!-- Executive P&L Snapshot -->
          <div style="background: #1e293b; border-radius: 10px; padding: 16px 20px; margin: 20px 0; border: 1px solid #334155;">
            <table style="width: 100%; border-collapse: collapse;">
              <tr>
                <td style="color: #94a3b8; font-size: 13px; padding-bottom: 6px;">Live Market Price:</td>
                <td style="text-align: right; font-weight: bold; font-size: 18px; color: #ffffff;">₹{advisory['live_price']:,.2f}</td>
              </tr>
              <tr>
                <td style="color: #94a3b8; font-size: 13px; padding-bottom: 6px;">Average Buy Price:</td>
                <td style="text-align: right; font-weight: 600; font-size: 14px; color: #cbd5e1;">₹{advisory['buy_price']:,.2f} ({advisory['quantity']} shares)</td>
              </tr>
              <tr>
                <td style="color: #94a3b8; font-size: 13px; padding-bottom: 6px;">Unrealized P&amp;L:</td>
                <td style="text-align: right; font-weight: bold; font-size: 16px; color: {pnl_color};">
                  {advisory['unrealized_pnl']:+,.2f} ({advisory['unrealized_pct']:+.2f}%)
                </td>
              </tr>
              <tr>
                <td style="color: #94a3b8; font-size: 13px;">Total Position Value:</td>
                <td style="text-align: right; font-weight: bold; font-size: 14px; color: #ffffff;">₹{advisory['current_value']:,.2f}</td>
              </tr>
            </table>
          </div>

          <!-- Neural & Technical Signals Table -->
          <h4 style="color: #60a5fa; margin-bottom: 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">
            AI Forecast &amp; Technical Breakdown
          </h4>
          <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px;">
            <tr style="border-bottom: 1px solid #1f2937;">
              <td style="padding: 8px 0; color: #94a3b8;">Next-Day LSTM Forecast:</td>
              <td style="padding: 8px 0; text-align: right; font-weight: 600; color: #ffffff;">₹{advisory['next_day_forecast']:,.2f} ({advisory['next_day_move_pct']:+.2f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1f2937;">
              <td style="padding: 8px 0; color: #94a3b8;">{advisory['forecast_days']}-Day Target Forecast:</td>
              <td style="padding: 8px 0; text-align: right; font-weight: 600; color: #ffffff;">₹{advisory['n_day_target']:,.2f} ({advisory['n_day_move_pct']:+.2f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1f2937;">
              <td style="padding: 8px 0; color: #94a3b8;">Stop-Loss Boundary:</td>
              <td style="padding: 8px 0; text-align: right; color: #f87171; font-weight: 600;">₹{advisory['stop_loss_price']:,.2f} ({advisory['stop_loss_pct']:+.1f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #1f2937;">
              <td style="padding: 8px 0; color: #94a3b8;">Take-Profit Target:</td>
              <td style="padding: 8px 0; text-align: right; color: #34d399; font-weight: 600;">₹{advisory['take_profit_price']:,.2f} ({advisory['take_profit_pct']:+.1f}%)</td>
            </tr>
            <tr>
              <td style="padding: 8px 0; color: #94a3b8;">14-Day RSI Indicator:</td>
              <td style="padding: 8px 0; text-align: right; font-weight: 600; color: #38bdf8;">{advisory['rsi_14']:.1f} ({advisory['rsi_status']})</td>
            </tr>
          </table>

          <!-- Actionable Reasoning -->
          <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="font-weight: 700; color: {banner_color}; margin-bottom: 10px; font-size: 14px;">
              Actionable Recommendation Details:
            </div>
            <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 13px;">
              {reasons_html}
            </ul>
          </div>

          <p style="font-size: 12px; color: #64748b; line-height: 1.5; margin-bottom: 0;">
            ⚠️ <em>Disclaimer: Stock-AI projections and signals are machine learning algorithmic estimates based on historical sequences and are intended strictly for educational and informational analysis. Consult a certified financial advisor before placing market trades.</em>
          </p>
        </div>

        <!-- Footer -->
        <div style="background: #0f172a; padding: 14px 24px; text-align: center; border-top: 1px solid #1f2937; font-size: 11px; color: #475569;">
          Automated Report sent to {receiver_email} &bull; Stock-AI Neural Intelligence Platform
        </div>
      </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Stock-AI Advisory Engine <{sender_email}>"
    msg["To"] = receiver_email

    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True, f"Advisory email successfully dispatched to {receiver_email}!"
    except smtplib.SMTPAuthenticationError:
        return False, "Gmail SMTP Authentication Failed: Please verify that you are using a 16-character App Password (not your normal account password)."
    except Exception as e:
        return False, f"Failed to dispatch advisory email: {str(e)}"
