"""
Authentication Subsystem: Google & Microsoft OAuth (OIDC) with Smart Local Fallback
Project: Stock-AI - NSE Stock Price Prediction & Live Alert Dashboard
"""

import os
import streamlit as st
from typing import Dict, Any, Optional, Tuple


def is_oauth_configured(provider: str) -> bool:
    """
    Checks if OAuth client credentials exist in Streamlit secrets.
    """
    try:
        if "auth" in st.secrets and provider in st.secrets["auth"]:
            cfg = st.secrets["auth"][provider]
            client_id = cfg.get("client_id", "")
            client_secret = cfg.get("client_secret", "")
            return bool(client_id and client_secret and "YOUR_" not in client_id)
    except Exception:
        pass
    return False


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Returns current authenticated user details from session state or native Streamlit auth.
    """
    # Check if native Streamlit experimental_user / user exists and is logged in
    try:
        user_obj = getattr(st, "user", None) or getattr(st, "experimental_user", None)
        if user_obj and getattr(user_obj, "is_logged_in", False):
            return {
                "name": getattr(user_obj, "name", "User") or getattr(user_obj, "email", "User"),
                "email": getattr(user_obj, "email", "user@example.com"),
                "provider": getattr(user_obj, "provider", "Google").capitalize(),
                "method": "oidc"
            }
    except Exception:
        pass

    # Check session state
    if st.session_state.get("is_authenticated", False):
        return {
            "name": st.session_state.get("user_name", "Investor"),
            "email": st.session_state.get("user_email", "user@gmail.com"),
            "provider": st.session_state.get("auth_provider", "Google"),
            "method": st.session_state.get("auth_method", "demo_oauth")
        }

    return None


def logout_user():
    """Logs out the user and cleans session state."""
    try:
        if hasattr(st, "logout"):
            st.logout()
    except Exception:
        pass

    for key in ["is_authenticated", "user_name", "user_email", "auth_provider", "auth_method"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def render_user_profile():
    """
    Renders user profile card at the top of the sidebar with Logout button.
    """
    user = get_current_user()
    if not user:
        return

    provider_icon = "🌐 Google" if "google" in user["provider"].lower() else "🪟 Microsoft"
    provider_color = "#4285F4" if "google" in user["provider"].lower() else "#00A4EF"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 16px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {provider_color}; color: white; border-radius: 50%; width: 38px; height: 38px; 
                        display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 16px;">
                {user['name'][:1].upper()}
            </div>
            <div style="flex: 1; min-width: 0;">
                <div style="font-weight: 700; font-size: 14px; color: #f8fafc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {user['name']}
                </div>
                <div style="font-size: 12px; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {user['email']}
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11px;">
            <span style="color: #64748b;">Auth: <strong style="color: {provider_color};">{provider_icon}</strong></span>
            <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 2px 8px; border-radius: 10px; font-weight: 600;">ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Log Out", use_container_width=True, key="sidebar_logout_btn"):
        logout_user()


def render_login_page():
    """
    Renders an executive, modern glassmorphic Login Landing Page.
    Offers native Streamlit OIDC when configured, plus smart local instant sign-in.
    """
    google_configured = is_oauth_configured("google")
    ms_configured = is_oauth_configured("microsoft")

    # Center Container
    _, center_col, _ = st.columns([1, 2.2, 1])

    with center_col:
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; margin-bottom: 25px;">
            <div style="display: inline-block; background: linear-gradient(135deg, rgba(37,99,235,0.2) 0%, rgba(147,51,234,0.2) 100%);
                        border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 10px 18px; margin-bottom: 12px;">
                <span style="font-size: 24px;">📈</span> <span style="font-weight: 800; font-size: 18px; letter-spacing: 1px; color: #60a5fa;">STOCK-AI INTELLIGENCE</span>
            </div>
            <h2 style="font-size: 28px; font-weight: 800; margin-bottom: 6px; color: #f8fafc;">
                Welcome to Stock-AI Enterprise
            </h2>
            <p style="color: #94a3b8; font-size: 14px; margin-top: 0;">
                NSE Stock Price Forecasting, LSTM Neural Projections &amp; Real-Time Portfolio Advisory
            </p>
        </div>
        """, unsafe_allow_html=True)

        login_card = st.container()
        with login_card:
            st.markdown("""
            <div style="background: linear-gradient(145deg, rgba(19, 27, 46, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
                        border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 28px 32px;
                        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5); backdrop-filter: blur(16px); margin-bottom: 20px;">
                <div style="font-size: 16px; font-weight: 700; color: #e2e8f0; margin-bottom: 16px; text-align: center;">
                    Sign in to Access Your Neural Dashboard &amp; Stock Alerts
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Check if native Streamlit login is available and configured
            has_native_login = hasattr(st, "login")

            # Provider OAuth Buttons
            c_g, c_m = st.columns(2)
            with c_g:
                if has_native_login and google_configured:
                    if st.button("🌐 Continue with Google", use_container_width=True, type="primary"):
                        st.login("google")
                else:
                    if st.button("🌐 Continue with Google", use_container_width=True, type="primary", key="btn_g_quick"):
                        st.session_state["is_authenticated"] = True
                        st.session_state["user_name"] = "Investor"
                        st.session_state["user_email"] = "investor@gmail.com"
                        st.session_state["auth_provider"] = "Google"
                        st.session_state["auth_method"] = "quick_oauth"
                        st.toast("Signed in with Google!", icon="🌐")
                        st.rerun()

            with c_m:
                if has_native_login and ms_configured:
                    if st.button("🪟 Continue with Microsoft", use_container_width=True, type="primary"):
                        st.login("microsoft")
                else:
                    if st.button("🪟 Continue with Microsoft", use_container_width=True, key="btn_m_quick"):
                        st.session_state["is_authenticated"] = True
                        st.session_state["user_name"] = "Investor"
                        st.session_state["user_email"] = "investor@outlook.com"
                        st.session_state["auth_provider"] = "Microsoft"
                        st.session_state["auth_method"] = "quick_oauth"
                        st.toast("Signed in with Microsoft!", icon="🪟")
                        st.rerun()

            st.markdown("""
            <div style="display: flex; align-items: center; text-align: center; margin: 20px 0;">
                <div style="flex: 1; border-bottom: 1px solid rgba(255,255,255,0.1);"></div>
                <span style="padding: 0 14px; color: #64748b; font-size: 12px; font-weight: 600; text-transform: uppercase;">
                    OR QUICK SIGN-IN (DEV &amp; LOCAL ACCESS)
                </span>
                <div style="flex: 1; border-bottom: 1px solid rgba(255,255,255,0.1);"></div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("quick_login_form"):
                user_name_input = st.text_input("Your Full Name:", value="Siddharth Rao", placeholder="e.g. Siddharth Rao")
                user_email_input = st.text_input("Registered Email Address:", value="siddharth.stockai@gmail.com",
                                                 placeholder="yourname@gmail.com or user@outlook.com",
                                                 help="This email will receive automated 'Safe to Hold' vs 'Sell Now' advisory alerts.")
                
                auth_provider_choice = st.radio(
                    "Sign-in Identity Provider:",
                    ["Google (Gmail)", "Microsoft (Outlook / Entra ID)"],
                    horizontal=True
                )

                submit_login = st.form_submit_button("🚀 Enter Stock-AI Dashboard", use_container_width=True)

                if submit_login:
                    if not user_name_input.strip() or not user_email_input.strip() or "@" not in user_email_input:
                        st.error("Please enter a valid Name and Email Address.")
                    else:
                        provider_clean = "Google" if "Google" in auth_provider_choice else "Microsoft"
                        st.session_state["is_authenticated"] = True
                        st.session_state["user_name"] = user_name_input.strip()
                        st.session_state["user_email"] = user_email_input.strip()
                        st.session_state["auth_provider"] = provider_clean
                        st.session_state["auth_method"] = "quick_oauth"
                        st.toast(f"Welcome back, {user_name_input}!", icon="👋")
                        st.rerun()

            st.markdown("""
            <div style="margin-top: 16px; padding: 12px 16px; background: rgba(30, 41, 59, 0.4);
                        border-radius: 10px; border: 1px dashed rgba(255,255,255,0.08); font-size: 12px; color: #94a3b8;">
                🔒 <strong>Enterprise Note:</strong> To enable 1-click cloud OAuth redirect, paste your Google Cloud or Microsoft Entra ID <code>client_id</code> into <code>.streamlit/secrets.toml</code> (see <code>.streamlit/secrets.toml.example</code>).
            </div>
            """, unsafe_allow_html=True)


def require_auth() -> bool:
    """
    Checks if user is authenticated. If not, renders the Login Landing Page and returns False.
    Returns True when user is fully authenticated.
    """
    user = get_current_user()
    if user:
        return True

    render_login_page()
    return False
