"""Firebase Authentication helpers for the Streamlit dashboard."""

import os
from typing import Any, Dict

import requests
import streamlit as st


def _api_key() -> str:
    """Read the Firebase Web API key from Streamlit secrets or environment."""
    try:
        value = st.secrets.get("firebase", {}).get("api_key", "")
    except Exception:
        value = ""
    return value or os.getenv("FIREBASE_API_KEY", "")


def _post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    api_key = _api_key()
    if not api_key:
        raise ValueError("Firebase is not configured. Set firebase.api_key in Streamlit secrets.")
    response = requests.post(
        f"https://identitytoolkit.googleapis.com/v1/{endpoint}?key={api_key}",
        json=payload,
        timeout=15,
    )
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        code = data.get("error", {}).get("message", "Firebase authentication failed")
        messages = {
            "EMAIL_EXISTS": "An account with this email already exists.",
            "EMAIL_NOT_FOUND": "No account was found for this email.",
            "INVALID_PASSWORD": "The password is incorrect.",
            "INVALID_LOGIN_CREDENTIALS": "The email or password is incorrect.",
            "WEAK_PASSWORD : Password should be at least 6 characters": "Choose a password with at least 6 characters.",
            "OPERATION_NOT_ALLOWED": "Enable Email/Password sign-in in Firebase Authentication settings.",
            "INVALID_ID_TOKEN": "Your session expired. Please sign in again.",
            "USER_DISABLED": "This account has been disabled.",
        }
        raise ValueError(messages.get(code, code.replace("_", " ").capitalize()))
    return data


def sign_in(email: str, password: str) -> Dict[str, str]:
    data = _post("accounts:signInWithPassword", {
        "email": email.strip(), "password": password,
        "returnSecureToken": True,
    })
    return {
        "id_token": data["idToken"],
        "refresh_token": data["refreshToken"],
        "uid": data["localId"],
        "email": data.get("email", email.strip()),
        "name": data.get("displayName") or data.get("email", email.strip()).split("@")[0],
    }


def sign_up(email: str, password: str, name: str) -> Dict[str, str]:
    data = _post("accounts:signUp", {
        "email": email.strip(), "password": password,
        "displayName": name.strip(), "returnSecureToken": True,
    })
    return {
        "id_token": data["idToken"],
        "refresh_token": data["refreshToken"],
        "uid": data["localId"],
        "email": data.get("email", email.strip()),
        "name": data.get("displayName") or name.strip(),
    }


def refresh_session(refresh_token: str) -> Dict[str, str]:
    api_key = _api_key()
    if not api_key:
        raise ValueError("Firebase is not configured. Set firebase.api_key in Streamlit secrets.")
    response = requests.post(
        f"https://securetoken.googleapis.com/v1/token?key={api_key}",
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        timeout=15,
    )
    try:
        data = response.json()
    except ValueError:
        data = {}
    if not response.ok:
        raise ValueError("Your session expired. Please sign in again.")
    return {"id_token": data["id_token"], "refresh_token": data["refresh_token"]}


def verify_session(id_token: str) -> Dict[str, str]:
    data = _post("accounts:lookup", {"idToken": id_token})
    user = data.get("users", [{}])[0]
    return {
        "uid": user.get("localId", ""),
        "email": user.get("email", ""),
        "name": user.get("displayName") or user.get("email", "Investor").split("@")[0],
    }
