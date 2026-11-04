"""Email/password account and login tracking API (raw MySQL SQL)."""
from datetime import datetime, timedelta, timezone
from functools import wraps
import hashlib
import os
import re
import secrets

import jwt
import mysql.connector
from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request
from mysql.connector import Error
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()
app = Flask(__name__)
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
TOKEN_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

def db():
    keys = ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME")
    missing = [k for k in keys if os.getenv(k) is None]
    if missing:
        raise RuntimeError("Missing environment settings: " + ", ".join(missing))
    return mysql.connector.connect(host=os.environ["DB_HOST"], port=int(os.environ["DB_PORT"]),
        user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"], database=os.environ["DB_NAME"])

def token_for(user_id, email):
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not configured")
    now = datetime.now(timezone.utc)
    return jwt.encode({"id": user_id, "email": email, "iat": now,
                       "exp": now + timedelta(hours=TOKEN_HOURS)}, JWT_SECRET, algorithm=JWT_ALGORITHM)

def token_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            return jsonify(error="Bearer token required"), 401
        try:
            claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            g.current_user = {"id": claims["id"], "email": claims["email"]}
        except jwt.ExpiredSignatureError:
            return jsonify(error="Token expired"), 401
        except (jwt.InvalidTokenError, KeyError):
            return jsonify(error="Invalid token"), 401
        return fn(*args, **kwargs)
    return wrapped

def record_login(cursor, user_id):
    # Table created by schema.sql; IP and user agent are recorded for security auditing.
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")[:255]
    agent = request.headers.get("User-Agent", "")[:1000]
    cursor.execute("INSERT INTO login_history (user_id, login_at, ip_address, user_agent) VALUES (%s, CURRENT_TIMESTAMP, %s, %s)",
                   (user_id, ip, agent))

@app.get("/")
def home():
    return render_template("auth.html")

@app.post("/api/register")
def register():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "")
    password = data.get("password", "")
    if not isinstance(email, str) or len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email.strip()):
        return jsonify(error="Valid email required"), 400
    if not isinstance(password, str) or len(password) < 10 or len(password) > 1024:
        return jsonify(error="Password must be at least 10 characters"), 400
    email = email.strip().lower()
    conn = cur = None
    try:
        conn = db(); cur = conn.cursor()
        cur.execute("INSERT INTO users (email, password, is_verified, verification_token) VALUES (%s, %s, TRUE, NULL)",
                    (email, generate_password_hash(password)))
        conn.commit()
        return jsonify(message="Account created. You can now sign in."), 201
    except mysql.connector.IntegrityError:
        if conn and conn.is_connected(): conn.rollback()
        return jsonify(error="An account with this email already exists"), 409
    except (Error, RuntimeError):
        app.logger.exception("Registration failed")
        if conn and conn.is_connected(): conn.rollback()
        return jsonify(error="Could not register or send verification email"), 500
    finally:
        if cur: cur.close()
        if conn and conn.is_connected(): conn.close()

@app.post("/api/verify-email")
def verify_email():
    data = request.get_json(silent=True) or {}
    email, code = data.get("email", ""), data.get("code", "")
    if not isinstance(email, str) or not isinstance(code, str) or not re.fullmatch(r"\d{6}", code):
        return jsonify(error="Email and 6-digit code required"), 400
    conn = cur = None
    try:
        conn = db(); conn.start_transaction(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, verification_token, is_verified FROM users WHERE email=%s FOR UPDATE", (email.strip().lower(),))
        user = cur.fetchone()
        expected = hashlib.sha256(code.encode()).hexdigest()
        if not user or user["is_verified"] or not secrets.compare_digest(user["verification_token"] or "", expected):
            conn.rollback(); return jsonify(error="Invalid verification code"), 400
        cur.execute("UPDATE users SET is_verified=TRUE, verification_token=NULL WHERE id=%s", (user["id"],))
        conn.commit()
        return jsonify(message="Email verified. You can now log in."), 200
    except (Error, RuntimeError):
        app.logger.exception("Email verification failed")
        if conn and conn.is_connected(): conn.rollback()
        return jsonify(error="Could not verify email"), 500
    finally:
        if cur: cur.close()
        if conn and conn.is_connected(): conn.close()

@app.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    email, password = data.get("email", ""), data.get("password", "")
    if not isinstance(email, str) or not isinstance(password, str):
        return jsonify(error="Email and password required"), 400
    conn = cur = None
    try:
        conn = db(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, email, password, is_verified FROM users WHERE email=%s", (email.strip().lower(),))
        user = cur.fetchone()
        if not user or not check_password_hash(user["password"], password):
            return jsonify(error="Invalid email or password"), 401
        record_login(cur, user["id"]); conn.commit()
        return jsonify(token=token_for(user["id"], user["email"]), user={"id": user["id"], "email": user["email"]}), 200
    except (Error, RuntimeError):
        app.logger.exception("Login failed")
        if conn and conn.is_connected(): conn.rollback()
        return jsonify(error="Could not log in"), 500
    finally:
        if cur: cur.close()
        if conn and conn.is_connected(): conn.close()

@app.get("/api/me/login-history")
@token_required
def my_login_history():
    conn = cur = None
    try:
        conn = db(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT login_at, ip_address, user_agent FROM login_history WHERE user_id=%s ORDER BY login_at DESC LIMIT 100", (g.current_user["id"],))
        return jsonify(logins=cur.fetchall()), 200
    except (Error, RuntimeError):
        app.logger.exception("Could not read login history")
        return jsonify(error="Could not read login history"), 500
    finally:
        if cur: cur.close()
        if conn and conn.is_connected(): conn.close()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
