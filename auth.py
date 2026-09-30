"""Password hashing, JWT tokens and in-memory session management."""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Set

import jwt
from fastapi import HTTPException, Request, status

from config import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY, SESSION_IDLE_SECONDS
from database import get_user
from models import UserInDB, UserSession

PBKDF2_ITERATIONS = 200_000

active_sessions: Dict[str, UserSession] = {}
blacklisted_tokens: Set[str] = set()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------- Passwords ----------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        _, iterations, salt, expected = stored_hash.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
    except ValueError:
        return False
    return hmac.compare_digest(digest.hex(), expected)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if user is None or user.disabled or not verify_password(password, user.hashed_password):
        return None
    return user


# ---------- Tokens ----------

def create_access_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "jti": secrets.token_hex(8),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_token(request: Request) -> Optional[str]:
    token = request.cookies.get("access_token")
    if not token:
        header = request.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
    return token or None


def username_from_token(token: Optional[str]) -> Optional[str]:
    if not token or token in blacklisted_tokens:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return payload.get("sub")


# ---------- Sessions ----------

def create_session(username: str, token: str) -> None:
    existing = active_sessions.get(username)
    user_data = {}
    if existing:
        user_data = existing.user_data
        blacklisted_tokens.add(existing.token)  # only the newest login stays valid
    now = utcnow()
    active_sessions[username] = UserSession(
        username=username, login_time=now, last_activity=now, token=token, user_data=user_data
    )


def end_session(token: Optional[str]) -> None:
    if not token:
        return
    username = username_from_token(token)
    blacklisted_tokens.add(token)
    if username and username in active_sessions:
        del active_sessions[username]


def remove_expired_sessions() -> None:
    now = utcnow()
    expired = [
        name for name, session in active_sessions.items()
        if (now - session.last_activity).total_seconds() > SESSION_IDLE_SECONDS
    ]
    for name in expired:
        blacklisted_tokens.add(active_sessions[name].token)
        del active_sessions[name]
        print(f"Removing expired session for {name}")


def _user_from_request(request: Request) -> Optional[UserInDB]:
    token = get_token(request)
    username = username_from_token(token)
    if not username:
        return None
    user = get_user(username)
    if user is None or user.disabled:
        return None
    session = active_sessions.get(user.username)
    if session is None:
        # Valid token but no session (e.g. the server restarted): start a new one.
        create_session(user.username, token)
    else:
        session.last_activity = utcnow()
    return user


def get_optional_user(request: Request) -> Optional[UserInDB]:
    return _user_from_request(request)


async def get_current_active_user(request: Request) -> UserInDB:
    user = _user_from_request(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Your session has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
