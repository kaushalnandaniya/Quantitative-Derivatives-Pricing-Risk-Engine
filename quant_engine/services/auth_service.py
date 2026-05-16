"""
Authentication Service
========================
JWT token management, password hashing, and OTP verification.
"""

import os
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from sqlalchemy.orm import Session

from db.models import User, UserRole

logger = logging.getLogger(__name__)

# Config from environment
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE_MIN = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

OTP_EXPIRE_SECONDS = 300  # 5 minutes

import hashlib
import bcrypt


# =============================================================================
# Redis Helper (for OTP storage)
# =============================================================================

_redis_client = None


def _get_redis():
    """Lazily connect to Redis for OTP storage."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    import redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable for OTP storage: {e}")
        return None


# In-memory fallback for OTP when Redis is unavailable (dev only)
_otp_fallback: dict[str, tuple[str, float]] = {}


# =============================================================================
# Password Utilities
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password using bcrypt with SHA-256 prehash."""
    pw = hashlib.sha256(password.encode()).hexdigest().encode()
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    pw = hashlib.sha256(plain.encode()).hexdigest().encode()
    return bcrypt.checkpw(pw, hashed.encode())


# =============================================================================
# JWT Token Utilities
# =============================================================================

def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_EXPIRE_MIN)
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.debug(f"JWT decode failed: {e}")
        return None


# =============================================================================
# OTP Generation & Verification
# =============================================================================

def generate_otp() -> str:
    """Generate a cryptographically secure 6-digit OTP."""
    return f"{secrets.randbelow(900000) + 100000}"


def store_otp(email: str, otp: str) -> bool:
    """
    Store the OTP in Redis with a 5-minute TTL.
    Falls back to in-memory dict if Redis is unavailable.
    """
    email_key = f"otp:{email.lower().strip()}"
    r = _get_redis()

    if r is not None:
        try:
            r.setex(email_key, OTP_EXPIRE_SECONDS, otp)
            logger.info(f"OTP stored in Redis for {email}")
            return True
        except Exception as e:
            logger.error(f"Failed to store OTP in Redis: {e}")

    # Fallback: in-memory storage (dev only)
    import time
    _otp_fallback[email_key] = (otp, time.time() + OTP_EXPIRE_SECONDS)
    logger.warning(f"OTP stored in-memory fallback for {email}")
    return True


def verify_otp(email: str, otp: str) -> bool:
    """
    Verify the OTP against the stored value.
    Deletes the OTP after successful verification (one-time use).
    """
    email_key = f"otp:{email.lower().strip()}"
    r = _get_redis()

    if r is not None:
        try:
            stored_otp = r.get(email_key)
            if stored_otp and stored_otp == otp:
                r.delete(email_key)
                logger.info(f"OTP verified for {email}")
                return True
            logger.warning(f"OTP mismatch for {email}")
            return False
        except Exception as e:
            logger.error(f"Failed to verify OTP in Redis: {e}")

    # Fallback: check in-memory
    import time
    entry = _otp_fallback.get(email_key)
    if entry:
        stored_otp, expires_at = entry
        if time.time() < expires_at and stored_otp == otp:
            del _otp_fallback[email_key]
            logger.info(f"OTP verified (in-memory) for {email}")
            return True
        if time.time() >= expires_at:
            del _otp_fallback[email_key]
            logger.warning(f"OTP expired for {email}")
        else:
            logger.warning(f"OTP mismatch (in-memory) for {email}")

    return False


# =============================================================================
# User Service Functions
# =============================================================================

def register_user(db: Session, email: str, password: str, full_name: str, role: str = "trader") -> User:
    """Create a new user account."""
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise ValueError(f"Email '{email}' is already registered")

    user = User(
        email=email.lower().strip(),
        password_hash=hash_password(password),
        full_name=full_name.strip(),
        role=UserRole(role),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"User registered: {user.email} (role={user.role})")
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Verify email/password and return the user, or None."""
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user or not verify_password(password, user.password_hash):
        return None
    if not user.is_active:
        return None

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    return user


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    """Fetch user by ID."""
    return db.query(User).filter(User.id == user_id).first()
