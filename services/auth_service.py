"""
Authentication Service
========================
JWT token management and password hashing.
"""

import os
import logging
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

import hashlib
import bcrypt


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
