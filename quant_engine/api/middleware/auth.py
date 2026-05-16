"""
Auth Middleware
================
FastAPI dependency injection for authentication and role-based access.

Usage:
    @router.get("/protected")
    def protected(user: User = Depends(get_current_user)):
        ...

    @router.get("/admin-only")
    def admin_only(user: User = Depends(require_role("admin"))):
        ...
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from services.auth_service import decode_token, get_user_by_id

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Extract and validate JWT from Authorization header.
    Returns the authenticated User or raises 401.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — provide a Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(db, payload["sub"])
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Like get_current_user but returns None instead of 401 if no token."""
    if credentials is None:
        return None
    try:
        payload = decode_token(credentials.credentials)
        if payload and payload.get("type") == "access":
            return get_user_by_id(db, payload["sub"])
    except Exception:
        pass
    return None


def require_role(*roles: str):
    """Dependency factory — restricts access to specific roles."""
    async def _check(user: User = Depends(get_current_user)):
        if user.role.value not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(roles)}. Your role: {user.role.value}",
            )
        return user
    return _check
