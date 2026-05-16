"""
Auth API Routes
=================
POST /auth/register  — Create account
POST /auth/login     — Login → JWT tokens
POST /auth/refresh   — Refresh access token
GET  /auth/me        — Current user profile
PUT  /auth/me        — Update profile
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from schemas.auth import (
    RegisterRequest, LoginRequest, TokenResponse,
    RefreshRequest, UserResponse, UserUpdateRequest,
)
from services.auth_service import (
    register_user, authenticate_user, get_user_by_id,
    create_access_token, create_refresh_token, decode_token,
)
from api.middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=201, summary="Register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account."""
    try:
        user = register_user(db, data.email, data.password, data.full_name, data.role)
        return UserResponse(
            id=user.id, email=user.email, full_name=user.full_name,
            role=user.role.value, is_active=user.is_active,
            created_at=user.created_at.isoformat(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=TokenResponse, summary="Login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate and receive JWT tokens."""
    user = authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token(user.id, user.role.value)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user.id, email=user.email, full_name=user.full_name,
            role=user.role.value, is_active=user.is_active,
            created_at=user.created_at.isoformat(),
        ),
    )


@router.post("/refresh", summary="Refresh Token")
def refresh(data: RefreshRequest, db: Session = Depends(get_db)):
    """Get a new access token using a refresh token."""
    payload = decode_token(data.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = get_user_by_id(db, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    new_access = create_access_token(user.id, user.role.value)
    return {"access_token": new_access, "token_type": "bearer"}


@router.get("/me", response_model=UserResponse, summary="Current User")
def get_me(user: User = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    return UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        role=user.role.value, is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )


@router.put("/me", response_model=UserResponse, summary="Update Profile")
def update_me(
    data: UserUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the current user's profile."""
    if data.full_name:
        user.full_name = data.full_name.strip()
    if data.email:
        user.email = data.email.lower().strip()
    db.commit()
    db.refresh(user)
    return UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        role=user.role.value, is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )
