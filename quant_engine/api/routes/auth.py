"""
Auth API Routes
=================
POST /auth/send-otp  — Send OTP to email for registration
POST /auth/register  — Create account (requires verified OTP)
POST /auth/login     — Login → JWT tokens (Bloom filter pre-check)
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
    SendOtpRequest, RegisterRequest, LoginRequest, TokenResponse,
    RefreshRequest, UserResponse, UserUpdateRequest,
    ForgotPasswordRequest, ResetPasswordRequest,
)
from services.auth_service import (
    register_user, authenticate_user, get_user_by_id, reset_password,
    create_access_token, create_refresh_token, decode_token,
    generate_otp, store_otp, verify_otp,
)
from services.bloom_filter import add_email, might_contain
from services.email_service import send_otp_email
from api.middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/send-otp", summary="Send OTP", status_code=200)
def send_otp(data: SendOtpRequest, db: Session = Depends(get_db)):
    """
    Send a 6-digit OTP to the user's email for registration verification.
    
    Uses the Bloom filter to quickly reject emails that are already registered.
    """
    email = data.email.lower().strip()

    # Step 1: Bloom filter pre-check — is this email possibly already registered?
    if might_contain(email):
        # Bloom filter says "maybe" — confirm with the database
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="An account with this email already exists. Please sign in.",
            )

    # Step 2: Generate OTP and store in Redis
    otp = generate_otp()
    store_otp(email, otp)

    # Step 3: Send OTP via email
    sent = send_otp_email(email, otp)
    if not sent:
        raise HTTPException(
            status_code=500,
            detail="Failed to send verification email. Please try again.",
        )

    return {
        "message": "Verification code sent to your email",
        "email": email,
    }


@router.post("/register", response_model=UserResponse, status_code=201, summary="Register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    """
    Create a new user account after OTP verification.
    
    The OTP must have been sent via /auth/send-otp first.
    """
    email = data.email.lower().strip()

    # Step 1: Verify the OTP
    if not verify_otp(email, data.otp):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired verification code. Please request a new one.",
        )

    # Step 2: Register the user
    try:
        user = register_user(db, email, data.password, data.full_name, data.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Step 3: Add email to Bloom filter for future fast lookups
    add_email(email)

    return UserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        role=user.role.value, is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )


@router.post("/login", response_model=TokenResponse, summary="Login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate and receive JWT tokens.
    
    Uses Bloom filter for fast rejection of unregistered emails,
    preventing brute-force DB enumeration attacks.
    """
    email = data.email.lower().strip()

    # Step 1: Bloom filter pre-check
    if not might_contain(email):
        # DEFINITELY not registered — reject instantly without hitting DB
        logger.info(f"Bloom filter rejected login for {email} (not registered)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Step 2: Standard DB authentication
    user = authenticate_user(db, email, data.password)
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


@router.post("/forgot-password", summary="Forgot Password", status_code=200)
def forgot_password(data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Send a password reset OTP to the user's email.
    
    Silently succeeds even if email doesn't exist (security best practice).
    """
    email = data.email.lower().strip()

    # Only send if user exists (but don't reveal this to the client)
    user = db.query(User).filter(User.email == email).first()
    if user:
        otp = generate_otp()
        store_otp(email, otp)
        send_otp_email(email, otp)

    return {"message": "If an account exists with this email, a reset code has been sent."}


@router.post("/reset-password", summary="Reset Password", status_code=200)
def reset_password_route(data: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset password after OTP verification.
    """
    email = data.email.lower().strip()

    if not verify_otp(email, data.otp):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired verification code.",
        )

    try:
        reset_password(db, email, data.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Password has been reset successfully. You can now sign in."}
