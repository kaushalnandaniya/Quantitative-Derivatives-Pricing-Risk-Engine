"""
Auth Schemas
==============
Pydantic models for authentication endpoints.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, EmailStr


class SendOtpRequest(BaseModel):
    email: str = Field(..., description="Email address to send OTP to")


class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 chars)")
    full_name: str = Field(..., min_length=2, max_length=255, description="Full name")
    otp: str = Field(..., min_length=6, max_length=6, description="6-digit OTP verification code")
    role: Literal["trader", "risk_manager", "admin"] = Field("trader", description="User role")


class LoginRequest(BaseModel):
    email: str = Field(..., description="Email")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
