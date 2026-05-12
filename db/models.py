"""
Database ORM Models
=====================
SQLAlchemy models for the quant engine platform.

Tables:
    - users: Authentication & role management
    - portfolios: Saved portfolio configurations
    - trades: Trade booking & position tracking
    - alerts: VaR breach & price trigger alerts
    - audit_log: Full request audit trail
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    ForeignKey, Text, Enum, JSON, Index,
)
from sqlalchemy.orm import relationship
import enum

from db.database import Base


# =============================================================================
# Enums
# =============================================================================

class UserRole(str, enum.Enum):
    trader = "trader"
    risk_manager = "risk_manager"
    admin = "admin"


class TradeStatus(str, enum.Enum):
    open = "open"
    closed = "closed"
    expired = "expired"


class TradeSide(str, enum.Enum):
    buy = "buy"
    sell = "sell"


class OptionType(str, enum.Enum):
    call = "call"
    put = "put"


class AlertType(str, enum.Enum):
    var_breach = "var_breach"
    price_trigger = "price_trigger"
    expiry_warning = "expiry_warning"
    margin_call = "margin_call"


# =============================================================================
# Helper
# =============================================================================

def generate_uuid():
    return str(uuid.uuid4())


def utc_now():
    return datetime.now(timezone.utc)


# =============================================================================
# Models
# =============================================================================

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.trader, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email} role={self.role}>"


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    positions = Column(JSON, nullable=False, default=list)  # [{type, S, K, T, r, sigma, qty}]
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    user = relationship("User", back_populates="portfolios")
    trades = relationship("Trade", back_populates="portfolio", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="portfolio", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_portfolios_user_active", "user_id", "is_active"),
    )

    def __repr__(self):
        return f"<Portfolio '{self.name}' positions={len(self.positions or [])}>"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="SET NULL"), nullable=True)
    side = Column(Enum(TradeSide), nullable=False)
    option_type = Column(Enum(OptionType), nullable=False)
    spot_at_entry = Column(Float, nullable=False)
    strike = Column(Float, nullable=False)
    premium = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    sigma_at_entry = Column(Float, nullable=False)
    T_at_entry = Column(Float, nullable=False)
    r_at_entry = Column(Float, nullable=False, default=0.05)
    traded_at = Column(DateTime, default=utc_now, nullable=False)
    closed_at = Column(DateTime, nullable=True)
    close_premium = Column(Float, nullable=True)
    status = Column(Enum(TradeStatus), default=TradeStatus.open, nullable=False)
    notes = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="trades")
    portfolio = relationship("Portfolio", back_populates="trades")

    __table_args__ = (
        Index("ix_trades_user_status", "user_id", "status"),
        Index("ix_trades_portfolio", "portfolio_id"),
    )

    @property
    def notional(self):
        return self.premium * self.quantity

    def __repr__(self):
        return f"<Trade {self.side.value} {self.quantity}x {self.option_type.value} K={self.strike} status={self.status.value}>"


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=True)
    alert_type = Column(Enum(AlertType), nullable=False)
    condition = Column(JSON, nullable=False)  # {threshold, direction, symbol, etc.}
    message = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    triggered = Column(Boolean, default=False, nullable=False)
    triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    user = relationship("User", back_populates="alerts")
    portfolio = relationship("Portfolio", back_populates="alerts")

    def __repr__(self):
        return f"<Alert {self.alert_type.value} triggered={self.triggered}>"


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(50), nullable=False)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    request_body = Column(JSON, nullable=True)
    response_status = Column(Integer, nullable=True)
    elapsed_ms = Column(Float, nullable=True)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_user_time", "user_id", "created_at"),
    )

    def __repr__(self):
        return f"<AuditLog {self.method} {self.endpoint}>"
