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


class TenantPlan(str, enum.Enum):
    basic = "basic"
    professional = "professional"
    enterprise = "enterprise"


class OrderStatus(str, enum.Enum):
    pending = "pending"
    validated = "validated"
    rejected = "rejected"
    submitted = "submitted"
    partial_fill = "partial_fill"
    filled = "filled"
    cancelled = "cancelled"


class OrderType(str, enum.Enum):
    market = "market"
    limit = "limit"


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

class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, unique=True)
    domain = Column(String(255), nullable=True)
    config = Column(JSON, nullable=False, default=dict)  # {features, branding, etc.}
    plan = Column(Enum(TenantPlan), default=TenantPlan.basic, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    risk_limits = relationship("RiskLimits", back_populates="tenant", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tenant '{self.name}' plan={self.plan}>"


class RiskLimits(Base):
    __tablename__ = "risk_limits"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, unique=True)
    max_portfolio_var = Column(Float, nullable=False, default=100000.0)
    max_position_size = Column(Integer, nullable=False, default=1000)
    max_notional = Column(Float, nullable=False, default=10000000.0)
    margin_requirement = Column(Float, nullable=False, default=0.1)  # 10%
    custom_rules = Column(JSON, nullable=False, default=dict)

    # Relationships
    tenant = relationship("Tenant", back_populates="risk_limits")

    def __repr__(self):
        return f"<RiskLimits tenant={self.tenant_id} maxVaR={self.max_portfolio_var}>"


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(36), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.trader, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="user", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="user", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    saved_strategies = relationship("SavedStrategy", back_populates="user", cascade="all, delete-orphan")

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


class Order(Base):
    __tablename__ = "orders"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="SET NULL"), nullable=True)
    side = Column(Enum(TradeSide), nullable=False)
    option_type = Column(Enum(OptionType), nullable=False)
    order_type = Column(Enum(OrderType), default=OrderType.market, nullable=False)
    spot_price = Column(Float, nullable=False)
    strike = Column(Float, nullable=False)
    T = Column(Float, nullable=False)
    r = Column(Float, nullable=False, default=0.05)
    sigma = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    limit_price = Column(Float, nullable=True)  # For limit orders
    filled_quantity = Column(Integer, default=0, nullable=False)
    avg_fill_price = Column(Float, nullable=True)
    status = Column(Enum(OrderStatus), default=OrderStatus.pending, nullable=False)
    risk_check_result = Column(JSON, nullable=True)  # Pre-trade risk check details
    rejection_reason = Column(Text, nullable=True)
    submitted_at = Column(DateTime, default=utc_now, nullable=False)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="orders")
    portfolio = relationship("Portfolio")
    executions = relationship("Execution", back_populates="order", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_orders_user_status", "user_id", "status"),
    )

    @property
    def is_terminal(self):
        return self.status in (OrderStatus.filled, OrderStatus.rejected, OrderStatus.cancelled)

    def __repr__(self):
        return f"<Order {self.side.value} {self.quantity}x {self.option_type.value} K={self.strike} status={self.status.value}>"


class Execution(Base):
    __tablename__ = "executions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    order_id = Column(String(36), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False)
    fill_price = Column(Float, nullable=False)
    fill_quantity = Column(Integer, nullable=False)
    executed_at = Column(DateTime, default=utc_now, nullable=False)
    exchange_ref = Column(String(100), nullable=True)  # External reference

    # Relationships
    order = relationship("Order", back_populates="executions")

    def __repr__(self):
        return f"<Execution {self.fill_quantity}x @ {self.fill_price}>"


class SavedStrategy(Base):
    __tablename__ = "saved_strategies"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    pine_script = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    user = relationship("User", back_populates="saved_strategies")

    __table_args__ = (
        Index("ix_strategies_user", "user_id"),
    )

    def __repr__(self):
        return f"<SavedStrategy '{self.name}'>"
