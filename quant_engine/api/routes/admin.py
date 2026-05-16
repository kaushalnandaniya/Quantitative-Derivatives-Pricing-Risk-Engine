"""
Admin API Routes
==================
POST /admin/tenants           — Create tenant
GET  /admin/tenants           — List tenants
GET  /admin/tenants/{id}      — Tenant detail + risk limits
PUT  /admin/tenants/{id}/limits — Update risk limits
GET  /admin/users             — List all users (admin only)
PUT  /admin/users/{id}/role   — Update user role
GET  /admin/audit-log         — Query audit log
GET  /admin/system            — System metrics
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, Tenant, RiskLimits, AuditLog, UserRole, TenantPlan
from api.middleware.auth import get_current_user, require_role

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])


# =============================================================================
# Bootstrap: One-time admin promotion (only works if NO admin exists)
# =============================================================================

@router.post("/bootstrap", summary="Bootstrap First Admin", status_code=200)
def bootstrap_admin(db: Session = Depends(get_db)):
    """
    Promote the first registered user to admin.
    Only works if there are ZERO admin users in the system.
    This is a one-time setup endpoint.
    """
    existing_admin = db.query(User).filter(User.role == UserRole.admin).first()
    if existing_admin:
        raise HTTPException(403, "Admin already exists. Bootstrap disabled.")

    first_user = db.query(User).order_by(User.created_at.asc()).first()
    if not first_user:
        raise HTTPException(404, "No users found in the system")

    first_user.role = UserRole.admin
    db.commit()
    db.refresh(first_user)
    logger.info(f"Bootstrap: {first_user.email} promoted to admin")
    return {
        "message": f"User '{first_user.email}' has been promoted to admin",
        "user_id": first_user.id,
        "email": first_user.email,
        "role": "admin",
    }


# =============================================================================
# Schemas
# =============================================================================

class CreateTenantRequest(BaseModel):
    name: str
    domain: Optional[str] = None
    plan: str = "basic"
    config: dict = {}


class UpdateLimitsRequest(BaseModel):
    max_portfolio_var: Optional[float] = None
    max_position_size: Optional[int] = None
    max_notional: Optional[float] = None
    margin_requirement: Optional[float] = None
    custom_rules: Optional[dict] = None


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., description="trader, risk_manager, or admin")
    tenant_id: Optional[str] = None


# =============================================================================
# Tenant Endpoints
# =============================================================================

@router.post("/tenants", status_code=201, summary="Create Tenant")
def create_tenant(
    data: CreateTenantRequest,
    user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    existing = db.query(Tenant).filter(Tenant.name == data.name).first()
    if existing:
        raise HTTPException(409, "Tenant name already exists")

    tenant = Tenant(name=data.name, domain=data.domain, plan=TenantPlan(data.plan), config=data.config)
    db.add(tenant)
    db.flush()

    # Create default risk limits
    limits = RiskLimits(tenant_id=tenant.id)
    db.add(limits)
    db.commit()
    db.refresh(tenant)

    logger.info(f"Tenant created: {tenant.name} (plan={tenant.plan.value})")
    return _tenant_response(tenant, limits)


@router.get("/tenants", summary="List Tenants")
def list_tenants(user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    tenants = db.query(Tenant).filter(Tenant.is_active == True).all()
    result = []
    for t in tenants:
        limits = db.query(RiskLimits).filter(RiskLimits.tenant_id == t.id).first()
        n_users = db.query(User).filter(User.tenant_id == t.id).count()
        resp = _tenant_response(t, limits)
        resp["user_count"] = n_users
        result.append(resp)
    return {"tenants": result, "count": len(result)}


@router.get("/tenants/{tenant_id}", summary="Tenant Detail")
def get_tenant(tenant_id: str, user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(404, "Tenant not found")
    limits = db.query(RiskLimits).filter(RiskLimits.tenant_id == tenant.id).first()
    users = db.query(User).filter(User.tenant_id == tenant.id).all()
    resp = _tenant_response(tenant, limits)
    resp["users"] = [
        {"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role.value, "is_active": u.is_active}
        for u in users
    ]
    return resp


@router.put("/tenants/{tenant_id}/limits", summary="Update Risk Limits")
def update_limits(
    tenant_id: str, data: UpdateLimitsRequest,
    user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    limits = db.query(RiskLimits).filter(RiskLimits.tenant_id == tenant_id).first()
    if not limits:
        raise HTTPException(404, "Tenant limits not found")

    if data.max_portfolio_var is not None: limits.max_portfolio_var = data.max_portfolio_var
    if data.max_position_size is not None: limits.max_position_size = data.max_position_size
    if data.max_notional is not None: limits.max_notional = data.max_notional
    if data.margin_requirement is not None: limits.margin_requirement = data.margin_requirement
    if data.custom_rules is not None: limits.custom_rules = data.custom_rules

    db.commit()
    db.refresh(limits)
    logger.info(f"Risk limits updated for tenant={tenant_id}")
    return {"limits": _limits_dict(limits)}


# =============================================================================
# User Management
# =============================================================================

@router.get("/users", summary="List All Users")
def list_users(
    tenant_id: str = Query(None),
    user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    q = db.query(User)
    if tenant_id:
        q = q.filter(User.tenant_id == tenant_id)
    users = q.order_by(User.created_at.desc()).all()
    return {
        "users": [
            {
                "id": u.id, "email": u.email, "full_name": u.full_name,
                "role": u.role.value, "tenant_id": u.tenant_id,
                "is_active": u.is_active, "created_at": u.created_at.isoformat(),
                "last_login": u.last_login.isoformat() if u.last_login else None,
            }
            for u in users
        ],
        "count": len(users),
    }


@router.put("/users/{user_id}/role", summary="Update User Role")
def update_role(
    user_id: str, data: UpdateRoleRequest,
    admin: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(404, "User not found")

    target.role = UserRole(data.role)
    if data.tenant_id is not None:
        target.tenant_id = data.tenant_id
    db.commit()
    db.refresh(target)
    logger.info(f"User {target.email} role updated to {data.role}")
    return {"id": target.id, "email": target.email, "role": target.role.value, "tenant_id": target.tenant_id}


# =============================================================================
# Audit Log
# =============================================================================

@router.get("/audit-log", summary="Query Audit Log")
def query_audit(
    user_id: str = Query(None),
    endpoint: str = Query(None),
    limit: int = Query(100, le=1000),
    user: User = Depends(require_role("admin", "risk_manager")),
    db: Session = Depends(get_db),
):
    q = db.query(AuditLog)
    if user_id:
        q = q.filter(AuditLog.user_id == user_id)
    if endpoint:
        q = q.filter(AuditLog.endpoint.contains(endpoint))
    logs = q.order_by(AuditLog.created_at.desc()).limit(limit).all()
    return {
        "audit_logs": [
            {
                "id": l.id, "user_id": l.user_id, "action": l.action,
                "endpoint": l.endpoint, "method": l.method,
                "response_status": l.response_status,
                "elapsed_ms": l.elapsed_ms,
                "ip_address": l.ip_address,
                "created_at": l.created_at.isoformat(),
            }
            for l in logs
        ],
        "count": len(logs),
    }


# =============================================================================
# System Metrics
# =============================================================================

@router.get("/system", summary="System Metrics")
def system_metrics(user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    from db.models import Portfolio, Trade, Alert, Order
    return {
        "counts": {
            "tenants": db.query(Tenant).filter(Tenant.is_active == True).count(),
            "users": db.query(User).filter(User.is_active == True).count(),
            "portfolios": db.query(Portfolio).filter(Portfolio.is_active == True).count(),
            "trades_open": db.query(Trade).filter(Trade.status == "open").count(),
            "trades_total": db.query(Trade).count(),
            "orders_total": db.query(Order).count(),
            "alerts_active": db.query(Alert).filter(Alert.is_active == True).count(),
        },
    }


# =============================================================================
# Helpers
# =============================================================================

def _tenant_response(t, limits=None) -> dict:
    resp = {
        "id": t.id, "name": t.name, "domain": t.domain,
        "plan": t.plan.value, "config": t.config,
        "is_active": t.is_active, "created_at": t.created_at.isoformat(),
    }
    if limits:
        resp["risk_limits"] = _limits_dict(limits)
    return resp


def _limits_dict(l) -> dict:
    return {
        "max_portfolio_var": l.max_portfolio_var,
        "max_position_size": l.max_position_size,
        "max_notional": l.max_notional,
        "margin_requirement": l.margin_requirement,
        "custom_rules": l.custom_rules,
    }
