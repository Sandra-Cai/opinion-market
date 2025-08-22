from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.schemas.market import MarketResponse

router = APIRouter()

@router.get("/pending")
def get_pending_verifications(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get markets pending verification (moderator/admin only)"""
    
    # Check if user has verification permissions
    if current_user.reputation_score < 500:  # Require high reputation
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for verification"
        )
    
    query = db.query(Market).filter(Market.status == MarketStatus.PENDING_VERIFICATION)
    total = query.count()
    markets = query.order_by(Market.created_at).offset(skip).limit(limit).all()
    
    return {
        "markets": markets,
        "total": total,
        "page": skip // limit + 1,
        "per_page": limit
    }

@router.post("/{market_id}/verify")
def verify_market(
    market_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify a market (moderator/admin only)"""
    
    # Check if user has verification permissions
    if current_user.reputation_score < 500:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for verification"
        )
    
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    if market.status != MarketStatus.PENDING_VERIFICATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market is not pending verification"
        )
    
    # Verify the market
    market.status = MarketStatus.OPEN
    market.verified_by = current_user.id
    market.verified_at = datetime.utcnow()
    
    # Update market quality metrics
    market.calculate_quality_score()
    market.calculate_trending_score()
    
    db.commit()
    
    return {"message": "Market verified successfully", "market_id": market_id}

@router.post("/{market_id}/reject")
def reject_market(
    market_id: int,
    reason: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reject a market (moderator/admin only)"""
    
    # Check if user has verification permissions
    if current_user.reputation_score < 500:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for verification"
        )
    
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    if market.status != MarketStatus.PENDING_VERIFICATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market is not pending verification"
        )
    
    # Reject the market
    market.status = MarketStatus.SUSPENDED
    market.verified_by = current_user.id
    market.verified_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Market rejected", "market_id": market_id, "reason": reason}

@router.get("/stats")
def get_verification_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get verification statistics (moderator/admin only)"""
    
    # Check if user has verification permissions
    if current_user.reputation_score < 500:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Get verification statistics
    pending_count = db.query(Market).filter(
        Market.status == MarketStatus.PENDING_VERIFICATION
    ).count()
    
    verified_today = db.query(Market).filter(
        Market.verified_at >= datetime.utcnow().date()
    ).count()
    
    rejected_today = db.query(Market).filter(
        Market.status == MarketStatus.SUSPENDED,
        Market.verified_at >= datetime.utcnow().date()
    ).count()
    
    total_verified = db.query(Market).filter(
        Market.verified_by.isnot(None)
    ).count()
    
    return {
        "pending_count": pending_count,
        "verified_today": verified_today,
        "rejected_today": rejected_today,
        "total_verified": total_verified
    }

@router.get("/my-verifications")
def get_my_verifications(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get markets verified by the current user"""
    
    query = db.query(Market).filter(Market.verified_by == current_user.id)
    total = query.count()
    markets = query.order_by(desc(Market.verified_at)).offset(skip).limit(limit).all()
    
    return {
        "markets": markets,
        "total": total,
        "page": skip // limit + 1,
        "per_page": limit
    }

@router.post("/{market_id}/suspend")
def suspend_market(
    market_id: int,
    reason: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Suspend a verified market (moderator/admin only)"""
    
    # Check if user has verification permissions
    if current_user.reputation_score < 500:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for suspension"
        )
    
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    if market.status != MarketStatus.OPEN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market is not open"
        )
    
    # Suspend the market
    market.status = MarketStatus.SUSPENDED
    market.dispute_reason = reason
    market.dispute_resolved_by = current_user.id
    market.dispute_resolved_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Market suspended", "market_id": market_id, "reason": reason}

@router.post("/{market_id}/reactivate")
def reactivate_market(
    market_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reactivate a suspended market (moderator/admin only)"""
    
    # Check if user has verification permissions
    if current_user.reputation_score < 500:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for reactivation"
        )
    
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    if market.status != MarketStatus.SUSPENDED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market is not suspended"
        )
    
    # Reactivate the market
    market.status = MarketStatus.OPEN
    market.dispute_reason = None
    market.dispute_resolved_by = current_user.id
    market.dispute_resolved_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Market reactivated", "market_id": market_id}
