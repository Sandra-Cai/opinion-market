from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.models.dispute import MarketDispute, DisputeVote, DisputeStatus, DisputeType
from app.schemas.dispute import (
    DisputeCreate,
    DisputeUpdate,
    DisputeResponse,
    DisputeVoteCreate,
    DisputeListResponse,
)

router = APIRouter()


@router.post("/", response_model=DisputeResponse)
def create_dispute(
    dispute_data: DisputeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new market dispute"""

    # Check if market exists
    market = db.query(Market).filter(Market.id == dispute_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Check if market is resolved
    if market.status != MarketStatus.RESOLVED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only dispute resolved markets",
        )

    # Check if user already created a dispute for this market
    existing_dispute = (
        db.query(MarketDispute)
        .filter(
            MarketDispute.market_id == dispute_data.market_id,
            MarketDispute.created_by == current_user.id,
        )
        .first()
    )

    if existing_dispute:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already created a dispute for this market",
        )

    # Create dispute
    db_dispute = MarketDispute(
        market_id=dispute_data.market_id,
        created_by=current_user.id,
        dispute_type=dispute_data.dispute_type,
        reason=dispute_data.reason,
        evidence=dispute_data.evidence,
    )

    db.add(db_dispute)

    # Update market dispute count
    market.dispute_count += 1
    market.status = MarketStatus.DISPUTED

    db.commit()
    db.refresh(db_dispute)

    return db_dispute


@router.get("/", response_model=DisputeListResponse)
def get_disputes(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[DisputeStatus] = None,
    dispute_type: Optional[DisputeType] = None,
    db: Session = Depends(get_db),
):
    """Get all disputes with filtering"""
    query = db.query(MarketDispute)

    if status:
        query = query.filter(MarketDispute.status == status)

    if dispute_type:
        query = query.filter(MarketDispute.dispute_type == dispute_type)

    total = query.count()
    disputes = (
        query.order_by(desc(MarketDispute.created_at)).offset(skip).limit(limit).all()
    )

    return DisputeListResponse(
        disputes=disputes, total=total, page=skip // limit + 1, per_page=limit
    )


@router.get("/{dispute_id}", response_model=DisputeResponse)
def get_dispute(dispute_id: int, db: Session = Depends(get_db)):
    """Get specific dispute details"""
    dispute = db.query(MarketDispute).filter(MarketDispute.id == dispute_id).first()
    if not dispute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dispute not found"
        )
    return dispute


@router.post("/{dispute_id}/vote")
def vote_on_dispute(
    dispute_id: int,
    vote_data: DisputeVoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Vote on a dispute"""

    # Check if dispute exists
    dispute = db.query(MarketDispute).filter(MarketDispute.id == dispute_id).first()
    if not dispute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dispute not found"
        )

    # Check if dispute is still open
    if dispute.is_resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dispute is already resolved",
        )

    # Check if user already voted
    existing_vote = (
        db.query(DisputeVote)
        .filter(
            DisputeVote.dispute_id == dispute_id,
            DisputeVote.voter_id == current_user.id,
        )
        .first()
    )

    if existing_vote:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already voted on this dispute",
        )

    # Create vote
    db_vote = DisputeVote(
        dispute_id=dispute_id,
        voter_id=current_user.id,
        vote_for_dispute=vote_data.vote_for_dispute,
        reason=vote_data.reason,
    )

    db.add(db_vote)

    # Update dispute vote counts
    if vote_data.vote_for_dispute:
        dispute.votes_for_dispute += 1
    else:
        dispute.votes_against_dispute += 1

    # Check if consensus is reached
    if dispute.consensus_reached:
        if dispute.dispute_wins:
            dispute.status = DisputeStatus.RESOLVED
            # Update market status
            market = db.query(Market).filter(Market.id == dispute.market_id).first()
            if market:
                market.status = MarketStatus.DISPUTED
        else:
            dispute.status = DisputeStatus.DISMISSED
            # Revert market status
            market = db.query(Market).filter(Market.id == dispute.market_id).first()
            if market:
                market.status = MarketStatus.RESOLVED

    db.commit()

    return {"message": "Vote recorded successfully"}


@router.put("/{dispute_id}", response_model=DisputeResponse)
def update_dispute(
    dispute_id: int,
    dispute_update: DisputeUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update dispute (admin/moderator only)"""

    # Check if user is admin/moderator (simplified check)
    if current_user.reputation_score < 500:  # Require high reputation
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions"
        )

    dispute = db.query(MarketDispute).filter(MarketDispute.id == dispute_id).first()
    if not dispute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dispute not found"
        )

    # Update dispute fields
    for field, value in dispute_update.dict(exclude_unset=True).items():
        setattr(dispute, field, value)

    dispute.reviewed_by = current_user.id
    dispute.reviewed_at = datetime.utcnow()

    db.commit()
    db.refresh(dispute)

    return dispute


@router.get("/market/{market_id}")
def get_market_disputes(market_id: int, db: Session = Depends(get_db)):
    """Get all disputes for a specific market"""
    disputes = (
        db.query(MarketDispute)
        .filter(MarketDispute.market_id == market_id)
        .order_by(desc(MarketDispute.created_at))
        .all()
    )

    return {"market_id": market_id, "disputes": disputes, "total": len(disputes)}
