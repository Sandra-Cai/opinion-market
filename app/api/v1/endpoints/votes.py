from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.models.vote import Vote, VoteOutcome
from app.schemas.vote import VoteCreate, VoteResponse, VoteListResponse

router = APIRouter()


@router.post("/", response_model=VoteResponse)
def create_vote(
    vote_data: VoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Get market
    market = db.query(Market).filter(Market.id == vote_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Check if market is active
    if not market.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market is not active for voting",
        )

    # Check if user already voted
    existing_vote = (
        db.query(Vote)
        .filter(Vote.market_id == vote_data.market_id, Vote.user_id == current_user.id)
        .first()
    )

    if existing_vote:
        # Update existing vote
        existing_vote.outcome = vote_data.outcome
        existing_vote.confidence = vote_data.confidence
        existing_vote.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(existing_vote)
        return existing_vote

    # Create new vote
    db_vote = Vote(
        outcome=vote_data.outcome,
        confidence=vote_data.confidence,
        market_id=vote_data.market_id,
        user_id=current_user.id,
    )

    db.add(db_vote)
    db.commit()
    db.refresh(db_vote)

    return db_vote


@router.get("/", response_model=VoteListResponse)
def get_votes(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    market_id: int = None,
    user_id: int = None,
    db: Session = Depends(get_db),
):
    query = db.query(Vote)

    if market_id:
        query = query.filter(Vote.market_id == market_id)

    if user_id:
        query = query.filter(Vote.user_id == user_id)

    total = query.count()
    votes = query.offset(skip).limit(limit).all()

    return VoteListResponse(
        votes=votes, total=total, page=skip // limit + 1, per_page=limit
    )


@router.get("/{vote_id}", response_model=VoteResponse)
def get_vote(vote_id: int, db: Session = Depends(get_db)):
    vote = db.query(Vote).filter(Vote.id == vote_id).first()
    if not vote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Vote not found"
        )
    return vote


@router.get("/user/me", response_model=VoteListResponse)
def get_my_votes(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Vote).filter(Vote.user_id == current_user.id)
    total = query.count()
    votes = query.offset(skip).limit(limit).all()

    return VoteListResponse(
        votes=votes, total=total, page=skip // limit + 1, per_page=limit
    )


@router.delete("/{vote_id}")
def delete_vote(
    vote_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vote = db.query(Vote).filter(Vote.id == vote_id).first()
    if not vote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Vote not found"
        )

    # Only user can delete their own vote
    if vote.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only delete your own vote",
        )

    db.delete(vote)
    db.commit()

    return {"message": "Vote deleted successfully"}
