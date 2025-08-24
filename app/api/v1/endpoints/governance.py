from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.governance import (
    GovernanceProposal, GovernanceVote, GovernanceToken,
    ProposalType, ProposalStatus, VoteType
)
from app.schemas.governance import (
    GovernanceProposalCreate, GovernanceProposalUpdate, GovernanceProposalResponse,
    GovernanceProposalListResponse, GovernanceVoteCreate, GovernanceVoteResponse,
    GovernanceTokenResponse, GovernanceTokenStakeRequest, GovernanceTokenUnstakeRequest,
    GovernanceStatsResponse
)

router = APIRouter()

@router.post("/proposals", response_model=GovernanceProposalResponse)
def create_proposal(
    proposal_data: GovernanceProposalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new governance proposal"""
    
    # Check if user has enough voting power to create proposal
    user_tokens = db.query(GovernanceToken).filter(
        GovernanceToken.user_id == current_user.id
    ).first()
    
    if not user_tokens or user_tokens.total_voting_power < 1000:  # Minimum 1000 tokens required
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient voting power to create proposal"
        )
    
    # Validate voting period
    if proposal_data.voting_start <= datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voting start must be in the future"
        )
    
    if proposal_data.voting_end <= proposal_data.voting_start:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voting end must be after voting start"
        )
    
    # Create proposal
    db_proposal = GovernanceProposal(
        title=proposal_data.title,
        description=proposal_data.description,
        proposal_type=proposal_data.proposal_type,
        voting_start=proposal_data.voting_start,
        voting_end=proposal_data.voting_end,
        quorum_required=proposal_data.quorum_required,
        majority_required=proposal_data.majority_required,
        metadata=proposal_data.metadata or {},
        created_by=current_user.id
    )
    
    db.add(db_proposal)
    db.commit()
    db.refresh(db_proposal)
    
    return db_proposal

@router.get("/proposals", response_model=GovernanceProposalListResponse)
def get_proposals(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[ProposalStatus] = None,
    proposal_type: Optional[ProposalType] = None,
    db: Session = Depends(get_db)
):
    """Get governance proposals with filtering"""
    query = db.query(GovernanceProposal)
    
    if status:
        query = query.filter(GovernanceProposal.status == status)
    
    if proposal_type:
        query = query.filter(GovernanceProposal.proposal_type == proposal_type)
    
    total = query.count()
    proposals = query.order_by(desc(GovernanceProposal.created_at)).offset(skip).limit(limit).all()
    
    return GovernanceProposalListResponse(
        proposals=proposals,
        total=total,
        page=skip // limit + 1,
        per_page=limit
    )

@router.get("/proposals/{proposal_id}", response_model=GovernanceProposalResponse)
def get_proposal(proposal_id: int, db: Session = Depends(get_db)):
    """Get specific governance proposal"""
    proposal = db.query(GovernanceProposal).filter(GovernanceProposal.id == proposal_id).first()
    if not proposal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Proposal not found"
        )
    return proposal

@router.put("/proposals/{proposal_id}", response_model=GovernanceProposalResponse)
def update_proposal(
    proposal_id: int,
    proposal_update: GovernanceProposalUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a governance proposal (creator only)"""
    proposal = db.query(GovernanceProposal).filter(GovernanceProposal.id == proposal_id).first()
    if not proposal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Proposal not found"
        )
    
    if proposal.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only proposal creator can update"
        )
    
    if proposal.status != ProposalStatus.DRAFT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only update draft proposals"
        )
    
    # Update proposal fields
    for field, value in proposal_update.dict(exclude_unset=True).items():
        setattr(proposal, field, value)
    
    db.commit()
    db.refresh(proposal)
    
    return proposal

@router.post("/proposals/{proposal_id}/vote")
def vote_on_proposal(
    proposal_id: int,
    vote_data: GovernanceVoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Vote on a governance proposal"""
    
    # Check if proposal exists and is active
    proposal = db.query(GovernanceProposal).filter(GovernanceProposal.id == proposal_id).first()
    if not proposal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Proposal not found"
        )
    
    if not proposal.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Proposal is not active for voting"
        )
    
    # Check if user already voted
    existing_vote = db.query(GovernanceVote).filter(
        GovernanceVote.proposal_id == proposal_id,
        GovernanceVote.voter_id == current_user.id
    ).first()
    
    if existing_vote:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already voted on this proposal"
        )
    
    # Get user's voting power
    user_tokens = db.query(GovernanceToken).filter(
        GovernanceToken.user_id == current_user.id
    ).first()
    
    if not user_tokens or user_tokens.available_amount < vote_data.voting_power:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient voting power"
        )
    
    # Create vote
    db_vote = GovernanceVote(
        proposal_id=proposal_id,
        voter_id=current_user.id,
        vote_type=vote_data.vote_type,
        voting_power=vote_data.voting_power,
        reason=vote_data.reason
    )
    
    db.add(db_vote)
    
    # Lock tokens for voting
    user_tokens.lock_tokens(vote_data.voting_power)
    
    # Update proposal results
    proposal.calculate_results()
    
    db.commit()
    
    return {"message": "Vote recorded successfully"}

@router.get("/proposals/{proposal_id}/votes")
def get_proposal_votes(
    proposal_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get votes for a specific proposal"""
    proposal = db.query(GovernanceProposal).filter(GovernanceProposal.id == proposal_id).first()
    if not proposal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Proposal not found"
        )
    
    votes = db.query(GovernanceVote).filter(
        GovernanceVote.proposal_id == proposal_id
    ).order_by(desc(GovernanceVote.created_at)).offset(skip).limit(limit).all()
    
    return {
        "proposal_id": proposal_id,
        "votes": votes,
        "total": len(votes)
    }

@router.get("/tokens/me", response_model=GovernanceTokenResponse)
def get_my_tokens(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's governance tokens"""
    tokens = db.query(GovernanceToken).filter(
        GovernanceToken.user_id == current_user.id
    ).first()
    
    if not tokens:
        # Create default token record
        tokens = GovernanceToken(
            user_id=current_user.id,
            token_amount=100.0  # Default tokens for new users
        )
        db.add(tokens)
        db.commit()
        db.refresh(tokens)
    
    return tokens

@router.post("/tokens/stake")
def stake_tokens(
    stake_data: GovernanceTokenStakeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stake governance tokens for additional voting power"""
    tokens = db.query(GovernanceToken).filter(
        GovernanceToken.user_id == current_user.id
    ).first()
    
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No governance tokens found"
        )
    
    if tokens.available_amount < stake_data.amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient available tokens"
        )
    
    tokens.stake_tokens(stake_data.amount)
    db.commit()
    
    return {"message": f"Successfully staked {stake_data.amount} tokens"}

@router.post("/tokens/unstake")
def unstake_tokens(
    unstake_data: GovernanceTokenUnstakeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unstake governance tokens"""
    tokens = db.query(GovernanceToken).filter(
        GovernanceToken.user_id == current_user.id
    ).first()
    
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No governance tokens found"
        )
    
    if tokens.staked_amount < unstake_data.amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient staked tokens"
        )
    
    tokens.unstake_tokens(unstake_data.amount)
    db.commit()
    
    return {"message": f"Successfully unstaked {unstake_data.amount} tokens"}

@router.get("/stats", response_model=GovernanceStatsResponse)
def get_governance_stats(db: Session = Depends(get_db)):
    """Get governance statistics"""
    
    # Get proposal statistics
    total_proposals = db.query(GovernanceProposal).count()
    active_proposals = db.query(GovernanceProposal).filter(
        GovernanceProposal.status == ProposalStatus.ACTIVE
    ).count()
    passed_proposals = db.query(GovernanceProposal).filter(
        GovernanceProposal.status == ProposalStatus.PASSED
    ).count()
    rejected_proposals = db.query(GovernanceProposal).filter(
        ProposalStatus.REJECTED
    ).count()
    
    # Get voting statistics
    total_votes = db.query(GovernanceVote).count()
    total_voting_power = db.query(func.sum(GovernanceVote.voting_power)).scalar() or 0
    
    # Calculate participation rate
    total_users = db.query(User).count()
    users_voted = db.query(GovernanceVote.voter_id).distinct().count()
    participation_rate = (users_voted / total_users * 100) if total_users > 0 else 0
    
    # Get top voters
    top_voters = db.query(
        GovernanceVote.voter_id,
        func.sum(GovernanceVote.voting_power).label('total_power'),
        func.count(GovernanceVote.id).label('votes_cast')
    ).group_by(GovernanceVote.voter_id).order_by(
        desc('total_power')
    ).limit(10).all()
    
    return GovernanceStatsResponse(
        total_proposals=total_proposals,
        active_proposals=active_proposals,
        passed_proposals=passed_proposals,
        rejected_proposals=rejected_proposals,
        total_votes_cast=total_votes,
        total_voting_power=total_voting_power,
        average_participation_rate=participation_rate,
        top_voters=[
            {
                "user_id": voter.voter_id,
                "total_power": voter.total_power,
                "votes_cast": voter.votes_cast
            }
            for voter in top_voters
        ]
    )


