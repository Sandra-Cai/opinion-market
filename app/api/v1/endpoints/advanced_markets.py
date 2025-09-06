from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.models.advanced_markets import (
    FuturesContract,
    FuturesPosition,
    OptionsContract,
    OptionsPosition,
    ConditionalMarket,
    SpreadMarket,
    MarketInstrument,
    OptionType,
)
from app.schemas.advanced_markets import (
    FuturesContractCreate,
    FuturesContractResponse,
    FuturesPositionCreate,
    FuturesPositionResponse,
    OptionsContractCreate,
    OptionsContractResponse,
    OptionsPositionCreate,
    OptionsPositionResponse,
    ConditionalMarketCreate,
    ConditionalMarketResponse,
    SpreadMarketCreate,
    SpreadMarketResponse,
    MarketInstrumentResponse,
)

router = APIRouter()


# Futures Contracts
@router.post("/futures/contracts", response_model=FuturesContractResponse)
def create_futures_contract(
    contract_data: FuturesContractCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new futures contract"""

    # Check if market exists and is active
    market = db.query(Market).filter(Market.id == contract_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    if not market.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Market is not active"
        )

    # Check if user has permission to create futures contracts
    if current_user.reputation_score < 1000:  # High reputation required
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient reputation to create futures contracts",
        )

    # Create futures contract
    db_contract = FuturesContract(
        market_id=contract_data.market_id,
        contract_size=contract_data.contract_size,
        tick_size=contract_data.tick_size,
        margin_requirement=contract_data.margin_requirement,
        settlement_date=contract_data.settlement_date,
        cash_settlement=contract_data.cash_settlement,
        max_position_size=contract_data.max_position_size,
        daily_price_limit=contract_data.daily_price_limit,
    )

    db.add(db_contract)
    db.commit()
    db.refresh(db_contract)

    return db_contract


@router.get("/futures/contracts", response_model=List[FuturesContractResponse])
def get_futures_contracts(
    market_id: Optional[int] = None,
    active_only: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Get futures contracts"""
    query = db.query(FuturesContract)

    if market_id:
        query = query.filter(FuturesContract.market_id == market_id)

    if active_only:
        query = query.filter(FuturesContract.settlement_date > datetime.utcnow())

    contracts = query.order_by(FuturesContract.settlement_date).all()
    return contracts


@router.get("/futures/contracts/{contract_id}", response_model=FuturesContractResponse)
def get_futures_contract(contract_id: int, db: Session = Depends(get_db)):
    """Get specific futures contract"""
    contract = (
        db.query(FuturesContract).filter(FuturesContract.id == contract_id).first()
    )
    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Futures contract not found"
        )
    return contract


@router.post("/futures/positions", response_model=FuturesPositionResponse)
def create_futures_position(
    position_data: FuturesPositionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new futures position"""

    # Check if contract exists and is active
    contract = (
        db.query(FuturesContract)
        .filter(FuturesContract.id == position_data.contract_id)
        .first()
    )
    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Futures contract not found"
        )

    if contract.is_settled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Contract is already settled",
        )

    # Check position limits
    if contract.max_position_size:
        total_position = position_data.long_contracts + position_data.short_contracts
        if total_position > contract.max_position_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Position size exceeds maximum allowed",
            )

    # Calculate margin requirement
    current_price = contract.market.current_price_a
    margin_required = contract.calculate_margin_requirement(
        position_data.long_contracts + position_data.short_contracts, current_price
    )

    if current_user.available_balance < margin_required:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient balance for margin requirement",
        )

    # Create position
    db_position = FuturesPosition(
        user_id=current_user.id,
        contract_id=position_data.contract_id,
        long_contracts=position_data.long_contracts,
        short_contracts=position_data.short_contracts,
        average_entry_price=current_price,
        margin_used=margin_required,
    )

    db.add(db_position)

    # Deduct margin from user balance
    current_user.available_balance -= margin_required

    db.commit()
    db.refresh(db_position)

    return db_position


@router.get("/futures/positions", response_model=List[FuturesPositionResponse])
def get_futures_positions(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get user's futures positions"""
    positions = (
        db.query(FuturesPosition)
        .filter(FuturesPosition.user_id == current_user.id)
        .order_by(desc(FuturesPosition.created_at))
        .all()
    )

    # Update P&L for all positions
    for position in positions:
        current_price = position.contract.market.current_price_a
        position.update_pnl(current_price)

    db.commit()
    return positions


# Options Contracts
@router.post("/options/contracts", response_model=OptionsContractResponse)
def create_options_contract(
    contract_data: OptionsContractCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new options contract"""

    # Check if market exists and is active
    market = db.query(Market).filter(Market.id == contract_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    if not market.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Market is not active"
        )

    # Check if user has permission to create options contracts
    if current_user.reputation_score < 1500:  # Very high reputation required
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient reputation to create options contracts",
        )

    # Create options contract
    db_contract = OptionsContract(
        market_id=contract_data.market_id,
        option_type=contract_data.option_type,
        strike_price=contract_data.strike_price,
        expiration_date=contract_data.expiration_date,
        contract_size=contract_data.contract_size,
        premium=contract_data.premium,
    )

    db.add(db_contract)
    db.commit()
    db.refresh(db_contract)

    return db_contract


@router.get("/options/contracts", response_model=List[OptionsContractResponse])
def get_options_contracts(
    market_id: Optional[int] = None,
    option_type: Optional[OptionType] = None,
    active_only: bool = Query(True),
    db: Session = Depends(get_db),
):
    """Get options contracts"""
    query = db.query(OptionsContract)

    if market_id:
        query = query.filter(OptionsContract.market_id == market_id)

    if option_type:
        query = query.filter(OptionsContract.option_type == option_type)

    if active_only:
        query = query.filter(OptionsContract.expiration_date > datetime.utcnow())

    contracts = query.order_by(OptionsContract.expiration_date).all()
    return contracts


@router.post("/options/positions", response_model=OptionsPositionResponse)
def create_options_position(
    position_data: OptionsPositionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new options position"""

    # Check if contract exists and is active
    contract = (
        db.query(OptionsContract)
        .filter(OptionsContract.id == position_data.contract_id)
        .first()
    )
    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Options contract not found"
        )

    if contract.is_expired:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Contract is expired"
        )

    # Calculate required balance
    total_cost = (
        position_data.long_contracts + position_data.short_contracts
    ) * contract.premium

    if current_user.available_balance < total_cost:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient balance"
        )

    # Create position
    db_position = OptionsPosition(
        user_id=current_user.id,
        contract_id=position_data.contract_id,
        long_contracts=position_data.long_contracts,
        short_contracts=position_data.short_contracts,
        average_entry_price=contract.premium,
    )

    db.add(db_position)

    # Deduct cost from user balance
    current_user.available_balance -= total_cost

    db.commit()
    db.refresh(db_position)

    return db_position


# Conditional Markets
@router.post("/conditional", response_model=ConditionalMarketResponse)
def create_conditional_market(
    conditional_data: ConditionalMarketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a conditional market"""

    # Check if market exists
    market = db.query(Market).filter(Market.id == conditional_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Check if trigger market exists (if specified)
    if conditional_data.trigger_market_id:
        trigger_market = (
            db.query(Market)
            .filter(Market.id == conditional_data.trigger_market_id)
            .first()
        )
        if not trigger_market:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Trigger market not found"
            )

    # Create conditional market
    db_conditional = ConditionalMarket(
        market_id=conditional_data.market_id,
        condition_description=conditional_data.condition_description,
        trigger_condition=conditional_data.trigger_condition,
        trigger_market_id=conditional_data.trigger_market_id,
    )

    db.add(db_conditional)
    db.commit()
    db.refresh(db_conditional)

    return db_conditional


@router.post("/conditional/{conditional_id}/activate")
def activate_conditional_market(
    conditional_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Activate a conditional market"""

    conditional = (
        db.query(ConditionalMarket)
        .filter(ConditionalMarket.id == conditional_id)
        .first()
    )
    if not conditional:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conditional market not found"
        )

    if conditional.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conditional market is already active",
        )

    # Check trigger condition
    if conditional.check_trigger_condition():
        conditional.activate(current_user.id)
        db.commit()
        return {"message": "Conditional market activated successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Trigger condition not met"
        )


# Spread Markets
@router.post("/spread", response_model=SpreadMarketResponse)
def create_spread_market(
    spread_data: SpreadMarketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a spread market"""

    # Check if market exists
    market = db.query(Market).filter(Market.id == spread_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Create spread market
    db_spread = SpreadMarket(
        market_id=spread_data.market_id,
        spread_type=spread_data.spread_type,
        min_value=spread_data.min_value,
        max_value=spread_data.max_value,
        tick_size=spread_data.tick_size,
    )

    # Generate outcomes
    db_spread.generate_outcomes()

    db.add(db_spread)
    db.commit()
    db.refresh(db_spread)

    return db_spread


@router.get("/instruments/{market_id}")
def get_market_instruments(market_id: int, db: Session = Depends(get_db)):
    """Get all instruments for a specific market"""

    # Check if market exists
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Get futures contracts
    futures_contracts = (
        db.query(FuturesContract).filter(FuturesContract.market_id == market_id).all()
    )

    # Get options contracts
    options_contracts = (
        db.query(OptionsContract).filter(OptionsContract.market_id == market_id).all()
    )

    # Get conditional markets
    conditional_markets = (
        db.query(ConditionalMarket)
        .filter(ConditionalMarket.market_id == market_id)
        .all()
    )

    # Get spread markets
    spread_markets = (
        db.query(SpreadMarket).filter(SpreadMarket.market_id == market_id).all()
    )

    return {
        "market_id": market_id,
        "futures_contracts": futures_contracts,
        "options_contracts": options_contracts,
        "conditional_markets": conditional_markets,
        "spread_markets": spread_markets,
    }
