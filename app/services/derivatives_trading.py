"""
Derivatives Trading Service
Provides futures, forwards, swaps, and exotic derivatives trading capabilities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class DerivativeContract:
    """Derivative contract information"""
    contract_id: str
    contract_type: str  # 'futures', 'forwards', 'swaps', 'exotic'
    underlying_asset: str
    contract_size: float
    tick_size: float
    margin_requirement: float
    initial_margin: float
    maintenance_margin: float
    settlement_type: str  # 'physical', 'cash'
    settlement_date: datetime
    last_trading_date: datetime
    current_price: float
    underlying_price: float
    open_interest: int
    volume: int
    bid_price: float
    ask_price: float
    created_at: datetime
    last_updated: datetime


@dataclass
class FuturesContract(DerivativeContract):
    """Futures contract specific information"""
    exchange: str
    contract_month: str
    delivery_location: Optional[str]
    delivery_terms: Optional[str]
    price_limit_up: Optional[float]
    price_limit_down: Optional[float]


@dataclass
class ForwardContract(DerivativeContract):
    """Forward contract specific information"""
    counterparty: str
    delivery_price: float
    delivery_location: str
    quality_specifications: Dict[str, Any]


@dataclass
class SwapContract(DerivativeContract):
    """Swap contract specific information"""
    swap_type: str  # 'interest_rate', 'currency', 'commodity', 'credit'
    notional_amount: float
    fixed_rate: float
    floating_rate_index: str
    payment_frequency: str
    next_payment_date: datetime
    last_reset_date: datetime


@dataclass
class ExoticDerivative(DerivativeContract):
    """Exotic derivative contract information"""
    exotic_type: str  # 'barrier', 'binary', 'asian', 'lookback', 'spread'
    payoff_function: str
    barrier_levels: Optional[List[float]]
    knock_in_out: Optional[str]
    averaging_period: Optional[int]
    lookback_period: Optional[int]


@dataclass
class DerivativePosition:
    """Derivative position"""
    position_id: str
    user_id: int
    contract_id: str
    position_type: str  # 'long', 'short'
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    margin_available: float
    leverage: float
    mark_to_market: float
    created_at: datetime
    last_updated: datetime


@dataclass
class MarginAccount:
    """Margin account information"""
    account_id: str
    user_id: int
    total_equity: float
    total_margin: float
    available_margin: float
    used_margin: float
    margin_ratio: float
    margin_call_level: float
    liquidation_level: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    last_updated: datetime


@dataclass
class SettlementInstruction:
    """Settlement instruction"""
    instruction_id: str
    contract_id: str
    user_id: int
    settlement_type: str  # 'delivery', 'cash'
    settlement_amount: float
    settlement_date: datetime
    delivery_location: Optional[str]
    delivery_instructions: Optional[str]
    status: str  # 'pending', 'confirmed', 'completed', 'failed'
    created_at: datetime


class DerivativesTradingService:
    """Comprehensive derivatives trading service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.derivative_contracts: Dict[str, DerivativeContract] = {}
        self.derivative_positions: Dict[str, DerivativePosition] = {}
        self.margin_accounts: Dict[str, MarginAccount] = {}
        self.settlement_instructions: Dict[str, SettlementInstruction] = {}
        
        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.open_interest_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Risk management
        self.position_limits: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.margin_requirements: Dict[str, float] = {}
        
    async def initialize(self):
        """Initialize the derivatives trading service"""
        logger.info("Initializing Derivatives Trading Service")
        
        # Load existing data
        await self._load_derivative_contracts()
        await self._load_margin_accounts()
        await self._load_settlement_instructions()
        
        # Start background tasks
        asyncio.create_task(self._update_market_data())
        asyncio.create_task(self._monitor_positions())
        asyncio.create_task(self._monitor_margin_accounts())
        asyncio.create_task(self._process_settlements())
        
        logger.info("Derivatives Trading Service initialized successfully")
    
    async def create_futures_contract(self, underlying_asset: str, contract_size: float, tick_size: float,
                                    margin_requirement: float, settlement_type: str, settlement_date: datetime,
                                    exchange: str, contract_month: str, delivery_location: Optional[str] = None) -> FuturesContract:
        """Create a new futures contract"""
        try:
            contract_id = f"futures_{underlying_asset}_{contract_month}_{exchange}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate dates
            last_trading_date = settlement_date - timedelta(days=1)
            
            contract = FuturesContract(
                contract_id=contract_id,
                contract_type='futures',
                underlying_asset=underlying_asset,
                contract_size=contract_size,
                tick_size=tick_size,
                margin_requirement=margin_requirement,
                initial_margin=margin_requirement,
                maintenance_margin=margin_requirement * 0.8,
                settlement_type=settlement_type,
                settlement_date=settlement_date,
                last_trading_date=last_trading_date,
                current_price=0.0,
                underlying_price=0.0,
                open_interest=0,
                volume=0,
                bid_price=0.0,
                ask_price=0.0,
                exchange=exchange,
                contract_month=contract_month,
                delivery_location=delivery_location,
                delivery_terms=None,
                price_limit_up=None,
                price_limit_down=None,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.derivative_contracts[contract_id] = contract
            await self._cache_derivative_contract(contract)
            
            logger.info(f"Created futures contract {contract_id}")
            return contract
            
        except Exception as e:
            logger.error(f"Error creating futures contract: {e}")
            raise
    
    async def create_forward_contract(self, underlying_asset: str, contract_size: float, delivery_price: float,
                                    settlement_date: datetime, counterparty: str, delivery_location: str,
                                    quality_specifications: Dict[str, Any]) -> ForwardContract:
        """Create a new forward contract"""
        try:
            contract_id = f"forward_{underlying_asset}_{counterparty}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            contract = ForwardContract(
                contract_id=contract_id,
                contract_type='forwards',
                underlying_asset=underlying_asset,
                contract_size=contract_size,
                tick_size=0.01,
                margin_requirement=0.0,  # Forwards typically don't require margin
                initial_margin=0.0,
                maintenance_margin=0.0,
                settlement_type='physical',
                settlement_date=settlement_date,
                last_trading_date=settlement_date,
                current_price=delivery_price,
                underlying_price=0.0,
                open_interest=1,
                volume=0,
                bid_price=delivery_price,
                ask_price=delivery_price,
                counterparty=counterparty,
                delivery_price=delivery_price,
                delivery_location=delivery_location,
                quality_specifications=quality_specifications,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.derivative_contracts[contract_id] = contract
            await self._cache_derivative_contract(contract)
            
            logger.info(f"Created forward contract {contract_id}")
            return contract
            
        except Exception as e:
            logger.error(f"Error creating forward contract: {e}")
            raise
    
    async def create_swap_contract(self, underlying_asset: str, swap_type: str, notional_amount: float,
                                 fixed_rate: float, floating_rate_index: str, payment_frequency: str,
                                 settlement_date: datetime, maturity_date: datetime) -> SwapContract:
        """Create a new swap contract"""
        try:
            contract_id = f"swap_{swap_type}_{underlying_asset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate payment dates
            next_payment_date = self._calculate_next_payment_date(settlement_date, payment_frequency)
            
            contract = SwapContract(
                contract_id=contract_id,
                contract_type='swaps',
                underlying_asset=underlying_asset,
                contract_size=notional_amount,
                tick_size=0.0001,
                margin_requirement=0.0,  # Swaps typically don't require margin
                initial_margin=0.0,
                maintenance_margin=0.0,
                settlement_type='cash',
                settlement_date=settlement_date,
                last_trading_date=maturity_date,
                current_price=0.0,
                underlying_price=0.0,
                open_interest=1,
                volume=0,
                bid_price=0.0,
                ask_price=0.0,
                swap_type=swap_type,
                notional_amount=notional_amount,
                fixed_rate=fixed_rate,
                floating_rate_index=floating_rate_index,
                payment_frequency=payment_frequency,
                next_payment_date=next_payment_date,
                last_reset_date=settlement_date,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.derivative_contracts[contract_id] = contract_id
            await self._cache_derivative_contract(contract)
            
            logger.info(f"Created swap contract {contract_id}")
            return contract
            
        except Exception as e:
            logger.error(f"Error creating swap contract: {e}")
            raise
    
    async def create_exotic_derivative(self, underlying_asset: str, exotic_type: str, payoff_function: str,
                                     contract_size: float, settlement_date: datetime, barrier_levels: Optional[List[float]] = None,
                                     knock_in_out: Optional[str] = None, averaging_period: Optional[int] = None,
                                     lookback_period: Optional[int] = None) -> ExoticDerivative:
        """Create a new exotic derivative contract"""
        try:
            contract_id = f"exotic_{exotic_type}_{underlying_asset}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            contract = ExoticDerivative(
                contract_id=contract_id,
                contract_type='exotic',
                underlying_asset=underlying_asset,
                contract_size=contract_size,
                tick_size=0.01,
                margin_requirement=0.05,  # Higher margin for exotic derivatives
                initial_margin=0.05,
                maintenance_margin=0.04,
                settlement_type='cash',
                settlement_date=settlement_date,
                last_trading_date=settlement_date,
                current_price=0.0,
                underlying_price=0.0,
                open_interest=0,
                volume=0,
                bid_price=0.0,
                ask_price=0.0,
                exotic_type=exotic_type,
                payoff_function=payoff_function,
                barrier_levels=barrier_levels,
                knock_in_out=knock_in_out,
                averaging_period=averaging_period,
                lookback_period=lookback_period,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.derivative_contracts[contract_id] = contract
            await self._cache_derivative_contract(contract)
            
            logger.info(f"Created exotic derivative {contract_id}")
            return contract
            
        except Exception as e:
            logger.error(f"Error creating exotic derivative: {e}")
            raise
    
    async def open_position(self, user_id: int, contract_id: str, position_type: str, quantity: int,
                           entry_price: float, leverage: float = 1.0) -> DerivativePosition:
        """Open a derivative position"""
        try:
            contract = self.derivative_contracts.get(contract_id)
            if not contract:
                raise ValueError(f"Derivative contract {contract_id} not found")
            
            position_id = f"pos_{user_id}_{contract_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate margin requirements
            position_value = quantity * entry_price * contract.contract_size
            required_margin = position_value * contract.margin_requirement / leverage
            
            # Check margin availability
            margin_account = await self._get_margin_account(user_id)
            if margin_account.available_margin < required_margin:
                raise ValueError(f"Insufficient margin. Required: {required_margin}, Available: {margin_account.available_margin}")
            
            position = DerivativePosition(
                position_id=position_id,
                user_id=user_id,
                contract_id=contract_id,
                position_type=position_type,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                margin_used=required_margin,
                margin_available=margin_account.available_margin - required_margin,
                leverage=leverage,
                mark_to_market=position_value,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.derivative_positions[position_id] = position
            await self._cache_derivative_position(position)
            
            # Update margin account
            await self._update_margin_account(user_id, required_margin, True)
            
            logger.info(f"Opened derivative position {position_id}")
            return position
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            raise
    
    async def close_position(self, position_id: str, exit_price: float) -> DerivativePosition:
        """Close a derivative position"""
        try:
            position = self.derivative_positions.get(position_id)
            if not position:
                raise ValueError(f"Position {position_id} not found")
            
            contract = self.derivative_contracts.get(position.contract_id)
            if not contract:
                raise ValueError(f"Contract {position.contract_id} not found")
            
            # Calculate P&L
            if position.position_type == 'long':
                pnl = (exit_price - position.entry_price) * position.quantity * contract.contract_size
            else:  # short
                pnl = (position.entry_price - exit_price) * position.quantity * contract.contract_size
            
            # Update position
            position.current_price = exit_price
            position.realized_pnl = pnl
            position.unrealized_pnl = 0.0
            position.last_updated = datetime.utcnow()
            
            # Release margin
            await self._update_margin_account(position.user_id, position.margin_used, False)
            
            # Update position
            await self._cache_derivative_position(position)
            
            logger.info(f"Closed derivative position {position_id} with P&L: {pnl}")
            return position
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            raise
    
    async def calculate_mark_to_market(self, position_id: str) -> float:
        """Calculate mark to market value for a position"""
        try:
            position = self.derivative_positions.get(position_id)
            if not position:
                return 0.0
            
            contract = self.derivative_contracts.get(position.contract_id)
            if not contract:
                return 0.0
            
            # Get current market price
            current_price = contract.current_price
            if current_price == 0:
                current_price = contract.underlying_price
            
            # Calculate mark to market
            position_value = position.quantity * current_price * contract.contract_size
            
            # Calculate unrealized P&L
            if position.position_type == 'long':
                unrealized_pnl = (current_price - position.entry_price) * position.quantity * contract.contract_size
            else:  # short
                unrealized_pnl = (position.entry_price - current_price) * position.quantity * contract.contract_size
            
            position.unrealized_pnl = unrealized_pnl
            position.mark_to_market = position_value
            position.last_updated = datetime.utcnow()
            
            await self._cache_derivative_position(position)
            
            return position_value
            
        except Exception as e:
            logger.error(f"Error calculating mark to market: {e}")
            return 0.0
    
    async def calculate_margin_requirements(self, user_id: int) -> Dict[str, float]:
        """Calculate margin requirements for a user"""
        try:
            user_positions = [
                pos for pos in self.derivative_positions.values()
                if pos.user_id == user_id
            ]
            
            total_margin_used = sum(pos.margin_used for pos in user_positions)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in user_positions)
            
            # Calculate margin ratio
            margin_account = await self._get_margin_account(user_id)
            margin_ratio = total_margin_used / margin_account.total_equity if margin_account.total_equity > 0 else 0
            
            # Determine risk level
            if margin_ratio <= 0.3:
                risk_level = 'low'
            elif margin_ratio <= 0.6:
                risk_level = 'medium'
            elif margin_ratio <= 0.8:
                risk_level = 'high'
            else:
                risk_level = 'critical'
            
            return {
                'total_margin_used': total_margin_used,
                'total_unrealized_pnl': total_unrealized_pnl,
                'margin_ratio': margin_ratio,
                'risk_level': risk_level,
                'available_margin': margin_account.available_margin,
                'total_equity': margin_account.total_equity
            }
            
        except Exception as e:
            logger.error(f"Error calculating margin requirements: {e}")
            raise
    
    async def create_settlement_instruction(self, contract_id: str, user_id: int, settlement_type: str,
                                          settlement_amount: float, settlement_date: datetime,
                                          delivery_location: Optional[str] = None,
                                          delivery_instructions: Optional[str] = None) -> SettlementInstruction:
        """Create a settlement instruction"""
        try:
            instruction_id = f"settlement_{contract_id}_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            instruction = SettlementInstruction(
                instruction_id=instruction_id,
                contract_id=contract_id,
                user_id=user_id,
                settlement_type=settlement_type,
                settlement_amount=settlement_amount,
                settlement_date=settlement_date,
                delivery_location=delivery_location,
                delivery_instructions=delivery_instructions,
                status='pending',
                created_at=datetime.utcnow()
            )
            
            self.settlement_instructions[instruction_id] = instruction
            await self._cache_settlement_instruction(instruction)
            
            logger.info(f"Created settlement instruction {instruction_id}")
            return instruction
            
        except Exception as e:
            logger.error(f"Error creating settlement instruction: {e}")
            raise
    
    async def get_derivatives_chain(self, underlying_asset: str, contract_type: Optional[str] = None) -> Dict[str, Any]:
        """Get derivatives chain for an underlying asset"""
        try:
            contracts = []
            for contract in self.derivative_contracts.values():
                if contract.underlying_asset == underlying_asset:
                    if contract_type is None or contract.contract_type == contract_type:
                        contracts.append({
                            'contract_id': contract.contract_id,
                            'contract_type': contract.contract_type,
                            'contract_size': contract.contract_size,
                            'settlement_date': contract.settlement_date.isoformat(),
                            'current_price': contract.current_price,
                            'underlying_price': contract.underlying_price,
                            'open_interest': contract.open_interest,
                            'volume': contract.volume,
                            'bid_price': contract.bid_price,
                            'ask_price': contract.ask_price,
                            'margin_requirement': contract.margin_requirement
                        })
            
            # Sort by settlement date
            contracts.sort(key=lambda x: x['settlement_date'])
            
            return {
                'underlying_asset': underlying_asset,
                'contract_type': contract_type,
                'contracts': contracts,
                'total_contracts': len(contracts)
            }
            
        except Exception as e:
            logger.error(f"Error getting derivatives chain: {e}")
            raise
    
    def _calculate_next_payment_date(self, start_date: datetime, frequency: str) -> datetime:
        """Calculate next payment date based on frequency"""
        if frequency == 'monthly':
            return start_date + timedelta(days=30)
        elif frequency == 'quarterly':
            return start_date + timedelta(days=90)
        elif frequency == 'semi_annual':
            return start_date + timedelta(days=180)
        elif frequency == 'annual':
            return start_date + timedelta(days=365)
        else:
            return start_date + timedelta(days=30)
    
    async def _get_margin_account(self, user_id: int) -> MarginAccount:
        """Get margin account for a user"""
        account_id = f"margin_{user_id}"
        
        if account_id not in self.margin_accounts:
            # Create default margin account
            account = MarginAccount(
                account_id=account_id,
                user_id=user_id,
                total_equity=100000.0,  # Default starting equity
                total_margin=0.0,
                available_margin=100000.0,
                used_margin=0.0,
                margin_ratio=0.0,
                margin_call_level=0.8,
                liquidation_level=0.9,
                risk_level='low',
                last_updated=datetime.utcnow()
            )
            
            self.margin_accounts[account_id] = account
            await self._cache_margin_account(account)
        
        return self.margin_accounts[account_id]
    
    async def _update_margin_account(self, user_id: int, margin_amount: float, is_using: bool):
        """Update margin account"""
        try:
            account = await self._get_margin_account(user_id)
            
            if is_using:
                account.used_margin += margin_amount
                account.available_margin -= margin_amount
            else:
                account.used_margin -= margin_amount
                account.available_margin += margin_amount
            
            account.total_margin = account.used_margin
            account.margin_ratio = account.total_margin / account.total_equity if account.total_equity > 0 else 0
            
            # Update risk level
            if account.margin_ratio <= 0.3:
                account.risk_level = 'low'
            elif account.margin_ratio <= 0.6:
                account.risk_level = 'medium'
            elif account.margin_ratio <= 0.8:
                account.risk_level = 'high'
            else:
                account.risk_level = 'critical'
            
            account.last_updated = datetime.utcnow()
            await self._cache_margin_account(account)
            
        except Exception as e:
            logger.error(f"Error updating margin account: {e}")
    
    # Background tasks
    async def _update_market_data(self):
        """Update market data periodically"""
        while True:
            try:
                # Update market data for all contracts
                for contract in self.derivative_contracts.values():
                    # Simulate market data updates
                    if contract.contract_type == 'futures':
                        # Update futures prices based on underlying
                        contract.current_price = contract.underlying_price * (1 + np.random.normal(0, 0.01))
                        contract.bid_price = contract.current_price * 0.999
                        contract.ask_price = contract.current_price * 1.001
                        contract.volume += np.random.randint(0, 100)
                        contract.open_interest += np.random.randint(-10, 10)
                    
                    contract.last_updated = datetime.utcnow()
                    await self._cache_derivative_contract(contract)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(120)
    
    async def _monitor_positions(self):
        """Monitor derivative positions"""
        while True:
            try:
                for position in self.derivative_positions.values():
                    # Update mark to market
                    await self.calculate_mark_to_market(position.position_id)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_margin_accounts(self):
        """Monitor margin accounts"""
        while True:
            try:
                for account in self.margin_accounts.values():
                    # Check for margin calls
                    if account.margin_ratio >= account.margin_call_level:
                        logger.warning(f"Margin call for account {account.account_id}")
                    
                    # Check for liquidation
                    if account.margin_ratio >= account.liquidation_level:
                        logger.critical(f"Liquidation required for account {account.account_id}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring margin accounts: {e}")
                await asyncio.sleep(600)
    
    async def _process_settlements(self):
        """Process settlement instructions"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for instruction in self.settlement_instructions.values():
                    if instruction.status == 'pending' and instruction.settlement_date <= current_time:
                        # Process settlement
                        instruction.status = 'completed'
                        instruction.created_at = current_time
                        await self._cache_settlement_instruction(instruction)
                        
                        logger.info(f"Processed settlement instruction {instruction.instruction_id}")
                
                await asyncio.sleep(3600)  # Process every hour
                
            except Exception as e:
                logger.error(f"Error processing settlements: {e}")
                await asyncio.sleep(7200)
    
    # Helper methods (implementations would depend on your data models)
    async def _load_derivative_contracts(self):
        """Load derivative contracts from database"""
        pass
    
    async def _load_margin_accounts(self):
        """Load margin accounts from database"""
        pass
    
    async def _load_settlement_instructions(self):
        """Load settlement instructions from database"""
        pass
    
    # Caching methods
    async def _cache_derivative_contract(self, contract: DerivativeContract):
        """Cache derivative contract"""
        try:
            cache_key = f"derivative_contract:{contract.contract_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps({
                    'contract_type': contract.contract_type,
                    'underlying_asset': contract.underlying_asset,
                    'contract_size': contract.contract_size,
                    'tick_size': contract.tick_size,
                    'margin_requirement': contract.margin_requirement,
                    'settlement_type': contract.settlement_type,
                    'settlement_date': contract.settlement_date.isoformat(),
                    'last_trading_date': contract.last_trading_date.isoformat(),
                    'current_price': contract.current_price,
                    'underlying_price': contract.underlying_price,
                    'open_interest': contract.open_interest,
                    'volume': contract.volume,
                    'bid_price': contract.bid_price,
                    'ask_price': contract.ask_price,
                    'created_at': contract.created_at.isoformat(),
                    'last_updated': contract.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching derivative contract: {e}")
    
    async def _cache_derivative_position(self, position: DerivativePosition):
        """Cache derivative position"""
        try:
            cache_key = f"derivative_position:{position.position_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps({
                    'user_id': position.user_id,
                    'contract_id': position.contract_id,
                    'position_type': position.position_type,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'margin_used': position.margin_used,
                    'margin_available': position.margin_available,
                    'leverage': position.leverage,
                    'mark_to_market': position.mark_to_market,
                    'created_at': position.created_at.isoformat(),
                    'last_updated': position.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching derivative position: {e}")
    
    async def _cache_margin_account(self, account: MarginAccount):
        """Cache margin account"""
        try:
            cache_key = f"margin_account:{account.account_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps({
                    'user_id': account.user_id,
                    'total_equity': account.total_equity,
                    'total_margin': account.total_margin,
                    'available_margin': account.available_margin,
                    'used_margin': account.used_margin,
                    'margin_ratio': account.margin_ratio,
                    'margin_call_level': account.margin_call_level,
                    'liquidation_level': account.liquidation_level,
                    'risk_level': account.risk_level,
                    'last_updated': account.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching margin account: {e}")
    
    async def _cache_settlement_instruction(self, instruction: SettlementInstruction):
        """Cache settlement instruction"""
        try:
            cache_key = f"settlement_instruction:{instruction.instruction_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'contract_id': instruction.contract_id,
                    'user_id': instruction.user_id,
                    'settlement_type': instruction.settlement_type,
                    'settlement_amount': instruction.settlement_amount,
                    'settlement_date': instruction.settlement_date.isoformat(),
                    'delivery_location': instruction.delivery_location,
                    'delivery_instructions': instruction.delivery_instructions,
                    'status': instruction.status,
                    'created_at': instruction.created_at.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching settlement instruction: {e}")


# Factory function
async def get_derivatives_trading_service(redis_client: redis.Redis, db_session: Session) -> DerivativesTradingService:
    """Get derivatives trading service instance"""
    service = DerivativesTradingService(redis_client, db_session)
    await service.initialize()
    return service
