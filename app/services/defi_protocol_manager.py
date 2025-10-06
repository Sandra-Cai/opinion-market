"""
DeFi Protocol Manager
Advanced DeFi protocol management and automation
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DeFiStrategyType(Enum):
    """DeFi strategy types"""
    YIELD_FARMING = "yield_farming"
    LIQUIDITY_PROVISION = "liquidity_provision"
    ARBITRAGE = "arbitrage"
    LENDING = "lending"
    BORROWING = "borrowing"
    STAKING = "staking"
    GOVERNANCE = "governance"
    OPTIONS_TRADING = "options_trading"

class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class PositionStatus(Enum):
    """Position status"""
    ACTIVE = "active"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"
    PENDING = "pending"

@dataclass
class DeFiStrategy:
    """DeFi strategy configuration"""
    strategy_id: str
    name: str
    strategy_type: DeFiStrategyType
    protocol: str
    blockchain: str
    risk_level: RiskLevel
    expected_apy: Decimal
    min_investment: Decimal
    max_investment: Decimal
    auto_compound: bool = True
    rebalance_frequency: int = 24  # hours
    stop_loss_percentage: Optional[Decimal] = None
    take_profit_percentage: Optional[Decimal] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeFiPosition:
    """DeFi position"""
    position_id: str
    user_address: str
    strategy_id: str
    protocol: str
    blockchain: str
    position_type: str
    token_address: str
    token_symbol: str
    amount: Decimal
    value_usd: Decimal
    apy: Decimal
    status: PositionStatus
    opened_at: datetime
    closed_at: Optional[datetime] = None
    pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class YieldOpportunity:
    """Yield opportunity"""
    opportunity_id: str
    protocol: str
    blockchain: str
    strategy_type: DeFiStrategyType
    token_pair: str
    apy: Decimal
    tvl: Decimal  # Total Value Locked
    risk_score: float
    liquidity_score: float
    gas_cost: Decimal
    min_investment: Decimal
    max_investment: Decimal
    discovered_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeFiTransaction:
    """DeFi transaction"""
    tx_id: str
    position_id: str
    transaction_type: str
    protocol: str
    blockchain: str
    token_address: str
    amount: Decimal
    gas_used: int
    gas_price: Decimal
    status: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class DeFiProtocolManager:
    """Advanced DeFi Protocol Manager"""
    
    def __init__(self):
        self.manager_id = f"defi_manager_{secrets.token_hex(8)}"
        self.is_running = False
        
        # DeFi data
        self.strategies: Dict[str, DeFiStrategy] = {}
        self.positions: List[DeFiPosition] = []
        self.yield_opportunities: List[YieldOpportunity] = []
        self.transactions: List[DeFiTransaction] = []
        
        # Protocol configurations
        self.protocol_configs = {
            "uniswap": {
                "router_address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "factory_address": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "fee_tier": 0.003,  # 0.3%
                "min_liquidity": Decimal("1000"),
                "supported_tokens": ["ETH", "USDC", "USDT", "DAI", "WBTC"]
            },
            "aave": {
                "lending_pool_address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
                "max_ltv": 0.8,  # 80% LTV
                "liquidation_threshold": 0.85,
                "supported_tokens": ["ETH", "USDC", "USDT", "DAI", "WBTC"],
                "borrow_rates": {
                    "ETH": 0.02,
                    "USDC": 0.03,
                    "USDT": 0.03,
                    "DAI": 0.02,
                    "WBTC": 0.02
                }
            },
            "compound": {
                "comptroller_address": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",
                "max_ltv": 0.75,
                "liquidation_threshold": 0.8,
                "supported_tokens": ["ETH", "USDC", "USDT", "DAI"],
                "borrow_rates": {
                    "ETH": 0.025,
                    "USDC": 0.035,
                    "USDT": 0.035,
                    "DAI": 0.025
                }
            },
            "yearn": {
                "vault_addresses": {
                    "USDC": "0x5f18C75AbDAe578b483E5F43f12a39cF75b973a9",
                    "USDT": "0x2f08119C6f07c006695E079AAFc638b8789FAf18",
                    "DAI": "0x19D3364A399d251E894aC732651be8B0E4e85001"
                },
                "management_fee": 0.02,  # 2%
                "performance_fee": 0.20,  # 20%
                "supported_tokens": ["USDC", "USDT", "DAI", "WETH"]
            }
        }
        
        # Risk management
        self.risk_parameters = {
            "max_position_size": Decimal("100000"),  # $100k max per position
            "max_total_exposure": Decimal("500000"),  # $500k max total exposure
            "max_protocol_exposure": Decimal("100000"),  # $100k max per protocol
            "min_liquidity_ratio": 0.1,  # 10% minimum liquidity
            "max_leverage": 3.0,  # 3x max leverage
            "stop_loss_threshold": 0.15,  # 15% stop loss
            "take_profit_threshold": 0.30  # 30% take profit
        }
        
        # Processing tasks
        self.strategy_execution_task: Optional[asyncio.Task] = None
        self.yield_hunting_task: Optional[asyncio.Task] = None
        self.risk_monitoring_task: Optional[asyncio.Task] = None
        self.position_rebalancing_task: Optional[asyncio.Task] = None
        self.performance_tracking_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.strategy_performance: Dict[str, List[float]] = {}
        self.protocol_performance: Dict[str, List[float]] = {}
        self.risk_metrics: Dict[str, float] = {}
        
        logger.info(f"DeFi Protocol Manager {self.manager_id} initialized")

    async def start_defi_manager(self):
        """Start the DeFi manager"""
        if self.is_running:
            return
        
        logger.info("Starting DeFi Protocol Manager...")
        
        # Initialize DeFi data
        await self._initialize_strategies()
        await self._initialize_positions()
        await self._initialize_yield_opportunities()
        
        # Start processing tasks
        self.is_running = True
        
        self.strategy_execution_task = asyncio.create_task(self._strategy_execution_loop())
        self.yield_hunting_task = asyncio.create_task(self._yield_hunting_loop())
        self.risk_monitoring_task = asyncio.create_task(self._risk_monitoring_loop())
        self.position_rebalancing_task = asyncio.create_task(self._position_rebalancing_loop())
        self.performance_tracking_task = asyncio.create_task(self._performance_tracking_loop())
        
        logger.info("DeFi Protocol Manager started")

    async def stop_defi_manager(self):
        """Stop the DeFi manager"""
        if not self.is_running:
            return
        
        logger.info("Stopping DeFi Protocol Manager...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.strategy_execution_task,
            self.yield_hunting_task,
            self.risk_monitoring_task,
            self.position_rebalancing_task,
            self.performance_tracking_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("DeFi Protocol Manager stopped")

    async def _initialize_strategies(self):
        """Initialize DeFi strategies"""
        try:
            # Create default strategies
            default_strategies = [
                {
                    "name": "Conservative Yield Farming",
                    "strategy_type": DeFiStrategyType.YIELD_FARMING,
                    "protocol": "aave",
                    "blockchain": "ethereum",
                    "risk_level": RiskLevel.LOW,
                    "expected_apy": Decimal("0.05"),  # 5%
                    "min_investment": Decimal("1000"),
                    "max_investment": Decimal("50000")
                },
                {
                    "name": "Aggressive Liquidity Provision",
                    "strategy_type": DeFiStrategyType.LIQUIDITY_PROVISION,
                    "protocol": "uniswap",
                    "blockchain": "ethereum",
                    "risk_level": RiskLevel.HIGH,
                    "expected_apy": Decimal("0.15"),  # 15%
                    "min_investment": Decimal("5000"),
                    "max_investment": Decimal("100000")
                },
                {
                    "name": "Cross-Chain Arbitrage",
                    "strategy_type": DeFiStrategyType.ARBITRAGE,
                    "protocol": "uniswap",
                    "blockchain": "ethereum",
                    "risk_level": RiskLevel.MEDIUM,
                    "expected_apy": Decimal("0.20"),  # 20%
                    "min_investment": Decimal("10000"),
                    "max_investment": Decimal("200000")
                },
                {
                    "name": "Yearn Vault Strategy",
                    "strategy_type": DeFiStrategyType.YIELD_FARMING,
                    "protocol": "yearn",
                    "blockchain": "ethereum",
                    "risk_level": RiskLevel.MEDIUM,
                    "expected_apy": Decimal("0.12"),  # 12%
                    "min_investment": Decimal("2000"),
                    "max_investment": Decimal("75000")
                }
            ]
            
            for strategy_data in default_strategies:
                strategy = DeFiStrategy(
                    strategy_id=f"strategy_{secrets.token_hex(8)}",
                    name=strategy_data["name"],
                    strategy_type=strategy_data["strategy_type"],
                    protocol=strategy_data["protocol"],
                    blockchain=strategy_data["blockchain"],
                    risk_level=strategy_data["risk_level"],
                    expected_apy=strategy_data["expected_apy"],
                    min_investment=strategy_data["min_investment"],
                    max_investment=strategy_data["max_investment"],
                    stop_loss_percentage=Decimal("0.10"),  # 10% stop loss
                    take_profit_percentage=Decimal("0.25")  # 25% take profit
                )
                
                self.strategies[strategy.strategy_id] = strategy
            
            logger.info(f"Initialized {len(self.strategies)} DeFi strategies")
            
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")

    async def _initialize_positions(self):
        """Initialize DeFi positions"""
        try:
            # Generate mock positions
            for i in range(25):  # Generate 25 mock positions
                strategy = secrets.choice(list(self.strategies.values()))
                
                position = DeFiPosition(
                    position_id=f"position_{secrets.token_hex(8)}",
                    user_address=f"0x{secrets.token_hex(20)}",
                    strategy_id=strategy.strategy_id,
                    protocol=strategy.protocol,
                    blockchain=strategy.blockchain,
                    position_type=strategy.strategy_type.value,
                    token_address=f"0x{secrets.token_hex(20)}",
                    token_symbol=secrets.choice(["ETH", "USDC", "USDT", "DAI", "WBTC"]),
                    amount=Decimal(str(secrets.randbelow(10000) + 1000)),
                    value_usd=Decimal(str(secrets.randbelow(50000) + 5000)),
                    apy=strategy.expected_apy + Decimal(str(secrets.randbelow(5) - 2)) / 100,
                    status=PositionStatus.ACTIVE,
                    opened_at=datetime.now() - timedelta(days=secrets.randbelow(30))
                )
                
                self.positions.append(position)
            
            logger.info(f"Initialized {len(self.positions)} DeFi positions")
            
        except Exception as e:
            logger.error(f"Error initializing positions: {e}")

    async def _initialize_yield_opportunities(self):
        """Initialize yield opportunities"""
        try:
            # Generate mock yield opportunities
            protocols = ["uniswap", "aave", "compound", "yearn", "curve", "balancer"]
            blockchains = ["ethereum", "polygon", "avalanche", "arbitrum"]
            strategy_types = list(DeFiStrategyType)
            
            for i in range(50):  # Generate 50 mock opportunities
                opportunity = YieldOpportunity(
                    opportunity_id=f"opportunity_{secrets.token_hex(8)}",
                    protocol=secrets.choice(protocols),
                    blockchain=secrets.choice(blockchains),
                    strategy_type=secrets.choice(strategy_types),
                    token_pair=f"{secrets.choice(['ETH', 'USDC', 'USDT', 'DAI'])}/{secrets.choice(['ETH', 'USDC', 'USDT', 'DAI'])}",
                    apy=Decimal(str(secrets.randbelow(50) + 5)) / 100,  # 5-55% APY
                    tvl=Decimal(str(secrets.randbelow(10000000) + 100000)),  # $100k - $10M TVL
                    risk_score=secrets.uniform(0.1, 0.9),
                    liquidity_score=secrets.uniform(0.3, 1.0),
                    gas_cost=Decimal(str(secrets.randbelow(100) + 10)),  # $10-110 gas cost
                    min_investment=Decimal(str(secrets.randbelow(5000) + 1000)),  # $1k-6k min
                    max_investment=Decimal(str(secrets.randbelow(100000) + 10000)),  # $10k-110k max
                    discovered_at=datetime.now() - timedelta(hours=secrets.randbelow(24))
                )
                
                self.yield_opportunities.append(opportunity)
            
            logger.info(f"Initialized {len(self.yield_opportunities)} yield opportunities")
            
        except Exception as e:
            logger.error(f"Error initializing yield opportunities: {e}")

    async def _strategy_execution_loop(self):
        """Strategy execution loop"""
        while self.is_running:
            try:
                # Execute active strategies
                await self._execute_strategies()
                
                await asyncio.sleep(300)  # Execute every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in strategy execution loop: {e}")
                await asyncio.sleep(300)

    async def _yield_hunting_loop(self):
        """Yield hunting loop"""
        while self.is_running:
            try:
                # Hunt for new yield opportunities
                await self._hunt_yield_opportunities()
                
                await asyncio.sleep(1800)  # Hunt every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in yield hunting loop: {e}")
                await asyncio.sleep(1800)

    async def _risk_monitoring_loop(self):
        """Risk monitoring loop"""
        while self.is_running:
            try:
                # Monitor risk metrics
                await self._monitor_risk_metrics()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _position_rebalancing_loop(self):
        """Position rebalancing loop"""
        while self.is_running:
            try:
                # Rebalance positions
                await self._rebalance_positions()
                
                await asyncio.sleep(3600)  # Rebalance every hour
                
            except Exception as e:
                logger.error(f"Error in position rebalancing loop: {e}")
                await asyncio.sleep(3600)

    async def _performance_tracking_loop(self):
        """Performance tracking loop"""
        while self.is_running:
            try:
                # Track performance metrics
                await self._track_performance_metrics()
                
                await asyncio.sleep(600)  # Track every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(600)

    async def _execute_strategies(self):
        """Execute active strategies"""
        try:
            for strategy in self.strategies.values():
                if strategy.enabled:
                    # Simulate strategy execution
                    await self._execute_strategy(strategy)
            
            logger.info(f"Executed {len([s for s in self.strategies.values() if s.enabled])} strategies")
            
        except Exception as e:
            logger.error(f"Error executing strategies: {e}")

    async def _execute_strategy(self, strategy: DeFiStrategy):
        """Execute a specific strategy"""
        try:
            # Simulate strategy execution based on type
            if strategy.strategy_type == DeFiStrategyType.YIELD_FARMING:
                await self._execute_yield_farming(strategy)
            elif strategy.strategy_type == DeFiStrategyType.LIQUIDITY_PROVISION:
                await self._execute_liquidity_provision(strategy)
            elif strategy.strategy_type == DeFiStrategyType.ARBITRAGE:
                await self._execute_arbitrage(strategy)
            elif strategy.strategy_type == DeFiStrategyType.LENDING:
                await self._execute_lending(strategy)
            elif strategy.strategy_type == DeFiStrategyType.BORROWING:
                await self._execute_borrowing(strategy)
            
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.strategy_id}: {e}")

    async def _execute_yield_farming(self, strategy: DeFiStrategy):
        """Execute yield farming strategy"""
        try:
            # Simulate yield farming execution
            logger.info(f"Executing yield farming strategy: {strategy.name}")
            
        except Exception as e:
            logger.error(f"Error executing yield farming: {e}")

    async def _execute_liquidity_provision(self, strategy: DeFiStrategy):
        """Execute liquidity provision strategy"""
        try:
            # Simulate liquidity provision execution
            logger.info(f"Executing liquidity provision strategy: {strategy.name}")
            
        except Exception as e:
            logger.error(f"Error executing liquidity provision: {e}")

    async def _execute_arbitrage(self, strategy: DeFiStrategy):
        """Execute arbitrage strategy"""
        try:
            # Simulate arbitrage execution
            logger.info(f"Executing arbitrage strategy: {strategy.name}")
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")

    async def _execute_lending(self, strategy: DeFiStrategy):
        """Execute lending strategy"""
        try:
            # Simulate lending execution
            logger.info(f"Executing lending strategy: {strategy.name}")
            
        except Exception as e:
            logger.error(f"Error executing lending: {e}")

    async def _execute_borrowing(self, strategy: DeFiStrategy):
        """Execute borrowing strategy"""
        try:
            # Simulate borrowing execution
            logger.info(f"Executing borrowing strategy: {strategy.name}")
            
        except Exception as e:
            logger.error(f"Error executing borrowing: {e}")

    async def _hunt_yield_opportunities(self):
        """Hunt for new yield opportunities"""
        try:
            # Simulate finding new opportunities
            num_opportunities = secrets.randbelow(5) + 1
            
            for _ in range(num_opportunities):
                opportunity = YieldOpportunity(
                    opportunity_id=f"opportunity_{secrets.token_hex(8)}",
                    protocol=secrets.choice(["uniswap", "aave", "compound", "yearn"]),
                    blockchain=secrets.choice(["ethereum", "polygon", "avalanche"]),
                    strategy_type=secrets.choice(list(DeFiStrategyType)),
                    token_pair=f"{secrets.choice(['ETH', 'USDC', 'USDT'])}/{secrets.choice(['ETH', 'USDC', 'USDT'])}",
                    apy=Decimal(str(secrets.randbelow(30) + 10)) / 100,  # 10-40% APY
                    tvl=Decimal(str(secrets.randbelow(5000000) + 50000)),  # $50k-5M TVL
                    risk_score=secrets.uniform(0.2, 0.8),
                    liquidity_score=secrets.uniform(0.4, 1.0),
                    gas_cost=Decimal(str(secrets.randbelow(50) + 5)),  # $5-55 gas cost
                    min_investment=Decimal(str(secrets.randbelow(3000) + 500)),  # $500-3.5k min
                    max_investment=Decimal(str(secrets.randbelow(50000) + 5000)),  # $5k-55k max
                    discovered_at=datetime.now()
                )
                
                self.yield_opportunities.append(opportunity)
            
            # Keep only last 1000 opportunities
            if len(self.yield_opportunities) > 1000:
                self.yield_opportunities = self.yield_opportunities[-1000:]
            
            logger.info(f"Discovered {num_opportunities} new yield opportunities")
            
        except Exception as e:
            logger.error(f"Error hunting yield opportunities: {e}")

    async def _monitor_risk_metrics(self):
        """Monitor risk metrics"""
        try:
            # Calculate risk metrics
            total_exposure = sum(pos.value_usd for pos in self.positions if pos.status == PositionStatus.ACTIVE)
            protocol_exposure = {}
            
            for position in self.positions:
                if position.status == PositionStatus.ACTIVE:
                    if position.protocol not in protocol_exposure:
                        protocol_exposure[position.protocol] = Decimal("0")
                    protocol_exposure[position.protocol] += position.value_usd
            
            # Update risk metrics
            self.risk_metrics = {
                "total_exposure": float(total_exposure),
                "max_exposure_utilization": float(total_exposure / self.risk_parameters["max_total_exposure"]),
                "protocol_exposure": {protocol: float(exposure) for protocol, exposure in protocol_exposure.items()},
                "active_positions": len([p for p in self.positions if p.status == PositionStatus.ACTIVE]),
                "total_pnl": float(sum(pos.pnl for pos in self.positions)),
                "average_apy": float(sum(pos.apy for pos in self.positions if pos.status == PositionStatus.ACTIVE) / 
                                   max(1, len([p for p in self.positions if p.status == PositionStatus.ACTIVE])))
            }
            
        except Exception as e:
            logger.error(f"Error monitoring risk metrics: {e}")

    async def _rebalance_positions(self):
        """Rebalance positions"""
        try:
            # Simulate position rebalancing
            rebalanced_count = 0
            
            for position in self.positions:
                if position.status == PositionStatus.ACTIVE:
                    # Simulate rebalancing decision
                    if secrets.choice([True, False]):  # 50% chance to rebalance
                        # Update position value
                        value_change = Decimal(str(secrets.randbelow(20) - 10)) / 100  # -10% to +10%
                        position.value_usd *= (1 + value_change)
                        position.updated_at = datetime.now()
                        rebalanced_count += 1
            
            logger.info(f"Rebalanced {rebalanced_count} positions")
            
        except Exception as e:
            logger.error(f"Error rebalancing positions: {e}")

    async def _track_performance_metrics(self):
        """Track performance metrics"""
        try:
            # Calculate performance metrics
            for strategy in self.strategies.values():
                strategy_positions = [p for p in self.positions if p.strategy_id == strategy.strategy_id and p.status == PositionStatus.ACTIVE]
                
                if strategy_positions:
                    total_value = sum(pos.value_usd for pos in strategy_positions)
                    total_pnl = sum(pos.pnl for pos in strategy_positions)
                    average_apy = sum(pos.apy for pos in strategy_positions) / len(strategy_positions)
                    
                    performance = {
                        "total_value": float(total_value),
                        "total_pnl": float(total_pnl),
                        "average_apy": float(average_apy),
                        "position_count": len(strategy_positions)
                    }
                    
                    self.strategy_performance[strategy.strategy_id] = performance
            
            # Calculate protocol performance
            for protocol in self.protocol_configs.keys():
                protocol_positions = [p for p in self.positions if p.protocol == protocol and p.status == PositionStatus.ACTIVE]
                
                if protocol_positions:
                    total_value = sum(pos.value_usd for pos in protocol_positions)
                    total_pnl = sum(pos.pnl for pos in protocol_positions)
                    
                    performance = {
                        "total_value": float(total_value),
                        "total_pnl": float(total_pnl),
                        "position_count": len(protocol_positions)
                    }
                    
                    self.protocol_performance[protocol] = performance
            
        except Exception as e:
            logger.error(f"Error tracking performance metrics: {e}")

    # Public API methods
    async def get_strategies(self) -> List[Dict[str, Any]]:
        """Get DeFi strategies"""
        try:
            return [
                {
                    "strategy_id": strategy.strategy_id,
                    "name": strategy.name,
                    "strategy_type": strategy.strategy_type.value,
                    "protocol": strategy.protocol,
                    "blockchain": strategy.blockchain,
                    "risk_level": strategy.risk_level.value,
                    "expected_apy": float(strategy.expected_apy),
                    "min_investment": float(strategy.min_investment),
                    "max_investment": float(strategy.max_investment),
                    "auto_compound": strategy.auto_compound,
                    "rebalance_frequency": strategy.rebalance_frequency,
                    "stop_loss_percentage": float(strategy.stop_loss_percentage) if strategy.stop_loss_percentage else None,
                    "take_profit_percentage": float(strategy.take_profit_percentage) if strategy.take_profit_percentage else None,
                    "enabled": strategy.enabled,
                    "created_at": strategy.created_at.isoformat(),
                    "metadata": strategy.metadata
                }
                for strategy in self.strategies.values()
            ]
            
        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return []

    async def get_positions(self, user_address: Optional[str] = None,
                          strategy_id: Optional[str] = None,
                          status: Optional[PositionStatus] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get DeFi positions"""
        try:
            positions = self.positions
            
            # Filter by user address
            if user_address:
                positions = [p for p in positions if p.user_address == user_address]
            
            # Filter by strategy
            if strategy_id:
                positions = [p for p in positions if p.strategy_id == strategy_id]
            
            # Filter by status
            if status:
                positions = [p for p in positions if p.status == status]
            
            # Sort by opened time (most recent first)
            positions = sorted(positions, key=lambda x: x.opened_at, reverse=True)
            
            # Limit results
            positions = positions[:limit]
            
            return [
                {
                    "position_id": pos.position_id,
                    "user_address": pos.user_address,
                    "strategy_id": pos.strategy_id,
                    "protocol": pos.protocol,
                    "blockchain": pos.blockchain,
                    "position_type": pos.position_type,
                    "token_address": pos.token_address,
                    "token_symbol": pos.token_symbol,
                    "amount": float(pos.amount),
                    "value_usd": float(pos.value_usd),
                    "apy": float(pos.apy),
                    "status": pos.status.value,
                    "opened_at": pos.opened_at.isoformat(),
                    "closed_at": pos.closed_at.isoformat() if pos.closed_at else None,
                    "pnl": float(pos.pnl),
                    "fees_paid": float(pos.fees_paid),
                    "metadata": pos.metadata
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def get_yield_opportunities(self, protocol: Optional[str] = None,
                                    min_apy: Optional[float] = None,
                                    max_risk: Optional[float] = None,
                                    limit: int = 50) -> List[Dict[str, Any]]:
        """Get yield opportunities"""
        try:
            opportunities = self.yield_opportunities
            
            # Filter by protocol
            if protocol:
                opportunities = [o for o in opportunities if o.protocol == protocol]
            
            # Filter by minimum APY
            if min_apy:
                opportunities = [o for o in opportunities if float(o.apy) >= min_apy]
            
            # Filter by maximum risk
            if max_risk:
                opportunities = [o for o in opportunities if o.risk_score <= max_risk]
            
            # Sort by APY (highest first)
            opportunities = sorted(opportunities, key=lambda x: x.apy, reverse=True)
            
            # Limit results
            opportunities = opportunities[:limit]
            
            return [
                {
                    "opportunity_id": opp.opportunity_id,
                    "protocol": opp.protocol,
                    "blockchain": opp.blockchain,
                    "strategy_type": opp.strategy_type.value,
                    "token_pair": opp.token_pair,
                    "apy": float(opp.apy),
                    "tvl": float(opp.tvl),
                    "risk_score": opp.risk_score,
                    "liquidity_score": opp.liquidity_score,
                    "gas_cost": float(opp.gas_cost),
                    "min_investment": float(opp.min_investment),
                    "max_investment": float(opp.max_investment),
                    "discovered_at": opp.discovered_at.isoformat(),
                    "expires_at": opp.expires_at.isoformat() if opp.expires_at else None,
                    "metadata": opp.metadata
                }
                for opp in opportunities
            ]
            
        except Exception as e:
            logger.error(f"Error getting yield opportunities: {e}")
            return []

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        try:
            return self.risk_metrics
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            return {
                "strategy_performance": self.strategy_performance,
                "protocol_performance": self.protocol_performance,
                "total_positions": len(self.positions),
                "active_positions": len([p for p in self.positions if p.status == PositionStatus.ACTIVE]),
                "total_strategies": len(self.strategies),
                "active_strategies": len([s for s in self.strategies.values() if s.enabled])
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    async def get_manager_metrics(self) -> Dict[str, Any]:
        """Get manager metrics"""
        try:
            return {
                "manager_id": self.manager_id,
                "is_running": self.is_running,
                "total_strategies": len(self.strategies),
                "total_positions": len(self.positions),
                "total_opportunities": len(self.yield_opportunities),
                "total_transactions": len(self.transactions),
                "supported_protocols": list(self.protocol_configs.keys()),
                "risk_parameters": {k: float(v) if isinstance(v, Decimal) else v for k, v in self.risk_parameters.items()},
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting manager metrics: {e}")
            return {}

    async def add_strategy(self, name: str, strategy_type: DeFiStrategyType, protocol: str,
                          blockchain: str, risk_level: RiskLevel, expected_apy: Decimal,
                          min_investment: Decimal, max_investment: Decimal) -> str:
        """Add a new strategy"""
        try:
            strategy = DeFiStrategy(
                strategy_id=f"strategy_{secrets.token_hex(8)}",
                name=name,
                strategy_type=strategy_type,
                protocol=protocol,
                blockchain=blockchain,
                risk_level=risk_level,
                expected_apy=expected_apy,
                min_investment=min_investment,
                max_investment=max_investment
            )
            
            self.strategies[strategy.strategy_id] = strategy
            
            logger.info(f"Added strategy: {strategy.strategy_id}")
            return strategy.strategy_id
            
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            raise

    async def update_strategy(self, strategy_id: str, **kwargs) -> bool:
        """Update a strategy"""
        try:
            if strategy_id not in self.strategies:
                return False
            
            strategy = self.strategies[strategy_id]
            
            # Update allowed fields
            allowed_fields = ['name', 'expected_apy', 'min_investment', 'max_investment', 
                            'auto_compound', 'rebalance_frequency', 'stop_loss_percentage', 
                            'take_profit_percentage', 'enabled']
            for field, value in kwargs.items():
                if field in allowed_fields:
                    setattr(strategy, field, value)
            
            logger.info(f"Updated strategy: {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating strategy {strategy_id}: {e}")
            return False

# Global instance
defi_protocol_manager = DeFiProtocolManager()
