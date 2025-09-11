"""
Market Microstructure Service
Advanced market microstructure analysis, liquidity management, and market making
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market states"""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    HALTED = "halted"
    AUCTION = "auction"
    VOLATILITY_PAUSE = "volatility_pause"


class LiquidityType(Enum):
    """Liquidity types"""

    TIGHT = "tight"
    NORMAL = "normal"
    ABUNDANT = "abundant"
    CRISIS = "crisis"


class MarketRegime(Enum):
    """Market regimes"""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    CALM = "calm"
    CRISIS = "crisis"


class OrderFlowType(Enum):
    """Order flow types"""

    AGGRESSIVE_BUY = "aggressive_buy"
    AGGRESSIVE_SELL = "aggressive_sell"
    PASSIVE_BUY = "passive_buy"
    PASSIVE_SELL = "passive_sell"
    MIXED = "mixed"


@dataclass
class MarketDepth:
    """Market depth information"""

    symbol: str
    timestamp: datetime
    bid_levels: List[Tuple[float, float]]  # (price, quantity)
    ask_levels: List[Tuple[float, float]]  # (price, quantity)
    total_bid_volume: float
    total_ask_volume: float
    bid_ask_spread: float
    mid_price: float
    weighted_mid_price: float
    imbalance_ratio: float
    depth_score: float


@dataclass
class LiquidityMetrics:
    """Liquidity metrics"""

    symbol: str
    timestamp: datetime
    bid_ask_spread: float
    effective_spread: float
    realized_spread: float
    price_impact: float
    market_impact: float
    liquidity_score: float
    depth_score: float
    resilience_score: float
    turnover_ratio: float
    volume_weighted_price: float
    time_weighted_price: float


@dataclass
class OrderFlow:
    """Order flow analysis"""

    symbol: str
    timestamp: datetime
    aggressive_buy_volume: float
    aggressive_sell_volume: float
    passive_buy_volume: float
    passive_sell_volume: float
    net_order_flow: float
    order_flow_imbalance: float
    order_flow_pressure: float
    flow_type: OrderFlowType
    flow_strength: float


@dataclass
class MarketRegimeAnalysis:
    """Market regime analysis"""

    symbol: str
    timestamp: datetime
    regime: MarketRegime
    regime_confidence: float
    volatility: float
    trend_strength: float
    mean_reversion_strength: float
    persistence: float
    jump_probability: float
    regime_duration: float


@dataclass
class MarketMakingStrategy:
    """Market making strategy"""

    strategy_id: str
    symbol: str
    user_id: int
    strategy_type: str  # 'basic', 'adaptive', 'aggressive', 'conservative'
    parameters: Dict[str, Any]
    is_active: bool
    performance_metrics: Dict[str, float]
    risk_limits: Dict[str, float]
    created_at: datetime
    last_updated: datetime


@dataclass
class MarketMakingQuote:
    """Market making quote"""

    quote_id: str
    strategy_id: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    mid_price: float
    skew: float
    timestamp: datetime
    is_active: bool


@dataclass
class MarketImpact:
    """Market impact analysis"""

    symbol: str
    timestamp: datetime
    trade_size: float
    price_impact: float
    temporary_impact: float
    permanent_impact: float
    market_impact_cost: float
    implementation_shortfall: float
    volume_impact: float
    time_impact: float


class MarketMicrostructureService:
    """Comprehensive Market Microstructure Service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # Market data
        self.market_depth: Dict[str, MarketDepth] = {}
        self.liquidity_metrics: Dict[str, LiquidityMetrics] = {}
        self.order_flow: Dict[str, OrderFlow] = {}
        self.market_regimes: Dict[str, MarketRegimeAnalysis] = {}
        self.market_impact: Dict[str, List[MarketImpact]] = defaultdict(list)

        # Market making
        self.market_making_strategies: Dict[str, MarketMakingStrategy] = {}
        self.active_quotes: Dict[str, List[MarketMakingQuote]] = defaultdict(list)

        # Historical data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Analytics
        self.volatility_estimates: Dict[str, float] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.liquidity_scores: Dict[str, float] = {}

        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the Market Microstructure Service"""
        logger.info("Initializing Market Microstructure Service")

        # Load historical data
        await self._load_historical_data()

        # Initialize analytics
        await self._initialize_analytics()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_market_depth()),
            asyncio.create_task(self._calculate_liquidity_metrics()),
            asyncio.create_task(self._analyze_order_flow()),
            asyncio.create_task(self._detect_market_regimes()),
            asyncio.create_task(self._update_market_making_quotes()),
            asyncio.create_task(self._calculate_market_impact()),
            asyncio.create_task(self._update_performance_metrics()),
        ]

        logger.info("Market Microstructure Service initialized successfully")

    async def get_market_depth(self, symbol: str) -> Optional[MarketDepth]:
        """Get market depth for a symbol"""
        try:
            return self.market_depth.get(symbol)

        except Exception as e:
            logger.error(f"Error getting market depth: {e}")
            return None

    async def get_liquidity_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Get liquidity metrics for a symbol"""
        try:
            return self.liquidity_metrics.get(symbol)

        except Exception as e:
            logger.error(f"Error getting liquidity metrics: {e}")
            return None

    async def get_order_flow(self, symbol: str) -> Optional[OrderFlow]:
        """Get order flow analysis for a symbol"""
        try:
            return self.order_flow.get(symbol)

        except Exception as e:
            logger.error(f"Error getting order flow: {e}")
            return None

    async def get_market_regime(self, symbol: str) -> Optional[MarketRegimeAnalysis]:
        """Get market regime analysis for a symbol"""
        try:
            return self.market_regimes.get(symbol)

        except Exception as e:
            logger.error(f"Error getting market regime: {e}")
            return None

    async def get_market_impact(
        self, symbol: str, limit: int = 100
    ) -> List[MarketImpact]:
        """Get market impact analysis for a symbol"""
        try:
            return self.market_impact.get(symbol, [])[-limit:]

        except Exception as e:
            logger.error(f"Error getting market impact: {e}")
            return []

    async def create_market_making_strategy(
        self, symbol: str, user_id: int, strategy_type: str, parameters: Dict[str, Any]
    ) -> MarketMakingStrategy:
        """Create a market making strategy"""
        try:
            strategy_id = f"MM_{symbol}_{uuid.uuid4().hex[:8]}"

            strategy = MarketMakingStrategy(
                strategy_id=strategy_id,
                symbol=symbol,
                user_id=user_id,
                strategy_type=strategy_type,
                parameters=parameters,
                is_active=True,
                performance_metrics={},
                risk_limits={
                    "max_position": 10000,
                    "max_daily_loss": 1000,
                    "max_spread": 0.01,
                    "min_volume": 100,
                },
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.market_making_strategies[strategy_id] = strategy

            # Start market making
            asyncio.create_task(self._start_market_making(strategy))

            logger.info(f"Created market making strategy {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Error creating market making strategy: {e}")
            raise

    async def get_market_making_quotes(
        self, strategy_id: str
    ) -> List[MarketMakingQuote]:
        """Get market making quotes for a strategy"""
        try:
            return self.active_quotes.get(strategy_id, [])

        except Exception as e:
            logger.error(f"Error getting market making quotes: {e}")
            return []

    async def update_market_making_parameters(
        self, strategy_id: str, parameters: Dict[str, Any]
    ) -> bool:
        """Update market making strategy parameters"""
        try:
            if strategy_id not in self.market_making_strategies:
                return False

            strategy = self.market_making_strategies[strategy_id]
            strategy.parameters.update(parameters)
            strategy.last_updated = datetime.utcnow()

            logger.info(f"Updated market making strategy {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating market making parameters: {e}")
            return False

    async def stop_market_making(self, strategy_id: str) -> bool:
        """Stop market making strategy"""
        try:
            if strategy_id not in self.market_making_strategies:
                return False

            strategy = self.market_making_strategies[strategy_id]
            strategy.is_active = False
            strategy.last_updated = datetime.utcnow()

            # Cancel all active quotes
            if strategy_id in self.active_quotes:
                for quote in self.active_quotes[strategy_id]:
                    quote.is_active = False

            logger.info(f"Stopped market making strategy {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Error stopping market making strategy: {e}")
            return False

    async def calculate_optimal_spread(
        self, symbol: str, trade_size: float
    ) -> Dict[str, float]:
        """Calculate optimal spread for a given trade size"""
        try:
            liquidity_metrics = self.liquidity_metrics.get(symbol)
            if not liquidity_metrics:
                return {"optimal_spread": 0.01, "confidence": 0.0}

            # Calculate optimal spread based on market conditions
            base_spread = liquidity_metrics.bid_ask_spread
            volatility = self.volatility_estimates.get(symbol, 0.02)
            liquidity_score = liquidity_metrics.liquidity_score

            # Adjust spread based on trade size
            size_adjustment = min(trade_size / 10000, 2.0)  # Cap at 2x

            # Adjust spread based on volatility
            volatility_adjustment = 1 + (volatility / 0.02)

            # Adjust spread based on liquidity
            liquidity_adjustment = 1 / max(liquidity_score, 0.1)

            optimal_spread = (
                base_spread
                * size_adjustment
                * volatility_adjustment
                * liquidity_adjustment
            )

            # Calculate confidence based on data quality
            confidence = min(liquidity_score, 1.0)

            return {
                "optimal_spread": optimal_spread,
                "confidence": confidence,
                "base_spread": base_spread,
                "size_adjustment": size_adjustment,
                "volatility_adjustment": volatility_adjustment,
                "liquidity_adjustment": liquidity_adjustment,
            }

        except Exception as e:
            logger.error(f"Error calculating optimal spread: {e}")
            return {"optimal_spread": 0.01, "confidence": 0.0}

    async def estimate_market_impact(
        self, symbol: str, trade_size: float, execution_time: float = 60
    ) -> Dict[str, float]:
        """Estimate market impact for a trade"""
        try:
            # Get recent market impact data
            recent_impacts = self.market_impact.get(symbol, [])[-100:]

            if not recent_impacts:
                return {"estimated_impact": 0.001, "confidence": 0.0}

            # Calculate average impact by trade size
            size_buckets = {}
            for impact in recent_impacts:
                bucket = int(impact.trade_size / 1000) * 1000
                if bucket not in size_buckets:
                    size_buckets[bucket] = []
                size_buckets[bucket].append(impact.price_impact)

            # Find closest size bucket
            closest_bucket = min(size_buckets.keys(), key=lambda x: abs(x - trade_size))
            avg_impact = np.mean(size_buckets[closest_bucket])

            # Adjust for execution time
            time_adjustment = min(execution_time / 60, 2.0)  # Cap at 2x

            # Adjust for current market conditions
            liquidity_metrics = self.liquidity_metrics.get(symbol)
            if liquidity_metrics:
                liquidity_adjustment = 1 / max(liquidity_metrics.liquidity_score, 0.1)
            else:
                liquidity_adjustment = 1.0

            estimated_impact = avg_impact * time_adjustment * liquidity_adjustment

            # Calculate confidence
            confidence = min(len(recent_impacts) / 100, 1.0)

            return {
                "estimated_impact": estimated_impact,
                "confidence": confidence,
                "avg_impact": avg_impact,
                "time_adjustment": time_adjustment,
                "liquidity_adjustment": liquidity_adjustment,
                "sample_size": len(recent_impacts),
            }

        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return {"estimated_impact": 0.001, "confidence": 0.0}

    # Background tasks
    async def _update_market_depth(self):
        """Update market depth information"""
        while True:
            try:
                # Simulate market depth updates
                for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
                    await self._simulate_market_depth_update(symbol)

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error updating market depth: {e}")
                await asyncio.sleep(5)

    async def _simulate_market_depth_update(self, symbol: str):
        """Simulate market depth update"""
        try:
            # Generate realistic market depth
            # Use hashlib for deterministic hash instead of built-in hash()
            deterministic_hash = int(hashlib.md5(symbol.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            base_price = 100.0 + deterministic_hash % 1000
            spread = 0.01

            # Generate bid levels
            bid_levels = []
            bid_price = base_price - spread / 2
            for i in range(10):
                quantity = np.random.uniform(100, 1000)
                bid_levels.append((bid_price, quantity))
                bid_price -= 0.01

            # Generate ask levels
            ask_levels = []
            ask_price = base_price + spread / 2
            for i in range(10):
                quantity = np.random.uniform(100, 1000)
                ask_levels.append((ask_price, quantity))
                ask_price += 0.01

            # Calculate metrics
            total_bid_volume = sum(q for _, q in bid_levels)
            total_ask_volume = sum(q for _, q in ask_levels)
            mid_price = (bid_levels[0][0] + ask_levels[0][0]) / 2

            # Weighted mid price
            weighted_mid_price = (
                bid_levels[0][0] * bid_levels[0][1]
                + ask_levels[0][0] * ask_levels[0][1]
            ) / (bid_levels[0][1] + ask_levels[0][1])

            # Imbalance ratio
            imbalance_ratio = (total_bid_volume - total_ask_volume) / (
                total_bid_volume + total_ask_volume
            )

            # Depth score
            depth_score = min((total_bid_volume + total_ask_volume) / 10000, 1.0)

            market_depth = MarketDepth(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                total_bid_volume=total_bid_volume,
                total_ask_volume=total_ask_volume,
                bid_ask_spread=spread,
                mid_price=mid_price,
                weighted_mid_price=weighted_mid_price,
                imbalance_ratio=imbalance_ratio,
                depth_score=depth_score,
            )

            self.market_depth[symbol] = market_depth

        except Exception as e:
            logger.error(f"Error simulating market depth update: {e}")

    async def _calculate_liquidity_metrics(self):
        """Calculate liquidity metrics"""
        while True:
            try:
                for symbol, depth in self.market_depth.items():
                    await self._calculate_symbol_liquidity_metrics(symbol, depth)

                await asyncio.sleep(5)  # Calculate every 5 seconds

            except Exception as e:
                logger.error(f"Error calculating liquidity metrics: {e}")
                await asyncio.sleep(10)

    async def _calculate_symbol_liquidity_metrics(
        self, symbol: str, depth: MarketDepth
    ):
        """Calculate liquidity metrics for a symbol"""
        try:
            # Basic metrics
            bid_ask_spread = depth.bid_ask_spread
            effective_spread = bid_ask_spread * 0.8  # Assume 80% effective
            realized_spread = bid_ask_spread * 0.6  # Assume 60% realized

            # Price impact (simplified)
            price_impact = bid_ask_spread * 0.5

            # Market impact (simplified)
            market_impact = bid_ask_spread * 0.3

            # Liquidity score
            liquidity_score = depth.depth_score * (
                1 - bid_ask_spread / 0.1
            )  # Normalize by spread

            # Depth score
            depth_score = depth.depth_score

            # Resilience score (how quickly order book recovers)
            resilience_score = min(depth_score * 1.2, 1.0)

            # Turnover ratio (simplified)
            turnover_ratio = np.random.uniform(0.1, 2.0)

            # Volume weighted price
            volume_weighted_price = depth.weighted_mid_price

            # Time weighted price
            time_weighted_price = depth.mid_price

            liquidity_metrics = LiquidityMetrics(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                bid_ask_spread=bid_ask_spread,
                effective_spread=effective_spread,
                realized_spread=realized_spread,
                price_impact=price_impact,
                market_impact=market_impact,
                liquidity_score=liquidity_score,
                depth_score=depth_score,
                resilience_score=resilience_score,
                turnover_ratio=turnover_ratio,
                volume_weighted_price=volume_weighted_price,
                time_weighted_price=time_weighted_price,
            )

            self.liquidity_metrics[symbol] = liquidity_metrics

        except Exception as e:
            logger.error(f"Error calculating symbol liquidity metrics: {e}")

    async def _analyze_order_flow(self):
        """Analyze order flow"""
        while True:
            try:
                for symbol in self.market_depth.keys():
                    await self._analyze_symbol_order_flow(symbol)

                await asyncio.sleep(2)  # Analyze every 2 seconds

            except Exception as e:
                logger.error(f"Error analyzing order flow: {e}")
                await asyncio.sleep(5)

    async def _analyze_symbol_order_flow(self, symbol: str):
        """Analyze order flow for a symbol"""
        try:
            # Simulate order flow data
            aggressive_buy_volume = np.random.uniform(0, 1000)
            aggressive_sell_volume = np.random.uniform(0, 1000)
            passive_buy_volume = np.random.uniform(0, 2000)
            passive_sell_volume = np.random.uniform(0, 2000)

            # Calculate metrics
            net_order_flow = aggressive_buy_volume - aggressive_sell_volume
            total_volume = (
                aggressive_buy_volume
                + aggressive_sell_volume
                + passive_buy_volume
                + passive_sell_volume
            )

            if total_volume > 0:
                order_flow_imbalance = net_order_flow / total_volume
            else:
                order_flow_imbalance = 0

            # Order flow pressure
            order_flow_pressure = abs(order_flow_imbalance)

            # Determine flow type
            if net_order_flow > 100:
                flow_type = OrderFlowType.AGGRESSIVE_BUY
            elif net_order_flow < -100:
                flow_type = OrderFlowType.AGGRESSIVE_SELL
            elif abs(net_order_flow) < 50:
                flow_type = OrderFlowType.MIXED
            else:
                flow_type = OrderFlowType.MIXED

            # Flow strength
            flow_strength = min(order_flow_pressure * 2, 1.0)

            order_flow = OrderFlow(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                aggressive_buy_volume=aggressive_buy_volume,
                aggressive_sell_volume=aggressive_sell_volume,
                passive_buy_volume=passive_buy_volume,
                passive_sell_volume=passive_sell_volume,
                net_order_flow=net_order_flow,
                order_flow_imbalance=order_flow_imbalance,
                order_flow_pressure=order_flow_pressure,
                flow_type=flow_type,
                flow_strength=flow_strength,
            )

            self.order_flow[symbol] = order_flow

        except Exception as e:
            logger.error(f"Error analyzing symbol order flow: {e}")

    async def _detect_market_regimes(self):
        """Detect market regimes"""
        while True:
            try:
                for symbol in self.market_depth.keys():
                    await self._detect_symbol_market_regime(symbol)

                await asyncio.sleep(10)  # Detect every 10 seconds

            except Exception as e:
                logger.error(f"Error detecting market regimes: {e}")
                await asyncio.sleep(20)

    async def _detect_symbol_market_regime(self, symbol: str):
        """Detect market regime for a symbol"""
        try:
            # Get recent price data
            price_data = list(self.price_history[symbol])[-100:]

            if len(price_data) < 20:
                return

            # Calculate volatility
            returns = np.diff(price_data) / price_data[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Calculate trend strength
            if len(price_data) >= 20:
                short_ma = np.mean(price_data[-5:])
                long_ma = np.mean(price_data[-20:])
                trend_strength = abs(short_ma - long_ma) / long_ma
            else:
                trend_strength = 0

            # Calculate mean reversion strength
            if len(price_data) >= 10:
                recent_prices = price_data[-10:]
                mean_price = np.mean(recent_prices)
                mean_reversion_strength = np.mean(
                    [abs(p - mean_price) / mean_price for p in recent_prices]
                )
            else:
                mean_reversion_strength = 0

            # Calculate persistence (autocorrelation)
            if len(returns) >= 10:
                persistence = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if np.isnan(persistence):
                    persistence = 0
            else:
                persistence = 0

            # Jump probability (simplified)
            jump_probability = min(volatility * 10, 1.0)

            # Determine regime
            if volatility > 0.3:
                regime = MarketRegime.CRISIS
                regime_confidence = 0.9
            elif volatility > 0.2:
                regime = MarketRegime.VOLATILE
                regime_confidence = 0.8
            elif trend_strength > 0.05:
                regime = MarketRegime.TRENDING
                regime_confidence = 0.7
            elif mean_reversion_strength > 0.02:
                regime = MarketRegime.MEAN_REVERTING
                regime_confidence = 0.6
            else:
                regime = MarketRegime.CALM
                regime_confidence = 0.5

            # Regime duration (simplified)
            regime_duration = np.random.uniform(60, 3600)  # 1 minute to 1 hour

            market_regime = MarketRegimeAnalysis(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                regime=regime,
                regime_confidence=regime_confidence,
                volatility=volatility,
                trend_strength=trend_strength,
                mean_reversion_strength=mean_reversion_strength,
                persistence=persistence,
                jump_probability=jump_probability,
                regime_duration=regime_duration,
            )

            self.market_regimes[symbol] = market_regime

        except Exception as e:
            logger.error(f"Error detecting symbol market regime: {e}")

    async def _update_market_making_quotes(self):
        """Update market making quotes"""
        while True:
            try:
                for strategy_id, strategy in self.market_making_strategies.items():
                    if strategy.is_active:
                        await self._update_strategy_quotes(strategy)

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error updating market making quotes: {e}")
                await asyncio.sleep(5)

    async def _update_strategy_quotes(self, strategy: MarketMakingStrategy):
        """Update quotes for a market making strategy"""
        try:
            symbol = strategy.symbol
            market_depth = self.market_depth.get(symbol)
            liquidity_metrics = self.liquidity_metrics.get(symbol)

            if not market_depth or not liquidity_metrics:
                return

            # Calculate quote parameters
            mid_price = market_depth.mid_price
            base_spread = liquidity_metrics.bid_ask_spread

            # Adjust spread based on strategy type
            if strategy.strategy_type == "aggressive":
                spread_multiplier = 0.8
            elif strategy.strategy_type == "conservative":
                spread_multiplier = 1.5
            else:
                spread_multiplier = 1.0

            spread = base_spread * spread_multiplier

            # Calculate bid and ask prices
            bid_price = mid_price - spread / 2
            ask_price = mid_price + spread / 2

            # Calculate quote sizes
            base_size = strategy.parameters.get("base_size", 100)
            max_size = strategy.parameters.get("max_size", 1000)

            # Adjust size based on market conditions
            liquidity_adjustment = liquidity_metrics.liquidity_score
            size_multiplier = 1 + liquidity_adjustment

            bid_size = min(base_size * size_multiplier, max_size)
            ask_size = min(base_size * size_multiplier, max_size)

            # Calculate skew (price adjustment for inventory)
            skew = strategy.parameters.get("skew", 0.0)

            # Apply skew
            if skew > 0:  # Long inventory, reduce bid
                bid_price -= skew * spread
            elif skew < 0:  # Short inventory, increase ask
                ask_price += abs(skew) * spread

            # Create quote
            quote = MarketMakingQuote(
                quote_id=f"QUOTE_{uuid.uuid4().hex[:8]}",
                strategy_id=strategy.strategy_id,
                symbol=symbol,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                spread=spread,
                mid_price=mid_price,
                skew=skew,
                timestamp=datetime.utcnow(),
                is_active=True,
            )

            # Update active quotes
            if strategy.strategy_id not in self.active_quotes:
                self.active_quotes[strategy.strategy_id] = []

            # Remove old quotes
            self.active_quotes[strategy.strategy_id] = [
                q
                for q in self.active_quotes[strategy.strategy_id]
                if (datetime.utcnow() - q.timestamp).total_seconds() < 60
            ]

            # Add new quote
            self.active_quotes[strategy.strategy_id].append(quote)

        except Exception as e:
            logger.error(f"Error updating strategy quotes: {e}")

    async def _start_market_making(self, strategy: MarketMakingStrategy):
        """Start market making for a strategy"""
        try:
            logger.info(f"Started market making for strategy {strategy.strategy_id}")

            # Market making logic would be implemented here
            # This is a simplified version

        except Exception as e:
            logger.error(f"Error starting market making: {e}")

    async def _calculate_market_impact(self):
        """Calculate market impact"""
        while True:
            try:
                # Simulate market impact calculations
                for symbol in self.market_depth.keys():
                    if np.random.random() < 0.1:  # 10% chance of trade
                        await self._simulate_market_impact_calculation(symbol)

                await asyncio.sleep(5)  # Calculate every 5 seconds

            except Exception as e:
                logger.error(f"Error calculating market impact: {e}")
                await asyncio.sleep(10)

    async def _simulate_market_impact_calculation(self, symbol: str):
        """Simulate market impact calculation"""
        try:
            # Simulate trade
            trade_size = np.random.uniform(100, 5000)

            # Calculate market impact
            liquidity_metrics = self.liquidity_metrics.get(symbol)
            if not liquidity_metrics:
                return

            # Price impact (simplified)
            price_impact = liquidity_metrics.price_impact * (trade_size / 1000)

            # Temporary impact
            temporary_impact = price_impact * 0.6

            # Permanent impact
            permanent_impact = price_impact * 0.4

            # Market impact cost
            market_impact_cost = price_impact * trade_size

            # Implementation shortfall
            implementation_shortfall = price_impact

            # Volume impact
            volume_impact = min(trade_size / 10000, 1.0)

            # Time impact
            time_impact = np.random.uniform(0.001, 0.01)

            market_impact = MarketImpact(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                trade_size=trade_size,
                price_impact=price_impact,
                temporary_impact=temporary_impact,
                permanent_impact=permanent_impact,
                market_impact_cost=market_impact_cost,
                implementation_shortfall=implementation_shortfall,
                volume_impact=volume_impact,
                time_impact=time_impact,
            )

            self.market_impact[symbol].append(market_impact)

            # Keep only recent impacts
            if len(self.market_impact[symbol]) > 1000:
                self.market_impact[symbol] = self.market_impact[symbol][-1000:]

        except Exception as e:
            logger.error(f"Error simulating market impact calculation: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        while True:
            try:
                # Update performance metrics for market making strategies
                for strategy_id, strategy in self.market_making_strategies.items():
                    await self._update_strategy_performance(strategy)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                await asyncio.sleep(60)

    async def _update_strategy_performance(self, strategy: MarketMakingStrategy):
        """Update performance metrics for a strategy"""
        try:
            # Simulate performance metrics
            performance_metrics = {
                "total_pnl": np.random.uniform(-100, 500),
                "daily_pnl": np.random.uniform(-50, 100),
                "trades_count": np.random.randint(10, 100),
                "fill_rate": np.random.uniform(0.7, 0.95),
                "average_spread": np.random.uniform(0.005, 0.02),
                "inventory_risk": np.random.uniform(0, 1000),
                "last_updated": datetime.utcnow(),
            }

            strategy.performance_metrics = performance_metrics
            strategy.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")

    # Helper methods
    async def _load_historical_data(self):
        """Load historical data"""
        try:
            # Initialize price history with some sample data
            for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
                # Use hashlib for deterministic hash instead of built-in hash()
                deterministic_hash = int(
                    hashlib.md5(symbol.encode(), usedforsecurity=False).hexdigest()[:8], 16
                )
                base_price = 100.0 + deterministic_hash % 1000
                for i in range(100):
                    price = base_price + np.random.normal(0, 2)
                    self.price_history[symbol].append(price)

                    volume = np.random.uniform(1000, 10000)
                    self.volume_history[symbol].append(volume)

            logger.info("Loaded historical data")

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

    async def _initialize_analytics(self):
        """Initialize analytics"""
        try:
            # Initialize volatility estimates
            for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
                self.volatility_estimates[symbol] = np.random.uniform(0.15, 0.35)

            # Initialize correlation matrix
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
            for i, symbol1 in enumerate(symbols):
                self.correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        self.correlation_matrix[symbol1][symbol2] = 1.0
                    else:
                        self.correlation_matrix[symbol1][symbol2] = np.random.uniform(
                            0.3, 0.8
                        )

            # Initialize liquidity scores
            for symbol in symbols:
                self.liquidity_scores[symbol] = np.random.uniform(0.6, 0.9)

            logger.info("Initialized analytics")

        except Exception as e:
            logger.error(f"Error initializing analytics: {e}")


# Factory function
async def get_market_microstructure_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> MarketMicrostructureService:
    """Get Market Microstructure Service instance"""
    service = MarketMicrostructureService(redis_client, db_session)
    await service.initialize()
    return service


# Synchronous wrapper for FastAPI dependencies
def get_market_microstructure_service_sync() -> MarketMicrostructureService:
    """Synchronous wrapper for Market Microstructure Service"""
    # Create a mock service for FastAPI dependencies
    return MarketMicrostructureService(None, None)
