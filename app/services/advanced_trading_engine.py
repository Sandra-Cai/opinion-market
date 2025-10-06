"""
Advanced Trading Engine
Comprehensive multi-strategy trading system with AI-powered decision making
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
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class TradingStrategy(Enum):
    """Trading strategy types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"
    ALGORITHMIC = "algorithmic"
    QUANTITATIVE = "quantitative"
    MACHINE_LEARNING = "machine_learning"
    HIGH_FREQUENCY = "high_frequency"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class MarketCondition(Enum):
    """Market conditions"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGING = "ranging"

@dataclass
class TradingSignal:
    """Trading signal"""
    signal_id: str
    strategy: TradingStrategy
    asset: str
    side: OrderSide
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price_target: float
    stop_loss: float
    take_profit: float
    time_horizon: timedelta
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TradingOrder:
    """Trading order"""
    order_id: str
    strategy: TradingStrategy
    asset: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TradingPosition:
    """Trading position"""
    position_id: str
    asset: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: Optional[TradingStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TradingPerformance:
    """Trading performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0

class AdvancedTradingEngine:
    """Advanced Trading Engine with multi-strategy support"""
    
    def __init__(self):
        self.engine_id = f"trading_engine_{secrets.token_hex(8)}"
        self.is_running = False
        self.trading_active = False
        
        # Trading data
        self.signals: Dict[str, TradingSignal] = {}
        self.orders: Dict[str, TradingOrder] = {}
        self.positions: Dict[str, TradingPosition] = {}
        self.performance: TradingPerformance = TradingPerformance()
        
        # Strategy configurations
        self.strategies: Dict[TradingStrategy, Dict[str, Any]] = {}
        self.strategy_weights: Dict[TradingStrategy, float] = {}
        
        # Market data
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.market_conditions: Dict[str, MarketCondition] = {}
        
        # Risk management
        self.risk_limits: Dict[str, float] = {
            "max_position_size": 0.1,  # 10% of portfolio
            "max_daily_loss": 0.02,    # 2% daily loss limit
            "max_drawdown": 0.05,      # 5% max drawdown
            "max_correlation": 0.7,    # Max correlation between positions
            "max_leverage": 2.0        # Max leverage
        }
        
        # Performance tracking
        self.daily_pnl: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Processing tasks
        self.signal_processing_task: Optional[asyncio.Task] = None
        self.order_management_task: Optional[asyncio.Task] = None
        self.risk_monitoring_task: Optional[asyncio.Task] = None
        self.performance_tracking_task: Optional[asyncio.Task] = None
        
        logger.info(f"Advanced Trading Engine {self.engine_id} initialized")

    async def start_trading_engine(self):
        """Start the trading engine"""
        if self.is_running:
            return
        
        logger.info("Starting Advanced Trading Engine...")
        
        # Initialize strategies
        await self._initialize_strategies()
        
        # Start processing tasks
        self.is_running = True
        self.trading_active = True
        
        self.signal_processing_task = asyncio.create_task(self._signal_processing_loop())
        self.order_management_task = asyncio.create_task(self._order_management_loop())
        self.risk_monitoring_task = asyncio.create_task(self._risk_monitoring_loop())
        self.performance_tracking_task = asyncio.create_task(self._performance_tracking_loop())
        
        logger.info("Advanced Trading Engine started")

    async def stop_trading_engine(self):
        """Stop the trading engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Advanced Trading Engine...")
        
        self.is_running = False
        self.trading_active = False
        
        # Cancel all processing tasks
        tasks = [
            self.signal_processing_task,
            self.order_management_task,
            self.risk_monitoring_task,
            self.performance_tracking_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all positions
        await self._close_all_positions()
        
        logger.info("Advanced Trading Engine stopped")

    async def _initialize_strategies(self):
        """Initialize trading strategies"""
        strategies_config = {
            TradingStrategy.MOMENTUM: {
                "enabled": True,
                "weight": 0.2,
                "parameters": {
                    "lookback_period": 20,
                    "threshold": 0.02,
                    "stop_loss": 0.05,
                    "take_profit": 0.1
                }
            },
            TradingStrategy.MEAN_REVERSION: {
                "enabled": True,
                "weight": 0.2,
                "parameters": {
                    "lookback_period": 14,
                    "bollinger_bands": 2.0,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70
                }
            },
            TradingStrategy.ARBITRAGE: {
                "enabled": True,
                "weight": 0.15,
                "parameters": {
                    "min_spread": 0.001,
                    "max_hold_time": 300,  # 5 minutes
                    "slippage_tolerance": 0.0005
                }
            },
            TradingStrategy.SCALPING: {
                "enabled": True,
                "weight": 0.15,
                "parameters": {
                    "quick_profit_target": 0.002,
                    "quick_stop_loss": 0.001,
                    "max_hold_time": 60  # 1 minute
                }
            },
            TradingStrategy.MACHINE_LEARNING: {
                "enabled": True,
                "weight": 0.3,
                "parameters": {
                    "model_confidence_threshold": 0.7,
                    "feature_window": 50,
                    "prediction_horizon": 5
                }
            }
        }
        
        self.strategies = strategies_config
        self.strategy_weights = {
            strategy: config["weight"] 
            for strategy, config in strategies_config.items() 
            if config["enabled"]
        }
        
        logger.info(f"Initialized {len(self.strategies)} trading strategies")

    async def _signal_processing_loop(self):
        """Process trading signals"""
        while self.is_running:
            try:
                # Generate signals from each strategy
                for strategy, config in self.strategies.items():
                    if config["enabled"]:
                        signals = await self._generate_strategy_signals(strategy)
                        for signal in signals:
                            await self._process_signal(signal)
                
                await asyncio.sleep(1)  # Process signals every second
                
            except Exception as e:
                logger.error(f"Error in signal processing loop: {e}")
                await asyncio.sleep(5)

    async def _order_management_loop(self):
        """Manage trading orders"""
        while self.is_running:
            try:
                # Process pending orders
                for order_id, order in list(self.orders.items()):
                    if order.status == OrderStatus.PENDING:
                        await self._execute_order(order)
                    elif order.status == OrderStatus.SUBMITTED:
                        await self._monitor_order(order)
                
                await asyncio.sleep(0.1)  # Check orders every 100ms
                
            except Exception as e:
                logger.error(f"Error in order management loop: {e}")
                await asyncio.sleep(1)

    async def _risk_monitoring_loop(self):
        """Monitor risk metrics"""
        while self.is_running:
            try:
                # Check risk limits
                await self._check_risk_limits()
                
                # Update position risk metrics
                await self._update_position_risk()
                
                await asyncio.sleep(5)  # Check risk every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _performance_tracking_loop(self):
        """Track trading performance"""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Log performance summary
                if len(self.daily_pnl) > 0:
                    logger.info(f"Daily PnL: {self.daily_pnl[-1]:.4f}, "
                              f"Total PnL: {self.performance.total_pnl:.4f}, "
                              f"Win Rate: {self.performance.win_rate:.2%}")
                
                await asyncio.sleep(60)  # Update performance every minute
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(60)

    async def _generate_strategy_signals(self, strategy: TradingStrategy) -> List[TradingSignal]:
        """Generate signals for a specific strategy"""
        signals = []
        
        try:
            if strategy == TradingStrategy.MOMENTUM:
                signals = await self._generate_momentum_signals()
            elif strategy == TradingStrategy.MEAN_REVERSION:
                signals = await self._generate_mean_reversion_signals()
            elif strategy == TradingStrategy.ARBITRAGE:
                signals = await self._generate_arbitrage_signals()
            elif strategy == TradingStrategy.SCALPING:
                signals = await self._generate_scalping_signals()
            elif strategy == TradingStrategy.MACHINE_LEARNING:
                signals = await self._generate_ml_signals()
            
        except Exception as e:
            logger.error(f"Error generating signals for {strategy}: {e}")
        
        return signals

    async def _generate_momentum_signals(self) -> List[TradingSignal]:
        """Generate momentum trading signals"""
        signals = []
        
        # Mock momentum signal generation
        for asset in ["BTC", "ETH", "AAPL", "TSLA"]:
            if asset in self.market_data:
                price_data = self.market_data[asset].get("prices", [])
                if len(price_data) >= 20:
                    # Calculate momentum
                    recent_prices = price_data[-20:]
                    momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    
                    if abs(momentum) > 0.02:  # 2% momentum threshold
                        signal = TradingSignal(
                            signal_id=f"momentum_{secrets.token_hex(8)}",
                            strategy=TradingStrategy.MOMENTUM,
                            asset=asset,
                            side=OrderSide.BUY if momentum > 0 else OrderSide.SELL,
                            strength=min(abs(momentum) * 10, 1.0),
                            confidence=0.8,
                            price_target=recent_prices[-1] * (1 + momentum * 0.5),
                            stop_loss=recent_prices[-1] * (1 - abs(momentum) * 0.5),
                            take_profit=recent_prices[-1] * (1 + momentum * 1.5),
                            time_horizon=timedelta(hours=4),
                            risk_score=0.3
                        )
                        signals.append(signal)
        
        return signals

    async def _generate_mean_reversion_signals(self) -> List[TradingSignal]:
        """Generate mean reversion trading signals"""
        signals = []
        
        # Mock mean reversion signal generation
        for asset in ["BTC", "ETH", "AAPL", "TSLA"]:
            if asset in self.market_data:
                price_data = self.market_data[asset].get("prices", [])
                if len(price_data) >= 14:
                    # Calculate RSI
                    recent_prices = price_data[-14:]
                    gains = [max(0, recent_prices[i] - recent_prices[i-1]) for i in range(1, len(recent_prices))]
                    losses = [max(0, recent_prices[i-1] - recent_prices[i]) for i in range(1, len(recent_prices))]
                    
                    avg_gain = sum(gains) / len(gains) if gains else 0
                    avg_loss = sum(losses) / len(losses) if losses else 0
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        current_price = recent_prices[-1]
                        
                        if rsi < 30:  # Oversold
                            signal = TradingSignal(
                                signal_id=f"mean_reversion_{secrets.token_hex(8)}",
                                strategy=TradingStrategy.MEAN_REVERSION,
                                asset=asset,
                                side=OrderSide.BUY,
                                strength=0.7,
                                confidence=0.75,
                                price_target=current_price * 1.05,
                                stop_loss=current_price * 0.95,
                                take_profit=current_price * 1.1,
                                time_horizon=timedelta(hours=8),
                                risk_score=0.4
                            )
                            signals.append(signal)
                        elif rsi > 70:  # Overbought
                            signal = TradingSignal(
                                signal_id=f"mean_reversion_{secrets.token_hex(8)}",
                                strategy=TradingStrategy.MEAN_REVERSION,
                                asset=asset,
                                side=OrderSide.SELL,
                                strength=0.7,
                                confidence=0.75,
                                price_target=current_price * 0.95,
                                stop_loss=current_price * 1.05,
                                take_profit=current_price * 0.9,
                                time_horizon=timedelta(hours=8),
                                risk_score=0.4
                            )
                            signals.append(signal)
        
        return signals

    async def _generate_arbitrage_signals(self) -> List[TradingSignal]:
        """Generate arbitrage trading signals"""
        signals = []
        
        # Mock arbitrage signal generation
        exchanges = ["binance", "coinbase", "kraken"]
        assets = ["BTC", "ETH"]
        
        for asset in assets:
            prices = {}
            for exchange in exchanges:
                if f"{asset}_{exchange}" in self.market_data:
                    prices[exchange] = self.market_data[f"{asset}_{exchange}"].get("price", 0)
            
            if len(prices) >= 2:
                min_price = min(prices.values())
                max_price = max(prices.values())
                spread = (max_price - min_price) / min_price
                
                if spread > 0.001:  # 0.1% spread threshold
                    buy_exchange = min(prices, key=prices.get)
                    sell_exchange = max(prices, key=prices.get)
                    
                    signal = TradingSignal(
                        signal_id=f"arbitrage_{secrets.token_hex(8)}",
                        strategy=TradingStrategy.ARBITRAGE,
                        asset=asset,
                        side=OrderSide.BUY,
                        strength=min(spread * 100, 1.0),
                        confidence=0.9,
                        price_target=max_price,
                        stop_loss=min_price * 0.999,
                        take_profit=max_price * 0.999,
                        time_horizon=timedelta(minutes=5),
                        risk_score=0.2,
                        metadata={
                            "buy_exchange": buy_exchange,
                            "sell_exchange": sell_exchange,
                            "spread": spread
                        }
                    )
                    signals.append(signal)
        
        return signals

    async def _generate_scalping_signals(self) -> List[TradingSignal]:
        """Generate scalping trading signals"""
        signals = []
        
        # Mock scalping signal generation
        for asset in ["BTC", "ETH"]:
            if asset in self.market_data:
                price_data = self.market_data[asset].get("prices", [])
                if len(price_data) >= 5:
                    # Look for quick price movements
                    recent_prices = price_data[-5:]
                    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    
                    if abs(price_change) > 0.001:  # 0.1% quick movement
                        signal = TradingSignal(
                            signal_id=f"scalping_{secrets.token_hex(8)}",
                            strategy=TradingStrategy.SCALPING,
                            asset=asset,
                            side=OrderSide.BUY if price_change > 0 else OrderSide.SELL,
                            strength=min(abs(price_change) * 1000, 1.0),
                            confidence=0.6,
                            price_target=recent_prices[-1] * (1 + price_change * 0.5),
                            stop_loss=recent_prices[-1] * (1 - abs(price_change) * 0.5),
                            take_profit=recent_prices[-1] * (1 + price_change * 1.0),
                            time_horizon=timedelta(minutes=1),
                            risk_score=0.1
                        )
                        signals.append(signal)
        
        return signals

    async def _generate_ml_signals(self) -> List[TradingSignal]:
        """Generate machine learning trading signals"""
        signals = []
        
        # Mock ML signal generation
        for asset in ["BTC", "ETH", "AAPL", "TSLA"]:
            if asset in self.market_data:
                # Simulate ML model prediction
                confidence = np.random.uniform(0.6, 0.95)
                prediction = np.random.choice([-1, 1])  # -1 for sell, 1 for buy
                
                if confidence > 0.7:
                    current_price = self.market_data[asset].get("price", 100)
                    
                    signal = TradingSignal(
                        signal_id=f"ml_{secrets.token_hex(8)}",
                        strategy=TradingStrategy.MACHINE_LEARNING,
                        asset=asset,
                        side=OrderSide.BUY if prediction > 0 else OrderSide.SELL,
                        strength=confidence,
                        confidence=confidence,
                        price_target=current_price * (1 + prediction * 0.03),
                        stop_loss=current_price * (1 - prediction * 0.02),
                        take_profit=current_price * (1 + prediction * 0.05),
                        time_horizon=timedelta(hours=2),
                        risk_score=0.3,
                        metadata={
                            "model_version": "v2.1",
                            "features_used": 50,
                            "prediction_horizon": 2
                        }
                    )
                    signals.append(signal)
        
        return signals

    async def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            # Check if signal meets minimum criteria
            if signal.strength < 0.5 or signal.confidence < 0.6:
                return
            
            # Check risk limits
            if not await self._check_signal_risk(signal):
                return
            
            # Create order from signal
            order = await self._create_order_from_signal(signal)
            
            # Store signal and order
            self.signals[signal.signal_id] = signal
            self.orders[order.order_id] = order
            
            logger.info(f"Processed signal {signal.signal_id} for {signal.asset} "
                       f"({signal.side.value}, strength: {signal.strength:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing signal {signal.signal_id}: {e}")

    async def _check_signal_risk(self, signal: TradingSignal) -> bool:
        """Check if signal meets risk criteria"""
        # Check position size limits
        current_exposure = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values() 
            if pos.asset == signal.asset
        )
        
        # Mock portfolio value
        portfolio_value = 100000  # $100k
        max_position_value = portfolio_value * self.risk_limits["max_position_size"]
        
        if current_exposure >= max_position_value:
            return False
        
        # Check correlation limits
        if len(self.positions) > 0:
            # Mock correlation check
            correlation = np.random.uniform(0.3, 0.9)
            if correlation > self.risk_limits["max_correlation"]:
                return False
        
        return True

    async def _create_order_from_signal(self, signal: TradingSignal) -> TradingOrder:
        """Create trading order from signal"""
        # Calculate position size based on risk
        portfolio_value = 100000  # Mock portfolio value
        risk_amount = portfolio_value * 0.01  # 1% risk per trade
        
        if signal.stop_loss and signal.price_target:
            risk_per_share = abs(signal.price_target - signal.stop_loss)
            if risk_per_share > 0:
                quantity = risk_amount / risk_per_share
            else:
                quantity = 100  # Default quantity
        else:
            quantity = 100
        
        order = TradingOrder(
            order_id=f"order_{secrets.token_hex(8)}",
            strategy=signal.strategy,
            asset=signal.asset,
            side=signal.side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=signal.price_target,
            stop_price=signal.stop_loss,
            metadata={
                "signal_id": signal.signal_id,
                "risk_score": signal.risk_score,
                "confidence": signal.confidence
            }
        )
        
        return order

    async def _execute_order(self, order: TradingOrder):
        """Execute a trading order"""
        try:
            # Mock order execution
            if order.order_type == OrderType.MARKET:
                # Market order - execute immediately
                execution_price = self.market_data.get(order.asset, {}).get("price", 100)
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_price = execution_price
                order.commission = order.quantity * execution_price * 0.001  # 0.1% commission
                
                # Create position
                await self._create_position_from_order(order)
                
            elif order.order_type == OrderType.LIMIT:
                # Limit order - check if price is reached
                current_price = self.market_data.get(order.asset, {}).get("price", 100)
                
                if order.side == OrderSide.BUY and current_price <= order.price:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_price = order.price
                    order.commission = order.quantity * order.price * 0.001
                    
                    await self._create_position_from_order(order)
                    
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_price = order.price
                    order.commission = order.quantity * order.price * 0.001
                    
                    await self._create_position_from_order(order)
                else:
                    order.status = OrderStatus.SUBMITTED
            
            order.updated_at = datetime.now()
            
            logger.info(f"Executed order {order.order_id} for {order.asset} "
                       f"({order.side.value}, {order.quantity} @ {order.average_price})")
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED

    async def _monitor_order(self, order: TradingOrder):
        """Monitor submitted orders"""
        try:
            # Check for order expiration
            if order.created_at < datetime.now() - timedelta(hours=1):
                order.status = OrderStatus.EXPIRED
                return
            
            # Check stop loss and take profit
            if order.stop_price:
                current_price = self.market_data.get(order.asset, {}).get("price", 100)
                
                if order.side == OrderSide.BUY and current_price <= order.stop_price:
                    # Stop loss triggered
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_price = order.stop_price
                    order.commission = order.quantity * order.stop_price * 0.001
                    
                    await self._create_position_from_order(order)
                    
                elif order.side == OrderSide.SELL and current_price >= order.stop_price:
                    # Stop loss triggered
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_price = order.stop_price
                    order.commission = order.quantity * order.stop_price * 0.001
                    
                    await self._create_position_from_order(order)
            
        except Exception as e:
            logger.error(f"Error monitoring order {order.order_id}: {e}")

    async def _create_position_from_order(self, order: TradingOrder):
        """Create position from filled order"""
        position = TradingPosition(
            position_id=f"pos_{secrets.token_hex(8)}",
            asset=order.asset,
            side=order.side,
            quantity=order.filled_quantity,
            entry_price=order.average_price,
            current_price=order.average_price,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            stop_loss=order.stop_price,
            strategy=order.strategy,
            metadata={
                "order_id": order.order_id,
                "commission": order.commission
            }
        )
        
        self.positions[position.position_id] = position
        
        # Add to trade history
        self.trade_history.append({
            "order_id": order.order_id,
            "position_id": position.position_id,
            "asset": order.asset,
            "side": order.side.value,
            "quantity": order.filled_quantity,
            "price": order.average_price,
            "commission": order.commission,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Created position {position.position_id} for {order.asset}")

    async def _check_risk_limits(self):
        """Check risk limits"""
        try:
            # Calculate current portfolio metrics
            total_value = 0
            total_pnl = 0
            
            for position in self.positions.values():
                position_value = position.quantity * position.current_price
                total_value += position_value
                total_pnl += position.unrealized_pnl
            
            # Check daily loss limit
            if len(self.daily_pnl) > 0:
                daily_loss = -min(self.daily_pnl[-1], 0)
                if daily_loss > self.risk_limits["max_daily_loss"]:
                    logger.warning(f"Daily loss limit exceeded: {daily_loss:.2%}")
                    await self._emergency_stop_trading()
            
            # Check max drawdown
            if len(self.daily_pnl) > 0:
                peak = max(self.daily_pnl)
                current = self.daily_pnl[-1]
                drawdown = (peak - current) / peak if peak > 0 else 0
                
                if drawdown > self.risk_limits["max_drawdown"]:
                    logger.warning(f"Max drawdown exceeded: {drawdown:.2%}")
                    await self._emergency_stop_trading()
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")

    async def _update_position_risk(self):
        """Update position risk metrics"""
        try:
            for position in self.positions.values():
                # Update current price (mock)
                if position.asset in self.market_data:
                    position.current_price = self.market_data[position.asset].get("price", position.entry_price)
                
                # Calculate unrealized PnL
                if position.side == OrderSide.BUY:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                
                position.updated_at = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating position risk: {e}")

    async def _update_performance_metrics(self):
        """Update trading performance metrics"""
        try:
            # Calculate daily PnL
            daily_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            self.daily_pnl.append(daily_pnl)
            
            # Keep only last 30 days
            if len(self.daily_pnl) > 30:
                self.daily_pnl = self.daily_pnl[-30:]
            
            # Update performance metrics
            self.performance.total_trades = len(self.trade_history)
            self.performance.total_pnl = sum(self.daily_pnl)
            
            if len(self.daily_pnl) > 1:
                # Calculate win rate
                winning_days = sum(1 for pnl in self.daily_pnl if pnl > 0)
                self.performance.win_rate = winning_days / len(self.daily_pnl)
                
                # Calculate volatility
                if len(self.daily_pnl) > 1:
                    returns = [self.daily_pnl[i] - self.daily_pnl[i-1] for i in range(1, len(self.daily_pnl))]
                    self.performance.volatility = np.std(returns) if returns else 0
                
                # Calculate max drawdown
                peak = max(self.daily_pnl)
                trough = min(self.daily_pnl)
                self.performance.max_drawdown = (peak - trough) / peak if peak > 0 else 0
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def _emergency_stop_trading(self):
        """Emergency stop trading"""
        logger.warning("Emergency stop trading triggered!")
        self.trading_active = False
        await self._close_all_positions()

    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            for position_id, position in list(self.positions.items()):
                await self._close_position(position_id)
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")

    async def _close_position(self, position_id: str):
        """Close a specific position"""
        try:
            if position_id not in self.positions:
                return
            
            position = self.positions[position_id]
            
            # Create closing order
            closing_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
            
            closing_order = TradingOrder(
                order_id=f"close_{secrets.token_hex(8)}",
                strategy=position.strategy,
                asset=position.asset,
                side=closing_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                metadata={
                    "position_id": position_id,
                    "is_closing": True
                }
            )
            
            # Execute closing order
            await self._execute_order(closing_order)
            
            # Calculate realized PnL
            if closing_order.status == OrderStatus.FILLED:
                position.realized_pnl = position.unrealized_pnl
                position.unrealized_pnl = 0
                
                # Remove position
                del self.positions[position_id]
                
                logger.info(f"Closed position {position_id} with PnL: {position.realized_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")

    # Public API methods
    async def submit_trading_order(self, asset: str, side: OrderSide, order_type: OrderType, 
                                 quantity: float, price: Optional[float] = None, 
                                 stop_price: Optional[float] = None, 
                                 strategy: TradingStrategy = TradingStrategy.ALGORITHMIC) -> str:
        """Submit a trading order"""
        order = TradingOrder(
            order_id=f"manual_{secrets.token_hex(8)}",
            strategy=strategy,
            asset=asset,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            metadata={"manual": True}
        )
        
        self.orders[order.order_id] = order
        return order.order_id

    async def get_trading_positions(self) -> List[Dict[str, Any]]:
        """Get all trading positions"""
        return [
            {
                "position_id": pos.position_id,
                "asset": pos.asset,
                "side": pos.side.value,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "realized_pnl": pos.realized_pnl,
                "strategy": pos.strategy.value if pos.strategy else None,
                "created_at": pos.created_at.isoformat()
            }
            for pos in self.positions.values()
        ]

    async def get_trading_performance(self) -> Dict[str, Any]:
        """Get trading performance metrics"""
        return {
            "performance_metrics": {
                "total_trades": self.performance.total_trades,
                "win_rate": self.performance.win_rate,
                "total_pnl": self.performance.total_pnl,
                "volatility": self.performance.volatility,
                "max_drawdown": self.performance.max_drawdown,
                "sharpe_ratio": self.performance.sharpe_ratio
            },
            "current_positions": len(self.positions),
            "active_orders": len([o for o in self.orders.values() if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]]),
            "daily_pnl": self.daily_pnl[-1] if self.daily_pnl else 0,
            "trading_active": self.trading_active
        }

    async def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Get recent trading signals"""
        recent_signals = sorted(
            self.signals.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:10]  # Last 10 signals
        
        return [
            {
                "signal_id": signal.signal_id,
                "strategy": signal.strategy.value,
                "asset": signal.asset,
                "side": signal.side.value,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "price_target": signal.price_target,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_score": signal.risk_score,
                "created_at": signal.created_at.isoformat()
            }
            for signal in recent_signals
        ]

    async def update_market_data(self, asset: str, price: float, volume: float = 0):
        """Update market data for an asset"""
        if asset not in self.market_data:
            self.market_data[asset] = {"prices": [], "volumes": []}
        
        self.market_data[asset]["price"] = price
        self.market_data[asset]["prices"].append(price)
        self.market_data[asset]["volumes"].append(volume)
        
        # Keep only last 100 price points
        if len(self.market_data[asset]["prices"]) > 100:
            self.market_data[asset]["prices"] = self.market_data[asset]["prices"][-100:]
            self.market_data[asset]["volumes"] = self.market_data[asset]["volumes"][-100:]

# Global instance
advanced_trading_engine = AdvancedTradingEngine()
