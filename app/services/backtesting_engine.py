"""
Backtesting Engine Service
Advanced backtesting engine for quantitative strategies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class BacktestStatus(Enum):
    """Backtest status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StrategyType(Enum):
    """Strategy types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MARKET_NEUTRAL = "market_neutral"
    LONG_SHORT = "long_short"
    FACTOR_BASED = "factor_based"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    QUANTITATIVE = "quantitative"


class OrderType(Enum):
    """Order types for backtesting"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    config_id: str
    strategy_name: str
    strategy_type: StrategyType
    universe: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    benchmark: str
    rebalance_frequency: str  # 'daily', 'weekly', 'monthly'
    transaction_costs: float
    slippage: float
    parameters: Dict[str, Any]
    risk_limits: Dict[str, float]
    created_at: datetime


@dataclass
class BacktestResult:
    """Backtest result"""
    backtest_id: str
    config_id: str
    status: BacktestStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float
    beta: float
    alpha: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall: float
    tail_ratio: float
    
    # Portfolio metrics
    final_portfolio_value: float
    peak_portfolio_value: float
    final_positions: Dict[str, float]
    
    # Benchmark comparison
    benchmark_return: float
    excess_return: float
    tracking_error: float
    
    # Results data
    equity_curve: List[Dict[str, Any]]
    trade_log: List[Dict[str, Any]]
    daily_returns: List[Dict[str, Any]]
    risk_metrics_history: List[Dict[str, Any]]
    
    created_at: datetime


@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    backtest_id: str
    symbol: str
    side: str  # 'buy', 'sell'
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: Optional[float]
    commission: float
    slippage: float
    holding_period: Optional[float]
    is_open: bool
    metadata: Dict[str, Any]


@dataclass
class Portfolio:
    """Portfolio state"""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, float]
    weights: Dict[str, float]
    returns: float
    cumulative_return: float


class BacktestingEngine:
    """Advanced Backtesting Engine"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        
        # Backtest management
        self.backtest_configs: Dict[str, BacktestConfig] = {}
        self.backtest_results: Dict[str, BacktestResult] = {}
        self.active_backtests: Dict[str, asyncio.Task] = {}
        
        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.fundamental_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Dict[str, pd.DataFrame] = {}
        
        # Portfolio tracking
        self.portfolios: Dict[str, List[Portfolio]] = defaultdict(list)
        self.trades: Dict[str, List[Trade]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the Backtesting Engine"""
        logger.info("Initializing Backtesting Engine")
        
        # Load historical data
        await self._load_historical_data()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._monitor_backtests()),
            asyncio.create_task(self._cleanup_completed_backtests())
        ]
        
        logger.info("Backtesting Engine initialized successfully")
    
    async def create_backtest_config(self, strategy_name: str, strategy_type: StrategyType,
                                   universe: List[str], start_date: datetime, end_date: datetime,
                                   initial_capital: float, benchmark: str = 'SPY',
                                   rebalance_frequency: str = 'daily',
                                   transaction_costs: float = 0.001,
                                   slippage: float = 0.0005,
                                   parameters: Optional[Dict[str, Any]] = None,
                                   risk_limits: Optional[Dict[str, float]] = None) -> BacktestConfig:
        """Create a backtest configuration"""
        try:
            config_id = f"CONFIG_{uuid.uuid4().hex[:8]}"
            
            config = BacktestConfig(
                config_id=config_id,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                benchmark=benchmark,
                rebalance_frequency=rebalance_frequency,
                transaction_costs=transaction_costs,
                slippage=slippage,
                parameters=parameters or {},
                risk_limits=risk_limits or {},
                created_at=datetime.utcnow()
            )
            
            self.backtest_configs[config_id] = config
            
            logger.info(f"Created backtest config {config_id}")
            return config
            
        except Exception as e:
            logger.error(f"Error creating backtest config: {e}")
            raise
    
    async def run_backtest(self, config_id: str) -> str:
        """Run a backtest"""
        try:
            config = self.backtest_configs.get(config_id)
            if not config:
                raise ValueError("Backtest config not found")
            
            backtest_id = f"BACKTEST_{uuid.uuid4().hex[:8]}"
            
            # Create initial result
            result = BacktestResult(
                backtest_id=backtest_id,
                config_id=config_id,
                status=BacktestStatus.PENDING,
                start_time=datetime.utcnow(),
                end_time=None,
                duration=None,
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                information_ratio=0.0,
                beta=0.0,
                alpha=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                var_95=0.0,
                var_99=0.0,
                expected_shortfall=0.0,
                tail_ratio=0.0,
                final_portfolio_value=config.initial_capital,
                peak_portfolio_value=config.initial_capital,
                final_positions={},
                benchmark_return=0.0,
                excess_return=0.0,
                tracking_error=0.0,
                equity_curve=[],
                trade_log=[],
                daily_returns=[],
                risk_metrics_history=[],
                created_at=datetime.utcnow()
            )
            
            self.backtest_results[backtest_id] = result
            
            # Start backtest
            task = asyncio.create_task(self._execute_backtest(backtest_id, config))
            self.active_backtests[backtest_id] = task
            
            logger.info(f"Started backtest {backtest_id}")
            return backtest_id
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    async def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """Get backtest result"""
        try:
            return self.backtest_results.get(backtest_id)
            
        except Exception as e:
            logger.error(f"Error getting backtest result: {e}")
            return None
    
    async def get_backtest_status(self, backtest_id: str) -> Optional[BacktestStatus]:
        """Get backtest status"""
        try:
            result = self.backtest_results.get(backtest_id)
            return result.status if result else None
            
        except Exception as e:
            logger.error(f"Error getting backtest status: {e}")
            return None
    
    async def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel a running backtest"""
        try:
            if backtest_id in self.active_backtests:
                task = self.active_backtests[backtest_id]
                task.cancel()
                del self.active_backtests[backtest_id]
                
                # Update result status
                result = self.backtest_results.get(backtest_id)
                if result:
                    result.status = BacktestStatus.CANCELLED
                    result.end_time = datetime.utcnow()
                    result.duration = (result.end_time - result.start_time).total_seconds()
                
                logger.info(f"Cancelled backtest {backtest_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling backtest: {e}")
            return False
    
    async def get_backtest_trades(self, backtest_id: str) -> List[Trade]:
        """Get trades for a backtest"""
        try:
            return self.trades.get(backtest_id, [])
            
        except Exception as e:
            logger.error(f"Error getting backtest trades: {e}")
            return []
    
    async def get_backtest_portfolios(self, backtest_id: str) -> List[Portfolio]:
        """Get portfolio history for a backtest"""
        try:
            return self.portfolios.get(backtest_id, [])
            
        except Exception as e:
            logger.error(f"Error getting backtest portfolios: {e}")
            return []
    
    async def _execute_backtest(self, backtest_id: str, config: BacktestConfig):
        """Execute a backtest"""
        try:
            result = self.backtest_results[backtest_id]
            result.status = BacktestStatus.RUNNING
            
            # Get price data
            price_data = await self._get_price_data(config.universe, config.start_date, config.end_date)
            benchmark_data = await self._get_benchmark_data(config.benchmark, config.start_date, config.end_date)
            
            if price_data.empty:
                result.status = BacktestStatus.FAILED
                return
            
            # Initialize portfolio
            portfolio = Portfolio(
                timestamp=config.start_date,
                total_value=config.initial_capital,
                cash=config.initial_capital,
                positions={},
                weights={},
                returns=0.0,
                cumulative_return=0.0
            )
            
            self.portfolios[backtest_id].append(portfolio)
            
            # Run backtest
            await self._run_strategy_backtest(backtest_id, config, price_data, benchmark_data)
            
            # Calculate final metrics
            await self._calculate_final_metrics(backtest_id, config, benchmark_data)
            
            # Update result
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
            logger.info(f"Completed backtest {backtest_id}")
            
        except Exception as e:
            logger.error(f"Error executing backtest: {e}")
            result = self.backtest_results.get(backtest_id)
            if result:
                result.status = BacktestStatus.FAILED
                result.end_time = datetime.utcnow()
                result.duration = (result.end_time - result.start_time).total_seconds()
    
    async def _run_strategy_backtest(self, backtest_id: str, config: BacktestConfig,
                                   price_data: pd.DataFrame, benchmark_data: pd.DataFrame):
        """Run strategy backtest"""
        try:
            result = self.backtest_results[backtest_id]
            
            # Generate date range
            date_range = pd.date_range(start=config.start_date, end=config.end_date, freq='D')
            
            for i, current_date in enumerate(date_range):
                if current_date not in price_data.index:
                    continue
                
                # Get current prices
                current_prices = price_data.loc[current_date]
                
                # Generate signals based on strategy type
                signals = await self._generate_signals(config, price_data, current_date)
                
                # Execute trades
                await self._execute_trades(backtest_id, config, signals, current_prices, current_date)
                
                # Update portfolio
                await self._update_portfolio(backtest_id, config, current_prices, current_date)
                
                # Check risk limits
                await self._check_risk_limits(backtest_id, config, current_date)
                
                # Update progress
                if i % 10 == 0:  # Update every 10 days
                    progress = (i / len(date_range)) * 100
                    logger.info(f"Backtest {backtest_id} progress: {progress:.1f}%")
            
        except Exception as e:
            logger.error(f"Error running strategy backtest: {e}")
            raise
    
    async def _generate_signals(self, config: BacktestConfig, price_data: pd.DataFrame,
                              current_date: datetime) -> Dict[str, float]:
        """Generate trading signals"""
        try:
            signals = {}
            
            if config.strategy_type == StrategyType.MOMENTUM:
                signals = await self._generate_momentum_signals(config, price_data, current_date)
            elif config.strategy_type == StrategyType.MEAN_REVERSION:
                signals = await self._generate_mean_reversion_signals(config, price_data, current_date)
            elif config.strategy_type == StrategyType.MARKET_NEUTRAL:
                signals = await self._generate_market_neutral_signals(config, price_data, current_date)
            else:
                signals = await self._generate_default_signals(config, price_data, current_date)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {}
    
    async def _generate_momentum_signals(self, config: BacktestConfig, price_data: pd.DataFrame,
                                       current_date: datetime) -> Dict[str, float]:
        """Generate momentum signals"""
        try:
            signals = {}
            lookback = config.parameters.get('lookback', 20)
            
            for symbol in config.universe:
                if symbol not in price_data.columns:
                    continue
                
                # Calculate momentum
                if len(price_data) >= lookback:
                    current_price = price_data[symbol].loc[current_date]
                    past_price = price_data[symbol].iloc[-lookback]
                    momentum = (current_price - past_price) / past_price
                    
                    # Generate signal
                    if momentum > 0.02:  # 2% momentum threshold
                        signals[symbol] = 1.0
                    elif momentum < -0.02:
                        signals[symbol] = -1.0
                    else:
                        signals[symbol] = 0.0
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return {}
    
    async def _generate_mean_reversion_signals(self, config: BacktestConfig, price_data: pd.DataFrame,
                                             current_date: datetime) -> Dict[str, float]:
        """Generate mean reversion signals"""
        try:
            signals = {}
            lookback = config.parameters.get('lookback', 20)
            threshold = config.parameters.get('threshold', 2.0)
            
            for symbol in config.universe:
                if symbol not in price_data.columns:
                    continue
                
                # Calculate z-score
                if len(price_data) >= lookback:
                    recent_prices = price_data[symbol].iloc[-lookback:]
                    current_price = price_data[symbol].loc[current_date]
                    mean_price = recent_prices.mean()
                    std_price = recent_prices.std()
                    
                    if std_price > 0:
                        z_score = (current_price - mean_price) / std_price
                        
                        # Generate signal
                        if z_score > threshold:
                            signals[symbol] = -1.0  # Sell (overvalued)
                        elif z_score < -threshold:
                            signals[symbol] = 1.0   # Buy (undervalued)
                        else:
                            signals[symbol] = 0.0
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signals: {e}")
            return {}
    
    async def _generate_market_neutral_signals(self, config: BacktestConfig, price_data: pd.DataFrame,
                                             current_date: datetime) -> Dict[str, float]:
        """Generate market neutral signals"""
        try:
            signals = {}
            
            # Calculate relative strength
            relative_strengths = {}
            for symbol in config.universe:
                if symbol not in price_data.columns:
                    continue
                
                # Calculate relative strength vs universe
                symbol_returns = price_data[symbol].pct_change().iloc[-20:].mean()
                universe_returns = price_data[config.universe].pct_change().iloc[-20:].mean().mean()
                relative_strengths[symbol] = symbol_returns - universe_returns
            
            # Rank and generate signals
            sorted_symbols = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)
            n_symbols = len(sorted_symbols)
            
            for i, (symbol, strength) in enumerate(sorted_symbols):
                if i < n_symbols // 2:
                    signals[symbol] = 1.0  # Long top half
                else:
                    signals[symbol] = -1.0  # Short bottom half
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating market neutral signals: {e}")
            return {}
    
    async def _generate_default_signals(self, config: BacktestConfig, price_data: pd.DataFrame,
                                      current_date: datetime) -> Dict[str, float]:
        """Generate default signals"""
        try:
            signals = {}
            
            # Simple random signals for demonstration
            for symbol in config.universe:
                if symbol in price_data.columns:
                    signals[symbol] = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating default signals: {e}")
            return {}
    
    async def _execute_trades(self, backtest_id: str, config: BacktestConfig,
                            signals: Dict[str, float], current_prices: pd.Series,
                            current_date: datetime):
        """Execute trades based on signals"""
        try:
            current_portfolio = self.portfolios[backtest_id][-1]
            
            for symbol, signal in signals.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                current_position = current_portfolio.positions.get(symbol, 0)
                
                # Calculate target position
                target_weight = signal * 0.1  # 10% position size
                target_value = current_portfolio.total_value * target_weight
                target_shares = target_value / current_price
                
                # Calculate trade
                trade_shares = target_shares - current_position
                
                if abs(trade_shares) > 0.01:  # Minimum trade size
                    # Create trade
                    trade = Trade(
                        trade_id=f"TRADE_{uuid.uuid4().hex[:8]}",
                        backtest_id=backtest_id,
                        symbol=symbol,
                        side='buy' if trade_shares > 0 else 'sell',
                        quantity=abs(trade_shares),
                        entry_price=current_price,
                        exit_price=None,
                        entry_time=current_date,
                        exit_time=None,
                        pnl=None,
                        commission=abs(trade_shares) * current_price * config.transaction_costs,
                        slippage=abs(trade_shares) * current_price * config.slippage,
                        holding_period=None,
                        is_open=True,
                        metadata={'signal': signal, 'target_weight': target_weight}
                    )
                    
                    self.trades[backtest_id].append(trade)
                    
                    # Update portfolio
                    trade_cost = trade_shares * current_price + trade.commission + trade.slippage
                    current_portfolio.cash -= trade_cost
                    current_portfolio.positions[symbol] = target_shares
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    async def _update_portfolio(self, backtest_id: str, config: BacktestConfig,
                              current_prices: pd.Series, current_date: datetime):
        """Update portfolio values"""
        try:
            current_portfolio = self.portfolios[backtest_id][-1]
            
            # Calculate portfolio value
            position_value = 0
            for symbol, shares in current_portfolio.positions.items():
                if symbol in current_prices:
                    position_value += shares * current_prices[symbol]
            
            total_value = current_portfolio.cash + position_value
            
            # Calculate returns
            if len(self.portfolios[backtest_id]) > 1:
                prev_portfolio = self.portfolios[backtest_id][-2]
                daily_return = (total_value - prev_portfolio.total_value) / prev_portfolio.total_value
                cumulative_return = (total_value - config.initial_capital) / config.initial_capital
            else:
                daily_return = 0.0
                cumulative_return = 0.0
            
            # Update weights
            weights = {}
            for symbol, shares in current_portfolio.positions.items():
                if symbol in current_prices and total_value > 0:
                    weights[symbol] = (shares * current_prices[symbol]) / total_value
            
            # Create new portfolio
            new_portfolio = Portfolio(
                timestamp=current_date,
                total_value=total_value,
                cash=current_portfolio.cash,
                positions=current_portfolio.positions.copy(),
                weights=weights,
                returns=daily_return,
                cumulative_return=cumulative_return
            )
            
            self.portfolios[backtest_id].append(new_portfolio)
            
            # Update result
            result = self.backtest_results[backtest_id]
            result.final_portfolio_value = total_value
            result.peak_portfolio_value = max(result.peak_portfolio_value, total_value)
            result.final_positions = current_portfolio.positions.copy()
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    async def _check_risk_limits(self, backtest_id: str, config: BacktestConfig, current_date: datetime):
        """Check risk limits"""
        try:
            # Implement risk limit checks
            # This is a simplified version
            pass
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    async def _calculate_final_metrics(self, backtest_id: str, config: BacktestConfig,
                                     benchmark_data: pd.DataFrame):
        """Calculate final performance metrics"""
        try:
            result = self.backtest_results[backtest_id]
            portfolios = self.portfolios[backtest_id]
            
            if not portfolios:
                return
            
            # Calculate returns
            portfolio_values = [p.total_value for p in portfolios]
            returns = [p.returns for p in portfolios[1:]]  # Skip first portfolio
            
            if not returns:
                return
            
            # Basic metrics
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            result.total_return = total_return
            
            # Annualized return
            days = (config.end_date - config.start_date).days
            result.annualized_return = (1 + total_return) ** (365 / days) - 1
            
            # Volatility
            result.volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio
            if result.volatility > 0:
                result.sharpe_ratio = result.annualized_return / result.volatility
            
            # Max drawdown
            peak = portfolio_values[0]
            max_dd = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd
            
            # Calmar ratio
            if result.max_drawdown > 0:
                result.calmar_ratio = result.annualized_return / result.max_drawdown
            
            # Trade statistics
            trades = self.trades[backtest_id]
            result.total_trades = len(trades)
            
            winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
            
            result.winning_trades = len(winning_trades)
            result.losing_trades = len(losing_trades)
            result.win_rate = len(winning_trades) / len(trades) if trades else 0
            
            if winning_trades:
                result.avg_win = np.mean([t.pnl for t in winning_trades])
            if losing_trades:
                result.avg_loss = np.mean([t.pnl for t in losing_trades])
            
            # Profit factor
            total_wins = sum([t.pnl for t in winning_trades]) if winning_trades else 0
            total_losses = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Risk metrics
            if returns:
                result.var_95 = np.percentile(returns, 5)
                result.var_99 = np.percentile(returns, 1)
                
                # Expected shortfall
                var_95_returns = [r for r in returns if r <= result.var_95]
                result.expected_shortfall = np.mean(var_95_returns) if var_95_returns else 0
            
            # Benchmark comparison
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data.pct_change().dropna()
                benchmark_total_return = (benchmark_data.iloc[-1] / benchmark_data.iloc[0]) - 1
                result.benchmark_return = benchmark_total_return
                result.excess_return = result.total_return - benchmark_total_return
                
                # Tracking error
                portfolio_returns_series = pd.Series(returns)
                if len(portfolio_returns_series) == len(benchmark_returns):
                    result.tracking_error = (portfolio_returns_series - benchmark_returns).std() * np.sqrt(252)
                
                # Beta and Alpha
                if len(portfolio_returns_series) == len(benchmark_returns):
                    covariance = np.cov(portfolio_returns_series, benchmark_returns)[0, 1]
                    benchmark_variance = np.var(benchmark_returns)
                    result.beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    result.alpha = result.annualized_return - (result.beta * benchmark_total_return)
            
            # Create equity curve
            result.equity_curve = [
                {'date': p.timestamp.isoformat(), 'value': p.total_value, 'return': p.returns}
                for p in portfolios
            ]
            
            # Create trade log
            result.trade_log = [
                {
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'side': t.side,
                    'quantity': t.quantity,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'pnl': t.pnl,
                    'commission': t.commission,
                    'slippage': t.slippage
                }
                for t in trades
            ]
            
            # Create daily returns
            result.daily_returns = [
                {'date': p.timestamp.isoformat(), 'return': p.returns}
                for p in portfolios[1:]  # Skip first portfolio
            ]
            
        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")
    
    # Background tasks
    async def _monitor_backtests(self):
        """Monitor active backtests"""
        while True:
            try:
                # Check for completed backtests
                completed_backtests = []
                for backtest_id, task in self.active_backtests.items():
                    if task.done():
                        completed_backtests.append(backtest_id)
                
                # Remove completed backtests
                for backtest_id in completed_backtests:
                    del self.active_backtests[backtest_id]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring backtests: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_completed_backtests(self):
        """Cleanup completed backtests"""
        while True:
            try:
                # Cleanup old completed backtests (older than 7 days)
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                backtests_to_remove = []
                for backtest_id, result in self.backtest_results.items():
                    if (result.status in [BacktestStatus.COMPLETED, BacktestStatus.FAILED, BacktestStatus.CANCELLED] and
                        result.end_time and result.end_time < cutoff_date):
                        backtests_to_remove.append(backtest_id)
                
                for backtest_id in backtests_to_remove:
                    del self.backtest_results[backtest_id]
                    if backtest_id in self.portfolios:
                        del self.portfolios[backtest_id]
                    if backtest_id in self.trades:
                        del self.trades[backtest_id]
                
                if backtests_to_remove:
                    logger.info(f"Cleaned up {len(backtests_to_remove)} old backtests")
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up completed backtests: {e}")
                await asyncio.sleep(7200)
    
    # Helper methods
    async def _get_price_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get price data for symbols"""
        try:
            # Simulate price data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            price_data = pd.DataFrame(index=date_range)
            
            for symbol in symbols:
                # Generate realistic price data
                base_price = 100 + hash(symbol) % 1000
                returns = np.random.normal(0, 0.02, len(date_range))
                prices = [base_price]
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                price_data[symbol] = prices
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()
    
    async def _get_benchmark_data(self, benchmark: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get benchmark data"""
        try:
            # Simulate benchmark data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            benchmark_data = pd.DataFrame(index=date_range)
            
            # Generate benchmark returns
            returns = np.random.normal(0.0005, 0.015, len(date_range))  # 0.05% daily return, 1.5% volatility
            prices = [100]  # Start at 100
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            benchmark_data[benchmark] = prices
            
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {e}")
            return pd.DataFrame()
    
    async def _load_historical_data(self):
        """Load historical data"""
        try:
            # Initialize with sample data
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
            
            for symbol in symbols:
                if symbol == 'SPY':
                    data = await self._get_benchmark_data(symbol, start_date, end_date)
                    self.benchmark_data[symbol] = data
                else:
                    data = await self._get_price_data([symbol], start_date, end_date)
                    self.price_data[symbol] = data
            
            logger.info("Loaded historical data")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")


# Factory function
async def get_backtesting_engine(redis_client: redis.Redis, db_session: Session) -> BacktestingEngine:
    """Get Backtesting Engine instance"""
    engine = BacktestingEngine(redis_client, db_session)
    await engine.initialize()
    return engine
