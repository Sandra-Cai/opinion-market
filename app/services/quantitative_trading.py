"""
Quantitative Trading Service
Provides algorithmic trading, backtesting, signal generation, and portfolio optimization
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
class TradingSignal:
    """Trading signal"""

    signal_id: str
    strategy_id: str
    market_id: int
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # Signal strength (0-1)
    confidence: float  # Confidence level (0-1)
    price_target: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    indicators: Dict[str, float]
    created_at: datetime
    expires_at: datetime


@dataclass
class TradingStrategy:
    """Trading strategy"""

    strategy_id: str
    strategy_name: str
    strategy_type: str  # 'momentum', 'mean_reversion', 'arbitrage', 'ml_based'
    parameters: Dict[str, Any]
    indicators: List[str]
    timeframes: List[str]
    markets: List[int]
    risk_management: Dict[str, Any]
    performance_metrics: Dict[str, float]
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class BacktestResult:
    """Backtest result"""

    backtest_id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trade_history: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    created_at: datetime


@dataclass
class PortfolioOptimization:
    """Portfolio optimization result"""

    optimization_id: str
    user_id: int
    optimization_type: str  # 'sharpe', 'min_variance', 'max_return', 'black_litterman'
    target_return: Optional[float]
    risk_tolerance: float
    constraints: Dict[str, Any]
    optimal_weights: Dict[int, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    efficient_frontier: List[Dict[str, float]]
    created_at: datetime


class QuantitativeTradingService:
    """Comprehensive quantitative trading service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.trading_signals: Dict[str, TradingSignal] = {}
        self.trading_strategies: Dict[str, TradingStrategy] = {}
        self.backtest_results: Dict[str, BacktestResult] = {}
        self.portfolio_optimizations: Dict[str, PortfolioOptimization] = {}

        # Market data
        self.price_data: Dict[int, pd.DataFrame] = {}
        self.technical_indicators: Dict[str, Dict[int, pd.Series]] = defaultdict(dict)

        # Strategy performance tracking
        self.strategy_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

    async def initialize(self):
        """Initialize the quantitative trading service"""
        logger.info("Initializing Quantitative Trading Service")

        # Load existing data
        await self._load_trading_strategies()
        await self._load_backtest_results()

        # Start background tasks
        asyncio.create_task(self._generate_signals())
        asyncio.create_task(self._update_indicators())
        asyncio.create_task(self._monitor_strategies())
        asyncio.create_task(self._cleanup_expired_signals())

        logger.info("Quantitative Trading Service initialized successfully")

    async def create_trading_strategy(
        self,
        strategy_name: str,
        strategy_type: str,
        parameters: Dict[str, Any],
        indicators: List[str],
        timeframes: List[str],
        markets: List[int],
        risk_management: Dict[str, Any],
    ) -> TradingStrategy:
        """Create a new trading strategy"""
        try:
            strategy_id = f"strategy_{strategy_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            strategy = TradingStrategy(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                parameters=parameters,
                indicators=indicators,
                timeframes=timeframes,
                markets=markets,
                risk_management=risk_management,
                performance_metrics={
                    "total_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                },
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.trading_strategies[strategy_id] = strategy
            await self._cache_trading_strategy(strategy)

            logger.info(f"Created trading strategy {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Error creating trading strategy: {e}")
            raise

    async def generate_signal(self, strategy_id: str, market_id: int) -> TradingSignal:
        """Generate trading signal for a strategy and market"""
        try:
            strategy = self.trading_strategies.get(strategy_id)
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")

            # Get market data
            market_data = await self._get_market_data(market_id)
            if market_data.empty:
                raise ValueError(f"No market data available for market {market_id}")

            # Calculate indicators
            indicators = await self._calculate_indicators(
                market_data, strategy.indicators
            )

            # Generate signal based on strategy type
            signal = await self._apply_strategy_logic(strategy, market_data, indicators)

            self.trading_signals[signal.signal_id] = signal
            await self._cache_trading_signal(signal)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            raise

    async def run_backtest(
        self,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
    ) -> BacktestResult:
        """Run backtest for a strategy"""
        try:
            strategy = self.trading_strategies.get(strategy_id)
            if not strategy:
                raise ValueError(f"Strategy {strategy_id} not found")

            backtest_id = (
                f"backtest_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            # Get historical data for all markets
            all_market_data = {}
            for market_id in strategy.markets:
                market_data = await self._get_historical_data(
                    market_id, start_date, end_date
                )
                if not market_data.empty:
                    all_market_data[market_id] = market_data

            # Run backtest simulation
            results = await self._simulate_backtest(
                strategy, all_market_data, initial_capital
            )

            backtest_result = BacktestResult(
                backtest_id=backtest_id,
                strategy_id=strategy_id,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=results["final_capital"],
                total_return=results["total_return"],
                annualized_return=results["annualized_return"],
                sharpe_ratio=results["sharpe_ratio"],
                max_drawdown=results["max_drawdown"],
                win_rate=results["win_rate"],
                profit_factor=results["profit_factor"],
                total_trades=results["total_trades"],
                winning_trades=results["winning_trades"],
                losing_trades=results["losing_trades"],
                avg_win=results["avg_win"],
                avg_loss=results["avg_loss"],
                trade_history=results["trade_history"],
                equity_curve=results["equity_curve"],
                created_at=datetime.utcnow(),
            )

            self.backtest_results[backtest_id] = backtest_result
            await self._cache_backtest_result(backtest_result)

            # Update strategy performance metrics
            strategy.performance_metrics.update(
                {
                    "total_return": results["total_return"],
                    "sharpe_ratio": results["sharpe_ratio"],
                    "max_drawdown": results["max_drawdown"],
                    "win_rate": results["win_rate"],
                    "profit_factor": results["profit_factor"],
                }
            )
            strategy.last_updated = datetime.utcnow()
            await self._cache_trading_strategy(strategy)

            logger.info(f"Completed backtest {backtest_id}")
            return backtest_result

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise

    async def optimize_portfolio(
        self,
        user_id: int,
        optimization_type: str,
        target_return: Optional[float] = None,
        risk_tolerance: float = 0.5,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> PortfolioOptimization:
        """Optimize portfolio allocation"""
        try:
            optimization_id = (
                f"optimization_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            # Get user's current positions and available markets
            user_positions = await self._get_user_positions(user_id)
            available_markets = await self._get_available_markets()

            # Calculate expected returns and covariance matrix
            expected_returns = await self._calculate_expected_returns(available_markets)
            covariance_matrix = await self._calculate_covariance_matrix(
                available_markets
            )

            # Run optimization
            optimal_weights = await self._run_portfolio_optimization(
                optimization_type,
                expected_returns,
                covariance_matrix,
                target_return,
                risk_tolerance,
                constraints,
            )

            # Calculate portfolio metrics
            portfolio_return = np.sum(
                expected_returns * np.array(list(optimal_weights.values()))
            )
            portfolio_volatility = np.sqrt(
                np.array(list(optimal_weights.values())).T
                @ covariance_matrix
                @ np.array(list(optimal_weights.values()))
            )
            sharpe_ratio = (
                portfolio_return / portfolio_volatility
                if portfolio_volatility > 0
                else 0
            )

            # Generate efficient frontier
            efficient_frontier = await self._generate_efficient_frontier(
                expected_returns, covariance_matrix, constraints
            )

            optimization = PortfolioOptimization(
                optimization_id=optimization_id,
                user_id=user_id,
                optimization_type=optimization_type,
                target_return=target_return,
                risk_tolerance=risk_tolerance,
                constraints=constraints or {},
                optimal_weights=optimal_weights,
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                efficient_frontier=efficient_frontier,
                created_at=datetime.utcnow(),
            )

            self.portfolio_optimizations[optimization_id] = optimization
            await self._cache_portfolio_optimization(optimization)

            logger.info(f"Completed portfolio optimization {optimization_id}")
            return optimization

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise

    async def calculate_technical_indicators(
        self, market_data: pd.DataFrame, indicators: List[str]
    ) -> Dict[str, pd.Series]:
        """Calculate technical indicators"""
        try:
            results = {}

            for indicator in indicators:
                if indicator == "sma":
                    for period in [20, 50, 200]:
                        results[f"sma_{period}"] = (
                            market_data["close"].rolling(window=period).mean()
                        )

                elif indicator == "ema":
                    for period in [12, 26]:
                        results[f"ema_{period}"] = (
                            market_data["close"].ewm(span=period).mean()
                        )

                elif indicator == "rsi":
                    results["rsi"] = self._calculate_rsi(market_data["close"])

                elif indicator == "macd":
                    macd, signal, histogram = self._calculate_macd(market_data["close"])
                    results["macd"] = macd
                    results["macd_signal"] = signal
                    results["macd_histogram"] = histogram

                elif indicator == "bollinger_bands":
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                        market_data["close"]
                    )
                    results["bb_upper"] = bb_upper
                    results["bb_middle"] = bb_middle
                    results["bb_lower"] = bb_lower

                elif indicator == "stochastic":
                    k_percent, d_percent = self._calculate_stochastic(market_data)
                    results["stoch_k"] = k_percent
                    results["stoch_d"] = d_percent

                elif indicator == "atr":
                    results["atr"] = self._calculate_atr(market_data)

                elif indicator == "volume_sma":
                    results["volume_sma"] = (
                        market_data["volume"].rolling(window=20).mean()
                    )

            return results

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

    async def _apply_strategy_logic(
        self,
        strategy: TradingStrategy,
        market_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
    ) -> TradingSignal:
        """Apply strategy logic to generate signal"""
        try:
            signal_id = f"signal_{strategy.strategy_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            current_price = market_data["close"].iloc[-1]
            signal_type = "hold"
            strength = 0.0
            confidence = 0.0
            reasoning = ""

            if strategy.strategy_type == "momentum":
                signal_type, strength, confidence, reasoning = self._momentum_strategy(
                    market_data, indicators, strategy.parameters
                )

            elif strategy.strategy_type == "mean_reversion":
                signal_type, strength, confidence, reasoning = (
                    self._mean_reversion_strategy(
                        market_data, indicators, strategy.parameters
                    )
                )

            elif strategy.strategy_type == "arbitrage":
                signal_type, strength, confidence, reasoning = self._arbitrage_strategy(
                    market_data, indicators, strategy.parameters
                )

            elif strategy.strategy_type == "ml_based":
                signal_type, strength, confidence, reasoning = (
                    await self._ml_based_strategy(
                        market_data, indicators, strategy.parameters
                    )
                )

            # Calculate price targets
            price_target = None
            stop_loss = None
            take_profit = None

            if signal_type in ["buy", "sell"]:
                atr = indicators.get("atr", pd.Series([current_price * 0.02]))
                if signal_type == "buy":
                    price_target = current_price * (1 + strength * 0.1)
                    stop_loss = current_price - atr.iloc[-1] * 2
                    take_profit = current_price + atr.iloc[-1] * 3
                else:
                    price_target = current_price * (1 - strength * 0.1)
                    stop_loss = current_price + atr.iloc[-1] * 2
                    take_profit = current_price - atr.iloc[-1] * 3

            signal = TradingSignal(
                signal_id=signal_id,
                strategy_id=strategy.strategy_id,
                market_id=market_data.get("market_id", 1),
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                indicators={
                    k: v.iloc[-1] if hasattr(v, "iloc") else v
                    for k, v in indicators.items()
                },
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),
            )

            return signal

        except Exception as e:
            logger.error(f"Error applying strategy logic: {e}")
            raise

    def _momentum_strategy(
        self,
        market_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        parameters: Dict[str, Any],
    ) -> Tuple[str, float, float, str]:
        """Momentum strategy logic"""
        try:
            current_price = market_data["close"].iloc[-1]
            sma_20 = indicators.get("sma_20", pd.Series([current_price]))
            sma_50 = indicators.get("sma_50", pd.Series([current_price]))
            rsi = indicators.get("rsi", pd.Series([50]))

            # Momentum signals
            price_above_sma20 = current_price > sma_20.iloc[-1]
            price_above_sma50 = current_price > sma_50.iloc[-1]
            rsi_not_overbought = rsi.iloc[-1] < 70
            rsi_not_oversold = rsi.iloc[-1] > 30

            if price_above_sma20 and price_above_sma50 and rsi_not_overbought:
                return (
                    "buy",
                    0.8,
                    0.7,
                    "Strong momentum with price above moving averages",
                )
            elif not price_above_sma20 and not price_above_sma50 and rsi_not_oversold:
                return (
                    "sell",
                    0.8,
                    0.7,
                    "Weak momentum with price below moving averages",
                )
            else:
                return "hold", 0.0, 0.5, "No clear momentum signal"

        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return "hold", 0.0, 0.0, "Error in momentum calculation"

    def _mean_reversion_strategy(
        self,
        market_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        parameters: Dict[str, Any],
    ) -> Tuple[str, float, float, str]:
        """Mean reversion strategy logic"""
        try:
            current_price = market_data["close"].iloc[-1]
            bb_upper = indicators.get("bb_upper", pd.Series([current_price * 1.02]))
            bb_lower = indicators.get("bb_lower", pd.Series([current_price * 0.98]))
            rsi = indicators.get("rsi", pd.Series([50]))

            # Mean reversion signals
            price_near_upper = current_price >= bb_upper.iloc[-1] * 0.98
            price_near_lower = current_price <= bb_lower.iloc[-1] * 1.02
            rsi_overbought = rsi.iloc[-1] > 70
            rsi_oversold = rsi.iloc[-1] < 30

            if price_near_upper and rsi_overbought:
                return (
                    "sell",
                    0.7,
                    0.6,
                    "Price near upper Bollinger Band with overbought RSI",
                )
            elif price_near_lower and rsi_oversold:
                return (
                    "buy",
                    0.7,
                    0.6,
                    "Price near lower Bollinger Band with oversold RSI",
                )
            else:
                return "hold", 0.0, 0.5, "No clear mean reversion signal"

        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return "hold", 0.0, 0.0, "Error in mean reversion calculation"

    def _arbitrage_strategy(
        self,
        market_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        parameters: Dict[str, Any],
    ) -> Tuple[str, float, float, str]:
        """Arbitrage strategy logic"""
        try:
            # Simplified arbitrage logic - in practice, this would be more complex
            return "hold", 0.0, 0.5, "Arbitrage opportunities require real-time data"

        except Exception as e:
            logger.error(f"Error in arbitrage strategy: {e}")
            return "hold", 0.0, 0.0, "Error in arbitrage calculation"

    async def _ml_based_strategy(
        self,
        market_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        parameters: Dict[str, Any],
    ) -> Tuple[str, float, float, str]:
        """Machine learning based strategy logic"""
        try:
            # Simplified ML strategy - in practice, this would use trained models
            return "hold", 0.0, 0.5, "ML strategy requires trained models"

        except Exception as e:
            logger.error(f"Error in ML strategy: {e}")
            return "hold", 0.0, 0.0, "Error in ML calculation"

    async def _simulate_backtest(
        self,
        strategy: TradingStrategy,
        market_data: Dict[int, pd.DataFrame],
        initial_capital: float,
    ) -> Dict[str, Any]:
        """Simulate backtest"""
        try:
            capital = initial_capital
            positions = {}
            trades = []
            equity_curve = []

            # Get all dates
            all_dates = set()
            for data in market_data.values():
                all_dates.update(data.index)
            all_dates = sorted(all_dates)

            for date in all_dates:
                # Update positions and calculate P&L
                for market_id, position in positions.items():
                    if (
                        market_id in market_data
                        and date in market_data[market_id].index
                    ):
                        current_price = market_data[market_id].loc[date, "close"]
                        position["unrealized_pnl"] = (
                            current_price - position["entry_price"]
                        ) * position["quantity"]

                # Generate signals
                for market_id in strategy.markets:
                    if (
                        market_id in market_data
                        and date in market_data[market_id].index
                    ):
                        # Get data up to current date
                        current_data = market_data[market_id].loc[:date]
                        if len(current_data) > 50:  # Need enough data for indicators
                            indicators = await self.calculate_technical_indicators(
                                current_data, strategy.indicators
                            )
                            signal = await self._apply_strategy_logic(
                                strategy, current_data, indicators
                            )

                            if signal.signal_type in ["buy", "sell"]:
                                # Execute trade
                                trade = await self._execute_trade(
                                    market_id, signal, capital, positions, date
                                )
                                if trade:
                                    trades.append(trade)
                                    capital = trade["capital_after"]

                # Record equity
                total_equity = capital + sum(
                    pos["unrealized_pnl"] for pos in positions.values()
                )
                equity_curve.append(
                    {
                        "date": date.isoformat(),
                        "equity": total_equity,
                        "capital": capital,
                        "positions": len(positions),
                    }
                )

            # Calculate performance metrics
            final_equity = (
                equity_curve[-1]["equity"] if equity_curve else initial_capital
            )
            total_return = (final_equity - initial_capital) / initial_capital

            # Calculate other metrics
            returns = [
                equity_curve[i]["equity"] / equity_curve[i - 1]["equity"] - 1
                for i in range(1, len(equity_curve))
            ]

            annualized_return = (
                (1 + total_return) ** (252 / len(equity_curve)) - 1
                if equity_curve
                else 0
            )
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if returns and np.std(returns) > 0
                else 0
            )

            # Calculate drawdown
            peak = initial_capital
            max_drawdown = 0
            for point in equity_curve:
                if point["equity"] > peak:
                    peak = point["equity"]
                drawdown = (peak - point["equity"]) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Calculate trade statistics
            winning_trades = [t for t in trades if t["pnl"] > 0]
            losing_trades = [t for t in trades if t["pnl"] < 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = (
                np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
            )
            avg_loss = (
                np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
            )
            profit_factor = (
                abs(
                    sum(t["pnl"] for t in winning_trades)
                    / sum(t["pnl"] for t in losing_trades)
                )
                if losing_trades
                else float("inf")
            )

            return {
                "final_capital": final_equity,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": len(trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "trade_history": trades,
                "equity_curve": equity_curve,
            }

        except Exception as e:
            logger.error(f"Error simulating backtest: {e}")
            raise

    async def _execute_trade(
        self,
        market_id: int,
        signal: TradingSignal,
        capital: float,
        positions: Dict,
        date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Execute a trade"""
        try:
            current_price = signal.indicators.get("close", 100.0)
            position_size = capital * 0.1  # 10% of capital per trade
            quantity = int(position_size / current_price)

            if quantity <= 0:
                return None

            if signal.signal_type == "buy":
                if market_id in positions:
                    # Close existing short position
                    old_position = positions[market_id]
                    pnl = (old_position["entry_price"] - current_price) * old_position[
                        "quantity"
                    ]
                    capital += pnl
                    del positions[market_id]

                # Open long position
                positions[market_id] = {
                    "type": "long",
                    "entry_price": current_price,
                    "quantity": quantity,
                    "unrealized_pnl": 0.0,
                }
                capital -= quantity * current_price

            elif signal.signal_type == "sell":
                if market_id in positions:
                    # Close existing long position
                    old_position = positions[market_id]
                    pnl = (current_price - old_position["entry_price"]) * old_position[
                        "quantity"
                    ]
                    capital += pnl
                    del positions[market_id]

                # Open short position
                positions[market_id] = {
                    "type": "short",
                    "entry_price": current_price,
                    "quantity": quantity,
                    "unrealized_pnl": 0.0,
                }
                capital += quantity * current_price

            return {
                "date": date.isoformat(),
                "market_id": market_id,
                "signal_type": signal.signal_type,
                "price": current_price,
                "quantity": quantity,
                "pnl": 0.0,  # Will be calculated later
                "capital_after": capital,
            }

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def _calculate_stochastic(
        self, market_data: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = market_data["low"].rolling(window=k_period).min()
        high_max = market_data["high"].rolling(window=k_period).max()
        k_percent = 100 * ((market_data["close"] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = market_data["high"] - market_data["low"]
        high_close = np.abs(market_data["high"] - market_data["close"].shift())
        low_close = np.abs(market_data["low"] - market_data["close"].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr

    # Background tasks
    async def _generate_signals(self):
        """Generate signals for active strategies"""
        while True:
            try:
                for strategy in self.trading_strategies.values():
                    if strategy.is_active:
                        for market_id in strategy.markets:
                            await self.generate_signal(strategy.strategy_id, market_id)

                await asyncio.sleep(300)  # Generate signals every 5 minutes

            except Exception as e:
                logger.error(f"Error generating signals: {e}")
                await asyncio.sleep(600)

    async def _update_indicators(self):
        """Update technical indicators"""
        while True:
            try:
                # Update indicators for all markets
                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating indicators: {e}")
                await asyncio.sleep(120)

    async def _monitor_strategies(self):
        """Monitor strategy performance"""
        while True:
            try:
                # Monitor and update strategy performance
                await asyncio.sleep(3600)  # Monitor every hour

            except Exception as e:
                logger.error(f"Error monitoring strategies: {e}")
                await asyncio.sleep(7200)

    async def _cleanup_expired_signals(self):
        """Clean up expired signals"""
        while True:
            try:
                expired_signals = [
                    signal_id
                    for signal_id, signal in self.trading_signals.items()
                    if signal.expires_at <= datetime.utcnow()
                ]

                for signal_id in expired_signals:
                    del self.trading_signals[signal_id]
                    await self.redis.delete(f"trading_signal:{signal_id}")

                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                logger.error(f"Error cleaning up expired signals: {e}")
                await asyncio.sleep(7200)

    # Helper methods (implementations would depend on your data models)
    async def _load_trading_strategies(self):
        """Load trading strategies from database"""
        pass

    async def _load_backtest_results(self):
        """Load backtest results from database"""
        pass

    async def _get_market_data(self, market_id: int) -> pd.DataFrame:
        """Get market data"""
        return pd.DataFrame()  # Placeholder

    async def _get_historical_data(
        self, market_id: int, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get historical market data"""
        return pd.DataFrame()  # Placeholder

    async def _calculate_indicators(
        self, market_data: pd.DataFrame, indicators: List[str]
    ) -> Dict[str, pd.Series]:
        """Calculate indicators"""
        return await self.calculate_technical_indicators(market_data, indicators)

    async def _get_user_positions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user positions"""
        return []  # Placeholder

    async def _get_available_markets(self) -> List[int]:
        """Get available markets"""
        return [1, 2, 3]  # Placeholder

    async def _calculate_expected_returns(self, markets: List[int]) -> pd.Series:
        """Calculate expected returns"""
        return pd.Series([0.1] * len(markets), index=markets)  # Placeholder

    async def _calculate_covariance_matrix(self, markets: List[int]) -> pd.DataFrame:
        """Calculate covariance matrix"""
        return pd.DataFrame(
            np.eye(len(markets)), index=markets, columns=markets
        )  # Placeholder

    async def _run_portfolio_optimization(
        self,
        optimization_type: str,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        target_return: Optional[float],
        risk_tolerance: float,
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[int, float]:
        """Run portfolio optimization"""
        # Simplified optimization - in practice, this would use scipy.optimize
        n_assets = len(expected_returns)
        weights = np.ones(n_assets) / n_assets  # Equal weight
        return dict(zip(expected_returns.index, weights))

    async def _generate_efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """Generate efficient frontier"""
        # Simplified frontier - in practice, this would calculate multiple portfolios
        return [{"return": 0.1, "volatility": 0.15, "sharpe": 0.67}]

    # Caching methods
    async def _cache_trading_signal(self, signal: TradingSignal):
        """Cache trading signal"""
        try:
            cache_key = f"trading_signal:{signal.signal_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "strategy_id": signal.strategy_id,
                        "market_id": signal.market_id,
                        "signal_type": signal.signal_type,
                        "strength": signal.strength,
                        "confidence": signal.confidence,
                        "price_target": signal.price_target,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "reasoning": signal.reasoning,
                        "indicators": signal.indicators,
                        "created_at": signal.created_at.isoformat(),
                        "expires_at": signal.expires_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching trading signal: {e}")

    async def _cache_trading_strategy(self, strategy: TradingStrategy):
        """Cache trading strategy"""
        try:
            cache_key = f"trading_strategy:{strategy.strategy_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "strategy_name": strategy.strategy_name,
                        "strategy_type": strategy.strategy_type,
                        "parameters": strategy.parameters,
                        "indicators": strategy.indicators,
                        "timeframes": strategy.timeframes,
                        "markets": strategy.markets,
                        "risk_management": strategy.risk_management,
                        "performance_metrics": strategy.performance_metrics,
                        "is_active": strategy.is_active,
                        "created_at": strategy.created_at.isoformat(),
                        "last_updated": strategy.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching trading strategy: {e}")

    async def _cache_backtest_result(self, result: BacktestResult):
        """Cache backtest result"""
        try:
            cache_key = f"backtest_result:{result.backtest_id}"
            await self.redis.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps(
                    {
                        "strategy_id": result.strategy_id,
                        "start_date": result.start_date.isoformat(),
                        "end_date": result.end_date.isoformat(),
                        "initial_capital": result.initial_capital,
                        "final_capital": result.final_capital,
                        "total_return": result.total_return,
                        "annualized_return": result.annualized_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "win_rate": result.win_rate,
                        "profit_factor": result.profit_factor,
                        "total_trades": result.total_trades,
                        "winning_trades": result.winning_trades,
                        "losing_trades": result.losing_trades,
                        "avg_win": result.avg_win,
                        "avg_loss": result.avg_loss,
                        "created_at": result.created_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching backtest result: {e}")

    async def _cache_portfolio_optimization(self, optimization: PortfolioOptimization):
        """Cache portfolio optimization"""
        try:
            cache_key = f"portfolio_optimization:{optimization.optimization_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "user_id": optimization.user_id,
                        "optimization_type": optimization.optimization_type,
                        "target_return": optimization.target_return,
                        "risk_tolerance": optimization.risk_tolerance,
                        "constraints": optimization.constraints,
                        "optimal_weights": optimization.optimal_weights,
                        "expected_return": optimization.expected_return,
                        "expected_volatility": optimization.expected_volatility,
                        "sharpe_ratio": optimization.sharpe_ratio,
                        "created_at": optimization.created_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching portfolio optimization: {e}")


# Factory function
async def get_quantitative_trading_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> QuantitativeTradingService:
    """Get quantitative trading service instance"""
    service = QuantitativeTradingService(redis_client, db_session)
    await service.initialize()
    return service
