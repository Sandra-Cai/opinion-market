"""
Foreign Exchange Trading Service
Provides spot FX, forwards, swaps, and comprehensive FX analytics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


@dataclass
class CurrencyPair:
    """Currency pair information"""

    pair_id: str
    base_currency: str  # e.g., 'USD'
    quote_currency: str  # e.g., 'EUR'
    pair_name: str  # e.g., 'EUR/USD'
    pip_value: float
    lot_size: float
    min_trade_size: float
    max_trade_size: float
    margin_requirement: float
    swap_long: float  # Overnight interest for long positions
    swap_short: float  # Overnight interest for short positions
    is_active: bool
    trading_hours: Dict[str, List[str]]  # Trading sessions
    created_at: datetime
    last_updated: datetime


@dataclass
class FXPrice:
    """FX price information"""

    price_id: str
    pair_id: str
    bid_price: float
    ask_price: float
    mid_price: float
    spread: float
    pip_value: float
    timestamp: datetime
    source: str
    volume_24h: float
    high_24h: float
    low_24h: float
    change_24h: float
    change_pct_24h: float


@dataclass
class FXPosition:
    """FX trading position"""

    position_id: str
    user_id: int
    pair_id: str
    position_type: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    pip_value: float
    unrealized_pnl: float
    realized_pnl: float
    swap_charges: float
    margin_used: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: datetime
    last_updated: datetime


@dataclass
class ForwardContract:
    """FX forward contract"""

    contract_id: str
    pair_id: str
    user_id: int
    quantity: float
    forward_rate: float
    spot_rate: float
    forward_points: float
    value_date: datetime
    maturity_date: datetime
    contract_type: str  # 'buy' or 'sell'
    is_deliverable: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class SwapContract:
    """FX swap contract"""

    swap_id: str
    pair_id: str
    user_id: int
    near_leg: Dict[str, Any]  # Near leg details
    far_leg: Dict[str, Any]  # Far leg details
    swap_rate: float
    swap_points: float
    value_date: datetime
    maturity_date: datetime
    created_at: datetime
    last_updated: datetime


@dataclass
class FXOrder:
    """FX trading order"""

    order_id: str
    user_id: int
    pair_id: str
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    limit_price: Optional[float]
    time_in_force: str  # 'GTC', 'IOC', 'FOK'
    status: str  # 'pending', 'filled', 'cancelled', 'rejected'
    filled_quantity: float
    filled_price: float
    created_at: datetime
    last_updated: datetime


class ForexTradingService:
    """Comprehensive foreign exchange trading service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.currency_pairs: Dict[str, CurrencyPair] = {}
        self.fx_prices: Dict[str, List[FXPrice]] = defaultdict(list)
        self.fx_positions: Dict[str, FXPosition] = {}
        self.forward_contracts: Dict[str, ForwardContract] = {}
        self.swap_contracts: Dict[str, SwapContract] = {}
        self.fx_orders: Dict[str, FXOrder] = {}

        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # FX specific data
        self.interest_rates: Dict[str, Dict[str, float]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.volatility_surfaces: Dict[str, Dict[str, Any]] = {}
        self.cross_rates: Dict[str, Dict[str, float]] = {}

        # Trading sessions
        self.trading_sessions = {
            "sydney": {"start": "22:00", "end": "07:00"},
            "tokyo": {"start": "00:00", "end": "09:00"},
            "london": {"start": "08:00", "end": "17:00"},
            "new_york": {"start": "13:00", "end": "22:00"},
        }

    async def initialize(self):
        """Initialize the forex trading service"""
        logger.info("Initializing Forex Trading Service")

        # Load existing data
        await self._load_currency_pairs()
        await self._load_interest_rates()
        await self._load_correlation_matrix()

        # Start background tasks
        asyncio.create_task(self._update_fx_prices())
        asyncio.create_task(self._update_positions())
        asyncio.create_task(self._update_forward_contracts())
        asyncio.create_task(self._update_swap_contracts())
        asyncio.create_task(self._monitor_trading_sessions())

        logger.info("Forex Trading Service initialized successfully")

    async def create_currency_pair(
        self,
        base_currency: str,
        quote_currency: str,
        pip_value: float,
        lot_size: float,
        min_trade_size: float,
        max_trade_size: float,
        margin_requirement: float,
        swap_long: float = 0.0,
        swap_short: float = 0.0,
    ) -> CurrencyPair:
        """Create a new currency pair"""
        try:
            pair_id = f"{base_currency}{quote_currency}"
            pair_name = f"{base_currency}/{quote_currency}"

            # Set default trading hours (24/5 for major pairs)
            trading_hours = {
                "monday": ["00:00", "23:59"],
                "tuesday": ["00:00", "23:59"],
                "wednesday": ["00:00", "23:59"],
                "thursday": ["00:00", "23:59"],
                "friday": ["00:00", "23:59"],
                "saturday": ["00:00", "00:00"],
                "sunday": ["00:00", "00:00"],
            }

            pair = CurrencyPair(
                pair_id=pair_id,
                base_currency=base_currency,
                quote_currency=quote_currency,
                pair_name=pair_name,
                pip_value=pip_value,
                lot_size=lot_size,
                min_trade_size=min_trade_size,
                max_trade_size=max_trade_size,
                margin_requirement=margin_requirement,
                swap_long=swap_long,
                swap_short=swap_short,
                is_active=True,
                trading_hours=trading_hours,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.currency_pairs[pair_id] = pair
            await self._cache_currency_pair(pair)

            logger.info(f"Created currency pair {pair_name}")
            return pair

        except Exception as e:
            logger.error(f"Error creating currency pair: {e}")
            raise

    async def add_fx_price(
        self,
        pair_id: str,
        bid_price: float,
        ask_price: float,
        volume_24h: float,
        high_24h: float,
        low_24h: float,
        change_24h: float,
        source: str,
    ) -> FXPrice:
        """Add a new FX price"""
        try:
            if pair_id not in self.currency_pairs:
                raise ValueError(f"Currency pair {pair_id} not found")

            # Calculate derived values
            mid_price = (bid_price + ask_price) / 2
            spread = ask_price - bid_price
            pip_value = self.currency_pairs[pair_id].pip_value

            # Calculate percentage change
            change_pct_24h = (
                (change_24h / (mid_price - change_24h)) * 100
                if (mid_price - change_24h) != 0
                else 0
            )

            price = FXPrice(
                price_id=f"fx_price_{pair_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                pair_id=pair_id,
                bid_price=bid_price,
                ask_price=ask_price,
                mid_price=mid_price,
                spread=spread,
                pip_value=pip_value,
                timestamp=datetime.utcnow(),
                source=source,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                change_24h=change_24h,
                change_pct_24h=change_pct_24h,
            )

            self.fx_prices[pair_id].append(price)

            # Keep only recent prices
            if len(self.fx_prices[pair_id]) > 1000:
                self.fx_prices[pair_id] = self.fx_prices[pair_id][-1000:]

            await self._cache_fx_price(price)

            logger.info(f"Added price for {pair_id}: {bid_price}/{ask_price}")
            return price

        except Exception as e:
            logger.error(f"Error adding FX price: {e}")
            raise

    async def create_fx_position(
        self,
        user_id: int,
        pair_id: str,
        position_type: str,
        quantity: float,
        entry_price: float,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> FXPosition:
        """Create a new FX position"""
        try:
            if pair_id not in self.currency_pairs:
                raise ValueError(f"Currency pair {pair_id} not found")

            if position_type not in ["long", "short"]:
                raise ValueError("Position type must be 'long' or 'short'")

            currency_pair = self.currency_pairs[pair_id]

            # Calculate margin requirements
            margin_used = (
                quantity * entry_price * currency_pair.margin_requirement
            ) / leverage

            position_id = f"fx_pos_{user_id}_{pair_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            position = FXPosition(
                position_id=position_id,
                user_id=user_id,
                pair_id=pair_id,
                position_type=position_type,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                pip_value=currency_pair.pip_value,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                swap_charges=0.0,
                margin_used=margin_used,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.fx_positions[position_id] = position
            await self._cache_fx_position(position)

            logger.info(f"Created FX position {position_id}")
            return position

        except Exception as e:
            logger.error(f"Error creating FX position: {e}")
            raise

    async def create_forward_contract(
        self,
        user_id: int,
        pair_id: str,
        quantity: float,
        forward_rate: float,
        spot_rate: float,
        value_date: datetime,
        maturity_date: datetime,
        contract_type: str,
        is_deliverable: bool = True,
    ) -> ForwardContract:
        """Create a new FX forward contract"""
        try:
            if pair_id not in self.currency_pairs:
                raise ValueError(f"Currency pair {pair_id} not found")

            if contract_type not in ["buy", "sell"]:
                raise ValueError("Contract type must be 'buy' or 'sell'")

            # Calculate forward points
            forward_points = forward_rate - spot_rate

            contract_id = f"fx_fwd_{user_id}_{pair_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            contract = ForwardContract(
                contract_id=contract_id,
                pair_id=pair_id,
                user_id=user_id,
                quantity=quantity,
                forward_rate=forward_rate,
                spot_rate=spot_rate,
                forward_points=forward_points,
                value_date=value_date,
                maturity_date=maturity_date,
                contract_type=contract_type,
                is_deliverable=is_deliverable,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.forward_contracts[contract_id] = contract
            await self._cache_forward_contract(contract)

            logger.info(f"Created forward contract {contract_id}")
            return contract

        except Exception as e:
            logger.error(f"Error creating forward contract: {e}")
            raise

    async def create_swap_contract(
        self,
        user_id: int,
        pair_id: str,
        near_leg: Dict[str, Any],
        far_leg: Dict[str, Any],
        swap_rate: float,
        value_date: datetime,
        maturity_date: datetime,
    ) -> SwapContract:
        """Create a new FX swap contract"""
        try:
            if pair_id not in self.currency_pairs:
                raise ValueError(f"Currency pair {pair_id} not found")

            # Calculate swap points
            swap_points = swap_rate - near_leg.get("rate", 0)

            swap_id = f"fx_swap_{user_id}_{pair_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            swap = SwapContract(
                swap_id=swap_id,
                pair_id=pair_id,
                user_id=user_id,
                near_leg=near_leg,
                far_leg=far_leg,
                swap_rate=swap_rate,
                swap_points=swap_points,
                value_date=value_date,
                maturity_date=maturity_date,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.swap_contracts[swap_id] = swap
            await self._cache_swap_contract(swap)

            logger.info(f"Created swap contract {swap_id}")
            return swap

        except Exception as e:
            logger.error(f"Error creating swap contract: {e}")
            raise

    async def place_fx_order(
        self,
        user_id: int,
        pair_id: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> FXOrder:
        """Place a new FX order"""
        try:
            if pair_id not in self.currency_pairs:
                raise ValueError(f"Currency pair {pair_id} not found")

            if order_type not in ["market", "limit", "stop", "stop_limit"]:
                raise ValueError("Invalid order type")

            if side not in ["buy", "sell"]:
                raise ValueError("Side must be 'buy' or 'sell'")

            if time_in_force not in ["GTC", "IOC", "FOK"]:
                raise ValueError("Invalid time in force")

            order_id = f"fx_order_{user_id}_{pair_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            order = FXOrder(
                order_id=order_id,
                user_id=user_id,
                pair_id=pair_id,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                limit_price=limit_price,
                time_in_force=time_in_force,
                status="pending",
                filled_quantity=0.0,
                filled_price=0.0,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.fx_orders[order_id] = order
            await self._cache_fx_order(order)

            logger.info(f"Placed FX order {order_id}")
            return order

        except Exception as e:
            logger.error(f"Error placing FX order: {e}")
            raise

    async def calculate_fx_metrics(self, pair_id: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a currency pair"""
        try:
            if pair_id not in self.currency_pairs:
                raise ValueError(f"Currency pair {pair_id} not found")

            currency_pair = self.currency_pairs[pair_id]
            prices = self.fx_prices.get(pair_id, [])

            if not prices:
                return {"pair_id": pair_id, "error": "No price data available"}

            # Get current price data
            current_price = prices[-1]

            # Calculate volatility (standard deviation of returns)
            if len(prices) > 1:
                returns = []
                for i in range(1, len(prices)):
                    return_val = (
                        prices[i].mid_price - prices[i - 1].mid_price
                    ) / prices[i - 1].mid_price
                    returns.append(return_val)

                volatility = (
                    np.std(returns) * np.sqrt(252) * 100
                )  # Annualized volatility
            else:
                volatility = 0.0

            # Calculate average spread
            spreads = [price.spread for price in prices[-100:]]  # Last 100 prices
            avg_spread = sum(spreads) / len(spreads) if spreads else 0

            # Calculate correlation with other pairs
            correlations = self.correlation_matrix.get(pair_id, {})

            metrics = {
                "pair_id": pair_id,
                "pair_name": currency_pair.pair_name,
                "base_currency": currency_pair.base_currency,
                "quote_currency": currency_pair.quote_currency,
                "current_price": {
                    "bid": current_price.bid_price,
                    "ask": current_price.ask_price,
                    "mid": current_price.mid_price,
                    "spread": current_price.spread,
                },
                "market_data": {
                    "volume_24h": current_price.volume_24h,
                    "high_24h": current_price.high_24h,
                    "low_24h": current_price.low_24h,
                    "change_24h": current_price.change_24h,
                    "change_pct_24h": current_price.change_pct_24h,
                },
                "trading_metrics": {
                    "pip_value": currency_pair.pip_value,
                    "lot_size": currency_pair.lot_size,
                    "min_trade_size": currency_pair.min_trade_size,
                    "max_trade_size": currency_pair.max_trade_size,
                    "margin_requirement": currency_pair.margin_requirement,
                },
                "risk_metrics": {
                    "volatility": volatility,
                    "avg_spread": avg_spread,
                    "spread_pct": (avg_spread / current_price.mid_price) * 100,
                },
                "correlations": correlations,
                "swap_rates": {
                    "long": currency_pair.swap_long,
                    "short": currency_pair.swap_short,
                },
                "trading_hours": currency_pair.trading_hours,
                "last_updated": datetime.utcnow().isoformat(),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating FX metrics: {e}")
            raise

    async def get_cross_currency_rates(self, base_currency: str) -> Dict[str, Any]:
        """Get cross currency rates for a base currency"""
        try:
            cross_rates = {}

            for pair_id, currency_pair in self.currency_pairs.items():
                if currency_pair.base_currency == base_currency:
                    # Get current price
                    prices = self.fx_prices.get(pair_id, [])
                    if prices:
                        current_price = prices[-1]
                        cross_rates[currency_pair.quote_currency] = {
                            "rate": current_price.mid_price,
                            "bid": current_price.bid_price,
                            "ask": current_price.ask_price,
                            "spread": current_price.spread,
                            "timestamp": current_price.timestamp.isoformat(),
                        }

            return {
                "base_currency": base_currency,
                "cross_rates": cross_rates,
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting cross currency rates: {e}")
            raise

    async def calculate_forward_points(
        self,
        pair_id: str,
        spot_rate: float,
        interest_rate_base: float,
        interest_rate_quote: float,
        days_to_maturity: int,
    ) -> Dict[str, Any]:
        """Calculate forward points using interest rate parity"""
        try:
            if pair_id not in self.currency_pairs:
                raise ValueError(f"Currency pair {pair_id} not found")

            # Convert days to years
            years_to_maturity = days_to_maturity / 365.0

            # Calculate forward rate using interest rate parity
            # F = S * (1 + r_quote)^t / (1 + r_base)^t
            forward_rate = spot_rate * (
                (1 + interest_rate_quote / 100) ** years_to_maturity
                / (1 + interest_rate_base / 100) ** years_to_maturity
            )

            # Calculate forward points
            forward_points = forward_rate - spot_rate

            # Calculate annualized forward points
            annualized_points = (
                (forward_points / spot_rate) * (365 / days_to_maturity) * 100
            )

            return {
                "pair_id": pair_id,
                "spot_rate": spot_rate,
                "forward_rate": forward_rate,
                "forward_points": forward_points,
                "annualized_points": annualized_points,
                "interest_rate_base": interest_rate_base,
                "interest_rate_quote": interest_rate_quote,
                "days_to_maturity": days_to_maturity,
                "years_to_maturity": years_to_maturity,
                "calculation_method": "interest_rate_parity",
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating forward points: {e}")
            raise

    # Background tasks
    async def _update_fx_prices(self):
        """Update FX prices periodically"""
        while True:
            try:
                # Update prices for all currency pairs
                for pair in self.currency_pairs.values():
                    if pair.is_active:
                        # Simulate price updates (in practice, this would fetch from data providers)
                        current_price = await self._get_current_fx_price(pair.pair_id)
                        if current_price:
                            # Add small random price movement
                            movement = np.random.normal(0, 0.1)  # 0.1% volatility
                            new_bid = current_price.bid_price * (1 + movement / 100)
                            new_ask = current_price.ask_price * (1 + movement / 100)

                            # Ensure ask > bid
                            if new_ask <= new_bid:
                                new_ask = new_bid * 1.0001

                            # Simulate volume and high/low
                            volume_change = np.random.normal(0, 0.05)
                            new_volume = current_price.volume_24h * (1 + volume_change)

                            new_high = max(current_price.high_24h, new_ask)
                            new_low = min(current_price.low_24h, new_bid)

                            change_24h = new_bid - current_price.bid_price

                            await self.add_fx_price(
                                pair.pair_id,
                                new_bid,
                                new_ask,
                                new_volume,
                                new_high,
                                new_low,
                                change_24h,
                                "simulated",
                            )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating FX prices: {e}")
                await asyncio.sleep(120)

    async def _update_positions(self):
        """Update FX positions"""
        while True:
            try:
                for position in self.fx_positions.values():
                    # Get current price
                    current_price = await self._get_current_fx_price(position.pair_id)
                    if current_price:
                        position.current_price = current_price.mid_price

                        # Calculate unrealized P&L
                        if position.position_type == "long":
                            position.unrealized_pnl = (
                                position.current_price - position.entry_price
                            ) * position.quantity
                        else:  # short
                            position.unrealized_pnl = (
                                position.entry_price - position.current_price
                            ) * position.quantity

                        # Calculate swap charges
                        currency_pair = self.currency_pairs.get(position.pair_id)
                        if currency_pair:
                            swap_rate = (
                                currency_pair.swap_long
                                if position.position_type == "long"
                                else currency_pair.swap_short
                            )
                            position.swap_charges += (
                                swap_rate * position.quantity / 365
                            )  # Daily swap

                        position.last_updated = datetime.utcnow()
                        await self._cache_fx_position(position)

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating positions: {e}")
                await asyncio.sleep(120)

    async def _update_forward_contracts(self):
        """Update forward contracts"""
        while True:
            try:
                # Update forward contracts (in practice, this would update rates)
                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating forward contracts: {e}")
                await asyncio.sleep(7200)

    async def _update_swap_contracts(self):
        """Update swap contracts"""
        while True:
            try:
                # Update swap contracts (in practice, this would update rates)
                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating swap contracts: {e}")
                await asyncio.sleep(7200)

    async def _monitor_trading_sessions(self):
        """Monitor trading sessions"""
        while True:
            try:
                # Monitor trading sessions (in practice, this would check market hours)
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring trading sessions: {e}")
                await asyncio.sleep(600)

    async def _get_current_fx_price(self, pair_id: str) -> Optional[FXPrice]:
        """Get current FX price"""
        try:
            prices = self.fx_prices.get(pair_id, [])
            if prices:
                return prices[-1]
            return None

        except Exception as e:
            logger.error(f"Error getting current FX price: {e}")
            return None

    # Helper methods (implementations would depend on your data models)
    async def _load_currency_pairs(self):
        """Load currency pairs from database"""
        pass

    async def _load_interest_rates(self):
        """Load interest rates from database"""
        pass

    async def _load_correlation_matrix(self):
        """Load correlation matrix from database"""
        pass

    # Caching methods
    async def _cache_currency_pair(self, pair: CurrencyPair):
        """Cache currency pair"""
        try:
            cache_key = f"currency_pair:{pair.pair_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "base_currency": pair.base_currency,
                        "quote_currency": pair.quote_currency,
                        "pair_name": pair.pair_name,
                        "pip_value": pair.pip_value,
                        "lot_size": pair.lot_size,
                        "min_trade_size": pair.min_trade_size,
                        "max_trade_size": pair.max_trade_size,
                        "margin_requirement": pair.margin_requirement,
                        "swap_long": pair.swap_long,
                        "swap_short": pair.swap_short,
                        "is_active": pair.is_active,
                        "trading_hours": pair.trading_hours,
                        "created_at": pair.created_at.isoformat(),
                        "last_updated": pair.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching currency pair: {e}")

    async def _cache_fx_price(self, price: FXPrice):
        """Cache FX price"""
        try:
            cache_key = f"fx_price:{price.price_id}"
            await self.redis.setex(
                cache_key,
                300,  # 5 minutes TTL
                json.dumps(
                    {
                        "pair_id": price.pair_id,
                        "bid_price": price.bid_price,
                        "ask_price": price.ask_price,
                        "mid_price": price.mid_price,
                        "spread": price.spread,
                        "pip_value": price.pip_value,
                        "timestamp": price.timestamp.isoformat(),
                        "source": price.source,
                        "volume_24h": price.volume_24h,
                        "high_24h": price.high_24h,
                        "low_24h": price.low_24h,
                        "change_24h": price.change_24h,
                        "change_pct_24h": price.change_pct_24h,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching FX price: {e}")

    async def _cache_fx_position(self, position: FXPosition):
        """Cache FX position"""
        try:
            cache_key = f"fx_position:{position.position_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "user_id": position.user_id,
                        "pair_id": position.pair_id,
                        "position_type": position.position_type,
                        "quantity": position.quantity,
                        "entry_price": position.entry_price,
                        "current_price": position.current_price,
                        "pip_value": position.pip_value,
                        "unrealized_pnl": position.unrealized_pnl,
                        "realized_pnl": position.realized_pnl,
                        "swap_charges": position.swap_charges,
                        "margin_used": position.margin_used,
                        "leverage": position.leverage,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "created_at": position.created_at.isoformat(),
                        "last_updated": position.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching FX position: {e}")

    async def _cache_forward_contract(self, contract: ForwardContract):
        """Cache forward contract"""
        try:
            cache_key = f"forward_contract:{contract.contract_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "pair_id": contract.pair_id,
                        "user_id": contract.user_id,
                        "quantity": contract.quantity,
                        "forward_rate": contract.forward_rate,
                        "spot_rate": contract.spot_rate,
                        "forward_points": contract.forward_points,
                        "value_date": contract.value_date.isoformat(),
                        "maturity_date": contract.maturity_date.isoformat(),
                        "contract_type": contract.contract_type,
                        "is_deliverable": contract.is_deliverable,
                        "created_at": contract.created_at.isoformat(),
                        "last_updated": contract.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching forward contract: {e}")

    async def _cache_swap_contract(self, swap: SwapContract):
        """Cache swap contract"""
        try:
            cache_key = f"swap_contract:{swap.swap_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "pair_id": swap.pair_id,
                        "user_id": swap.user_id,
                        "near_leg": swap.near_leg,
                        "far_leg": swap.far_leg,
                        "swap_rate": swap.swap_rate,
                        "swap_points": swap.swap_points,
                        "value_date": swap.value_date.isoformat(),
                        "maturity_date": swap.maturity_date.isoformat(),
                        "created_at": swap.created_at.isoformat(),
                        "last_updated": swap.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching swap contract: {e}")

    async def _cache_fx_order(self, order: FXOrder):
        """Cache FX order"""
        try:
            cache_key = f"fx_order:{order.order_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "user_id": order.user_id,
                        "pair_id": order.pair_id,
                        "order_type": order.order_type,
                        "side": order.side,
                        "quantity": order.quantity,
                        "price": order.price,
                        "stop_price": order.stop_price,
                        "limit_price": order.limit_price,
                        "time_in_force": order.time_in_force,
                        "status": order.status,
                        "filled_quantity": order.filled_quantity,
                        "filled_price": order.filled_price,
                        "created_at": order.created_at.isoformat(),
                        "last_updated": order.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching FX order: {e}")


# Factory function
async def get_forex_trading_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> ForexTradingService:
    """Get forex trading service instance"""
    service = ForexTradingService(redis_client, db_session)
    await service.initialize()
    return service
