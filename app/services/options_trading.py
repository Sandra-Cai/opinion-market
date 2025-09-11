"""
Options Trading Service
Provides options pricing, Greeks calculation, volatility surface modeling, and advanced options strategies
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

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Option contract information"""

    contract_id: str
    underlying_asset: str
    option_type: str  # 'call' or 'put'
    strike_price: float
    expiration_date: datetime
    current_price: float
    underlying_price: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float
    time_to_expiry: float
    created_at: datetime
    last_updated: datetime


@dataclass
class OptionGreeks:
    """Option Greeks"""

    contract_id: str
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    vanna: float
    volga: float
    charm: float
    calculated_at: datetime


@dataclass
class OptionPrice:
    """Option pricing information"""

    contract_id: str
    theoretical_price: float
    market_price: float
    bid_price: float
    ask_price: float
    implied_volatility: float
    intrinsic_value: float
    time_value: float
    calculated_at: datetime


@dataclass
class VolatilitySurface:
    """Volatility surface data"""

    underlying_asset: str
    expiration_dates: List[datetime]
    strike_prices: List[float]
    implied_volatilities: np.ndarray
    surface_type: str  # 'call', 'put', 'mid'
    created_at: datetime
    last_updated: datetime


@dataclass
class OptionsStrategy:
    """Options trading strategy"""

    strategy_id: str
    strategy_name: str
    strategy_type: (
        str  # 'spread', 'straddle', 'strangle', 'butterfly', 'iron_condor', etc.
    )
    legs: List[Dict[str, Any]]
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    risk_reward_ratio: float
    created_at: datetime
    last_updated: datetime


@dataclass
class OptionsPosition:
    """Options position"""

    position_id: str
    user_id: int
    contract_id: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    delta_exposure: float
    gamma_exposure: float
    theta_exposure: float
    vega_exposure: float
    position_type: str  # 'long', 'short'
    created_at: datetime
    last_updated: datetime


class OptionsTradingService:
    """Comprehensive options trading service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.option_contracts: Dict[str, OptionContract] = {}
        self.option_greeks: Dict[str, OptionGreeks] = {}
        self.option_prices: Dict[str, OptionPrice] = {}
        self.volatility_surfaces: Dict[str, VolatilitySurface] = {}
        self.options_strategies: Dict[str, OptionsStrategy] = {}
        self.options_positions: Dict[str, OptionsPosition] = {}

        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volatility_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Risk-free rates (could be fetched from Treasury yields)
        self.risk_free_rates = {
            "1M": 0.05,
            "3M": 0.055,
            "6M": 0.06,
            "1Y": 0.065,
            "2Y": 0.07,
            "5Y": 0.075,
            "10Y": 0.08,
        }

    async def initialize(self):
        """Initialize the options trading service"""
        logger.info("Initializing Options Trading Service")

        # Load existing data
        await self._load_option_contracts()
        await self._load_volatility_surfaces()
        await self._load_options_strategies()

        # Start background tasks
        asyncio.create_task(self._update_option_prices())
        asyncio.create_task(self._update_greeks())
        asyncio.create_task(self._update_volatility_surfaces())
        asyncio.create_task(self._monitor_positions())

        logger.info("Options Trading Service initialized successfully")

    async def create_option_contract(
        self,
        underlying_asset: str,
        option_type: str,
        strike_price: float,
        expiration_date: datetime,
        current_price: float,
        underlying_price: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> OptionContract:
        """Create a new option contract"""
        try:
            contract_id = f"{underlying_asset}_{option_type}_{strike_price}_{expiration_date.strftime('%Y%m%d')}"

            # Calculate time to expiry
            time_to_expiry = (expiration_date - datetime.utcnow()).days / 365.0

            # Get risk-free rate based on time to expiry
            risk_free_rate = self._get_risk_free_rate(time_to_expiry)

            contract = OptionContract(
                contract_id=contract_id,
                underlying_asset=underlying_asset,
                option_type=option_type,
                strike_price=strike_price,
                expiration_date=expiration_date,
                current_price=current_price,
                underlying_price=underlying_price,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                time_to_expiry=time_to_expiry,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.option_contracts[contract_id] = contract
            await self._cache_option_contract(contract)

            # Calculate initial pricing and Greeks
            await self._calculate_option_pricing(contract)
            await self._calculate_greeks(contract)

            logger.info(f"Created option contract {contract_id}")
            return contract

        except Exception as e:
            logger.error(f"Error creating option contract: {e}")
            raise

    async def calculate_black_scholes_price(self, contract: OptionContract) -> float:
        """Calculate Black-Scholes option price"""
        try:
            S = contract.underlying_price
            K = contract.strike_price
            T = contract.time_to_expiry
            r = contract.risk_free_rate
            sigma = contract.volatility
            q = contract.dividend_yield

            if T <= 0:
                return (
                    max(0, S - K) if contract.option_type == "call" else max(0, K - S)
                )

            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if contract.option_type == "call":
                price = S * np.exp(-q * T) * self._normal_cdf(d1) - K * np.exp(
                    -r * T
                ) * self._normal_cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * np.exp(
                    -q * T
                ) * self._normal_cdf(-d1)

            return max(0, price)

        except Exception as e:
            logger.error(f"Error calculating Black-Scholes price: {e}")
            return 0.0

    async def calculate_greeks(self, contract: OptionContract) -> OptionGreeks:
        """Calculate option Greeks"""
        try:
            S = contract.underlying_price
            K = contract.strike_price
            T = contract.time_to_expiry
            r = contract.risk_free_rate
            sigma = contract.volatility
            q = contract.dividend_yield

            if T <= 0:
                # At expiration, Greeks are simplified
                if contract.option_type == "call":
                    delta = 1.0 if S > K else 0.0
                    gamma = 0.0
                    theta = 0.0
                    vega = 0.0
                    rho = 0.0
                else:  # put
                    delta = -1.0 if S < K else 0.0
                    gamma = 0.0
                    theta = 0.0
                    vega = 0.0
                    rho = 0.0
            else:
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (
                    sigma * np.sqrt(T)
                )
                d2 = d1 - sigma * np.sqrt(T)

                # Calculate Greeks
                if contract.option_type == "call":
                    delta = np.exp(-q * T) * self._normal_cdf(d1)
                    gamma = (
                        np.exp(-q * T) * self._normal_pdf(d1) / (S * sigma * np.sqrt(T))
                    )
                    theta = (
                        -S
                        * np.exp(-q * T)
                        * self._normal_pdf(d1)
                        * sigma
                        / (2 * np.sqrt(T))
                        - r * K * np.exp(-r * T) * self._normal_cdf(d2)
                        + q * S * np.exp(-q * T) * self._normal_cdf(d1)
                    )
                    vega = S * np.exp(-q * T) * self._normal_pdf(d1) * np.sqrt(T)
                    rho = K * T * np.exp(-r * T) * self._normal_cdf(d2)
                else:  # put
                    delta = np.exp(-q * T) * (self._normal_cdf(d1) - 1)
                    gamma = (
                        np.exp(-q * T) * self._normal_pdf(d1) / (S * sigma * np.sqrt(T))
                    )
                    theta = (
                        -S
                        * np.exp(-q * T)
                        * self._normal_pdf(d1)
                        * sigma
                        / (2 * np.sqrt(T))
                        + r * K * np.exp(-r * T) * self._normal_cdf(-d2)
                        - q * S * np.exp(-q * T) * self._normal_cdf(-d1)
                    )
                    vega = S * np.exp(-q * T) * self._normal_pdf(d1) * np.sqrt(T)
                    rho = -K * T * np.exp(-r * T) * self._normal_cdf(-d2)

                # Second-order Greeks
                vanna = -np.exp(-q * T) * self._normal_pdf(d1) * d2 / sigma
                volga = (
                    S
                    * np.exp(-q * T)
                    * self._normal_pdf(d1)
                    * np.sqrt(T)
                    * d1
                    * d2
                    / sigma
                )
                charm = q * np.exp(-q * T) * self._normal_cdf(d1) - np.exp(
                    -q * T
                ) * self._normal_pdf(d1) * (
                    2 * (r - q) * T - d2 * sigma * np.sqrt(T)
                ) / (
                    2 * T * sigma * np.sqrt(T)
                )

            greeks = OptionGreeks(
                contract_id=contract.contract_id,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                vanna=vanna,
                volga=volga,
                charm=charm,
                calculated_at=datetime.utcnow(),
            )

            self.option_greeks[contract.contract_id] = greeks
            await self._cache_greeks(greeks)

            return greeks

        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            raise

    async def calculate_implied_volatility(
        self, contract: OptionContract, market_price: float
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:

            def black_scholes_price(vol):
                S = contract.underlying_price
                K = contract.strike_price
                T = contract.time_to_expiry
                r = contract.risk_free_rate
                q = contract.dividend_yield

                if T <= 0:
                    return (
                        max(0, S - K)
                        if contract.option_type == "call"
                        else max(0, K - S)
                    )

                d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                d2 = d1 - vol * np.sqrt(T)

                if contract.option_type == "call":
                    price = S * np.exp(-q * T) * self._normal_cdf(d1) - K * np.exp(
                        -r * T
                    ) * self._normal_cdf(d2)
                else:  # put
                    price = K * np.exp(-r * T) * self._normal_cdf(-d2) - S * np.exp(
                        -q * T
                    ) * self._normal_cdf(-d1)

                return max(0, price)

            def vega(vol):
                S = contract.underlying_price
                K = contract.strike_price
                T = contract.time_to_expiry
                r = contract.risk_free_rate
                q = contract.dividend_yield

                if T <= 0:
                    return 0.0

                d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                return S * np.exp(-q * T) * self._normal_pdf(d1) * np.sqrt(T)

            # Newton-Raphson iteration
            sigma = 0.3  # Initial guess
            tolerance = 1e-6
            max_iterations = 100

            for i in range(max_iterations):
                price = black_scholes_price(sigma)
                vega_val = vega(sigma)

                if abs(price - market_price) < tolerance:
                    break

                sigma = sigma - (price - market_price) / vega_val
                sigma = max(0.001, sigma)  # Ensure positive volatility

            return sigma

        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return contract.volatility

    async def create_volatility_surface(
        self,
        underlying_asset: str,
        expiration_dates: List[datetime],
        strike_prices: List[float],
        implied_volatilities: List[List[float]],
    ) -> VolatilitySurface:
        """Create volatility surface"""
        try:
            # Convert to numpy array
            vol_array = np.array(implied_volatilities)

            surface = VolatilitySurface(
                underlying_asset=underlying_asset,
                expiration_dates=expiration_dates,
                strike_prices=strike_prices,
                implied_volatilities=vol_array,
                surface_type="mid",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.volatility_surfaces[underlying_asset] = surface
            await self._cache_volatility_surface(surface)

            logger.info(f"Created volatility surface for {underlying_asset}")
            return surface

        except Exception as e:
            logger.error(f"Error creating volatility surface: {e}")
            raise

    async def interpolate_volatility(
        self, underlying_asset: str, strike: float, expiration: datetime
    ) -> float:
        """Interpolate volatility from surface"""
        try:
            surface = self.volatility_surfaces.get(underlying_asset)
            if not surface:
                return 0.3  # Default volatility

            # Find nearest expiration and strike
            exp_idx = min(
                range(len(surface.expiration_dates)),
                key=lambda i: abs((surface.expiration_dates[i] - expiration).days),
            )
            strike_idx = min(
                range(len(surface.strike_prices)),
                key=lambda i: abs(surface.strike_prices[i] - strike),
            )

            return surface.implied_volatilities[exp_idx, strike_idx]

        except Exception as e:
            logger.error(f"Error interpolating volatility: {e}")
            return 0.3

    async def create_options_strategy(
        self, strategy_name: str, strategy_type: str, legs: List[Dict[str, Any]]
    ) -> OptionsStrategy:
        """Create an options trading strategy"""
        try:
            strategy_id = f"strategy_{strategy_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Calculate strategy metrics
            max_profit, max_loss, breakeven_points = (
                await self._calculate_strategy_metrics(legs)
            )
            probability_of_profit = await self._calculate_probability_of_profit(legs)
            risk_reward_ratio = (
                abs(max_profit / max_loss) if max_loss != 0 else float("inf")
            )

            strategy = OptionsStrategy(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                legs=legs,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=breakeven_points,
                probability_of_profit=probability_of_profit,
                risk_reward_ratio=risk_reward_ratio,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.options_strategies[strategy_id] = strategy
            await self._cache_options_strategy(strategy)

            logger.info(f"Created options strategy {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Error creating options strategy: {e}")
            raise

    async def create_options_position(
        self,
        user_id: int,
        contract_id: str,
        quantity: int,
        entry_price: float,
        position_type: str,
    ) -> OptionsPosition:
        """Create an options position"""
        try:
            position_id = f"pos_{user_id}_{contract_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            contract = self.option_contracts.get(contract_id)
            if not contract:
                raise ValueError(f"Option contract {contract_id} not found")

            current_price = await self.calculate_black_scholes_price(contract)
            unrealized_pnl = (
                (current_price - entry_price) * quantity
                if position_type == "long"
                else (entry_price - current_price) * quantity
            )

            # Calculate position Greeks
            greeks = await self.calculate_greeks(contract)
            multiplier = quantity if position_type == "long" else -quantity

            position = OptionsPosition(
                position_id=position_id,
                user_id=user_id,
                contract_id=contract_id,
                quantity=quantity,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0.0,
                delta_exposure=greeks.delta * multiplier,
                gamma_exposure=greeks.gamma * multiplier,
                theta_exposure=greeks.theta * multiplier,
                vega_exposure=greeks.vega * multiplier,
                position_type=position_type,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.options_positions[position_id] = position
            await self._cache_options_position(position)

            logger.info(f"Created options position {position_id}")
            return position

        except Exception as e:
            logger.error(f"Error creating options position: {e}")
            raise

    async def get_options_chain(
        self, underlying_asset: str, expiration_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get options chain for an underlying asset"""
        try:
            contracts = []
            for contract in self.option_contracts.values():
                if contract.underlying_asset == underlying_asset:
                    if (
                        expiration_date is None
                        or contract.expiration_date == expiration_date
                    ):
                        # Get current pricing and Greeks
                        price = await self.calculate_black_scholes_price(contract)
                        greeks = await self.calculate_greeks(contract)

                        contracts.append(
                            {
                                "contract_id": contract.contract_id,
                                "option_type": contract.option_type,
                                "strike_price": contract.strike_price,
                                "expiration_date": contract.expiration_date.isoformat(),
                                "current_price": price,
                                "underlying_price": contract.underlying_price,
                                "implied_volatility": contract.volatility,
                                "delta": greeks.delta,
                                "gamma": greeks.gamma,
                                "theta": greeks.theta,
                                "vega": greeks.vega,
                                "rho": greeks.rho,
                                "time_to_expiry": contract.time_to_expiry,
                            }
                        )

            # Sort by strike price
            contracts.sort(key=lambda x: x["strike_price"])

            return {
                "underlying_asset": underlying_asset,
                "expiration_date": (
                    expiration_date.isoformat() if expiration_date else None
                ),
                "contracts": contracts,
                "total_contracts": len(contracts),
            }

        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            raise

    async def calculate_portfolio_greeks(self, user_id: int) -> Dict[str, float]:
        """Calculate portfolio Greeks for a user"""
        try:
            user_positions = [
                pos for pos in self.options_positions.values() if pos.user_id == user_id
            ]

            total_delta = sum(pos.delta_exposure for pos in user_positions)
            total_gamma = sum(pos.gamma_exposure for pos in user_positions)
            total_theta = sum(pos.theta_exposure for pos in user_positions)
            total_vega = sum(pos.vega_exposure for pos in user_positions)

            return {
                "total_delta": total_delta,
                "total_gamma": total_gamma,
                "total_theta": total_theta,
                "total_vega": total_vega,
                "position_count": len(user_positions),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            raise

    def _normal_cdf(self, x: float) -> float:
        """Normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _normal_pdf(self, x: float) -> float:
        """Normal probability density function"""
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

    def _get_risk_free_rate(self, time_to_expiry: float) -> float:
        """Get risk-free rate based on time to expiry"""
        if time_to_expiry <= 1 / 12:  # 1 month
            return self.risk_free_rates["1M"]
        elif time_to_expiry <= 3 / 12:  # 3 months
            return self.risk_free_rates["3M"]
        elif time_to_expiry <= 6 / 12:  # 6 months
            return self.risk_free_rates["6M"]
        elif time_to_expiry <= 1:  # 1 year
            return self.risk_free_rates["1Y"]
        elif time_to_expiry <= 2:  # 2 years
            return self.risk_free_rates["2Y"]
        elif time_to_expiry <= 5:  # 5 years
            return self.risk_free_rates["5Y"]
        else:  # 10+ years
            return self.risk_free_rates["10Y"]

    async def _calculate_option_pricing(self, contract: OptionContract):
        """Calculate option pricing"""
        try:
            theoretical_price = await self.calculate_black_scholes_price(contract)
            intrinsic_value = (
                max(0, contract.underlying_price - contract.strike_price)
                if contract.option_type == "call"
                else max(0, contract.strike_price - contract.underlying_price)
            )
            time_value = theoretical_price - intrinsic_value

            price = OptionPrice(
                contract_id=contract.contract_id,
                theoretical_price=theoretical_price,
                market_price=contract.current_price,
                bid_price=theoretical_price * 0.99,  # Approximate bid
                ask_price=theoretical_price * 1.01,  # Approximate ask
                implied_volatility=contract.volatility,
                intrinsic_value=intrinsic_value,
                time_value=time_value,
                calculated_at=datetime.utcnow(),
            )

            self.option_prices[contract.contract_id] = price
            await self._cache_option_price(price)

        except Exception as e:
            logger.error(f"Error calculating option pricing: {e}")

    async def _calculate_strategy_metrics(
        self, legs: List[Dict[str, Any]]
    ) -> Tuple[float, float, List[float]]:
        """Calculate strategy metrics"""
        try:
            # Simplified calculation - in practice, this would be more complex
            max_profit = sum(leg.get("max_profit", 0) for leg in legs)
            max_loss = sum(leg.get("max_loss", 0) for leg in legs)
            breakeven_points = [100.0]  # Simplified

            return max_profit, max_loss, breakeven_points

        except Exception as e:
            logger.error(f"Error calculating strategy metrics: {e}")
            return 0.0, 0.0, []

    async def _calculate_probability_of_profit(
        self, legs: List[Dict[str, Any]]
    ) -> float:
        """Calculate probability of profit for strategy"""
        try:
            # Simplified calculation
            return 0.5  # 50% probability

        except Exception as e:
            logger.error(f"Error calculating probability of profit: {e}")
            return 0.5

    async def _update_option_prices(self):
        """Update option prices periodically"""
        while True:
            try:
                for contract in self.option_contracts.values():
                    await self._calculate_option_pricing(contract)

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating option prices: {e}")
                await asyncio.sleep(120)

    async def _update_greeks(self):
        """Update Greeks periodically"""
        while True:
            try:
                for contract in self.option_contracts.values():
                    await self.calculate_greeks(contract)

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating Greeks: {e}")
                await asyncio.sleep(120)

    async def _update_volatility_surfaces(self):
        """Update volatility surfaces periodically"""
        while True:
            try:
                # Update volatility surfaces
                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error updating volatility surfaces: {e}")
                await asyncio.sleep(600)

    async def _monitor_positions(self):
        """Monitor options positions"""
        while True:
            try:
                for position in self.options_positions.values():
                    contract = self.option_contracts.get(position.contract_id)
                    if contract:
                        current_price = await self.calculate_black_scholes_price(
                            contract
                        )
                        position.current_price = current_price
                        position.unrealized_pnl = (
                            (current_price - position.entry_price) * position.quantity
                            if position.position_type == "long"
                            else (position.entry_price - current_price)
                            * position.quantity
                        )
                        position.last_updated = datetime.utcnow()

                        await self._cache_options_position(position)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(60)

    # Helper methods (implementations would depend on your data models)
    async def _load_option_contracts(self):
        """Load option contracts from database"""
        pass

    async def _load_volatility_surfaces(self):
        """Load volatility surfaces from database"""
        pass

    async def _load_options_strategies(self):
        """Load options strategies from database"""
        pass

    # Caching methods
    async def _cache_option_contract(self, contract: OptionContract):
        """Cache option contract"""
        try:
            cache_key = f"option_contract:{contract.contract_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "underlying_asset": contract.underlying_asset,
                        "option_type": contract.option_type,
                        "strike_price": contract.strike_price,
                        "expiration_date": contract.expiration_date.isoformat(),
                        "current_price": contract.current_price,
                        "underlying_price": contract.underlying_price,
                        "risk_free_rate": contract.risk_free_rate,
                        "volatility": contract.volatility,
                        "dividend_yield": contract.dividend_yield,
                        "time_to_expiry": contract.time_to_expiry,
                        "created_at": contract.created_at.isoformat(),
                        "last_updated": contract.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching option contract: {e}")

    async def _cache_greeks(self, greeks: OptionGreeks):
        """Cache Greeks"""
        try:
            cache_key = f"option_greeks:{greeks.contract_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "delta": greeks.delta,
                        "gamma": greeks.gamma,
                        "theta": greeks.theta,
                        "vega": greeks.vega,
                        "rho": greeks.rho,
                        "vanna": greeks.vanna,
                        "volga": greeks.volga,
                        "charm": greeks.charm,
                        "calculated_at": greeks.calculated_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching Greeks: {e}")

    async def _cache_option_price(self, price: OptionPrice):
        """Cache option price"""
        try:
            cache_key = f"option_price:{price.contract_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "theoretical_price": price.theoretical_price,
                        "market_price": price.market_price,
                        "bid_price": price.bid_price,
                        "ask_price": price.ask_price,
                        "implied_volatility": price.implied_volatility,
                        "intrinsic_value": price.intrinsic_value,
                        "time_value": price.time_value,
                        "calculated_at": price.calculated_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching option price: {e}")

    async def _cache_volatility_surface(self, surface: VolatilitySurface):
        """Cache volatility surface"""
        try:
            cache_key = f"volatility_surface:{surface.underlying_asset}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "expiration_dates": [
                            d.isoformat() for d in surface.expiration_dates
                        ],
                        "strike_prices": surface.strike_prices,
                        "implied_volatilities": surface.implied_volatilities.tolist(),
                        "surface_type": surface.surface_type,
                        "created_at": surface.created_at.isoformat(),
                        "last_updated": surface.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching volatility surface: {e}")

    async def _cache_options_strategy(self, strategy: OptionsStrategy):
        """Cache options strategy"""
        try:
            cache_key = f"options_strategy:{strategy.strategy_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "strategy_name": strategy.strategy_name,
                        "strategy_type": strategy.strategy_type,
                        "legs": strategy.legs,
                        "max_profit": strategy.max_profit,
                        "max_loss": strategy.max_loss,
                        "breakeven_points": strategy.breakeven_points,
                        "probability_of_profit": strategy.probability_of_profit,
                        "risk_reward_ratio": strategy.risk_reward_ratio,
                        "created_at": strategy.created_at.isoformat(),
                        "last_updated": strategy.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching options strategy: {e}")

    async def _cache_options_position(self, position: OptionsPosition):
        """Cache options position"""
        try:
            cache_key = f"options_position:{position.position_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "user_id": position.user_id,
                        "contract_id": position.contract_id,
                        "quantity": position.quantity,
                        "entry_price": position.entry_price,
                        "current_price": position.current_price,
                        "unrealized_pnl": position.unrealized_pnl,
                        "realized_pnl": position.realized_pnl,
                        "delta_exposure": position.delta_exposure,
                        "gamma_exposure": position.gamma_exposure,
                        "theta_exposure": position.theta_exposure,
                        "vega_exposure": position.vega_exposure,
                        "position_type": position.position_type,
                        "created_at": position.created_at.isoformat(),
                        "last_updated": position.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching options position: {e}")


# Factory function
async def get_options_trading_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> OptionsTradingService:
    """Get options trading service instance"""
    service = OptionsTradingService(redis_client, db_session)
    await service.initialize()
    return service
