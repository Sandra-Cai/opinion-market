"""
Derivatives Trading Service
Advanced derivatives trading, pricing, and risk management
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
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DerivativeType(Enum):
    """Derivative types"""
    OPTION = "option"
    FUTURE = "future"
    FORWARD = "forward"
    SWAP = "swap"
    WARRANT = "warrant"
    CONVERTIBLE = "convertible"
    STRUCTURED_PRODUCT = "structured_product"


class OptionType(Enum):
    """Option types"""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Exercise styles"""
    AMERICAN = "american"
    EUROPEAN = "european"
    BERMUDAN = "bermudan"


class SwapType(Enum):
    """Swap types"""
    INTEREST_RATE = "interest_rate"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    EQUITY = "equity"
    CREDIT_DEFAULT = "credit_default"
    TOTAL_RETURN = "total_return"


@dataclass
class Derivative:
    """Derivative instrument"""
    derivative_id: str
    symbol: str
    derivative_type: DerivativeType
    underlying_asset: str
    strike_price: Optional[float]
    expiration_date: Optional[datetime]
    option_type: Optional[OptionType]
    exercise_style: Optional[ExerciseStyle]
    contract_size: float
    multiplier: float
    currency: str
    exchange: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class OptionGreeks:
    """Option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    vanna: float
    volga: float
    charm: float
    veta: float
    speed: float
    zomma: float
    color: float
    ultima: float
    dual_delta: float
    dual_gamma: float


@dataclass
class DerivativePrice:
    """Derivative price"""
    derivative_id: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    mid_price: float
    last_price: float
    volume: float
    open_interest: float
    implied_volatility: Optional[float]
    greeks: Optional[OptionGreeks]
    theoretical_price: float
    price_source: str


@dataclass
class DerivativePosition:
    """Derivative position"""
    position_id: str
    user_id: int
    derivative_id: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_required: float
    delta_exposure: float
    gamma_exposure: float
    theta_exposure: float
    vega_exposure: float
    created_at: datetime
    last_updated: datetime


@dataclass
class DerivativeOrder:
    """Derivative order"""
    order_id: str
    user_id: int
    derivative_id: str
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    side: str  # 'buy', 'sell'
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str  # 'GTC', 'IOC', 'FOK', 'DAY'
    status: str  # 'pending', 'filled', 'cancelled', 'rejected'
    filled_quantity: float
    average_fill_price: float
    commission: float
    created_at: datetime
    updated_at: datetime


@dataclass
class VolatilitySurface:
    """Volatility surface"""
    underlying_asset: str
    timestamp: datetime
    strikes: List[float]
    expirations: List[datetime]
    implied_volatilities: List[List[float]]
    risk_free_rate: float
    dividend_yield: float


class DerivativesTradingService:
    """Comprehensive Derivatives Trading Service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        
        # Derivatives management
        self.derivatives: Dict[str, Derivative] = {}
        self.derivative_prices: Dict[str, List[DerivativePrice]] = defaultdict(list)
        self.derivative_positions: Dict[str, List[DerivativePosition]] = defaultdict(list)
        self.derivative_orders: Dict[str, List[DerivativeOrder]] = defaultdict(list)
        self.volatility_surfaces: Dict[str, VolatilitySurface] = {}
        
        # Pricing models
        self.pricing_models: Dict[str, Any] = {}
        
        # Risk management
        self.risk_limits: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.portfolio_greeks: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Market data
        self.underlying_prices: Dict[str, float] = {}
        self.risk_free_rates: Dict[str, float] = {}
        self.dividend_yields: Dict[str, float] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the Derivatives Trading Service"""
        logger.info("Initializing Derivatives Trading Service")
        
        # Load derivatives
        await self._load_derivatives()
        
        # Initialize pricing models
        await self._initialize_pricing_models()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_derivative_prices()),
            asyncio.create_task(self._update_volatility_surfaces()),
            asyncio.create_task(self._calculate_portfolio_greeks()),
            asyncio.create_task(self._monitor_risk_limits()),
            asyncio.create_task(self._process_derivative_orders())
        ]
        
        logger.info("Derivatives Trading Service initialized successfully")
    
    async def create_derivative(self, symbol: str, derivative_type: DerivativeType,
                              underlying_asset: str, strike_price: Optional[float] = None,
                              expiration_date: Optional[datetime] = None,
                              option_type: Optional[OptionType] = None,
                              exercise_style: Optional[ExerciseStyle] = None,
                              contract_size: float = 1.0, multiplier: float = 1.0,
                              currency: str = 'USD', exchange: str = 'CBOE') -> Derivative:
        """Create a new derivative instrument"""
        try:
            derivative_id = f"DERIV_{uuid.uuid4().hex[:8]}"
            
            derivative = Derivative(
                derivative_id=derivative_id,
                symbol=symbol,
                derivative_type=derivative_type,
                underlying_asset=underlying_asset,
                strike_price=strike_price,
                expiration_date=expiration_date,
                option_type=option_type,
                exercise_style=exercise_style,
                contract_size=contract_size,
                multiplier=multiplier,
                currency=currency,
                exchange=exchange,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.derivatives[derivative_id] = derivative
            
            logger.info(f"Created derivative {derivative_id}")
            return derivative
            
        except Exception as e:
            logger.error(f"Error creating derivative: {e}")
            raise
    
    async def get_derivative_price(self, derivative_id: str) -> Optional[DerivativePrice]:
        """Get current derivative price"""
        try:
            prices = self.derivative_prices.get(derivative_id, [])
            return prices[-1] if prices else None
            
        except Exception as e:
            logger.error(f"Error getting derivative price: {e}")
            return None
    
    async def calculate_option_price(self, derivative_id: str, underlying_price: float,
                                   risk_free_rate: float, dividend_yield: float = 0.0,
                                   volatility: Optional[float] = None) -> Dict[str, Any]:
        """Calculate option price using Black-Scholes model"""
        try:
            derivative = self.derivatives.get(derivative_id)
            if not derivative or derivative.derivative_type != DerivativeType.OPTION:
                raise ValueError("Invalid option derivative")
            
            if not derivative.strike_price or not derivative.expiration_date:
                raise ValueError("Missing strike price or expiration date")
            
            # Calculate time to expiration
            time_to_expiry = (derivative.expiration_date - datetime.utcnow()).days / 365.0
            
            if time_to_expiry <= 0:
                return {'error': 'Option has expired'}
            
            # Use provided volatility or calculate implied volatility
            if volatility is None:
                volatility = await self._get_implied_volatility(derivative_id, underlying_price)
            
            # Calculate option price using Black-Scholes
            if derivative.exercise_style == ExerciseStyle.EUROPEAN:
                price, greeks = self._black_scholes_european(
                    underlying_price, derivative.strike_price, time_to_expiry,
                    risk_free_rate, dividend_yield, volatility, derivative.option_type
                )
            else:
                # American option - use binomial model approximation
                price, greeks = self._black_scholes_american_approximation(
                    underlying_price, derivative.strike_price, time_to_expiry,
                    risk_free_rate, dividend_yield, volatility, derivative.option_type
                )
            
            return {
                'derivative_id': derivative_id,
                'theoretical_price': price,
                'greeks': greeks,
                'underlying_price': underlying_price,
                'strike_price': derivative.strike_price,
                'time_to_expiry': time_to_expiry,
                'risk_free_rate': risk_free_rate,
                'dividend_yield': dividend_yield,
                'volatility': volatility,
                'option_type': derivative.option_type.value,
                'exercise_style': derivative.exercise_style.value
            }
            
        except Exception as e:
            logger.error(f"Error calculating option price: {e}")
            return {'error': str(e)}
    
    async def create_derivative_position(self, user_id: int, derivative_id: str,
                                       quantity: float, average_price: float) -> DerivativePosition:
        """Create a derivative position"""
        try:
            position_id = f"POS_{uuid.uuid4().hex[:8]}"
            
            # Get current price
            current_price = await self._get_current_price(derivative_id)
            
            position = DerivativePosition(
                position_id=position_id,
                user_id=user_id,
                derivative_id=derivative_id,
                quantity=quantity,
                average_price=average_price,
                current_price=current_price,
                unrealized_pnl=(current_price - average_price) * quantity,
                realized_pnl=0.0,
                margin_required=0.0,
                delta_exposure=0.0,
                gamma_exposure=0.0,
                theta_exposure=0.0,
                vega_exposure=0.0,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Calculate Greeks exposure
            await self._calculate_position_greeks(position)
            
            # Calculate margin requirement
            await self._calculate_margin_requirement(position)
            
            self.derivative_positions[user_id].append(position)
            
            logger.info(f"Created derivative position {position_id}")
            return position
            
        except Exception as e:
            logger.error(f"Error creating derivative position: {e}")
            raise
    
    async def place_derivative_order(self, user_id: int, derivative_id: str,
                                   order_type: str, side: str, quantity: float,
                                   price: Optional[float] = None,
                                   stop_price: Optional[float] = None,
                                   time_in_force: str = 'GTC') -> DerivativeOrder:
        """Place a derivative order"""
        try:
            order_id = f"ORDER_{uuid.uuid4().hex[:8]}"
            
            order = DerivativeOrder(
                order_id=order_id,
                user_id=user_id,
                derivative_id=derivative_id,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                status='pending',
                filled_quantity=0.0,
                average_fill_price=0.0,
                commission=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Validate order
            await self._validate_derivative_order(order)
            
            self.derivative_orders[user_id].append(order)
            
            logger.info(f"Placed derivative order {order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Error placing derivative order: {e}")
            raise
    
    async def get_volatility_surface(self, underlying_asset: str) -> Optional[VolatilitySurface]:
        """Get volatility surface for underlying asset"""
        try:
            return self.volatility_surfaces.get(underlying_asset)
            
        except Exception as e:
            logger.error(f"Error getting volatility surface: {e}")
            return None
    
    async def calculate_portfolio_greeks(self, user_id: int) -> Dict[str, float]:
        """Calculate portfolio Greeks"""
        try:
            positions = self.derivative_positions.get(user_id, [])
            
            total_delta = 0.0
            total_gamma = 0.0
            total_theta = 0.0
            total_vega = 0.0
            
            for position in positions:
                derivative = self.derivatives.get(position.derivative_id)
                if not derivative:
                    continue
                
                if derivative.derivative_type == DerivativeType.OPTION:
                    # Get option Greeks
                    price_data = await self.get_derivative_price(position.derivative_id)
                    if price_data and price_data.greeks:
                        total_delta += position.delta_exposure
                        total_gamma += position.gamma_exposure
                        total_theta += position.theta_exposure
                        total_vega += position.vega_exposure
            
            portfolio_greeks = {
                'delta': total_delta,
                'gamma': total_gamma,
                'theta': total_theta,
                'vega': total_vega,
                'timestamp': datetime.utcnow()
            }
            
            self.portfolio_greeks[user_id] = portfolio_greeks
            
            return portfolio_greeks
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            return {}
    
    # Pricing models
    def _black_scholes_european(self, S: float, K: float, T: float, r: float,
                              q: float, sigma: float, option_type: OptionType) -> Tuple[float, OptionGreeks]:
        """Black-Scholes European option pricing"""
        try:
            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate option price
            if option_type == OptionType.CALL:
                price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            
            # Calculate Greeks
            greeks = self._calculate_black_scholes_greeks(S, K, T, r, q, sigma, d1, d2, option_type)
            
            return price, greeks
            
        except Exception as e:
            logger.error(f"Error in Black-Scholes pricing: {e}")
            return 0.0, OptionGreeks(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _black_scholes_american_approximation(self, S: float, K: float, T: float, r: float,
                                            q: float, sigma: float, option_type: OptionType) -> Tuple[float, OptionGreeks]:
        """American option approximation using Barone-Adesi and Whaley"""
        try:
            # For simplicity, use European price as approximation
            # In practice, would implement Barone-Adesi and Whaley or binomial model
            return self._black_scholes_european(S, K, T, r, q, sigma, option_type)
            
        except Exception as e:
            logger.error(f"Error in American option pricing: {e}")
            return 0.0, OptionGreeks(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float,
                                      q: float, sigma: float, d1: float, d2: float,
                                      option_type: OptionType) -> OptionGreeks:
        """Calculate Black-Scholes Greeks"""
        try:
            # Delta
            if option_type == OptionType.CALL:
                delta = np.exp(-q * T) * norm.cdf(d1)
            else:
                delta = -np.exp(-q * T) * norm.cdf(-d1)
            
            # Gamma
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            theta1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            theta2 = -r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == OptionType.CALL else r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta3 = q * S * np.exp(-q * T) * norm.cdf(d1) if option_type == OptionType.CALL else -q * S * np.exp(-q * T) * norm.cdf(-d1)
            theta = theta1 + theta2 + theta3
            
            # Vega
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
            
            # Rho
            if option_type == OptionType.CALL:
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
            # Higher-order Greeks (simplified)
            vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma
            volga = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) * d1 * d2 / sigma
            charm = -q * np.exp(-q * T) * norm.cdf(d1) + np.exp(-q * T) * norm.pdf(d1) * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
            
            return OptionGreeks(
                delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
                vanna=vanna, volga=volga, charm=charm,
                veta=0, speed=0, zomma=0, color=0, ultima=0, dual_delta=0, dual_gamma=0
            )
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return OptionGreeks(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    # Background tasks
    async def _update_derivative_prices(self):
        """Update derivative prices"""
        while True:
            try:
                for derivative_id, derivative in self.derivatives.items():
                    if derivative.is_active:
                        await self._update_derivative_price(derivative)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating derivative prices: {e}")
                await asyncio.sleep(300)
    
    async def _update_derivative_price(self, derivative: Derivative):
        """Update price for a derivative"""
        try:
            # Get underlying price
            underlying_price = self.underlying_prices.get(derivative.underlying_asset, 100.0)
            
            # Calculate theoretical price
            if derivative.derivative_type == DerivativeType.OPTION:
                price_result = await self.calculate_option_price(
                    derivative.derivative_id, underlying_price, 0.05, 0.02
                )
                theoretical_price = price_result.get('theoretical_price', 0.0)
                greeks = price_result.get('greeks')
            else:
                theoretical_price = underlying_price
                greeks = None
            
            # Add bid-ask spread
            spread = theoretical_price * 0.01  # 1% spread
            bid_price = theoretical_price - spread / 2
            ask_price = theoretical_price + spread / 2
            
            # Create price data
            price_data = DerivativePrice(
                derivative_id=derivative.derivative_id,
                timestamp=datetime.utcnow(),
                bid_price=bid_price,
                ask_price=ask_price,
                mid_price=theoretical_price,
                last_price=theoretical_price,
                volume=np.random.uniform(100, 1000),
                open_interest=np.random.uniform(1000, 10000),
                implied_volatility=0.2,
                greeks=greeks,
                theoretical_price=theoretical_price,
                price_source='theoretical'
            )
            
            self.derivative_prices[derivative.derivative_id].append(price_data)
            
            # Keep only recent prices
            if len(self.derivative_prices[derivative.derivative_id]) > 1000:
                self.derivative_prices[derivative.derivative_id] = self.derivative_prices[derivative.derivative_id][-1000:]
            
        except Exception as e:
            logger.error(f"Error updating derivative price: {e}")
    
    async def _update_volatility_surfaces(self):
        """Update volatility surfaces"""
        while True:
            try:
                # Get unique underlying assets
                underlying_assets = set()
                for derivative in self.derivatives.values():
                    if derivative.derivative_type == DerivativeType.OPTION:
                        underlying_assets.add(derivative.underlying_asset)
                
                for underlying in underlying_assets:
                    await self._update_volatility_surface(underlying)
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating volatility surfaces: {e}")
                await asyncio.sleep(7200)
    
    async def _update_volatility_surface(self, underlying_asset: str):
        """Update volatility surface for underlying asset"""
        try:
            # Generate sample volatility surface
            strikes = np.linspace(80, 120, 9)  # 80% to 120% of current price
            expirations = [datetime.utcnow() + timedelta(days=d) for d in [7, 14, 30, 60, 90, 180, 365]]
            
            # Generate implied volatilities (simplified)
            implied_vols = []
            for exp in expirations:
                time_to_exp = (exp - datetime.utcnow()).days / 365.0
                vol_row = []
                for strike in strikes:
                    # Simple volatility smile
                    moneyness = strike / 100.0  # Assuming current price is 100
                    vol = 0.2 + 0.1 * (moneyness - 1) ** 2 + 0.05 * time_to_exp
                    vol_row.append(vol)
                implied_vols.append(vol_row)
            
            volatility_surface = VolatilitySurface(
                underlying_asset=underlying_asset,
                timestamp=datetime.utcnow(),
                strikes=strikes.tolist(),
                expirations=expirations,
                implied_volatilities=implied_vols,
                risk_free_rate=0.05,
                dividend_yield=0.02
            )
            
            self.volatility_surfaces[underlying_asset] = volatility_surface
            
        except Exception as e:
            logger.error(f"Error updating volatility surface: {e}")
    
    async def _calculate_portfolio_greeks(self):
        """Calculate portfolio Greeks for all users"""
        while True:
            try:
                for user_id in self.derivative_positions.keys():
                    await self.calculate_portfolio_greeks(user_id)
                
                await asyncio.sleep(300)  # Calculate every 5 minutes
                
            except Exception as e:
                logger.error(f"Error calculating portfolio Greeks: {e}")
                await asyncio.sleep(600)
    
    async def _monitor_risk_limits(self):
        """Monitor risk limits"""
        while True:
            try:
                for user_id, positions in self.derivative_positions.items():
                    await self._check_user_risk_limits(user_id, positions)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring risk limits: {e}")
                await asyncio.sleep(300)
    
    async def _check_user_risk_limits(self, user_id: int, positions: List[DerivativePosition]):
        """Check risk limits for a user"""
        try:
            # Calculate total exposure
            total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions)
            
            # Check exposure limit
            exposure_limit = self.risk_limits[user_id].get('max_exposure', 1000000.0)
            if total_exposure > exposure_limit:
                logger.warning(f"User {user_id} exceeded exposure limit: {total_exposure} > {exposure_limit}")
            
            # Check Greeks limits
            portfolio_greeks = self.portfolio_greeks.get(user_id, {})
            delta_limit = self.risk_limits[user_id].get('max_delta', 10000.0)
            if abs(portfolio_greeks.get('delta', 0)) > delta_limit:
                logger.warning(f"User {user_id} exceeded delta limit: {portfolio_greeks.get('delta', 0)} > {delta_limit}")
            
        except Exception as e:
            logger.error(f"Error checking user risk limits: {e}")
    
    async def _process_derivative_orders(self):
        """Process derivative orders"""
        while True:
            try:
                for user_id, orders in self.derivative_orders.items():
                    for order in orders:
                        if order.status == 'pending':
                            await self._process_derivative_order(order)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing derivative orders: {e}")
                await asyncio.sleep(30)
    
    async def _process_derivative_order(self, order: DerivativeOrder):
        """Process a derivative order"""
        try:
            # Get current price
            current_price = await self._get_current_price(order.derivative_id)
            
            # Check if order can be filled
            can_fill = False
            fill_price = 0.0
            
            if order.order_type == 'market':
                can_fill = True
                fill_price = current_price
            elif order.order_type == 'limit':
                if order.side == 'buy' and current_price <= order.price:
                    can_fill = True
                    fill_price = order.price
                elif order.side == 'sell' and current_price >= order.price:
                    can_fill = True
                    fill_price = order.price
            
            if can_fill:
                # Fill order
                order.status = 'filled'
                order.filled_quantity = order.quantity
                order.average_fill_price = fill_price
                order.commission = order.quantity * fill_price * 0.001  # 0.1% commission
                order.updated_at = datetime.utcnow()
                
                # Create position
                await self.create_derivative_position(
                    order.user_id, order.derivative_id, order.quantity, fill_price
                )
                
                logger.info(f"Filled derivative order {order.order_id}")
            
        except Exception as e:
            logger.error(f"Error processing derivative order: {e}")
    
    # Helper methods
    async def _get_implied_volatility(self, derivative_id: str, underlying_price: float) -> float:
        """Get implied volatility for derivative"""
        try:
            # Simplified - return market volatility
            return 0.2
            
        except Exception as e:
            logger.error(f"Error getting implied volatility: {e}")
            return 0.2
    
    async def _get_current_price(self, derivative_id: str) -> float:
        """Get current price for derivative"""
        try:
            price_data = await self.get_derivative_price(derivative_id)
            return price_data.mid_price if price_data else 0.0
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    async def _calculate_position_greeks(self, position: DerivativePosition):
        """Calculate Greeks for a position"""
        try:
            derivative = self.derivatives.get(position.derivative_id)
            if not derivative or derivative.derivative_type != DerivativeType.OPTION:
                return
            
            price_data = await self.get_derivative_price(position.derivative_id)
            if not price_data or not price_data.greeks:
                return
            
            # Calculate position Greeks
            position.delta_exposure = position.quantity * price_data.greeks.delta
            position.gamma_exposure = position.quantity * price_data.greeks.gamma
            position.theta_exposure = position.quantity * price_data.greeks.theta
            position.vega_exposure = position.quantity * price_data.greeks.vega
            
        except Exception as e:
            logger.error(f"Error calculating position Greeks: {e}")
    
    async def _calculate_margin_requirement(self, position: DerivativePosition):
        """Calculate margin requirement for position"""
        try:
            derivative = self.derivatives.get(position.derivative_id)
            if not derivative:
                return
            
            if derivative.derivative_type == DerivativeType.OPTION:
                # Option margin calculation (simplified)
                position.margin_required = position.quantity * position.current_price * 0.1  # 10% margin
            else:
                # Future margin calculation (simplified)
                position.margin_required = position.quantity * position.current_price * 0.05  # 5% margin
            
        except Exception as e:
            logger.error(f"Error calculating margin requirement: {e}")
    
    async def _validate_derivative_order(self, order: DerivativeOrder):
        """Validate derivative order"""
        try:
            # Check if derivative exists
            derivative = self.derivatives.get(order.derivative_id)
            if not derivative or not derivative.is_active:
                raise ValueError("Invalid or inactive derivative")
            
            # Check order parameters
            if order.quantity <= 0:
                raise ValueError("Invalid quantity")
            
            if order.order_type == 'limit' and not order.price:
                raise ValueError("Limit order requires price")
            
            if order.order_type == 'stop_limit' and (not order.price or not order.stop_price):
                raise ValueError("Stop-limit order requires both price and stop price")
            
        except Exception as e:
            logger.error(f"Error validating derivative order: {e}")
            raise
    
    async def _load_derivatives(self):
        """Load sample derivatives"""
        try:
            # Create sample options
            sample_options = [
                {
                    'symbol': 'AAPL_C_150_20241220',
                    'derivative_type': DerivativeType.OPTION,
                    'underlying_asset': 'AAPL',
                    'strike_price': 150.0,
                    'expiration_date': datetime.utcnow() + timedelta(days=30),
                    'option_type': OptionType.CALL,
                    'exercise_style': ExerciseStyle.AMERICAN
                },
                {
                    'symbol': 'AAPL_P_150_20241220',
                    'derivative_type': DerivativeType.OPTION,
                    'underlying_asset': 'AAPL',
                    'strike_price': 150.0,
                    'expiration_date': datetime.utcnow() + timedelta(days=30),
                    'option_type': OptionType.PUT,
                    'exercise_style': ExerciseStyle.AMERICAN
                },
                {
                    'symbol': 'GOOGL_C_2800_20250117',
                    'derivative_type': DerivativeType.OPTION,
                    'underlying_asset': 'GOOGL',
                    'strike_price': 2800.0,
                    'expiration_date': datetime.utcnow() + timedelta(days=60),
                    'option_type': OptionType.CALL,
                    'exercise_style': ExerciseStyle.EUROPEAN
                }
            ]
            
            for option_data in sample_options:
                await self.create_derivative(**option_data)
            
            logger.info("Loaded sample derivatives")
            
        except Exception as e:
            logger.error(f"Error loading derivatives: {e}")
    
    async def _initialize_pricing_models(self):
        """Initialize pricing models"""
        try:
            # Initialize market data
            self.underlying_prices = {
                'AAPL': 150.0,
                'GOOGL': 2800.0,
                'MSFT': 400.0,
                'TSLA': 250.0
            }
            
            self.risk_free_rates = {
                'USD': 0.05,
                'EUR': 0.03,
                'GBP': 0.04
            }
            
            self.dividend_yields = {
                'AAPL': 0.005,
                'GOOGL': 0.0,
                'MSFT': 0.007,
                'TSLA': 0.0
            }
            
            logger.info("Initialized pricing models")
            
        except Exception as e:
            logger.error(f"Error initializing pricing models: {e}")


# Factory function
async def get_derivatives_trading_service(redis_client: redis.Redis, db_session: Session) -> DerivativesTradingService:
    """Get Derivatives Trading Service instance"""
    service = DerivativesTradingService(redis_client, db_session)
    await service.initialize()
    return service