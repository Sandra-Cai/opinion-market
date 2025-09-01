"""
Foreign Exchange Trading Service
Provides spot FX, forwards, swaps, options, and FX-specific analytics
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
class CurrencyPair:
    """Currency pair information"""
    pair_id: str
    base_currency: str  # e.g., 'USD', 'EUR', 'GBP'
    quote_currency: str  # e.g., 'JPY', 'CHF', 'AUD'
    pair_name: str  # e.g., 'EUR/USD', 'GBP/JPY'
    pip_value: float
    min_trade_size: float
    max_trade_size: float
    trading_hours: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class FXRate:
    """Foreign exchange rate"""
    rate_id: str
    currency_pair: str
    bid_rate: float
    ask_rate: float
    mid_rate: float
    spread: float
    timestamp: datetime
    source: str
    volume_24h: Optional[float]
    high_24h: Optional[float]
    low_24h: Optional[float]


@dataclass
class FXPosition:
    """FX trading position"""
    position_id: str
    user_id: int
    currency_pair: str
    position_type: str  # 'long', 'short'
    size: float
    entry_rate: float
    current_rate: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: datetime
    last_updated: datetime


@dataclass
class FXForward:
    """FX forward contract"""
    forward_id: str
    user_id: int
    currency_pair: str
    contract_size: float
    forward_rate: float
    spot_rate: float
    forward_points: float
    value_date: datetime
    maturity_date: datetime
    interest_rate_base: float
    interest_rate_quote: float
    status: str  # 'active', 'settled', 'cancelled'
    created_at: datetime
    last_updated: datetime


@dataclass
class FXSwap:
    """FX swap contract"""
    swap_id: str
    user_id: int
    currency_pair: str
    near_leg_size: float
    far_leg_size: float
    near_rate: float
    far_rate: float
    swap_points: float
    near_value_date: datetime
    far_value_date: datetime
    interest_rate_differential: float
    status: str  # 'active', 'settled', 'cancelled'
    created_at: datetime
    last_updated: datetime


@dataclass
class FXOption:
    """FX option contract"""
    option_id: str
    user_id: int
    currency_pair: str
    option_type: str  # 'call', 'put'
    strike_rate: float
    notional_amount: float
    premium: float
    expiration_date: datetime
    value_date: datetime
    volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    status: str  # 'active', 'expired', 'exercised'
    created_at: datetime
    last_updated: datetime


@dataclass
class FXOrder:
    """FX trading order"""
    order_id: str
    user_id: int
    currency_pair: str
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    side: str  # 'buy', 'sell'
    size: float
    rate: Optional[float]
    stop_rate: Optional[float]
    limit_rate: Optional[float]
    time_in_force: str  # 'GTC', 'IOC', 'FOK'
    status: str  # 'pending', 'filled', 'cancelled', 'rejected'
    filled_size: float
    filled_rate: float
    commission: float
    created_at: datetime
    last_updated: datetime


class ForexTradingService:
    """Comprehensive foreign exchange trading service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.currency_pairs: Dict[str, CurrencyPair] = {}
        self.fx_rates: Dict[str, List[FXRate]] = defaultdict(list)
        self.fx_positions: Dict[str, FXPosition] = {}
        self.fx_forwards: Dict[str, FXForward] = {}
        self.fx_swaps: Dict[str, FXSwap] = {}
        self.fx_options: Dict[str, FXOption] = {}
        self.fx_orders: Dict[str, FXOrder] = {}
        
        # Market data
        self.rate_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # FX-specific data
        self.interest_rates: Dict[str, Dict[str, float]] = {}
        self.central_bank_rates: Dict[str, Dict[str, Any]] = {}
        self.economic_calendar: Dict[str, List[Dict[str, Any]]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
    async def initialize(self):
        """Initialize the forex trading service"""
        logger.info("Initializing Forex Trading Service")
        
        # Load existing data
        await self._load_currency_pairs()
        await self._load_interest_rates()
        await self._load_central_bank_rates()
        
        # Start background tasks
        asyncio.create_task(self._update_fx_rates())
        asyncio.create_task(self._update_positions())
        asyncio.create_task(self._update_volatility())
        asyncio.create_task(self._monitor_orders())
        
        logger.info("Forex Trading Service initialized successfully")
    
    async def create_currency_pair(self, base_currency: str, quote_currency: str, pip_value: float,
                                  min_trade_size: float, max_trade_size: float, trading_hours: str) -> CurrencyPair:
        """Create a new currency pair"""
        try:
            pair_name = f"{base_currency}/{quote_currency}"
            pair_id = f"pair_{base_currency}_{quote_currency}"
            
            pair = CurrencyPair(
                pair_id=pair_id,
                base_currency=base_currency,
                quote_currency=quote_currency,
                pair_name=pair_name,
                pip_value=pip_value,
                min_trade_size=min_trade_size,
                max_trade_size=max_trade_size,
                trading_hours=trading_hours,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.currency_pairs[pair_id] = pair
            await self._cache_currency_pair(pair)
            
            logger.info(f"Created currency pair {pair_id}")
            return pair
            
        except Exception as e:
            logger.error(f"Error creating currency pair: {e}")
            raise
    
    async def add_fx_rate(self, currency_pair: str, bid_rate: float, ask_rate: float,
                          source: str, volume_24h: Optional[float] = None,
                          high_24h: Optional[float] = None, low_24h: Optional[float] = None) -> FXRate:
        """Add a new FX rate"""
        try:
            if currency_pair not in [p.pair_id for p in self.currency_pairs.values()]:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            mid_rate = (bid_rate + ask_rate) / 2
            spread = ask_rate - bid_rate
            
            rate = FXRate(
                rate_id=f"rate_{currency_pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                currency_pair=currency_pair,
                bid_rate=bid_rate,
                ask_rate=ask_rate,
                mid_rate=mid_rate,
                spread=spread,
                timestamp=datetime.utcnow(),
                source=source,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h
            )
            
            self.fx_rates[currency_pair].append(rate)
            
            # Keep only recent rates
            if len(self.fx_rates[currency_pair]) > 1000:
                self.fx_rates[currency_pair] = self.fx_rates[currency_pair][-1000:]
            
            await self._cache_fx_rate(rate)
            
            logger.info(f"Added FX rate for {currency_pair}: {bid_rate}/{ask_rate}")
            return rate
            
        except Exception as e:
            logger.error(f"Error adding FX rate: {e}")
            raise
    
    async def create_fx_position(self, user_id: int, currency_pair: str, position_type: str,
                                size: float, entry_rate: float, leverage: float = 1.0,
                                stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> FXPosition:
        """Create a new FX position"""
        try:
            if currency_pair not in [p.pair_id for p in self.currency_pairs.values()]:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            # Calculate margin requirement
            margin_used = size * entry_rate / leverage
            
            position_id = f"fx_pos_{user_id}_{currency_pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            position = FXPosition(
                position_id=position_id,
                user_id=user_id,
                currency_pair=currency_pair,
                position_type=position_type,
                size=size,
                entry_rate=entry_rate,
                current_rate=entry_rate,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                margin_used=margin_used,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.fx_positions[position_id] = position
            await self._cache_fx_position(position)
            
            logger.info(f"Created FX position {position_id}")
            return position
            
        except Exception as e:
            logger.error(f"Error creating FX position: {e}")
            raise
    
    async def create_fx_forward(self, user_id: int, currency_pair: str, contract_size: float,
                               forward_rate: float, value_date: datetime, maturity_date: datetime,
                               interest_rate_base: float, interest_rate_quote: float) -> FXForward:
        """Create a new FX forward contract"""
        try:
            if currency_pair not in [p.pair_id for p in self.currency_pairs.values()]:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            # Get current spot rate
            current_rate = await self._get_current_fx_rate(currency_pair)
            if not current_rate:
                raise ValueError(f"No current rate available for {currency_pair}")
            
            spot_rate = current_rate.mid_rate
            forward_points = forward_rate - spot_rate
            
            forward_id = f"fx_fwd_{user_id}_{currency_pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            forward = FXForward(
                forward_id=forward_id,
                user_id=user_id,
                currency_pair=currency_pair,
                contract_size=contract_size,
                forward_rate=forward_rate,
                spot_rate=spot_rate,
                forward_points=forward_points,
                value_date=value_date,
                maturity_date=maturity_date,
                interest_rate_base=interest_rate_base,
                interest_rate_quote=interest_rate_quote,
                status='active',
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.fx_forwards[forward_id] = forward
            await self._cache_fx_forward(forward)
            
            logger.info(f"Created FX forward {forward_id}")
            return forward
            
        except Exception as e:
            logger.error(f"Error creating FX forward: {e}")
            raise
    
    async def create_fx_swap(self, user_id: int, currency_pair: str, near_leg_size: float,
                            far_leg_size: float, near_rate: float, far_rate: float,
                            near_value_date: datetime, far_value_date: datetime) -> FXSwap:
        """Create a new FX swap contract"""
        try:
            if currency_pair not in [p.pair_id for p in self.currency_pairs.values()]:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            swap_points = far_rate - near_rate
            
            # Calculate interest rate differential
            days_between = (far_value_date - near_value_date).days
            interest_rate_differential = (swap_points / near_rate) * (365 / days_between) if days_between > 0 else 0
            
            swap_id = f"fx_swap_{user_id}_{currency_pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            swap = FXSwap(
                swap_id=swap_id,
                user_id=user_id,
                currency_pair=currency_pair,
                near_leg_size=near_leg_size,
                far_leg_size=far_leg_size,
                near_rate=near_rate,
                far_rate=far_rate,
                swap_points=swap_points,
                near_value_date=near_value_date,
                far_value_date=far_value_date,
                interest_rate_differential=interest_rate_differential,
                status='active',
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.fx_swaps[swap_id] = swap
            await self._cache_fx_swap(swap)
            
            logger.info(f"Created FX swap {swap_id}")
            return swap
            
        except Exception as e:
            logger.error(f"Error creating FX swap: {e}")
            raise
    
    async def create_fx_option(self, user_id: int, currency_pair: str, option_type: str,
                              strike_rate: float, notional_amount: float, premium: float,
                              expiration_date: datetime, value_date: datetime,
                              volatility: float) -> FXOption:
        """Create a new FX option contract"""
        try:
            if currency_pair not in [p.pair_id for p in self.currency_pairs.values()]:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            option_id = f"fx_opt_{user_id}_{currency_pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate option Greeks (simplified)
            current_rate = await self._get_current_fx_rate(currency_pair)
            if not current_rate:
                raise ValueError(f"No current rate available for {currency_pair}")
            
            spot_rate = current_rate.mid_rate
            time_to_expiry = (expiration_date - datetime.utcnow()).days / 365.0
            
            # Simplified Greeks calculation
            delta = self._calculate_fx_option_delta(option_type, spot_rate, strike_rate, time_to_expiry, volatility)
            gamma = self._calculate_fx_option_gamma(spot_rate, strike_rate, time_to_expiry, volatility)
            theta = self._calculate_fx_option_theta(option_type, spot_rate, strike_rate, time_to_expiry, volatility)
            vega = self._calculate_fx_option_vega(spot_rate, strike_rate, time_to_expiry, volatility)
            rho = self._calculate_fx_option_rho(option_type, spot_rate, strike_rate, time_to_expiry, volatility)
            
            option = FXOption(
                option_id=option_id,
                user_id=user_id,
                currency_pair=currency_pair,
                option_type=option_type,
                strike_rate=strike_rate,
                notional_amount=notional_amount,
                premium=premium,
                expiration_date=expiration_date,
                value_date=value_date,
                volatility=volatility,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                status='active',
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.fx_options[option_id] = option
            await self._cache_fx_option(option)
            
            logger.info(f"Created FX option {option_id}")
            return option
            
        except Exception as e:
            logger.error(f"Error creating FX option: {e}")
            raise
    
    async def create_fx_order(self, user_id: int, currency_pair: str, order_type: str, side: str,
                             size: float, rate: Optional[float] = None, stop_rate: Optional[float] = None,
                             limit_rate: Optional[float] = None, time_in_force: str = 'GTC') -> FXOrder:
        """Create a new FX order"""
        try:
            if currency_pair not in [p.pair_id for p in self.currency_pairs.values()]:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            order_id = f"fx_order_{user_id}_{currency_pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            order = FXOrder(
                order_id=order_id,
                user_id=user_id,
                currency_pair=currency_pair,
                order_type=order_type,
                side=side,
                size=size,
                rate=rate,
                stop_rate=stop_rate,
                limit_rate=limit_rate,
                time_in_force=time_in_force,
                status='pending',
                filled_size=0.0,
                filled_rate=0.0,
                commission=0.0,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.fx_orders[order_id] = order
            await self._cache_fx_order(order)
            
            logger.info(f"Created FX order {order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating FX order: {e}")
            raise
    
    async def calculate_pip_value(self, currency_pair: str, trade_size: float) -> Dict[str, Any]:
        """Calculate pip value for a trade"""
        try:
            pair = self.currency_pairs.get(currency_pair)
            if not pair:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            # Calculate pip value
            pip_value = pair.pip_value * trade_size
            
            # Get current rate for USD value calculation
            current_rate = await self._get_current_fx_rate(currency_pair)
            if current_rate:
                pip_value_usd = pip_value * current_rate.mid_rate
            else:
                pip_value_usd = pip_value
            
            return {
                'currency_pair': currency_pair,
                'trade_size': trade_size,
                'pip_value': pip_value,
                'pip_value_usd': pip_value_usd,
                'currency': pair.quote_currency
            }
            
        except Exception as e:
            logger.error(f"Error calculating pip value: {e}")
            raise
    
    async def calculate_margin_requirement(self, currency_pair: str, trade_size: float,
                                         leverage: float) -> Dict[str, Any]:
        """Calculate margin requirement for a trade"""
        try:
            current_rate = await self._get_current_fx_rate(currency_pair)
            if not current_rate:
                raise ValueError(f"No current rate available for {currency_pair}")
            
            # Calculate margin requirement
            notional_value = trade_size * current_rate.mid_rate
            margin_required = notional_value / leverage
            margin_percentage = (margin_required / notional_value) * 100
            
            return {
                'currency_pair': currency_pair,
                'trade_size': trade_size,
                'notional_value': notional_value,
                'leverage': leverage,
                'margin_required': margin_required,
                'margin_percentage': margin_percentage,
                'currency': 'USD'
            }
            
        except Exception as e:
            logger.error(f"Error calculating margin requirement: {e}")
            raise
    
    async def get_fx_analytics(self, currency_pair: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a currency pair"""
        try:
            if currency_pair not in [p.pair_id for p in self.currency_pairs.values()]:
                raise ValueError(f"Currency pair {currency_pair} not found")
            
            pair = self.currency_pairs[currency_pair]
            rates = self.fx_rates.get(currency_pair, [])
            
            if not rates:
                return {'currency_pair': currency_pair, 'error': 'No rate data available'}
            
            # Calculate rate statistics
            bid_rates = [r.bid_rate for r in rates]
            ask_rates = [r.ask_rate for r in rates]
            spreads = [r.spread for r in rates]
            
            analytics = {
                'currency_pair': currency_pair,
                'pair_name': pair.pair_name,
                'base_currency': pair.base_currency,
                'quote_currency': pair.quote_currency,
                'current_bid': bid_rates[-1] if bid_rates else None,
                'current_ask': ask_rates[-1] if ask_rates else None,
                'current_mid': (bid_rates[-1] + ask_rates[-1]) / 2 if bid_rates and ask_rates else None,
                'current_spread': spreads[-1] if spreads else None,
                'rate_statistics': {
                    'bid_mean': np.mean(bid_rates) if bid_rates else None,
                    'bid_std': np.std(bid_rates) if bid_rates else None,
                    'ask_mean': np.mean(ask_rates) if ask_rates else None,
                    'ask_std': np.std(ask_rates) if ask_rates else None,
                    'spread_mean': np.mean(spreads) if spreads else None,
                    'spread_std': np.std(spreads) if spreads else None
                },
                'rate_trends': await self._calculate_rate_trends(currency_pair),
                'volatility': await self._calculate_volatility(currency_pair),
                'correlation': self.correlation_matrix.get(currency_pair, {}),
                'interest_rates': self.interest_rates.get(currency_pair, {}),
                'central_bank_info': self.central_bank_rates.get(currency_pair, {}),
                'economic_events': self.economic_calendar.get(currency_pair, []),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting FX analytics: {e}")
            raise
    
    async def get_currency_correlation(self, currency_pairs: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation between currency pairs"""
        try:
            correlations = {}
            
            for pair1 in currency_pairs:
                correlations[pair1] = {}
                rates1 = self.fx_rates.get(pair1, [])
                
                if not rates1:
                    continue
                
                # Get recent rates for correlation calculation
                recent_rates1 = [r.mid_rate for r in rates1[-100:]]  # Last 100 rates
                
                for pair2 in currency_pairs:
                    if pair1 == pair2:
                        correlations[pair1][pair2] = 1.0
                        continue
                    
                    rates2 = self.fx_rates.get(pair2, [])
                    if not rates2:
                        correlations[pair1][pair2] = 0.0
                        continue
                    
                    # Get recent rates for pair2
                    recent_rates2 = [r.mid_rate for r in rates2[-100:]]
                    
                    # Calculate correlation (ensure same length)
                    min_length = min(len(recent_rates1), len(recent_rates2))
                    if min_length > 1:
                        corr = np.corrcoef(recent_rates1[:min_length], recent_rates2[:min_length])[0, 1]
                        correlations[pair1][pair2] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlations[pair1][pair2] = 0.0
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating currency correlation: {e}")
            return {}
    
    # FX Option Greeks calculations
    def _calculate_fx_option_delta(self, option_type: str, spot_rate: float, strike_rate: float,
                                  time_to_expiry: float, volatility: float) -> float:
        """Calculate FX option delta"""
        try:
            if time_to_expiry <= 0:
                return 1.0 if (option_type == 'call' and spot_rate > strike_rate) or (option_type == 'put' and spot_rate < strike_rate) else 0.0
            
            # Simplified Black-Scholes delta calculation
            d1 = (np.log(spot_rate / strike_rate) + (0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            
            if option_type == 'call':
                delta = self._normal_cdf(d1)
            else:  # put
                delta = self._normal_cdf(d1) - 1
            
            return delta
            
        except Exception as e:
            logger.error(f"Error calculating FX option delta: {e}")
            return 0.0
    
    def _calculate_fx_option_gamma(self, spot_rate: float, strike_rate: float,
                                  time_to_expiry: float, volatility: float) -> float:
        """Calculate FX option gamma"""
        try:
            if time_to_expiry <= 0:
                return 0.0
            
            d1 = (np.log(spot_rate / strike_rate) + (0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            gamma = self._normal_pdf(d1) / (spot_rate * volatility * np.sqrt(time_to_expiry))
            
            return gamma
            
        except Exception as e:
            logger.error(f"Error calculating FX option gamma: {e}")
            return 0.0
    
    def _calculate_fx_option_theta(self, option_type: str, spot_rate: float, strike_rate: float,
                                   time_to_expiry: float, volatility: float) -> float:
        """Calculate FX option theta"""
        try:
            if time_to_expiry <= 0:
                return 0.0
            
            # Simplified theta calculation
            theta = -0.5 * volatility**2 * spot_rate**2 * self._calculate_fx_option_gamma(spot_rate, strike_rate, time_to_expiry, volatility)
            
            return theta
            
        except Exception as e:
            logger.error(f"Error calculating FX option theta: {e}")
            return 0.0
    
    def _calculate_fx_option_vega(self, spot_rate: float, strike_rate: float,
                                  time_to_expiry: float, volatility: float) -> float:
        """Calculate FX option vega"""
        try:
            if time_to_expiry <= 0:
                return 0.0
            
            d1 = (np.log(spot_rate / strike_rate) + (0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            vega = spot_rate * np.sqrt(time_to_expiry) * self._normal_pdf(d1)
            
            return vega
            
        except Exception as e:
            logger.error(f"Error calculating FX option vega: {e}")
            return 0.0
    
    def _calculate_fx_option_rho(self, option_type: str, spot_rate: float, strike_rate: float,
                                 time_to_expiry: float, volatility: float) -> float:
        """Calculate FX option rho"""
        try:
            if time_to_expiry <= 0:
                return 0.0
            
            # Simplified rho calculation
            risk_free_rate = 0.05  # 5% annual rate
            rho = risk_free_rate * time_to_expiry * spot_rate * self._calculate_fx_option_delta(option_type, spot_rate, strike_rate, time_to_expiry, volatility)
            
            return rho
            
        except Exception as e:
            logger.error(f"Error calculating FX option rho: {e}")
            return 0.0
    
    def _normal_cdf(self, x: float) -> float:
        """Normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Normal probability density function"""
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)
    
    async def _get_current_fx_rate(self, currency_pair: str) -> Optional[FXRate]:
        """Get current FX rate for a currency pair"""
        try:
            rates = self.fx_rates.get(currency_pair, [])
            if rates:
                return rates[-1]
            return None
            
        except Exception as e:
            logger.error(f"Error getting current FX rate: {e}")
            return None
    
    async def _calculate_rate_trends(self, currency_pair: str) -> Dict[str, Any]:
        """Calculate rate trends for a currency pair"""
        try:
            rates = self.fx_rates.get(currency_pair, [])
            if len(rates) < 2:
                return {}
            
            # Get recent rates
            recent_rates = rates[-30:]  # Last 30 rates
            mid_rates = [r.mid_rate for r in recent_rates]
            
            # Calculate trends
            if len(mid_rates) >= 2:
                rate_change = mid_rates[-1] - mid_rates[0]
                rate_change_percent = (rate_change / mid_rates[0]) * 100 if mid_rates[0] != 0 else 0
                
                # Simple moving averages
                sma_5 = np.mean(mid_rates[-5:]) if len(mid_rates) >= 5 else None
                sma_10 = np.mean(mid_rates[-10:]) if len(mid_rates) >= 10 else None
                sma_20 = np.mean(mid_rates[-20:]) if len(mid_rates) >= 20 else None
                
                return {
                    'rate_change': rate_change,
                    'rate_change_percent': rate_change_percent,
                    'trend_direction': 'up' if rate_change > 0 else 'down' if rate_change < 0 else 'flat',
                    'sma_5': sma_5,
                    'sma_10': sma_10,
                    'sma_20': sma_20,
                    'volatility': np.std(mid_rates) if len(mid_rates) > 1 else 0
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating rate trends: {e}")
            return {}
    
    async def _calculate_volatility(self, currency_pair: str) -> Dict[str, float]:
        """Calculate volatility for a currency pair"""
        try:
            rates = self.fx_rates.get(currency_pair, [])
            if len(rates) < 2:
                return {}
            
            # Calculate returns
            mid_rates = [r.mid_rate for r in rates]
            returns = [np.log(mid_rates[i] / mid_rates[i-1]) for i in range(1, len(mid_rates))]
            
            if returns:
                # Calculate volatility measures
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                realized_volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else volatility
                
                return {
                    'volatility': volatility,
                    'realized_volatility': realized_volatility,
                    'volatility_20d': realized_volatility
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return {}
    
    # Background tasks
    async def _update_fx_rates(self):
        """Update FX rates periodically"""
        while True:
            try:
                # Update rates for all currency pairs
                for pair in self.currency_pairs.values():
                    if pair.is_active:
                        # Simulate rate updates (in practice, this would fetch from data providers)
                        current_rate = await self._get_current_fx_rate(pair.pair_id)
                        if current_rate:
                            # Add small random rate movement
                            movement = np.random.normal(0, 0.0001)  # Small pip movement
                            new_bid = current_rate.bid_rate * (1 + movement)
                            new_ask = current_rate.ask_rate * (1 + movement)
                            
                            await self.add_fx_rate(
                                pair.pair_id, new_bid, new_ask, 'simulated'
                            )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating FX rates: {e}")
                await asyncio.sleep(120)
    
    async def _update_positions(self):
        """Update FX positions"""
        while True:
            try:
                for position in self.fx_positions.values():
                    # Get current rate
                    current_rate = await self._get_current_fx_rate(position.currency_pair)
                    if current_rate:
                        position.current_rate = current_rate.mid_rate
                        
                        # Calculate unrealized P&L
                        if position.position_type == 'long':
                            position.unrealized_pnl = (position.current_rate - position.entry_rate) * position.size
                        else:  # short
                            position.unrealized_pnl = (position.entry_rate - position.current_rate) * position.size
                        
                        position.last_updated = datetime.utcnow()
                        await self._cache_fx_position(position)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating positions: {e}")
                await asyncio.sleep(60)
    
    async def _update_volatility(self):
        """Update volatility calculations"""
        while True:
            try:
                # Update volatility for all currency pairs
                for pair in self.currency_pairs.values():
                    if pair.is_active:
                        volatility = await self._calculate_volatility(pair.pair_id)
                        if volatility:
                            self.volatility_history[pair.pair_id].append(volatility.get('volatility', 0))
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error updating volatility: {e}")
                await asyncio.sleep(600)
    
    async def _monitor_orders(self):
        """Monitor FX orders"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for order in self.fx_orders.values():
                    if order.status == 'pending':
                        # Check if order should be filled (simplified logic)
                        current_rate = await self._get_current_fx_rate(order.currency_pair)
                        if current_rate:
                            # Simple order execution logic
                            if order.order_type == 'market':
                                order.status = 'filled'
                                order.filled_size = order.size
                                order.filled_rate = current_rate.mid_rate
                                order.last_updated = current_time
                                
                                await self._cache_fx_order(order)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(30)
    
    # Helper methods (implementations would depend on your data models)
    async def _load_currency_pairs(self):
        """Load currency pairs from database"""
        pass
    
    async def _load_interest_rates(self):
        """Load interest rates from database"""
        pass
    
    async def _load_central_bank_rates(self):
        """Load central bank rates from database"""
        pass
    
    # Caching methods
    async def _cache_currency_pair(self, pair: CurrencyPair):
        """Cache currency pair"""
        try:
            cache_key = f"currency_pair:{pair.pair_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'base_currency': pair.base_currency,
                    'quote_currency': pair.quote_currency,
                    'pair_name': pair.pair_name,
                    'pip_value': pair.pip_value,
                    'min_trade_size': pair.min_trade_size,
                    'max_trade_size': pair.max_trade_size,
                    'trading_hours': pair.trading_hours,
                    'is_active': pair.is_active,
                    'created_at': pair.created_at.isoformat(),
                    'last_updated': pair.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching currency pair: {e}")
    
    async def _cache_fx_rate(self, rate: FXRate):
        """Cache FX rate"""
        try:
            cache_key = f"fx_rate:{rate.rate_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps({
                    'currency_pair': rate.currency_pair,
                    'bid_rate': rate.bid_rate,
                    'ask_rate': rate.ask_rate,
                    'mid_rate': rate.mid_rate,
                    'spread': rate.spread,
                    'timestamp': rate.timestamp.isoformat(),
                    'source': rate.source,
                    'volume_24h': rate.volume_24h,
                    'high_24h': rate.high_24h,
                    'low_24h': rate.low_24h
                })
            )
        except Exception as e:
            logger.error(f"Error caching FX rate: {e}")
    
    async def _cache_fx_position(self, position: FXPosition):
        """Cache FX position"""
        try:
            cache_key = f"fx_position:{position.position_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps({
                    'user_id': position.user_id,
                    'currency_pair': position.currency_pair,
                    'position_type': position.position_type,
                    'size': position.size,
                    'entry_rate': position.entry_rate,
                    'current_rate': position.current_rate,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'margin_used': position.margin_used,
                    'leverage': position.leverage,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'created_at': position.created_at.isoformat(),
                    'last_updated': position.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching FX position: {e}")
    
    async def _cache_fx_forward(self, forward: FXForward):
        """Cache FX forward"""
        try:
            cache_key = f"fx_forward:{forward.forward_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'user_id': forward.user_id,
                    'currency_pair': forward.currency_pair,
                    'contract_size': forward.contract_size,
                    'forward_rate': forward.forward_rate,
                    'spot_rate': forward.spot_rate,
                    'forward_points': forward.forward_points,
                    'value_date': forward.value_date.isoformat(),
                    'maturity_date': forward.maturity_date.isoformat(),
                    'interest_rate_base': forward.interest_rate_base,
                    'interest_rate_quote': forward.interest_rate_quote,
                    'status': forward.status,
                    'created_at': forward.created_at.isoformat(),
                    'last_updated': forward.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching FX forward: {e}")
    
    async def _cache_fx_swap(self, swap: FXSwap):
        """Cache FX swap"""
        try:
            cache_key = f"fx_swap:{swap.swap_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'user_id': swap.user_id,
                    'currency_pair': swap.currency_pair,
                    'near_leg_size': swap.near_leg_size,
                    'far_leg_size': swap.far_leg_size,
                    'near_rate': swap.near_rate,
                    'far_rate': swap.far_rate,
                    'swap_points': swap.swap_points,
                    'near_value_date': swap.near_value_date.isoformat(),
                    'far_value_date': swap.far_value_date.isoformat(),
                    'interest_rate_differential': swap.interest_rate_differential,
                    'status': swap.status,
                    'created_at': swap.created_at.isoformat(),
                    'last_updated': swap.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching FX swap: {e}")
    
    async def _cache_fx_option(self, option: FXOption):
        """Cache FX option"""
        try:
            cache_key = f"fx_option:{option.option_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'user_id': option.user_id,
                    'currency_pair': option.currency_pair,
                    'option_type': option.option_type,
                    'strike_rate': option.strike_rate,
                    'notional_amount': option.notional_amount,
                    'premium': option.premium,
                    'expiration_date': option.expiration_date.isoformat(),
                    'value_date': option.value_date.isoformat(),
                    'volatility': option.volatility,
                    'delta': option.delta,
                    'gamma': option.gamma,
                    'theta': option.theta,
                    'vega': option.vega,
                    'rho': option.rho,
                    'status': option.status,
                    'created_at': option.created_at.isoformat(),
                    'last_updated': option.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching FX option: {e}")
    
    async def _cache_fx_order(self, order: FXOrder):
        """Cache FX order"""
        try:
            cache_key = f"fx_order:{order.order_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps({
                    'user_id': order.user_id,
                    'currency_pair': order.currency_pair,
                    'order_type': order.order_type,
                    'side': order.side,
                    'size': order.size,
                    'rate': order.rate,
                    'stop_rate': order.stop_rate,
                    'limit_rate': order.limit_rate,
                    'time_in_force': order.time_in_force,
                    'status': order.status,
                    'filled_size': order.filled_size,
                    'filled_rate': order.filled_rate,
                    'commission': order.commission,
                    'created_at': order.created_at.isoformat(),
                    'last_updated': order.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching FX order: {e}")


# Factory function
async def get_forex_trading_service(redis_client: redis.Redis, db_session: Session) -> ForexTradingService:
    """Get forex trading service instance"""
    service = ForexTradingService(redis_client, db_session)
    await service.initialize()
    return service
