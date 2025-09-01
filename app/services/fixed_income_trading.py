"""
Fixed Income Trading Service
Provides government bonds, corporate bonds, municipal bonds, and fixed income analytics
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
class Bond:
    """Bond information"""
    bond_id: str
    issuer: str
    bond_type: str  # 'government', 'corporate', 'municipal', 'agency'
    cusip: str
    isin: str
    face_value: float
    coupon_rate: float
    coupon_frequency: str  # 'annual', 'semi_annual', 'quarterly'
    maturity_date: datetime
    issue_date: datetime
    callable: bool
    putable: bool
    convertible: bool
    credit_rating: str
    currency: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class BondPrice:
    """Bond price information"""
    price_id: str
    bond_id: str
    clean_price: float
    dirty_price: float
    yield_to_maturity: float
    yield_to_call: Optional[float]
    yield_to_put: Optional[float]
    current_yield: float
    bid_price: float
    ask_price: float
    bid_yield: float
    ask_yield: float
    spread_to_treasury: float
    duration: float
    modified_duration: float
    convexity: float
    timestamp: datetime
    source: str


@dataclass
class BondPosition:
    """Bond trading position"""
    position_id: str
    user_id: int
    bond_id: str
    quantity: float
    entry_price: float
    current_price: float
    face_value: float
    coupon_income: float
    unrealized_pnl: float
    realized_pnl: float
    accrued_interest: float
    yield_to_maturity: float
    duration_exposure: float
    created_at: datetime
    last_updated: datetime


@dataclass
class YieldCurve:
    """Yield curve information"""
    curve_id: str
    curve_type: str  # 'treasury', 'corporate', 'municipal'
    currency: str
    tenors: List[str]  # ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
    yields: List[float]
    spreads: List[float]
    curve_date: datetime
    created_at: datetime
    last_updated: datetime


class FixedIncomeTradingService:
    """Comprehensive fixed income trading service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.bonds: Dict[str, Bond] = {}
        self.bond_prices: Dict[str, List[BondPrice]] = defaultdict(list)
        self.bond_positions: Dict[str, BondPosition] = {}
        self.yield_curves: Dict[str, YieldCurve] = {}
        
        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.yield_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.spread_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Fixed income specific data
        self.credit_spreads: Dict[str, Dict[str, float]] = {}
        self.liquidity_metrics: Dict[str, Dict[str, Any]] = {}
        self.issuer_ratings: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the fixed income trading service"""
        logger.info("Initializing Fixed Income Trading Service")
        
        # Load existing data
        await self._load_bonds()
        await self._load_yield_curves()
        await self._load_credit_spreads()
        
        # Start background tasks
        asyncio.create_task(self._update_bond_prices())
        asyncio.create_task(self._update_positions())
        asyncio.create_task(self._update_yield_curves())
        asyncio.create_task(self._monitor_credit_ratings())
        
        logger.info("Fixed Income Trading Service initialized successfully")
    
    async def create_bond(self, issuer: str, bond_type: str, cusip: str, isin: str,
                          face_value: float, coupon_rate: float, coupon_frequency: str,
                          maturity_date: datetime, issue_date: datetime, callable: bool = False,
                          putable: bool = False, convertible: bool = False, credit_rating: str = "BBB",
                          currency: str = "USD") -> Bond:
        """Create a new bond"""
        try:
            bond_id = f"bond_{cusip}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            bond = Bond(
                bond_id=bond_id,
                issuer=issuer,
                bond_type=bond_type,
                cusip=cusip,
                isin=isin,
                face_value=face_value,
                coupon_rate=coupon_rate,
                coupon_frequency=coupon_frequency,
                maturity_date=maturity_date,
                issue_date=issue_date,
                callable=callable,
                putable=putable,
                convertible=convertible,
                credit_rating=credit_rating,
                currency=currency,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.bonds[bond_id] = bond
            await self._cache_bond(bond)
            
            logger.info(f"Created bond {bond_id}")
            return bond
            
        except Exception as e:
            logger.error(f"Error creating bond: {e}")
            raise
    
    async def add_bond_price(self, bond_id: str, clean_price: float, yield_to_maturity: float,
                            bid_price: float, ask_price: float, bid_yield: float, ask_yield: float,
                            spread_to_treasury: float, source: str) -> BondPrice:
        """Add a new bond price"""
        try:
            if bond_id not in self.bonds:
                raise ValueError(f"Bond {bond_id} not found")
            
            bond = self.bonds[bond_id]
            
            # Calculate additional metrics
            dirty_price = clean_price + self._calculate_accrued_interest(bond)
            current_yield = (bond.coupon_rate * bond.face_value) / (clean_price * 100)
            
            # Calculate duration and convexity
            duration = self._calculate_duration(bond, yield_to_maturity, clean_price)
            modified_duration = duration / (1 + yield_to_maturity / 100)
            convexity = self._calculate_convexity(bond, yield_to_maturity, clean_price)
            
            price = BondPrice(
                price_id=f"price_{bond_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                bond_id=bond_id,
                clean_price=clean_price,
                dirty_price=dirty_price,
                yield_to_maturity=yield_to_maturity,
                yield_to_call=None,  # Would be calculated if callable
                yield_to_put=None,   # Would be calculated if putable
                current_yield=current_yield,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_yield=bid_yield,
                ask_yield=ask_yield,
                spread_to_treasury=spread_to_treasury,
                duration=duration,
                modified_duration=modified_duration,
                convexity=convexity,
                timestamp=datetime.utcnow(),
                source=source
            )
            
            self.bond_prices[bond_id].append(price)
            
            # Keep only recent prices
            if len(self.bond_prices[bond_id]) > 1000:
                self.bond_prices[bond_id] = self.bond_prices[bond_id][-1000:]
            
            await self._cache_bond_price(price)
            
            logger.info(f"Added price for bond {bond_id}: ${clean_price}")
            return price
            
        except Exception as e:
            logger.error(f"Error adding bond price: {e}")
            raise
    
    async def create_bond_position(self, user_id: int, bond_id: str, quantity: float,
                                  entry_price: float) -> BondPosition:
        """Create a new bond position"""
        try:
            if bond_id not in self.bonds:
                raise ValueError(f"Bond {bond_id} not found")
            
            bond = self.bonds[bond_id]
            face_value = quantity * bond.face_value
            
            position_id = f"bond_pos_{user_id}_{bond_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            position = BondPosition(
                position_id=position_id,
                user_id=user_id,
                bond_id=bond_id,
                quantity=quantity,
                entry_price=entry_price,
                current_price=entry_price,
                face_value=face_value,
                coupon_income=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                accrued_interest=0.0,
                yield_to_maturity=0.0,
                duration_exposure=0.0,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.bond_positions[position_id] = position
            await self._cache_bond_position(position)
            
            logger.info(f"Created bond position {position_id}")
            return position
            
        except Exception as e:
            logger.error(f"Error creating bond position: {e}")
            raise
    
    async def create_yield_curve(self, curve_type: str, currency: str, tenors: List[str],
                                yields: List[float], spreads: Optional[List[float]] = None) -> YieldCurve:
        """Create a new yield curve"""
        try:
            curve_id = f"curve_{curve_type}_{currency}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            if spreads is None:
                spreads = [0.0] * len(yields)
            
            curve = YieldCurve(
                curve_id=curve_id,
                curve_type=curve_type,
                currency=currency,
                tenors=tenors,
                yields=yields,
                spreads=spreads,
                curve_date=datetime.utcnow(),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.yield_curves[curve_id] = curve
            await self._cache_yield_curve(curve)
            
            logger.info(f"Created yield curve {curve_id}")
            return curve
            
        except Exception as e:
            logger.error(f"Error creating yield curve: {e}")
            raise
    
    async def calculate_bond_metrics(self, bond_id: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a bond"""
        try:
            if bond_id not in self.bonds:
                raise ValueError(f"Bond {bond_id} not found")
            
            bond = self.bonds[bond_id]
            prices = self.bond_prices.get(bond_id, [])
            
            if not prices:
                return {'bond_id': bond_id, 'error': 'No price data available'}
            
            # Get current price data
            current_price = prices[-1]
            
            # Calculate time to maturity
            time_to_maturity = (bond.maturity_date - datetime.utcnow()).days / 365.0
            
            # Calculate metrics
            metrics = {
                'bond_id': bond_id,
                'issuer': bond.issuer,
                'bond_type': bond.bond_type,
                'cusip': bond.cusip,
                'face_value': bond.face_value,
                'coupon_rate': bond.coupon_rate,
                'maturity_date': bond.maturity_date.isoformat(),
                'time_to_maturity': time_to_maturity,
                'current_price': {
                    'clean_price': current_price.clean_price,
                    'dirty_price': current_price.dirty_price,
                    'bid_price': current_price.bid_price,
                    'ask_price': current_price.ask_price
                },
                'yields': {
                    'yield_to_maturity': current_price.yield_to_maturity,
                    'yield_to_call': current_price.yield_to_call,
                    'yield_to_put': current_price.yield_to_put,
                    'current_yield': current_price.current_yield,
                    'bid_yield': current_price.bid_yield,
                    'ask_yield': current_price.ask_yield
                },
                'risk_metrics': {
                    'duration': current_price.duration,
                    'modified_duration': current_price.modified_duration,
                    'convexity': current_price.convexity,
                    'spread_to_treasury': current_price.spread_to_treasury
                },
                'credit_metrics': {
                    'credit_rating': bond.credit_rating,
                    'credit_spread': self.credit_spreads.get(bond_id, {}),
                    'issuer_rating': self.issuer_ratings.get(bond.issuer, {})
                },
                'liquidity_metrics': self.liquidity_metrics.get(bond_id, {}),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating bond metrics: {e}")
            raise
    
    async def get_yield_curve_analysis(self, curve_type: str, currency: str) -> Dict[str, Any]:
        """Get comprehensive yield curve analysis"""
        try:
            # Find the most recent yield curve
            curves = [c for c in self.yield_curves.values() 
                     if c.curve_type == curve_type and c.currency == currency]
            
            if not curves:
                return {'curve_type': curve_type, 'currency': currency, 'error': 'No yield curve data available'}
            
            # Get the most recent curve
            curve = max(curves, key=lambda x: x.curve_date)
            
            # Calculate curve metrics
            yield_spread = max(curve.yields) - min(curve.yields)
            yield_slope = curve.yields[-1] - curve.yields[0] if len(curve.yields) > 1 else 0
            
            # Calculate duration-weighted average yield
            durations = [self._tenor_to_years(tenor) for tenor in curve.tenors]
            duration_weighted_yield = sum(y * d for y, d in zip(curve.yields, durations)) / sum(durations) if durations else 0
            
            analysis = {
                'curve_type': curve_type,
                'currency': currency,
                'curve_date': curve.curve_date.isoformat(),
                'tenors': curve.tenors,
                'yields': curve.yields,
                'spreads': curve.spreads,
                'curve_metrics': {
                    'yield_spread': yield_spread,
                    'yield_slope': yield_slope,
                    'duration_weighted_yield': duration_weighted_yield,
                    'curve_shape': self._classify_curve_shape(curve.yields)
                },
                'historical_comparison': await self._get_historical_curve_comparison(curve_type, currency),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting yield curve analysis: {e}")
            raise
    
    def _calculate_accrued_interest(self, bond: Bond) -> float:
        """Calculate accrued interest for a bond"""
        try:
            # Get last coupon date
            last_coupon = bond.issue_date
            while last_coupon < datetime.utcnow() and last_coupon < bond.maturity_date:
                if bond.coupon_frequency == 'annual':
                    last_coupon = last_coupon.replace(year=last_coupon.year + 1)
                elif bond.coupon_frequency == 'semi_annual':
                    last_coupon = last_coupon.replace(year=last_coupon.year + 1) if last_coupon.month >= 7 else last_coupon.replace(month=last_coupon.month + 6)
                elif bond.coupon_frequency == 'quarterly':
                    last_coupon = last_coupon.replace(month=last_coupon.month + 3)
            
            # Calculate days since last coupon
            days_since_coupon = (datetime.utcnow() - last_coupon).days
            days_in_period = 365 if bond.coupon_frequency == 'annual' else 182 if bond.coupon_frequency == 'semi_annual' else 91
            
            # Calculate accrued interest
            coupon_payment = bond.coupon_rate * bond.face_value / 100
            accrued_interest = coupon_payment * (days_since_coupon / days_in_period)
            
            return accrued_interest
            
        except Exception as e:
            logger.error(f"Error calculating accrued interest: {e}")
            return 0.0
    
    def _calculate_duration(self, bond: Bond, ytm: float, price: float) -> float:
        """Calculate Macaulay duration for a bond"""
        try:
            # Simplified duration calculation
            time_to_maturity = (bond.maturity_date - datetime.utcnow()).days / 365.0
            
            if time_to_maturity <= 0:
                return 0.0
            
            # For zero-coupon bonds or very short maturities
            if bond.coupon_rate == 0 or time_to_maturity < 0.1:
                return time_to_maturity
            
            # Simplified duration formula
            duration = time_to_maturity * 0.8  # Approximate for coupon bonds
            
            return duration
            
        except Exception as e:
            logger.error(f"Error calculating duration: {e}")
            return 0.0
    
    def _calculate_convexity(self, bond: Bond, ytm: float, price: float) -> float:
        """Calculate convexity for a bond"""
        try:
            # Simplified convexity calculation
            time_to_maturity = (bond.maturity_date - datetime.utcnow()).days / 365.0
            
            if time_to_maturity <= 0:
                return 0.0
            
            # Simplified convexity formula
            convexity = time_to_maturity ** 2 * 0.5
            
            return convexity
            
        except Exception as e:
            logger.error(f"Error calculating convexity: {e}")
            return 0.0
    
    def _tenor_to_years(self, tenor: str) -> float:
        """Convert tenor string to years"""
        try:
            if 'M' in tenor:
                return int(tenor.replace('M', '')) / 12
            elif 'Y' in tenor:
                return int(tenor.replace('Y', ''))
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Error converting tenor to years: {e}")
            return 0.0
    
    def _classify_curve_shape(self, yields: List[float]) -> str:
        """Classify yield curve shape"""
        try:
            if len(yields) < 2:
                return 'unknown'
            
            if yields[-1] > yields[0]:
                return 'normal'
            elif yields[-1] < yields[0]:
                return 'inverted'
            else:
                return 'flat'
                
        except Exception as e:
            logger.error(f"Error classifying curve shape: {e}")
            return 'unknown'
    
    async def _get_historical_curve_comparison(self, curve_type: str, currency: str) -> Dict[str, Any]:
        """Get historical yield curve comparison"""
        try:
            # Get historical curves for comparison
            historical_curves = [c for c in self.yield_curves.values() 
                               if c.curve_type == curve_type and c.currency == currency]
            
            if len(historical_curves) < 2:
                return {}
            
            # Sort by date
            historical_curves.sort(key=lambda x: x.curve_date)
            
            # Get current and previous curves
            current_curve = historical_curves[-1]
            previous_curve = historical_curves[-2]
            
            # Calculate changes
            yield_changes = [current - previous for current, previous in zip(current_curve.yields, previous_curve.yields)]
            
            return {
                'previous_curve_date': previous_curve.curve_date.isoformat(),
                'yield_changes': yield_changes,
                'max_change': max(yield_changes) if yield_changes else 0,
                'min_change': min(yield_changes) if yield_changes else 0,
                'avg_change': sum(yield_changes) / len(yield_changes) if yield_changes else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting historical curve comparison: {e}")
            return {}
    
    # Background tasks
    async def _update_bond_prices(self):
        """Update bond prices periodically"""
        while True:
            try:
                # Update prices for all bonds
                for bond in self.bonds.values():
                    if bond.is_active:
                        # Simulate price updates (in practice, this would fetch from data providers)
                        current_price = await self._get_current_bond_price(bond.bond_id)
                        if current_price:
                            # Add small random price movement
                            movement = np.random.normal(0, 0.5)  # 0.5% volatility
                            new_clean_price = current_price.clean_price * (1 + movement / 100)
                            new_ytm = current_price.yield_to_maturity + np.random.normal(0, 0.1)
                            
                            await self.add_bond_price(
                                bond.bond_id, new_clean_price, new_ytm,
                                new_clean_price * 0.999, new_clean_price * 1.001,
                                new_ytm + 0.05, new_ytm - 0.05,
                                current_price.spread_to_treasury, 'simulated'
                            )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error updating bond prices: {e}")
                await asyncio.sleep(600)
    
    async def _update_positions(self):
        """Update bond positions"""
        while True:
            try:
                for position in self.bond_positions.values():
                    # Get current price
                    current_price = await self._get_current_bond_price(position.bond_id)
                    if current_price:
                        position.current_price = current_price.clean_price
                        position.yield_to_maturity = current_price.yield_to_maturity
                        position.duration_exposure = current_price.duration * position.quantity
                        
                        # Calculate unrealized P&L
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                        
                        # Calculate accrued interest
                        bond = self.bonds.get(position.bond_id)
                        if bond:
                            position.accrued_interest = self._calculate_accrued_interest(bond) * position.quantity
                        
                        position.last_updated = datetime.utcnow()
                        await self._cache_bond_position(position)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating positions: {e}")
                await asyncio.sleep(120)
    
    async def _update_yield_curves(self):
        """Update yield curves periodically"""
        while True:
            try:
                # Update yield curves (in practice, this would fetch from data providers)
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating yield curves: {e}")
                await asyncio.sleep(7200)
    
    async def _monitor_credit_ratings(self):
        """Monitor credit ratings"""
        while True:
            try:
                # Monitor credit ratings (in practice, this would check for rating changes)
                await asyncio.sleep(86400)  # Check daily
                
            except Exception as e:
                logger.error(f"Error monitoring credit ratings: {e}")
                await asyncio.sleep(172800)
    
    async def _get_current_bond_price(self, bond_id: str) -> Optional[BondPrice]:
        """Get current bond price"""
        try:
            prices = self.bond_prices.get(bond_id, [])
            if prices:
                return prices[-1]
            return None
            
        except Exception as e:
            logger.error(f"Error getting current bond price: {e}")
            return None
    
    # Helper methods (implementations would depend on your data models)
    async def _load_bonds(self):
        """Load bonds from database"""
        pass
    
    async def _load_yield_curves(self):
        """Load yield curves from database"""
        pass
    
    async def _load_credit_spreads(self):
        """Load credit spreads from database"""
        pass
    
    # Caching methods
    async def _cache_bond(self, bond: Bond):
        """Cache bond"""
        try:
            cache_key = f"bond:{bond.bond_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'issuer': bond.issuer,
                    'bond_type': bond.bond_type,
                    'cusip': bond.cusip,
                    'isin': bond.isin,
                    'face_value': bond.face_value,
                    'coupon_rate': bond.coupon_rate,
                    'coupon_frequency': bond.coupon_frequency,
                    'maturity_date': bond.maturity_date.isoformat(),
                    'issue_date': bond.issue_date.isoformat(),
                    'callable': bond.callable,
                    'putable': bond.putable,
                    'convertible': bond.convertible,
                    'credit_rating': bond.credit_rating,
                    'currency': bond.currency,
                    'is_active': bond.is_active,
                    'created_at': bond.created_at.isoformat(),
                    'last_updated': bond.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching bond: {e}")
    
    async def _cache_bond_price(self, price: BondPrice):
        """Cache bond price"""
        try:
            cache_key = f"bond_price:{price.price_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps({
                    'bond_id': price.bond_id,
                    'clean_price': price.clean_price,
                    'dirty_price': price.dirty_price,
                    'yield_to_maturity': price.yield_to_maturity,
                    'yield_to_call': price.yield_to_call,
                    'yield_to_put': price.yield_to_put,
                    'current_yield': price.current_yield,
                    'bid_price': price.bid_price,
                    'ask_price': price.ask_price,
                    'bid_yield': price.bid_yield,
                    'ask_yield': price.ask_yield,
                    'spread_to_treasury': price.spread_to_treasury,
                    'duration': price.duration,
                    'modified_duration': price.modified_duration,
                    'convexity': price.convexity,
                    'timestamp': price.timestamp.isoformat(),
                    'source': price.source
                })
            )
        except Exception as e:
            logger.error(f"Error caching bond price: {e}")
    
    async def _cache_bond_position(self, position: BondPosition):
        """Cache bond position"""
        try:
            cache_key = f"bond_position:{position.position_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps({
                    'user_id': position.user_id,
                    'bond_id': position.bond_id,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'face_value': position.face_value,
                    'coupon_income': position.coupon_income,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'accrued_interest': position.accrued_interest,
                    'yield_to_maturity': position.yield_to_maturity,
                    'duration_exposure': position.duration_exposure,
                    'created_at': position.created_at.isoformat(),
                    'last_updated': position.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching bond position: {e}")
    
    async def _cache_yield_curve(self, curve: YieldCurve):
        """Cache yield curve"""
        try:
            cache_key = f"yield_curve:{curve.curve_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'curve_type': curve.curve_type,
                    'currency': curve.currency,
                    'tenors': curve.tenors,
                    'yields': curve.yields,
                    'spreads': curve.spreads,
                    'curve_date': curve.curve_date.isoformat(),
                    'created_at': curve.created_at.isoformat(),
                    'last_updated': curve.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching yield curve: {e}")


# Factory function
async def get_fixed_income_trading_service(redis_client: redis.Redis, db_session: Session) -> FixedIncomeTradingService:
    """Get fixed income trading service instance"""
    service = FixedIncomeTradingService(redis_client, db_session)
    await service.initialize()
    return service
