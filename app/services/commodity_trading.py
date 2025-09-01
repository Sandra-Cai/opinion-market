"""
Commodity Trading Service
Provides spot trading, futures, physical delivery, storage, and commodity-specific analytics
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
class Commodity:
    """Commodity information"""
    commodity_id: str
    name: str
    category: str  # 'energy', 'metals', 'agriculture', 'softs'
    unit: str  # 'barrel', 'ton', 'bushel', 'ounce'
    tick_size: float
    contract_size: float
    delivery_months: List[str]
    trading_hours: str
    exchange: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class CommodityPrice:
    """Commodity price information"""
    commodity_id: str
    price_type: str  # 'spot', 'futures', 'forward'
    price: float
    currency: str
    timestamp: datetime
    source: str
    location: Optional[str]
    quality_specifications: Optional[Dict[str, Any]]


@dataclass
class StorageFacility:
    """Storage facility information"""
    facility_id: str
    name: str
    location: str
    commodity_types: List[str]
    total_capacity: float
    available_capacity: float
    storage_cost_per_unit: float
    storage_cost_currency: str
    insurance_cost_per_unit: float
    delivery_terms: Dict[str, Any]
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class PhysicalDelivery:
    """Physical delivery information"""
    delivery_id: str
    contract_id: str
    buyer_id: int
    seller_id: int
    commodity_id: str
    quantity: float
    delivery_date: datetime
    delivery_location: str
    quality_specifications: Dict[str, Any]
    delivery_instructions: str
    status: str  # 'pending', 'in_transit', 'delivered', 'completed'
    inspection_report: Optional[Dict[str, Any]]
    created_at: datetime
    last_updated: datetime


@dataclass
class CommodityInventory:
    """Commodity inventory information"""
    inventory_id: str
    user_id: int
    commodity_id: str
    quantity: float
    storage_facility_id: str
    acquisition_cost: float
    current_value: float
    storage_cost: float
    insurance_cost: float
    total_cost: float
    last_valuation: datetime
    created_at: datetime
    last_updated: datetime


@dataclass
class CommoditySpread:
    """Commodity spread information"""
    spread_id: str
    commodity_id: str
    spread_type: str  # 'calendar', 'inter_commodity', 'location', 'quality'
    leg1_contract: str
    leg2_contract: str
    spread_value: float
    spread_currency: str
    historical_spread: List[Dict[str, Any]]
    created_at: datetime
    last_updated: datetime


class CommodityTradingService:
    """Comprehensive commodity trading service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.commodities: Dict[str, Commodity] = {}
        self.commodity_prices: Dict[str, List[CommodityPrice]] = defaultdict(list)
        self.storage_facilities: Dict[str, StorageFacility] = {}
        self.physical_deliveries: Dict[str, PhysicalDelivery] = {}
        self.commodity_inventories: Dict[str, CommodityInventory] = {}
        self.commodity_spreads: Dict[str, CommoditySpread] = {}
        
        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.inventory_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Commodity-specific data
        self.seasonal_factors: Dict[str, List[float]] = {}
        self.weather_data: Dict[str, Dict[str, Any]] = {}
        self.supply_demand_data: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the commodity trading service"""
        logger.info("Initializing Commodity Trading Service")
        
        # Load existing data
        await self._load_commodities()
        await self._load_storage_facilities()
        await self._load_commodity_spreads()
        
        # Start background tasks
        asyncio.create_task(self._update_commodity_prices())
        asyncio.create_task(self._update_storage_capacity())
        asyncio.create_task(self._monitor_deliveries())
        asyncio.create_task(self._update_inventory_valuations())
        
        logger.info("Commodity Trading Service initialized successfully")
    
    async def create_commodity(self, name: str, category: str, unit: str, tick_size: float,
                              contract_size: float, delivery_months: List[str], trading_hours: str,
                              exchange: str) -> Commodity:
        """Create a new commodity"""
        try:
            commodity_id = f"commodity_{name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            commodity = Commodity(
                commodity_id=commodity_id,
                name=name,
                category=category,
                unit=unit,
                tick_size=tick_size,
                contract_size=contract_size,
                delivery_months=delivery_months,
                trading_hours=trading_hours,
                exchange=exchange,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.commodities[commodity_id] = commodity
            await self._cache_commodity(commodity)
            
            logger.info(f"Created commodity {commodity_id}")
            return commodity
            
        except Exception as e:
            logger.error(f"Error creating commodity: {e}")
            raise
    
    async def add_commodity_price(self, commodity_id: str, price_type: str, price: float,
                                 currency: str, source: str, location: Optional[str] = None,
                                 quality_specifications: Optional[Dict[str, Any]] = None) -> CommodityPrice:
        """Add a new commodity price"""
        try:
            if commodity_id not in self.commodities:
                raise ValueError(f"Commodity {commodity_id} not found")
            
            price_data = CommodityPrice(
                commodity_id=commodity_id,
                price_type=price_type,
                price=price,
                currency=currency,
                timestamp=datetime.utcnow(),
                source=source,
                location=location,
                quality_specifications=quality_specifications
            )
            
            self.commodity_prices[commodity_id].append(price_data)
            
            # Keep only recent prices
            if len(self.commodity_prices[commodity_id]) > 1000:
                self.commodity_prices[commodity_id] = self.commodity_prices[commodity_id][-1000:]
            
            await self._cache_commodity_price(price_data)
            
            logger.info(f"Added {price_type} price for commodity {commodity_id}: {price} {currency}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error adding commodity price: {e}")
            raise
    
    async def create_storage_facility(self, name: str, location: str, commodity_types: List[str],
                                    total_capacity: float, storage_cost_per_unit: float,
                                    storage_cost_currency: str, insurance_cost_per_unit: float,
                                    delivery_terms: Dict[str, Any]) -> StorageFacility:
        """Create a new storage facility"""
        try:
            facility_id = f"storage_{name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            facility = StorageFacility(
                facility_id=facility_id,
                name=name,
                location=location,
                commodity_types=commodity_types,
                total_capacity=total_capacity,
                available_capacity=total_capacity,
                storage_cost_per_unit=storage_cost_per_unit,
                storage_cost_currency=storage_cost_currency,
                insurance_cost_per_unit=insurance_cost_per_unit,
                delivery_terms=delivery_terms,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.storage_facilities[facility_id] = facility
            await self._cache_storage_facility(facility)
            
            logger.info(f"Created storage facility {facility_id}")
            return facility
            
        except Exception as e:
            logger.error(f"Error creating storage facility: {e}")
            raise
    
    async def create_physical_delivery(self, contract_id: str, buyer_id: int, seller_id: int,
                                     commodity_id: str, quantity: float, delivery_date: datetime,
                                     delivery_location: str, quality_specifications: Dict[str, Any],
                                     delivery_instructions: str) -> PhysicalDelivery:
        """Create a new physical delivery"""
        try:
            delivery_id = f"delivery_{contract_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            delivery = PhysicalDelivery(
                delivery_id=delivery_id,
                contract_id=contract_id,
                buyer_id=buyer_id,
                seller_id=seller_id,
                commodity_id=commodity_id,
                quantity=quantity,
                delivery_date=delivery_date,
                delivery_location=delivery_location,
                quality_specifications=quality_specifications,
                delivery_instructions=delivery_instructions,
                status='pending',
                inspection_report=None,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.physical_deliveries[delivery_id] = delivery
            await self._cache_physical_delivery(delivery)
            
            logger.info(f"Created physical delivery {delivery_id}")
            return delivery
            
        except Exception as e:
            logger.error(f"Error creating physical delivery: {e}")
            raise
    
    async def create_commodity_inventory(self, user_id: int, commodity_id: str, quantity: float,
                                       storage_facility_id: str, acquisition_cost: float) -> CommodityInventory:
        """Create a new commodity inventory"""
        try:
            if storage_facility_id not in self.storage_facilities:
                raise ValueError(f"Storage facility {storage_facility_id} not found")
            
            facility = self.storage_facilities[storage_facility_id]
            if facility.available_capacity < quantity:
                raise ValueError(f"Insufficient storage capacity. Available: {facility.available_capacity}, Required: {quantity}")
            
            inventory_id = f"inventory_{user_id}_{commodity_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate costs
            storage_cost = quantity * facility.storage_cost_per_unit
            insurance_cost = quantity * facility.insurance_cost_per_unit
            total_cost = acquisition_cost + storage_cost + insurance_cost
            
            # Get current commodity price for valuation
            current_price = await self._get_current_commodity_price(commodity_id)
            current_value = quantity * current_price if current_price else acquisition_cost
            
            inventory = CommodityInventory(
                inventory_id=inventory_id,
                user_id=user_id,
                commodity_id=commodity_id,
                quantity=quantity,
                storage_facility_id=storage_facility_id,
                acquisition_cost=acquisition_cost,
                current_value=current_value,
                storage_cost=storage_cost,
                insurance_cost=insurance_cost,
                total_cost=total_cost,
                last_valuation=datetime.utcnow(),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.commodity_inventories[inventory_id] = inventory
            
            # Update storage facility capacity
            facility.available_capacity -= quantity
            facility.last_updated = datetime.utcnow()
            await self._cache_storage_facility(facility)
            
            await self._cache_commodity_inventory(inventory)
            
            logger.info(f"Created commodity inventory {inventory_id}")
            return inventory
            
        except Exception as e:
            logger.error(f"Error creating commodity inventory: {e}")
            raise
    
    async def create_commodity_spread(self, commodity_id: str, spread_type: str, leg1_contract: str,
                                    leg2_contract: str, spread_value: float, spread_currency: str) -> CommoditySpread:
        """Create a new commodity spread"""
        try:
            spread_id = f"spread_{commodity_id}_{spread_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            spread = CommoditySpread(
                spread_id=spread_id,
                commodity_id=commodity_id,
                spread_type=spread_type,
                leg1_contract=leg1_contract,
                leg2_contract=leg2_contract,
                spread_value=spread_value,
                spread_currency=spread_currency,
                historical_spread=[],
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.commodity_spreads[spread_id] = spread
            await self._cache_commodity_spread(spread)
            
            logger.info(f"Created commodity spread {spread_id}")
            return spread
            
        except Exception as e:
            logger.error(f"Error creating commodity spread: {e}")
            raise
    
    async def calculate_storage_costs(self, commodity_id: str, quantity: float, storage_days: int,
                                    storage_facility_id: str) -> Dict[str, float]:
        """Calculate storage costs for a commodity"""
        try:
            facility = self.storage_facilities.get(storage_facility_id)
            if not facility:
                raise ValueError(f"Storage facility {storage_facility_id} not found")
            
            # Calculate daily costs
            daily_storage_cost = quantity * facility.storage_cost_per_unit
            daily_insurance_cost = quantity * facility.insurance_cost_per_unit
            
            # Calculate total costs
            total_storage_cost = daily_storage_cost * storage_days
            total_insurance_cost = daily_insurance_cost * storage_days
            total_cost = total_storage_cost + total_insurance_cost
            
            return {
                'daily_storage_cost': daily_storage_cost,
                'daily_insurance_cost': daily_insurance_cost,
                'total_storage_cost': total_storage_cost,
                'total_insurance_cost': total_insurance_cost,
                'total_cost': total_cost,
                'currency': facility.storage_cost_currency
            }
            
        except Exception as e:
            logger.error(f"Error calculating storage costs: {e}")
            raise
    
    async def calculate_carry_costs(self, commodity_id: str, spot_price: float, futures_price: float,
                                  storage_cost: float, insurance_cost: float, time_to_delivery: float) -> Dict[str, float]:
        """Calculate carry costs for commodity arbitrage"""
        try:
            # Calculate theoretical futures price
            risk_free_rate = 0.05  # 5% annual rate
            daily_rate = risk_free_rate / 365
            
            # Carry cost components
            financing_cost = spot_price * daily_rate * time_to_delivery
            total_carry_cost = storage_cost + insurance_cost + financing_cost
            
            # Theoretical futures price
            theoretical_futures = spot_price + total_carry_cost
            
            # Arbitrage opportunity
            arbitrage_profit = futures_price - theoretical_futures
            
            return {
                'spot_price': spot_price,
                'futures_price': futures_price,
                'theoretical_futures': theoretical_futures,
                'storage_cost': storage_cost,
                'insurance_cost': insurance_cost,
                'financing_cost': financing_cost,
                'total_carry_cost': total_carry_cost,
                'arbitrage_profit': arbitrage_profit,
                'time_to_delivery': time_to_delivery
            }
            
        except Exception as e:
            logger.error(f"Error calculating carry costs: {e}")
            raise
    
    async def get_commodity_analytics(self, commodity_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a commodity"""
        try:
            if commodity_id not in self.commodities:
                raise ValueError(f"Commodity {commodity_id} not found")
            
            commodity = self.commodities[commodity_id]
            prices = self.commodity_prices.get(commodity_id, [])
            
            if not prices:
                return {'commodity_id': commodity_id, 'error': 'No price data available'}
            
            # Calculate price statistics
            spot_prices = [p.price for p in prices if p.price_type == 'spot']
            futures_prices = [p.price for p in prices if p.price_type == 'futures']
            
            analytics = {
                'commodity_id': commodity_id,
                'commodity_name': commodity.name,
                'category': commodity.category,
                'current_spot_price': spot_prices[-1] if spot_prices else None,
                'current_futures_price': futures_prices[-1] if futures_prices else None,
                'price_statistics': {
                    'spot_mean': np.mean(spot_prices) if spot_prices else None,
                    'spot_std': np.std(spot_prices) if spot_prices else None,
                    'spot_min': np.min(spot_prices) if spot_prices else None,
                    'spot_max': np.max(spot_prices) if spot_prices else None,
                    'futures_mean': np.mean(futures_prices) if futures_prices else None,
                    'futures_std': np.std(futures_prices) if futures_prices else None
                },
                'price_trends': await self._calculate_price_trends(commodity_id),
                'seasonal_factors': self.seasonal_factors.get(commodity_id, []),
                'supply_demand': self.supply_demand_data.get(commodity_id, {}),
                'weather_impact': self.weather_data.get(commodity_id, {}),
                'storage_capacity': await self._get_storage_capacity(commodity_id),
                'delivery_schedule': await self._get_delivery_schedule(commodity_id),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting commodity analytics: {e}")
            raise
    
    async def get_storage_facilities(self, commodity_type: Optional[str] = None, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get storage facilities with optional filtering"""
        try:
            facilities = []
            for facility in self.storage_facilities.values():
                if not facility.is_active:
                    continue
                
                # Apply filters
                if commodity_type and commodity_type not in facility.commodity_types:
                    continue
                if location and location.lower() not in facility.location.lower():
                    continue
                
                facilities.append({
                    'facility_id': facility.facility_id,
                    'name': facility.name,
                    'location': facility.location,
                    'commodity_types': facility.commodity_types,
                    'total_capacity': facility.total_capacity,
                    'available_capacity': facility.available_capacity,
                    'utilization_rate': (facility.total_capacity - facility.available_capacity) / facility.total_capacity,
                    'storage_cost_per_unit': facility.storage_cost_per_unit,
                    'storage_cost_currency': facility.storage_cost_currency,
                    'insurance_cost_per_unit': facility.insurance_cost_per_unit,
                    'delivery_terms': facility.delivery_terms,
                    'last_updated': facility.last_updated.isoformat()
                })
            
            # Sort by utilization rate
            facilities.sort(key=lambda x: x['utilization_rate'], reverse=True)
            
            return facilities
            
        except Exception as e:
            logger.error(f"Error getting storage facilities: {e}")
            raise
    
    async def _get_current_commodity_price(self, commodity_id: str) -> Optional[float]:
        """Get current commodity price"""
        try:
            prices = self.commodity_prices.get(commodity_id, [])
            if prices:
                # Get most recent spot price
                spot_prices = [p for p in prices if p.price_type == 'spot']
                if spot_prices:
                    return spot_prices[-1].price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current commodity price: {e}")
            return None
    
    async def _calculate_price_trends(self, commodity_id: str) -> Dict[str, Any]:
        """Calculate price trends for a commodity"""
        try:
            prices = self.commodity_prices.get(commodity_id, [])
            if len(prices) < 2:
                return {}
            
            # Get recent prices
            recent_prices = prices[-30:]  # Last 30 prices
            price_values = [p.price for p in recent_prices]
            
            # Calculate trends
            if len(price_values) >= 2:
                price_change = price_values[-1] - price_values[0]
                price_change_percent = (price_change / price_values[0]) * 100 if price_values[0] != 0 else 0
                
                # Simple moving averages
                sma_5 = np.mean(price_values[-5:]) if len(price_values) >= 5 else None
                sma_10 = np.mean(price_values[-10:]) if len(price_values) >= 10 else None
                sma_20 = np.mean(price_values[-20:]) if len(price_values) >= 20 else None
                
                return {
                    'price_change': price_change,
                    'price_change_percent': price_change_percent,
                    'trend_direction': 'up' if price_change > 0 else 'down' if price_change < 0 else 'flat',
                    'sma_5': sma_5,
                    'sma_10': sma_10,
                    'sma_20': sma_20,
                    'volatility': np.std(price_values) if len(price_values) > 1 else 0
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating price trends: {e}")
            return {}
    
    async def _get_storage_capacity(self, commodity_id: str) -> Dict[str, Any]:
        """Get storage capacity for a commodity"""
        try:
            total_capacity = 0
            available_capacity = 0
            facilities_count = 0
            
            for facility in self.storage_facilities.values():
                if facility.is_active and commodity_id in facility.commodity_types:
                    total_capacity += facility.total_capacity
                    available_capacity += facility.available_capacity
                    facilities_count += 1
            
            utilization_rate = (total_capacity - available_capacity) / total_capacity if total_capacity > 0 else 0
            
            return {
                'total_capacity': total_capacity,
                'available_capacity': available_capacity,
                'utilized_capacity': total_capacity - available_capacity,
                'utilization_rate': utilization_rate,
                'facilities_count': facilities_count
            }
            
        except Exception as e:
            logger.error(f"Error getting storage capacity: {e}")
            return {}
    
    async def _get_delivery_schedule(self, commodity_id: str) -> List[Dict[str, Any]]:
        """Get delivery schedule for a commodity"""
        try:
            deliveries = []
            for delivery in self.physical_deliveries.values():
                if delivery.commodity_id == commodity_id:
                    deliveries.append({
                        'delivery_id': delivery.delivery_id,
                        'contract_id': delivery.contract_id,
                        'buyer_id': delivery.buyer_id,
                        'seller_id': delivery.seller_id,
                        'quantity': delivery.quantity,
                        'delivery_date': delivery.delivery_date.isoformat(),
                        'delivery_location': delivery.delivery_location,
                        'status': delivery.status,
                        'created_at': delivery.created_at.isoformat()
                    })
            
            # Sort by delivery date
            deliveries.sort(key=lambda x: x['delivery_date'])
            
            return deliveries
            
        except Exception as e:
            logger.error(f"Error getting delivery schedule: {e}")
            return []
    
    # Background tasks
    async def _update_commodity_prices(self):
        """Update commodity prices periodically"""
        while True:
            try:
                # Update prices for all commodities
                for commodity in self.commodities.values():
                    if commodity.is_active:
                        # Simulate price updates (in practice, this would fetch from data providers)
                        current_price = await self._get_current_commodity_price(commodity.commodity_id)
                        if current_price:
                            # Add small random price movement
                            new_price = current_price * (1 + np.random.normal(0, 0.01))
                            await self.add_commodity_price(
                                commodity.commodity_id, 'spot', new_price, 'USD', 'simulated'
                            )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error updating commodity prices: {e}")
                await asyncio.sleep(600)
    
    async def _update_storage_capacity(self):
        """Update storage capacity periodically"""
        while True:
            try:
                # Update storage facility capacities
                for facility in self.storage_facilities.values():
                    if facility.is_active:
                        # Simulate capacity changes (in practice, this would be real-time updates)
                        facility.last_updated = datetime.utcnow()
                        await self._cache_storage_facility(facility)
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating storage capacity: {e}")
                await asyncio.sleep(7200)
    
    async def _monitor_deliveries(self):
        """Monitor physical deliveries"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for delivery in self.physical_deliveries.values():
                    if delivery.status == 'pending' and delivery.delivery_date <= current_time:
                        # Update delivery status
                        delivery.status = 'in_transit'
                        delivery.last_updated = current_time
                        await self._cache_physical_delivery(delivery)
                        
                        logger.info(f"Updated delivery {delivery.delivery_id} status to in_transit")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring deliveries: {e}")
                await asyncio.sleep(3600)
    
    async def _update_inventory_valuations(self):
        """Update inventory valuations"""
        while True:
            try:
                for inventory in self.commodity_inventories.values():
                    # Update current value based on latest price
                    current_price = await self._get_current_commodity_price(inventory.commodity_id)
                    if current_price:
                        inventory.current_value = inventory.quantity * current_price
                        inventory.last_valuation = datetime.utcnow()
                        inventory.last_updated = datetime.utcnow()
                        
                        await self._cache_commodity_inventory(inventory)
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating inventory valuations: {e}")
                await asyncio.sleep(7200)
    
    # Helper methods (implementations would depend on your data models)
    async def _load_commodities(self):
        """Load commodities from database"""
        pass
    
    async def _load_storage_facilities(self):
        """Load storage facilities from database"""
        pass
    
    async def _load_commodity_spreads(self):
        """Load commodity spreads from database"""
        pass
    
    # Caching methods
    async def _cache_commodity(self, commodity: Commodity):
        """Cache commodity"""
        try:
            cache_key = f"commodity:{commodity.commodity_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'name': commodity.name,
                    'category': commodity.category,
                    'unit': commodity.unit,
                    'tick_size': commodity.tick_size,
                    'contract_size': commodity.contract_size,
                    'delivery_months': commodity.delivery_months,
                    'trading_hours': commodity.trading_hours,
                    'exchange': commodity.exchange,
                    'is_active': commodity.is_active,
                    'created_at': commodity.created_at.isoformat(),
                    'last_updated': commodity.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching commodity: {e}")
    
    async def _cache_commodity_price(self, price: CommodityPrice):
        """Cache commodity price"""
        try:
            cache_key = f"commodity_price:{price.commodity_id}:{price.timestamp.isoformat()}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps({
                    'price_type': price.price_type,
                    'price': price.price,
                    'currency': price.currency,
                    'timestamp': price.timestamp.isoformat(),
                    'source': price.source,
                    'location': price.location,
                    'quality_specifications': price.quality_specifications
                })
            )
        except Exception as e:
            logger.error(f"Error caching commodity price: {e}")
    
    async def _cache_storage_facility(self, facility: StorageFacility):
        """Cache storage facility"""
        try:
            cache_key = f"storage_facility:{facility.facility_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'name': facility.name,
                    'location': facility.location,
                    'commodity_types': facility.commodity_types,
                    'total_capacity': facility.total_capacity,
                    'available_capacity': facility.available_capacity,
                    'storage_cost_per_unit': facility.storage_cost_per_unit,
                    'storage_cost_currency': facility.storage_cost_currency,
                    'insurance_cost_per_unit': facility.insurance_cost_per_unit,
                    'delivery_terms': facility.delivery_terms,
                    'is_active': facility.is_active,
                    'created_at': facility.created_at.isoformat(),
                    'last_updated': facility.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching storage facility: {e}")
    
    async def _cache_physical_delivery(self, delivery: PhysicalDelivery):
        """Cache physical delivery"""
        try:
            cache_key = f"physical_delivery:{delivery.delivery_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'contract_id': delivery.contract_id,
                    'buyer_id': delivery.buyer_id,
                    'seller_id': delivery.seller_id,
                    'commodity_id': delivery.commodity_id,
                    'quantity': delivery.quantity,
                    'delivery_date': delivery.delivery_date.isoformat(),
                    'delivery_location': delivery.delivery_location,
                    'quality_specifications': delivery.quality_specifications,
                    'delivery_instructions': delivery.delivery_instructions,
                    'status': delivery.status,
                    'inspection_report': delivery.inspection_report,
                    'created_at': delivery.created_at.isoformat(),
                    'last_updated': delivery.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching physical delivery: {e}")
    
    async def _cache_commodity_inventory(self, inventory: CommodityInventory):
        """Cache commodity inventory"""
        try:
            cache_key = f"commodity_inventory:{inventory.inventory_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps({
                    'user_id': inventory.user_id,
                    'commodity_id': inventory.commodity_id,
                    'quantity': inventory.quantity,
                    'storage_facility_id': inventory.storage_facility_id,
                    'acquisition_cost': inventory.acquisition_cost,
                    'current_value': inventory.current_value,
                    'storage_cost': inventory.storage_cost,
                    'insurance_cost': inventory.insurance_cost,
                    'total_cost': inventory.total_cost,
                    'last_valuation': inventory.last_valuation.isoformat(),
                    'created_at': inventory.created_at.isoformat(),
                    'last_updated': inventory.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching commodity inventory: {e}")
    
    async def _cache_commodity_spread(self, spread: CommoditySpread):
        """Cache commodity spread"""
        try:
            cache_key = f"commodity_spread:{spread.spread_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'commodity_id': spread.commodity_id,
                    'spread_type': spread.spread_type,
                    'leg1_contract': spread.leg1_contract,
                    'leg2_contract': spread.leg2_contract,
                    'spread_value': spread.spread_value,
                    'spread_currency': spread.spread_currency,
                    'historical_spread': spread.historical_spread,
                    'created_at': spread.created_at.isoformat(),
                    'last_updated': spread.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching commodity spread: {e}")


# Factory function
async def get_commodity_trading_service(redis_client: redis.Redis, db_session: Session) -> CommodityTradingService:
    """Get commodity trading service instance"""
    service = CommodityTradingService(redis_client, db_session)
    await service.initialize()
    return service
