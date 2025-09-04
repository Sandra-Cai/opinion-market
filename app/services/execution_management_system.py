"""
Execution Management System (EMS)
Advanced execution algorithms and smart order routing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Execution algorithms"""
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"
    ICEBERG = "iceberg"
    PEG = "peg"
    HIDDEN = "hidden"
    DISPLAY = "display"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"


class ExecutionStrategy(Enum):
    """Execution strategies"""
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    NEUTRAL = "neutral"
    ADAPTIVE = "adaptive"


class VenueType(Enum):
    """Venue types"""
    EXCHANGE = "exchange"
    ECN = "ecn"
    DARK_POOL = "dark_pool"
    MARKET_MAKER = "market_maker"
    INTERNAL = "internal"
    CROSSING_NETWORK = "crossing_network"


@dataclass
class ExecutionOrder:
    """Execution order"""
    execution_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    algorithm: ExecutionAlgorithm
    strategy: ExecutionStrategy
    parameters: Dict[str, Any]
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_price: float
    venues: List[str]
    created_at: datetime
    last_updated: datetime


@dataclass
class ExecutionSlice:
    """Execution slice"""
    slice_id: str
    execution_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    venue: str
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_price: float
    start_time: datetime
    end_time: Optional[datetime]
    created_at: datetime


@dataclass
class Venue:
    """Trading venue"""
    venue_id: str
    name: str
    venue_type: VenueType
    is_active: bool
    latency_ms: float
    commission_rate: float
    min_order_size: float
    max_order_size: float
    supported_algorithms: List[ExecutionAlgorithm]
    market_data_feed: str
    order_routing: str
    created_at: datetime
    last_updated: datetime


@dataclass
class ExecutionMetrics:
    """Execution metrics"""
    execution_id: str
    symbol: str
    total_quantity: float
    filled_quantity: float
    average_price: float
    benchmark_price: float
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    opportunity_cost: float
    total_cost: float
    vwap_deviation: float
    participation_rate: float
    fill_rate: float
    execution_time: float
    created_at: datetime


class ExecutionManagementSystem:
    """Advanced Execution Management System"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        
        # Execution management
        self.executions: Dict[str, ExecutionOrder] = {}
        self.slices: Dict[str, List[ExecutionSlice]] = defaultdict(list)
        self.venues: Dict[str, Venue] = {}
        self.metrics: Dict[str, ExecutionMetrics] = {}
        
        # Market data
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.volume_profiles: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Algorithms
        self.algorithm_engines: Dict[ExecutionAlgorithm, Any] = {}
        
        # Performance tracking
        self.execution_performance: Dict[str, Dict[str, float]] = {}
        self.venue_performance: Dict[str, Dict[str, float]] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the Execution Management System"""
        logger.info("Initializing Execution Management System")
        
        # Load venues
        await self._load_venues()
        
        # Initialize algorithms
        await self._initialize_algorithms()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._process_executions()),
            asyncio.create_task(self._update_market_data()),
            asyncio.create_task(self._monitor_venue_performance()),
            asyncio.create_task(self._calculate_metrics()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        logger.info("Execution Management System initialized successfully")
    
    async def create_execution(self, parent_order_id: str, symbol: str, side: str,
                              quantity: float, algorithm: ExecutionAlgorithm,
                              strategy: ExecutionStrategy = ExecutionStrategy.NEUTRAL,
                              parameters: Optional[Dict[str, Any]] = None) -> ExecutionOrder:
        """Create a new execution order"""
        try:
            execution_id = f"EXEC_{uuid.uuid4().hex[:12].upper()}"
            
            execution = ExecutionOrder(
                execution_id=execution_id,
                parent_order_id=parent_order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=None,
                algorithm=algorithm,
                strategy=strategy,
                parameters=parameters or {},
                status='pending',
                filled_quantity=0.0,
                remaining_quantity=quantity,
                average_price=0.0,
                venues=[],
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Select venues
            execution.venues = await self._select_venues(execution)
            
            # Store execution
            self.executions[execution_id] = execution
            
            # Start execution
            asyncio.create_task(self._start_execution(execution))
            
            logger.info(f"Created execution {execution_id} for order {parent_order_id}")
            return execution
            
        except Exception as e:
            logger.error(f"Error creating execution: {e}")
            raise
    
    async def get_execution(self, execution_id: str) -> Optional[ExecutionOrder]:
        """Get execution details"""
        try:
            return self.executions.get(execution_id)
            
        except Exception as e:
            logger.error(f"Error getting execution: {e}")
            return None
    
    async def get_execution_slices(self, execution_id: str) -> List[ExecutionSlice]:
        """Get execution slices"""
        try:
            return self.slices.get(execution_id, [])
            
        except Exception as e:
            logger.error(f"Error getting execution slices: {e}")
            return []
    
    async def get_execution_metrics(self, execution_id: str) -> Optional[ExecutionMetrics]:
        """Get execution metrics"""
        try:
            return self.metrics.get(execution_id)
            
        except Exception as e:
            logger.error(f"Error getting execution metrics: {e}")
            return None
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel execution"""
        try:
            if execution_id not in self.executions:
                return False
            
            execution = self.executions[execution_id]
            execution.status = 'cancelled'
            execution.last_updated = datetime.utcnow()
            
            # Cancel all slices
            for slice_obj in self.slices.get(execution_id, []):
                slice_obj.status = 'cancelled'
                slice_obj.end_time = datetime.utcnow()
            
            logger.info(f"Cancelled execution {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling execution: {e}")
            return False
    
    async def _select_venues(self, execution: ExecutionOrder) -> List[str]:
        """Select optimal venues for execution"""
        try:
            selected_venues = []
            
            # Filter venues by algorithm support
            for venue_id, venue in self.venues.items():
                if not venue.is_active:
                    continue
                
                if execution.algorithm in venue.supported_algorithms:
                    selected_venues.append(venue_id)
            
            # Sort by performance and latency
            selected_venues.sort(key=lambda v: (
                self.venue_performance.get(v, {}).get('fill_rate', 0),
                -self.venues[v].latency_ms
            ), reverse=True)
            
            return selected_venues[:3]  # Top 3 venues
            
        except Exception as e:
            logger.error(f"Error selecting venues: {e}")
            return []
    
    async def _start_execution(self, execution: ExecutionOrder):
        """Start execution"""
        try:
            execution.status = 'active'
            execution.last_updated = datetime.utcnow()
            
            # Create initial slices based on algorithm
            if execution.algorithm == ExecutionAlgorithm.TWAP:
                await self._create_twap_slices(execution)
            elif execution.algorithm == ExecutionAlgorithm.VWAP:
                await self._create_vwap_slices(execution)
            elif execution.algorithm == ExecutionAlgorithm.POV:
                await self._create_pov_slices(execution)
            else:
                await self._create_default_slices(execution)
            
            logger.info(f"Started execution {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Error starting execution: {e}")
            execution.status = 'error'
            execution.last_updated = datetime.utcnow()
    
    async def _create_twap_slices(self, execution: ExecutionOrder):
        """Create TWAP slices"""
        try:
            duration_minutes = execution.parameters.get('duration_minutes', 60)
            num_slices = execution.parameters.get('num_slices', 12)
            
            slice_quantity = execution.quantity / num_slices
            slice_interval = duration_minutes / num_slices
            
            for i in range(num_slices):
                slice_obj = ExecutionSlice(
                    slice_id=f"SLICE_{uuid.uuid4().hex[:12].upper()}",
                    execution_id=execution.execution_id,
                    symbol=execution.symbol,
                    side=execution.side,
                    quantity=slice_quantity,
                    price=None,
                    venue=execution.venues[i % len(execution.venues)] if execution.venues else 'default',
                    status='pending',
                    filled_quantity=0.0,
                    remaining_quantity=slice_quantity,
                    average_price=0.0,
                    start_time=datetime.utcnow() + timedelta(minutes=i * slice_interval),
                    end_time=None,
                    created_at=datetime.utcnow()
                )
                
                if execution.execution_id not in self.slices:
                    self.slices[execution.execution_id] = []
                
                self.slices[execution.execution_id].append(slice_obj)
            
        except Exception as e:
            logger.error(f"Error creating TWAP slices: {e}")
    
    async def _create_vwap_slices(self, execution: ExecutionOrder):
        """Create VWAP slices"""
        try:
            # Get volume profile
            volume_profile = self.volume_profiles.get(execution.symbol, [])
            
            if not volume_profile:
                # Default to TWAP if no volume profile
                await self._create_twap_slices(execution)
                return
            
            # Create slices based on volume profile
            total_volume = sum(vol for _, vol in volume_profile)
            remaining_quantity = execution.quantity
            
            for timestamp, volume in volume_profile:
                if remaining_quantity <= 0:
                    break
                
                # Calculate slice size based on volume
                slice_quantity = min(remaining_quantity, execution.quantity * (volume / total_volume))
                
                slice_obj = ExecutionSlice(
                    slice_id=f"SLICE_{uuid.uuid4().hex[:12].upper()}",
                    execution_id=execution.execution_id,
                    symbol=execution.symbol,
                    side=execution.side,
                    quantity=slice_quantity,
                    price=None,
                    venue=execution.venues[0] if execution.venues else 'default',
                    status='pending',
                    filled_quantity=0.0,
                    remaining_quantity=slice_quantity,
                    average_price=0.0,
                    start_time=timestamp,
                    end_time=None,
                    created_at=datetime.utcnow()
                )
                
                if execution.execution_id not in self.slices:
                    self.slices[execution.execution_id] = []
                
                self.slices[execution.execution_id].append(slice_obj)
                remaining_quantity -= slice_quantity
            
        except Exception as e:
            logger.error(f"Error creating VWAP slices: {e}")
    
    async def _create_pov_slices(self, execution: ExecutionOrder):
        """Create POV slices"""
        try:
            participation_rate = execution.parameters.get('participation_rate', 0.1)  # 10%
            duration_minutes = execution.parameters.get('duration_minutes', 60)
            
            # Create slices based on market volume
            slice_interval = 5  # 5 minutes
            num_slices = duration_minutes // slice_interval
            
            for i in range(num_slices):
                # Calculate expected market volume for this period
                expected_volume = await self._get_expected_volume(execution.symbol, slice_interval)
                slice_quantity = expected_volume * participation_rate
                
                slice_obj = ExecutionSlice(
                    slice_id=f"SLICE_{uuid.uuid4().hex[:12].upper()}",
                    execution_id=execution.execution_id,
                    symbol=execution.symbol,
                    side=execution.side,
                    quantity=slice_quantity,
                    price=None,
                    venue=execution.venues[i % len(execution.venues)] if execution.venues else 'default',
                    status='pending',
                    filled_quantity=0.0,
                    remaining_quantity=slice_quantity,
                    average_price=0.0,
                    start_time=datetime.utcnow() + timedelta(minutes=i * slice_interval),
                    end_time=None,
                    created_at=datetime.utcnow()
                )
                
                if execution.execution_id not in self.slices:
                    self.slices[execution.execution_id] = []
                
                self.slices[execution.execution_id].append(slice_obj)
            
        except Exception as e:
            logger.error(f"Error creating POV slices: {e}")
    
    async def _create_default_slices(self, execution: ExecutionOrder):
        """Create default slices"""
        try:
            slice_obj = ExecutionSlice(
                slice_id=f"SLICE_{uuid.uuid4().hex[:12].upper()}",
                execution_id=execution.execution_id,
                symbol=execution.symbol,
                side=execution.side,
                quantity=execution.quantity,
                price=execution.price,
                venue=execution.venues[0] if execution.venues else 'default',
                status='pending',
                filled_quantity=0.0,
                remaining_quantity=execution.quantity,
                average_price=0.0,
                start_time=datetime.utcnow(),
                end_time=None,
                created_at=datetime.utcnow()
            )
            
            if execution.execution_id not in self.slices:
                self.slices[execution.execution_id] = []
            
            self.slices[execution.execution_id].append(slice_obj)
            
        except Exception as e:
            logger.error(f"Error creating default slices: {e}")
    
    async def _get_expected_volume(self, symbol: str, period_minutes: int) -> float:
        """Get expected volume for period"""
        try:
            # Simple volume estimation
            # In practice, this would use historical volume data
            return 10000.0 * (period_minutes / 60)  # 10k shares per hour
            
        except Exception as e:
            logger.error(f"Error getting expected volume: {e}")
            return 1000.0
    
    # Background tasks
    async def _process_executions(self):
        """Process active executions"""
        while True:
            try:
                for execution in self.executions.values():
                    if execution.status == 'active':
                        await self._process_execution(execution)
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error processing executions: {e}")
                await asyncio.sleep(5)
    
    async def _process_execution(self, execution: ExecutionOrder):
        """Process individual execution"""
        try:
            slices = self.slices.get(execution.execution_id, [])
            
            for slice_obj in slices:
                if slice_obj.status == 'pending' and slice_obj.start_time <= datetime.utcnow():
                    await self._execute_slice(slice_obj)
                elif slice_obj.status == 'active':
                    await self._monitor_slice(slice_obj)
            
            # Update execution status
            total_filled = sum(s.filled_quantity for s in slices)
            execution.filled_quantity = total_filled
            execution.remaining_quantity = execution.quantity - total_filled
            
            if execution.remaining_quantity <= 0:
                execution.status = 'completed'
            elif all(s.status in ['completed', 'cancelled'] for s in slices):
                execution.status = 'completed'
            
            execution.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error processing execution: {e}")
    
    async def _execute_slice(self, slice_obj: ExecutionSlice):
        """Execute a slice"""
        try:
            slice_obj.status = 'active'
            
            # Simulate slice execution
            if np.random.random() < 0.8:  # 80% fill rate
                fill_quantity = slice_obj.remaining_quantity * np.random.uniform(0.5, 1.0)
                fill_price = await self._get_market_price(slice_obj.symbol)
                
                slice_obj.filled_quantity += fill_quantity
                slice_obj.remaining_quantity -= fill_quantity
                slice_obj.average_price = fill_price
                
                if slice_obj.remaining_quantity <= 0:
                    slice_obj.status = 'completed'
                    slice_obj.end_time = datetime.utcnow()
            
            logger.info(f"Executed slice {slice_obj.slice_id}")
            
        except Exception as e:
            logger.error(f"Error executing slice: {e}")
    
    async def _monitor_slice(self, slice_obj: ExecutionSlice):
        """Monitor active slice"""
        try:
            # Check if slice should be completed
            if slice_obj.start_time + timedelta(minutes=5) <= datetime.utcnow():
                if slice_obj.remaining_quantity > 0:
                    # Force complete remaining quantity
                    fill_price = await self._get_market_price(slice_obj.symbol)
                    slice_obj.filled_quantity += slice_obj.remaining_quantity
                    slice_obj.remaining_quantity = 0
                    slice_obj.average_price = fill_price
                    slice_obj.status = 'completed'
                    slice_obj.end_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error monitoring slice: {e}")
    
    async def _get_market_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            market_data = self.market_data.get(symbol, {})
            return market_data.get('last_price', 100.0)
            
        except Exception as e:
            logger.error(f"Error getting market price: {e}")
            return 100.0
    
    async def _update_market_data(self):
        """Update market data"""
        while True:
            try:
                # Simulate market data updates
                for symbol in ['AAPL', 'GOOGL', 'MSFT', 'TSLA']:
                    self.market_data[symbol] = {
                        'last_price': 100.0 + np.random.normal(0, 2),
                        'volume': np.random.uniform(10000, 100000),
                        'timestamp': datetime.utcnow()
                    }
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_venue_performance(self):
        """Monitor venue performance"""
        while True:
            try:
                # Update venue performance metrics
                for venue_id, venue in self.venues.items():
                    if venue_id not in self.venue_performance:
                        self.venue_performance[venue_id] = {}
                    
                    # Simulate performance metrics
                    self.venue_performance[venue_id].update({
                        'fill_rate': np.random.uniform(0.7, 0.95),
                        'latency': venue.latency_ms,
                        'commission': venue.commission_rate,
                        'last_updated': datetime.utcnow()
                    })
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error monitoring venue performance: {e}")
                await asyncio.sleep(120)
    
    async def _calculate_metrics(self):
        """Calculate execution metrics"""
        while True:
            try:
                for execution_id, execution in self.executions.items():
                    if execution.status == 'completed':
                        await self._calculate_execution_metrics(execution)
                
                await asyncio.sleep(30)  # Calculate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                await asyncio.sleep(60)
    
    async def _calculate_execution_metrics(self, execution: ExecutionOrder):
        """Calculate metrics for completed execution"""
        try:
            slices = self.slices.get(execution.execution_id, [])
            
            if not slices:
                return
            
            # Calculate metrics
            total_quantity = execution.quantity
            filled_quantity = execution.filled_quantity
            average_price = execution.average_price
            
            # Benchmark price (VWAP)
            benchmark_price = await self._calculate_benchmark_price(execution.symbol, execution.created_at)
            
            # Implementation shortfall
            implementation_shortfall = (average_price - benchmark_price) / benchmark_price if benchmark_price > 0 else 0
            
            # Market impact
            market_impact = abs(implementation_shortfall) * 0.5  # Simplified
            
            # Timing cost
            timing_cost = abs(implementation_shortfall) * 0.3  # Simplified
            
            # Opportunity cost
            opportunity_cost = abs(implementation_shortfall) * 0.2  # Simplified
            
            # Total cost
            total_cost = market_impact + timing_cost + opportunity_cost
            
            # VWAP deviation
            vwap_deviation = abs(average_price - benchmark_price) / benchmark_price if benchmark_price > 0 else 0
            
            # Participation rate
            participation_rate = filled_quantity / total_quantity if total_quantity > 0 else 0
            
            # Fill rate
            fill_rate = filled_quantity / total_quantity if total_quantity > 0 else 0
            
            # Execution time
            execution_time = (datetime.utcnow() - execution.created_at).total_seconds()
            
            metrics = ExecutionMetrics(
                execution_id=execution.execution_id,
                symbol=execution.symbol,
                total_quantity=total_quantity,
                filled_quantity=filled_quantity,
                average_price=average_price,
                benchmark_price=benchmark_price,
                implementation_shortfall=implementation_shortfall,
                market_impact=market_impact,
                timing_cost=timing_cost,
                opportunity_cost=opportunity_cost,
                total_cost=total_cost,
                vwap_deviation=vwap_deviation,
                participation_rate=participation_rate,
                fill_rate=fill_rate,
                execution_time=execution_time,
                created_at=datetime.utcnow()
            )
            
            self.metrics[execution.execution_id] = metrics
            
            logger.info(f"Calculated metrics for execution {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"Error calculating execution metrics: {e}")
    
    async def _calculate_benchmark_price(self, symbol: str, start_time: datetime) -> float:
        """Calculate benchmark price (VWAP)"""
        try:
            # Simple VWAP calculation
            # In practice, this would use actual market data
            return 100.0 + np.random.normal(0, 1)
            
        except Exception as e:
            logger.error(f"Error calculating benchmark price: {e}")
            return 100.0
    
    async def _cleanup_old_data(self):
        """Cleanup old data"""
        while True:
            try:
                # Cleanup old executions (older than 7 days)
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                executions_to_remove = []
                for execution_id, execution in self.executions.items():
                    if execution.created_at < cutoff_date and execution.status in ['completed', 'cancelled', 'error']:
                        executions_to_remove.append(execution_id)
                
                for execution_id in executions_to_remove:
                    del self.executions[execution_id]
                    if execution_id in self.slices:
                        del self.slices[execution_id]
                    if execution_id in self.metrics:
                        del self.metrics[execution_id]
                
                if executions_to_remove:
                    logger.info(f"Cleaned up {len(executions_to_remove)} old executions")
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(7200)
    
    # Helper methods
    async def _load_venues(self):
        """Load trading venues"""
        try:
            # Create default venues
            venues = [
                Venue(
                    venue_id="NYSE",
                    name="New York Stock Exchange",
                    venue_type=VenueType.EXCHANGE,
                    is_active=True,
                    latency_ms=5.0,
                    commission_rate=0.001,
                    min_order_size=1.0,
                    max_order_size=1000000.0,
                    supported_algorithms=[ExecutionAlgorithm.TWAP, ExecutionAlgorithm.VWAP, ExecutionAlgorithm.POV],
                    market_data_feed="NYSE_FEED",
                    order_routing="NYSE_ROUTER",
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                ),
                Venue(
                    venue_id="NASDAQ",
                    name="NASDAQ",
                    venue_type=VenueType.EXCHANGE,
                    is_active=True,
                    latency_ms=3.0,
                    commission_rate=0.0008,
                    min_order_size=1.0,
                    max_order_size=1000000.0,
                    supported_algorithms=[ExecutionAlgorithm.TWAP, ExecutionAlgorithm.VWAP, ExecutionAlgorithm.POV],
                    market_data_feed="NASDAQ_FEED",
                    order_routing="NASDAQ_ROUTER",
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                ),
                Venue(
                    venue_id="DARK_POOL_1",
                    name="Dark Pool 1",
                    venue_type=VenueType.DARK_POOL,
                    is_active=True,
                    latency_ms=10.0,
                    commission_rate=0.0005,
                    min_order_size=1000.0,
                    max_order_size=5000000.0,
                    supported_algorithms=[ExecutionAlgorithm.TWAP, ExecutionAlgorithm.VWAP],
                    market_data_feed="DARK_FEED",
                    order_routing="DARK_ROUTER",
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
            ]
            
            for venue in venues:
                self.venues[venue.venue_id] = venue
            
            logger.info(f"Loaded {len(venues)} venues")
            
        except Exception as e:
            logger.error(f"Error loading venues: {e}")
    
    async def _initialize_algorithms(self):
        """Initialize execution algorithms"""
        try:
            # Initialize algorithm engines
            self.algorithm_engines = {
                ExecutionAlgorithm.TWAP: "TWAP Engine",
                ExecutionAlgorithm.VWAP: "VWAP Engine",
                ExecutionAlgorithm.POV: "POV Engine",
                ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: "IS Engine",
                ExecutionAlgorithm.ADAPTIVE: "Adaptive Engine"
            }
            
            logger.info(f"Initialized {len(self.algorithm_engines)} algorithms")
            
        except Exception as e:
            logger.error(f"Error initializing algorithms: {e}")


# Factory function
async def get_execution_management_system(redis_client: redis.Redis, db_session: Session) -> ExecutionManagementSystem:
    """Get Execution Management System instance"""
    ems = ExecutionManagementSystem(redis_client, db_session)
    await ems.initialize()
    return ems
