"""
Order Management System (OMS)
Comprehensive order management, routing, and execution for institutional trading
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"
    PEG = "peg"
    HIDDEN = "hidden"
    DISPLAY = "display"


class OrderSide(Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderStatus(Enum):
    """Order statuses"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"


class TimeInForce(Enum):
    """Time in force"""

    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date
    ATC = "atc"  # At the Close
    ATO = "ato"  # At the Open


class OrderRoute(Enum):
    """Order routing destinations"""

    DIRECT = "direct"
    SMART = "smart"
    ALGO = "algo"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    MARKET_MAKER = "market_maker"
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class Order:
    """Order representation"""

    order_id: str
    user_id: int
    account_id: str
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    limit_price: Optional[float]
    time_in_force: TimeInForce
    status: OrderStatus
    route: OrderRoute

    # Execution details
    filled_quantity: float
    remaining_quantity: float
    average_price: float
    total_filled_value: float

    # Algo order parameters
    algo_type: Optional[str]
    algo_parameters: Dict[str, Any]

    # Risk management
    max_position_size: Optional[float]
    max_order_value: Optional[float]
    risk_limits: Dict[str, float]

    # Timestamps
    created_at: datetime
    submitted_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    filled_at: Optional[datetime]
    cancelled_at: Optional[datetime]
    last_updated: datetime

    # Additional metadata
    client_order_id: Optional[str]
    parent_order_id: Optional[str]
    child_orders: List[str]
    tags: Dict[str, str]
    notes: Optional[str]


@dataclass
class OrderFill:
    """Order fill representation"""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fill_value: float
    commission: float
    venue: str
    execution_id: str
    fill_time: datetime
    settlement_date: Optional[datetime]
    created_at: datetime


@dataclass
class OrderBook:
    """Order book representation"""

    symbol: str
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    last_trade_price: Optional[float]
    last_trade_quantity: Optional[float]
    last_trade_time: Optional[datetime]
    volume: float
    timestamp: datetime


@dataclass
class MarketData:
    """Market data snapshot"""

    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    high_price: float
    low_price: float
    open_price: float
    timestamp: datetime


@dataclass
class ExecutionReport:
    """Execution report"""

    report_id: str
    order_id: str
    execution_type: str  # 'new', 'partial_fill', 'fill', 'cancel', 'reject'
    order_status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    average_price: float
    last_fill_price: Optional[float]
    last_fill_quantity: Optional[float]
    commission: float
    venue: str
    execution_time: datetime
    text: Optional[str]
    created_at: datetime


@dataclass
class OrderRoute:
    """Order routing configuration"""

    route_id: str
    name: str
    route_type: OrderRoute
    venues: List[str]
    routing_rules: Dict[str, Any]
    priority: int
    is_active: bool
    created_at: datetime
    last_updated: datetime


class OrderManagementSystem:
    """Comprehensive Order Management System"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # Order management
        self.orders: Dict[str, Order] = {}
        self.order_fills: Dict[str, List[OrderFill]] = defaultdict(list)
        self.execution_reports: Dict[str, List[ExecutionReport]] = defaultdict(list)

        # Market data
        self.order_books: Dict[str, OrderBook] = {}
        self.market_data: Dict[str, MarketData] = {}

        # Routing
        self.routes: Dict[str, OrderRoute] = {}
        self.venue_configs: Dict[str, Dict[str, Any]] = {}

        # Risk management
        self.position_limits: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.order_limits: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.risk_checks: Dict[str, bool] = {}

        # Performance tracking
        self.order_performance: Dict[str, Dict[str, Any]] = {}
        self.execution_quality: Dict[str, Dict[str, float]] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the Order Management System"""
        logger.info("Initializing Order Management System")

        # Load configurations
        await self._load_routing_configs()
        await self._load_venue_configs()
        await self._load_risk_limits()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._process_orders()),
            asyncio.create_task(self._update_market_data()),
            asyncio.create_task(self._monitor_risk_limits()),
            asyncio.create_task(self._generate_execution_reports()),
            asyncio.create_task(self._cleanup_old_data()),
        ]

        logger.info("Order Management System initialized successfully")

    async def create_order(
        self,
        user_id: int,
        account_id: str,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        client_order_id: Optional[str] = None,
        algo_type: Optional[str] = None,
        algo_parameters: Optional[Dict[str, Any]] = None,
    ) -> Order:
        """Create a new order"""
        try:
            # Generate order ID
            order_id = f"ORD_{uuid.uuid4().hex[:12].upper()}"

            # Create order
            order = Order(
                order_id=order_id,
                user_id=user_id,
                account_id=account_id,
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                limit_price=price if order_type == OrderType.LIMIT else None,
                time_in_force=time_in_force,
                status=OrderStatus.PENDING,
                route=OrderRoute.SMART,
                filled_quantity=0.0,
                remaining_quantity=quantity,
                average_price=0.0,
                total_filled_value=0.0,
                algo_type=algo_type,
                algo_parameters=algo_parameters or {},
                max_position_size=None,
                max_order_value=None,
                risk_limits={},
                created_at=datetime.utcnow(),
                submitted_at=None,
                acknowledged_at=None,
                filled_at=None,
                cancelled_at=None,
                last_updated=datetime.utcnow(),
                client_order_id=client_order_id,
                parent_order_id=None,
                child_orders=[],
                tags={},
                notes=None,
            )

            # Risk checks
            if not await self._perform_risk_checks(order):
                order.status = OrderStatus.REJECTED
                order.last_updated = datetime.utcnow()
                logger.warning(f"Order {order_id} rejected due to risk checks")
                return order

            # Store order
            self.orders[order_id] = order

            # Cache order
            await self._cache_order(order)

            # Submit order for processing
            asyncio.create_task(self._submit_order(order))

            logger.info(f"Created order {order_id} for user {user_id}")
            return order

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            raise

    async def cancel_order(self, order_id: str, user_id: int) -> bool:
        """Cancel an order"""
        try:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False

            order = self.orders[order_id]

            # Check if order can be cancelled
            if order.status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            ]:
                logger.warning(
                    f"Order {order_id} cannot be cancelled in status {order.status}"
                )
                return False

            # Check user authorization
            if order.user_id != user_id:
                logger.warning(
                    f"User {user_id} not authorized to cancel order {order_id}"
                )
                return False

            # Update order status
            order.status = OrderStatus.PENDING_CANCEL
            order.last_updated = datetime.utcnow()

            # Cancel child orders if any
            for child_order_id in order.child_orders:
                await self.cancel_order(child_order_id, user_id)

            # Process cancellation
            asyncio.create_task(self._process_cancellation(order))

            logger.info(f"Cancelled order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def modify_order(
        self,
        order_id: str,
        user_id: int,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None,
    ) -> bool:
        """Modify an existing order"""
        try:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False

            order = self.orders[order_id]

            # Check if order can be modified
            if order.status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            ]:
                logger.warning(
                    f"Order {order_id} cannot be modified in status {order.status}"
                )
                return False

            # Check user authorization
            if order.user_id != user_id:
                logger.warning(
                    f"User {user_id} not authorized to modify order {order_id}"
                )
                return False

            # Update order
            if new_quantity is not None:
                order.quantity = new_quantity
                order.remaining_quantity = new_quantity - order.filled_quantity

            if new_price is not None:
                order.price = new_price
                order.limit_price = new_price

            order.status = OrderStatus.PENDING_REPLACE
            order.last_updated = datetime.utcnow()

            # Process modification
            asyncio.create_task(self._process_modification(order))

            logger.info(f"Modified order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return False

    async def get_order(self, order_id: str, user_id: int) -> Optional[Order]:
        """Get order details"""
        try:
            if order_id not in self.orders:
                return None

            order = self.orders[order_id]

            # Check user authorization
            if order.user_id != user_id:
                return None

            return order

        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None

    async def get_orders(
        self,
        user_id: int,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100,
    ) -> List[Order]:
        """Get orders for a user"""
        try:
            orders = []

            for order in self.orders.values():
                if order.user_id != user_id:
                    continue

                if account_id and order.account_id != account_id:
                    continue

                if symbol and order.symbol != symbol:
                    continue

                if status and order.status != status:
                    continue

                orders.append(order)

            # Sort by creation time (newest first)
            orders.sort(key=lambda x: x.created_at, reverse=True)

            return orders[:limit]

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    async def get_order_fills(self, order_id: str, user_id: int) -> List[OrderFill]:
        """Get fills for an order"""
        try:
            if order_id not in self.orders:
                return []

            order = self.orders[order_id]

            # Check user authorization
            if order.user_id != user_id:
                return []

            return self.order_fills.get(order_id, [])

        except Exception as e:
            logger.error(f"Error getting order fills: {e}")
            return []

    async def get_execution_reports(
        self, order_id: str, user_id: int
    ) -> List[ExecutionReport]:
        """Get execution reports for an order"""
        try:
            if order_id not in self.orders:
                return []

            order = self.orders[order_id]

            # Check user authorization
            if order.user_id != user_id:
                return []

            return self.execution_reports.get(order_id, [])

        except Exception as e:
            logger.error(f"Error getting execution reports: {e}")
            return []

    async def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for a symbol"""
        try:
            return self.order_books.get(symbol)

        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return None

    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for a symbol"""
        try:
            return self.market_data.get(symbol)

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    async def _perform_risk_checks(self, order: Order) -> bool:
        """Perform risk checks on an order"""
        try:
            # Check position limits
            if not await self._check_position_limits(order):
                return False

            # Check order limits
            if not await self._check_order_limits(order):
                return False

            # Check market hours
            if not await self._check_market_hours(order.symbol):
                return False

            # Check symbol validity
            if not await self._check_symbol_validity(order.symbol):
                return False

            return True

        except Exception as e:
            logger.error(f"Error performing risk checks: {e}")
            return False

    async def _check_position_limits(self, order: Order) -> bool:
        """Check position limits"""
        try:
            # Get current position
            current_position = await self._get_current_position(
                order.user_id, order.account_id, order.symbol
            )

            # Calculate new position
            if order.side == OrderSide.BUY:
                new_position = current_position + order.quantity
            else:
                new_position = current_position - order.quantity

            # Check limits
            limits = self.position_limits.get(order.account_id, {})
            max_position = limits.get(order.symbol, float("inf"))

            if abs(new_position) > max_position:
                logger.warning(f"Position limit exceeded for {order.symbol}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return False

    async def _check_order_limits(self, order: Order) -> bool:
        """Check order limits"""
        try:
            # Check order value
            if order.price:
                order_value = order.quantity * order.price
                limits = self.order_limits.get(order.account_id, {})
                max_order_value = limits.get("max_order_value", float("inf"))

                if order_value > max_order_value:
                    logger.warning(f"Order value limit exceeded")
                    return False

            # Check order size
            limits = self.order_limits.get(order.account_id, {})
            max_order_size = limits.get("max_order_size", float("inf"))

            if order.quantity > max_order_size:
                logger.warning(f"Order size limit exceeded")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking order limits: {e}")
            return False

    async def _check_market_hours(self, symbol: str) -> bool:
        """Check if market is open for symbol"""
        try:
            # Simple market hours check
            # In practice, this would check actual market hours
            current_hour = datetime.utcnow().hour
            return 9 <= current_hour <= 16  # 9 AM to 4 PM UTC

        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False

    async def _check_symbol_validity(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        try:
            # Simple symbol validation
            # In practice, this would check against a symbol database
            return len(symbol) >= 1 and symbol.isalnum()

        except Exception as e:
            logger.error(f"Error checking symbol validity: {e}")
            return False

    async def _get_current_position(
        self, user_id: int, account_id: str, symbol: str
    ) -> float:
        """Get current position for user/account/symbol"""
        try:
            # Calculate position from filled orders
            position = 0.0

            for order in self.orders.values():
                if (
                    order.user_id == user_id
                    and order.account_id == account_id
                    and order.symbol == symbol
                    and order.status == OrderStatus.FILLED
                ):

                    if order.side == OrderSide.BUY:
                        position += order.filled_quantity
                    else:
                        position -= order.filled_quantity

            return position

        except Exception as e:
            logger.error(f"Error getting current position: {e}")
            return 0.0

    async def _submit_order(self, order: Order):
        """Submit order for execution"""
        try:
            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
            order.last_updated = datetime.utcnow()

            # Generate execution report
            await self._generate_execution_report(order, "new", "Order submitted")

            # Route order
            await self._route_order(order)

            logger.info(f"Submitted order {order.order_id}")

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            order.last_updated = datetime.utcnow()

    async def _route_order(self, order: Order):
        """Route order to appropriate venue"""
        try:
            # Determine routing strategy
            route = await self._determine_route(order)

            # Execute routing
            if route == OrderRoute.ALGO:
                await self._route_to_algorithm(order)
            elif route == OrderRoute.DARK_POOL:
                await self._route_to_dark_pool(order)
            elif route == OrderRoute.ECN:
                await self._route_to_ecn(order)
            else:
                await self._route_to_venue(order, route)

            logger.info(f"Routed order {order.order_id} to {route}")

        except Exception as e:
            logger.error(f"Error routing order: {e}")
            order.status = OrderStatus.REJECTED
            order.last_updated = datetime.utcnow()

    async def _determine_route(self, order: Order) -> OrderRoute:
        """Determine optimal routing for order"""
        try:
            # Simple routing logic
            # In practice, this would use sophisticated routing algorithms

            if order.algo_type:
                return OrderRoute.ALGO

            if order.quantity > 10000:  # Large order
                return OrderRoute.DARK_POOL

            return OrderRoute.SMART

        except Exception as e:
            logger.error(f"Error determining route: {e}")
            return OrderRoute.DIRECT

    async def _route_to_algorithm(self, order: Order):
        """Route order to algorithmic execution"""
        try:
            # Create child orders for algo execution
            child_orders = await self._create_algo_child_orders(order)

            for child_order in child_orders:
                order.child_orders.append(child_order.order_id)
                self.orders[child_order.order_id] = child_order
                await self._cache_order(child_order)

            order.status = OrderStatus.ACKNOWLEDGED
            order.acknowledged_at = datetime.utcnow()
            order.last_updated = datetime.utcnow()

            logger.info(f"Routed order {order.order_id} to algorithm")

        except Exception as e:
            logger.error(f"Error routing to algorithm: {e}")
            order.status = OrderStatus.REJECTED
            order.last_updated = datetime.utcnow()

    async def _create_algo_child_orders(self, parent_order: Order) -> List[Order]:
        """Create child orders for algorithmic execution"""
        try:
            child_orders = []

            # Simple TWAP implementation
            if parent_order.algo_type == "twap":
                num_slices = 10
                slice_size = parent_order.quantity / num_slices

                for i in range(num_slices):
                    child_order = Order(
                        order_id=f"CHILD_{uuid.uuid4().hex[:12].upper()}",
                        user_id=parent_order.user_id,
                        account_id=parent_order.account_id,
                        symbol=parent_order.symbol,
                        order_type=parent_order.order_type,
                        side=parent_order.side,
                        quantity=slice_size,
                        price=parent_order.price,
                        stop_price=parent_order.stop_price,
                        limit_price=parent_order.limit_price,
                        time_in_force=parent_order.time_in_force,
                        status=OrderStatus.PENDING,
                        route=OrderRoute.SMART,
                        filled_quantity=0.0,
                        remaining_quantity=slice_size,
                        average_price=0.0,
                        total_filled_value=0.0,
                        algo_type=None,
                        algo_parameters={},
                        max_position_size=parent_order.max_position_size,
                        max_order_value=parent_order.max_order_value,
                        risk_limits=parent_order.risk_limits,
                        created_at=datetime.utcnow(),
                        submitted_at=None,
                        acknowledged_at=None,
                        filled_at=None,
                        cancelled_at=None,
                        last_updated=datetime.utcnow(),
                        client_order_id=None,
                        parent_order_id=parent_order.order_id,
                        child_orders=[],
                        tags=parent_order.tags,
                        notes=f"TWAP slice {i+1}/{num_slices}",
                    )

                    child_orders.append(child_order)

            return child_orders

        except Exception as e:
            logger.error(f"Error creating algo child orders: {e}")
            return []

    async def _route_to_dark_pool(self, order: Order):
        """Route order to dark pool"""
        try:
            # Simulate dark pool routing
            order.status = OrderStatus.ACKNOWLEDGED
            order.acknowledged_at = datetime.utcnow()
            order.last_updated = datetime.utcnow()

            logger.info(f"Routed order {order.order_id} to dark pool")

        except Exception as e:
            logger.error(f"Error routing to dark pool: {e}")
            order.status = OrderStatus.REJECTED
            order.last_updated = datetime.utcnow()

    async def _route_to_ecn(self, order: Order):
        """Route order to ECN"""
        try:
            # Simulate ECN routing
            order.status = OrderStatus.ACKNOWLEDGED
            order.acknowledged_at = datetime.utcnow()
            order.last_updated = datetime.utcnow()

            logger.info(f"Routed order {order.order_id} to ECN")

        except Exception as e:
            logger.error(f"Error routing to ECN: {e}")
            order.status = OrderStatus.REJECTED
            order.last_updated = datetime.utcnow()

    async def _route_to_venue(self, order: Order, venue: OrderRoute):
        """Route order to specific venue"""
        try:
            # Simulate venue routing
            order.status = OrderStatus.ACKNOWLEDGED
            order.acknowledged_at = datetime.utcnow()
            order.last_updated = datetime.utcnow()

            logger.info(f"Routed order {order.order_id} to venue {venue}")

        except Exception as e:
            logger.error(f"Error routing to venue: {e}")
            order.status = OrderStatus.REJECTED
            order.last_updated = datetime.utcnow()

    async def _process_cancellation(self, order: Order):
        """Process order cancellation"""
        try:
            # Cancel child orders
            for child_order_id in order.child_orders:
                if child_order_id in self.orders:
                    child_order = self.orders[child_order_id]
                    child_order.status = OrderStatus.CANCELLED
                    child_order.cancelled_at = datetime.utcnow()
                    child_order.last_updated = datetime.utcnow()
                    await self._cache_order(child_order)

            # Update parent order
            order.status = OrderStatus.CANCELLED
            order.cancelled_at = datetime.utcnow()
            order.last_updated = datetime.utcnow()

            # Generate execution report
            await self._generate_execution_report(order, "cancel", "Order cancelled")

            # Cache order
            await self._cache_order(order)

            logger.info(f"Processed cancellation for order {order.order_id}")

        except Exception as e:
            logger.error(f"Error processing cancellation: {e}")

    async def _process_modification(self, order: Order):
        """Process order modification"""
        try:
            # Cancel child orders
            for child_order_id in order.child_orders:
                if child_order_id in self.orders:
                    child_order = self.orders[child_order_id]
                    child_order.status = OrderStatus.CANCELLED
                    child_order.cancelled_at = datetime.utcnow()
                    child_order.last_updated = datetime.utcnow()
                    await self._cache_order(child_order)

            # Clear child orders
            order.child_orders = []

            # Update order status
            order.status = OrderStatus.ACKNOWLEDGED
            order.acknowledged_at = datetime.utcnow()
            order.last_updated = datetime.utcnow()

            # Generate execution report
            await self._generate_execution_report(order, "replace", "Order modified")

            # Re-route order
            await self._route_order(order)

            # Cache order
            await self._cache_order(order)

            logger.info(f"Processed modification for order {order.order_id}")

        except Exception as e:
            logger.error(f"Error processing modification: {e}")

    async def _generate_execution_report(
        self, order: Order, execution_type: str, text: Optional[str] = None
    ):
        """Generate execution report"""
        try:
            report = ExecutionReport(
                report_id=f"RPT_{uuid.uuid4().hex[:12].upper()}",
                order_id=order.order_id,
                execution_type=execution_type,
                order_status=order.status,
                filled_quantity=order.filled_quantity,
                remaining_quantity=order.remaining_quantity,
                average_price=order.average_price,
                last_fill_price=None,
                last_fill_quantity=None,
                commission=0.0,
                venue=order.route.value,
                execution_time=datetime.utcnow(),
                text=text,
                created_at=datetime.utcnow(),
            )

            if order.order_id not in self.execution_reports:
                self.execution_reports[order.order_id] = []

            self.execution_reports[order.order_id].append(report)

            # Cache report
            await self._cache_execution_report(report)

        except Exception as e:
            logger.error(f"Error generating execution report: {e}")

    # Background tasks
    async def _process_orders(self):
        """Process pending orders"""
        while True:
            try:
                # Process pending orders
                for order in self.orders.values():
                    if order.status == OrderStatus.PENDING:
                        await self._submit_order(order)
                    elif order.status == OrderStatus.PENDING_CANCEL:
                        await self._process_cancellation(order)
                    elif order.status == OrderStatus.PENDING_REPLACE:
                        await self._process_modification(order)

                await asyncio.sleep(1)  # Process every second

            except Exception as e:
                logger.error(f"Error processing orders: {e}")
                await asyncio.sleep(5)

    async def _update_market_data(self):
        """Update market data"""
        while True:
            try:
                # Simulate market data updates
                for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
                    await self._simulate_market_data_update(symbol)

                await asyncio.sleep(5)  # Update every 5 seconds

            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(10)

    async def _simulate_market_data_update(self, symbol: str):
        """Simulate market data update"""
        try:
            # Generate random market data
            # Use hashlib for deterministic hash instead of built-in hash()
            deterministic_hash = int(hashlib.md5(symbol.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            base_price = 100.0 + deterministic_hash % 1000

            market_data = MarketData(
                symbol=symbol,
                bid_price=base_price - 0.01,
                ask_price=base_price + 0.01,
                bid_size=1000.0,
                ask_size=1000.0,
                last_price=base_price,
                volume=np.random.uniform(10000, 100000),
                high_price=base_price + np.random.uniform(0, 2),
                low_price=base_price - np.random.uniform(0, 2),
                open_price=base_price,
                timestamp=datetime.utcnow(),
            )

            self.market_data[symbol] = market_data

            # Update order book
            order_book = OrderBook(
                symbol=symbol,
                bids=[(market_data.bid_price, market_data.bid_size)],
                asks=[(market_data.ask_price, market_data.ask_size)],
                last_trade_price=market_data.last_price,
                last_trade_quantity=100.0,
                last_trade_time=datetime.utcnow(),
                volume=market_data.volume,
                timestamp=datetime.utcnow(),
            )

            self.order_books[symbol] = order_book

        except Exception as e:
            logger.error(f"Error simulating market data update: {e}")

    async def _monitor_risk_limits(self):
        """Monitor risk limits"""
        while True:
            try:
                # Check position limits
                for user_id in set(order.user_id for order in self.orders.values()):
                    for account_id in set(
                        order.account_id
                        for order in self.orders.values()
                        if order.user_id == user_id
                    ):
                        for symbol in set(
                            order.symbol
                            for order in self.orders.values()
                            if order.user_id == user_id
                            and order.account_id == account_id
                        ):
                            position = await self._get_current_position(
                                user_id, account_id, symbol
                            )
                            limits = self.position_limits.get(account_id, {})
                            max_position = limits.get(symbol, float("inf"))

                            if abs(position) > max_position:
                                logger.warning(
                                    f"Position limit exceeded for {user_id}/{account_id}/{symbol}"
                                )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring risk limits: {e}")
                await asyncio.sleep(120)

    async def _generate_execution_reports(self):
        """Generate execution reports"""
        while True:
            try:
                # Generate reports for active orders
                for order in self.orders.values():
                    if order.status in [
                        OrderStatus.ACKNOWLEDGED,
                        OrderStatus.PARTIALLY_FILLED,
                    ]:
                        # Simulate fills
                        if np.random.random() < 0.1:  # 10% chance of fill
                            await self._simulate_order_fill(order)

                await asyncio.sleep(10)  # Generate every 10 seconds

            except Exception as e:
                logger.error(f"Error generating execution reports: {e}")
                await asyncio.sleep(30)

    async def _simulate_order_fill(self, order: Order):
        """Simulate order fill"""
        try:
            # Generate fill
            fill_quantity = min(order.remaining_quantity, np.random.uniform(100, 1000))
            fill_price = order.price or (
                self.market_data.get(
                    order.symbol,
                    MarketData(
                        symbol=order.symbol,
                        bid_price=100,
                        ask_price=100.01,
                        bid_size=1000,
                        ask_size=1000,
                        last_price=100,
                        volume=10000,
                        high_price=101,
                        low_price=99,
                        open_price=100,
                        timestamp=datetime.utcnow(),
                    ),
                ).last_price
            )

            fill = OrderFill(
                fill_id=f"FILL_{uuid.uuid4().hex[:12].upper()}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                fill_value=fill_quantity * fill_price,
                commission=fill_quantity * fill_price * 0.001,  # 0.1% commission
                venue=order.route.value,
                execution_id=f"EXEC_{uuid.uuid4().hex[:12].upper()}",
                fill_time=datetime.utcnow(),
                settlement_date=datetime.utcnow() + timedelta(days=2),
                created_at=datetime.utcnow(),
            )

            # Update order
            order.filled_quantity += fill_quantity
            order.remaining_quantity -= fill_quantity
            order.total_filled_value += fill.fill_value
            order.average_price = (
                order.total_filled_value / order.filled_quantity
                if order.filled_quantity > 0
                else 0
            )

            if order.remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

            order.last_updated = datetime.utcnow()

            # Store fill
            if order.order_id not in self.order_fills:
                self.order_fills[order.order_id] = []

            self.order_fills[order.order_id].append(fill)

            # Generate execution report
            await self._generate_execution_report(
                order,
                "partial_fill" if order.remaining_quantity > 0 else "fill",
                f"Fill: {fill_quantity} @ {fill_price}",
            )

            # Cache order and fill
            await self._cache_order(order)
            await self._cache_fill(fill)

            logger.info(
                f"Simulated fill for order {order.order_id}: {fill_quantity} @ {fill_price}"
            )

        except Exception as e:
            logger.error(f"Error simulating order fill: {e}")

    async def _cleanup_old_data(self):
        """Cleanup old data"""
        while True:
            try:
                # Cleanup old orders (older than 30 days)
                cutoff_date = datetime.utcnow() - timedelta(days=30)

                orders_to_remove = []
                for order_id, order in self.orders.items():
                    if order.created_at < cutoff_date and order.status in [
                        OrderStatus.FILLED,
                        OrderStatus.CANCELLED,
                        OrderStatus.REJECTED,
                    ]:
                        orders_to_remove.append(order_id)

                for order_id in orders_to_remove:
                    del self.orders[order_id]
                    if order_id in self.order_fills:
                        del self.order_fills[order_id]
                    if order_id in self.execution_reports:
                        del self.execution_reports[order_id]

                if orders_to_remove:
                    logger.info(f"Cleaned up {len(orders_to_remove)} old orders")

                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(7200)

    # Helper methods
    async def _load_routing_configs(self):
        """Load routing configurations"""
        pass

    async def _load_venue_configs(self):
        """Load venue configurations"""
        pass

    async def _load_risk_limits(self):
        """Load risk limits"""
        pass

    # Caching methods
    async def _cache_order(self, order: Order):
        """Cache order"""
        try:
            cache_key = f"order:{order.order_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "order_id": order.order_id,
                        "user_id": order.user_id,
                        "account_id": order.account_id,
                        "symbol": order.symbol,
                        "order_type": order.order_type.value,
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "price": order.price,
                        "status": order.status.value,
                        "filled_quantity": order.filled_quantity,
                        "remaining_quantity": order.remaining_quantity,
                        "average_price": order.average_price,
                        "created_at": order.created_at.isoformat(),
                        "last_updated": order.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching order: {e}")

    async def _cache_fill(self, fill: OrderFill):
        """Cache fill"""
        try:
            cache_key = f"fill:{fill.fill_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "fill_id": fill.fill_id,
                        "order_id": fill.order_id,
                        "symbol": fill.symbol,
                        "side": fill.side.value,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "fill_value": fill.fill_value,
                        "commission": fill.commission,
                        "venue": fill.venue,
                        "fill_time": fill.fill_time.isoformat(),
                        "created_at": fill.created_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching fill: {e}")

    async def _cache_execution_report(self, report: ExecutionReport):
        """Cache execution report"""
        try:
            cache_key = f"execution_report:{report.report_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "report_id": report.report_id,
                        "order_id": report.order_id,
                        "execution_type": report.execution_type,
                        "order_status": report.order_status.value,
                        "filled_quantity": report.filled_quantity,
                        "remaining_quantity": report.remaining_quantity,
                        "average_price": report.average_price,
                        "venue": report.venue,
                        "execution_time": report.execution_time.isoformat(),
                        "text": report.text,
                        "created_at": report.created_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching execution report: {e}")


# Factory function
async def get_order_management_system(
    redis_client: redis_sync.Redis, db_session: Session
) -> OrderManagementSystem:
    """Get Order Management System instance"""
    oms = OrderManagementSystem(redis_client, db_session)
    await oms.initialize()
    return oms
