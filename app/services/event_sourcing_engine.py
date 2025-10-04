"""
Event Sourcing Engine
Comprehensive event sourcing and CQRS patterns implementation
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import uuid
import hashlib
import pickle
import secrets

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event type enumeration"""
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    MARKET_CREATED = "market_created"
    MARKET_UPDATED = "market_updated"
    MARKET_CLOSED = "market_closed"
    TRADE_EXECUTED = "trade_executed"
    PRICE_UPDATED = "price_updated"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    PAYMENT_PROCESSED = "payment_processed"
    NOTIFICATION_SENT = "notification_sent"
    SYSTEM_EVENT = "system_event"


class EventStatus(Enum):
    """Event status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    COMPENSATED = "compensated"


class AggregateType(Enum):
    """Aggregate type enumeration"""
    USER = "user"
    MARKET = "market"
    TRADE = "trade"
    ORDER = "order"
    PAYMENT = "payment"
    NOTIFICATION = "notification"


@dataclass
class Event:
    """Event data structure"""
    event_id: str
    aggregate_id: str
    aggregate_type: AggregateType
    event_type: EventType
    event_data: Dict[str, Any]
    event_metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    status: EventStatus = EventStatus.PENDING
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class Aggregate:
    """Aggregate data structure"""
    aggregate_id: str
    aggregate_type: AggregateType
    version: int
    state: Dict[str, Any]
    events: List[Event] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Projection:
    """Projection data structure"""
    projection_id: str
    name: str
    description: str
    aggregate_type: AggregateType
    event_types: List[EventType]
    state: Dict[str, Any] = field(default_factory=dict)
    last_processed_event: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Command:
    """Command data structure"""
    command_id: str
    aggregate_id: str
    aggregate_type: AggregateType
    command_type: str
    command_data: Dict[str, Any]
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class EventSourcingEngine:
    """Event Sourcing Engine with CQRS patterns"""
    
    def __init__(self):
        self.events: Dict[str, Event] = {}
        self.aggregates: Dict[str, Aggregate] = {}
        self.projections: Dict[str, Projection] = {}
        self.commands: Dict[str, Command] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.command_handlers: Dict[str, Callable] = {}
        
        # Configuration
        self.config = {
            "event_sourcing_enabled": True,
            "cqrs_enabled": True,
            "event_store_enabled": True,
            "projection_rebuild_enabled": True,
            "snapshot_enabled": True,
            "event_retention_days": 365,
            "snapshot_interval": 100,  # events
            "projection_update_interval": 60,  # seconds
            "command_timeout": 30,  # seconds
            "event_processing_batch_size": 100,
            "compensation_enabled": True,
            "eventual_consistency_enabled": True
        }
        
        # Event schemas
        self.event_schemas = {
            EventType.USER_CREATED: {
                "required_fields": ["user_id", "email", "name"],
                "optional_fields": ["phone", "address"]
            },
            EventType.MARKET_CREATED: {
                "required_fields": ["market_id", "name", "description"],
                "optional_fields": ["category", "tags"]
            },
            EventType.TRADE_EXECUTED: {
                "required_fields": ["trade_id", "market_id", "quantity", "price"],
                "optional_fields": ["buyer_id", "seller_id"]
            },
            EventType.ORDER_PLACED: {
                "required_fields": ["order_id", "market_id", "side", "quantity", "price"],
                "optional_fields": ["user_id", "order_type"]
            }
        }
        
        # Monitoring
        self.event_sourcing_active = False
        self.event_sourcing_task: Optional[asyncio.Task] = None
        self.projection_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.event_sourcing_stats = {
            "events_created": 0,
            "events_processed": 0,
            "events_failed": 0,
            "aggregates_created": 0,
            "aggregates_updated": 0,
            "projections_updated": 0,
            "commands_processed": 0,
            "commands_failed": 0,
            "snapshots_created": 0,
            "compensations_executed": 0
        }
        
    async def start_event_sourcing_engine(self):
        """Start the event sourcing engine"""
        if self.event_sourcing_active:
            logger.warning("Event sourcing engine already active")
            return
            
        self.event_sourcing_active = True
        self.event_sourcing_task = asyncio.create_task(self._event_processing_loop())
        self.projection_task = asyncio.create_task(self._projection_update_loop())
        logger.info("Event Sourcing Engine started")
        
    async def stop_event_sourcing_engine(self):
        """Stop the event sourcing engine"""
        self.event_sourcing_active = False
        if self.event_sourcing_task:
            self.event_sourcing_task.cancel()
            try:
                await self.event_sourcing_task
            except asyncio.CancelledError:
                pass
        if self.projection_task:
            self.projection_task.cancel()
            try:
                await self.projection_task
            except asyncio.CancelledError:
                pass
        logger.info("Event Sourcing Engine stopped")
        
    async def _event_processing_loop(self):
        """Main event processing loop"""
        while self.event_sourcing_active:
            try:
                # Process pending events
                await self._process_pending_events()
                
                # Create snapshots
                if self.config["snapshot_enabled"]:
                    await self._create_snapshots()
                    
                # Clean up old events
                await self._cleanup_old_events()
                
                # Wait before next cycle
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _projection_update_loop(self):
        """Projection update loop"""
        while self.event_sourcing_active:
            try:
                # Update projections
                await self._update_projections()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["projection_update_interval"])
                
            except Exception as e:
                logger.error(f"Error in projection update loop: {e}")
                await asyncio.sleep(60)
                
    async def create_event(self, event_data: Dict[str, Any]) -> Event:
        """Create a new event"""
        try:
            event_id = f"evt_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Validate event data
            event_type = EventType(event_data.get("event_type", "system_event"))
            if not await self._validate_event_data(event_type, event_data.get("event_data", {})):
                raise ValueError(f"Invalid event data for type: {event_type.value}")
                
            # Create event
            event = Event(
                event_id=event_id,
                aggregate_id=event_data.get("aggregate_id", ""),
                aggregate_type=AggregateType(event_data.get("aggregate_type", "user")),
                event_type=event_type,
                event_data=event_data.get("event_data", {}),
                event_metadata=event_data.get("event_metadata", {}),
                version=event_data.get("version", 1),
                correlation_id=event_data.get("correlation_id"),
                causation_id=event_data.get("causation_id")
            )
            
            # Store event
            self.events[event_id] = event
            
            # Update aggregate
            await self._update_aggregate(event)
            
            self.event_sourcing_stats["events_created"] += 1
            
            logger.info(f"Event created: {event_id} - {event_type.value}")
            return event
            
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            raise
            
    async def process_command(self, command_data: Dict[str, Any]) -> Command:
        """Process a command"""
        try:
            command_id = f"cmd_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Create command
            command = Command(
                command_id=command_id,
                aggregate_id=command_data.get("aggregate_id", ""),
                aggregate_type=AggregateType(command_data.get("aggregate_type", "user")),
                command_type=command_data.get("command_type", "unknown"),
                command_data=command_data.get("command_data", {}),
                correlation_id=command_data.get("correlation_id"),
                causation_id=command_data.get("causation_id")
            )
            
            # Store command
            self.commands[command_id] = command
            
            # Process command
            success = await self._execute_command(command)
            
            # Update command status
            command.status = "completed" if success else "failed"
            
            # Update statistics
            if success:
                self.event_sourcing_stats["commands_processed"] += 1
            else:
                self.event_sourcing_stats["commands_failed"] += 1
                
            logger.info(f"Command processed: {command_id} - Success: {success}")
            return command
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            raise
            
    async def _validate_event_data(self, event_type: EventType, event_data: Dict[str, Any]) -> bool:
        """Validate event data against schema"""
        try:
            schema = self.event_schemas.get(event_type, {})
            required_fields = schema.get("required_fields", [])
            
            # Check required fields
            for field in required_fields:
                if field not in event_data:
                    logger.warning(f"Missing required field '{field}' for event type {event_type.value}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating event data: {e}")
            return False
            
    async def _update_aggregate(self, event: Event):
        """Update aggregate with event"""
        try:
            aggregate_id = event.aggregate_id
            
            # Get or create aggregate
            if aggregate_id in self.aggregates:
                aggregate = self.aggregates[aggregate_id]
                aggregate.version += 1
            else:
                aggregate = Aggregate(
                    aggregate_id=aggregate_id,
                    aggregate_type=event.aggregate_type,
                    version=1,
                    state={}
                )
                self.aggregates[aggregate_id] = aggregate
                self.event_sourcing_stats["aggregates_created"] += 1
                
            # Add event to aggregate
            aggregate.events.append(event)
            aggregate.updated_at = datetime.now()
            
            # Update aggregate state based on event
            await self._apply_event_to_aggregate(aggregate, event)
            
            self.event_sourcing_stats["aggregates_updated"] += 1
            
        except Exception as e:
            logger.error(f"Error updating aggregate: {e}")
            
    async def _apply_event_to_aggregate(self, aggregate: Aggregate, event: Event):
        """Apply event to aggregate state"""
        try:
            # Update aggregate state based on event type
            if event.event_type == EventType.USER_CREATED:
                aggregate.state.update({
                    "user_id": event.event_data.get("user_id"),
                    "email": event.event_data.get("email"),
                    "name": event.event_data.get("name"),
                    "created_at": event.timestamp.isoformat()
                })
            elif event.event_type == EventType.USER_UPDATED:
                aggregate.state.update(event.event_data)
                aggregate.state["updated_at"] = event.timestamp.isoformat()
            elif event.event_type == EventType.MARKET_CREATED:
                aggregate.state.update({
                    "market_id": event.event_data.get("market_id"),
                    "name": event.event_data.get("name"),
                    "description": event.event_data.get("description"),
                    "status": "active",
                    "created_at": event.timestamp.isoformat()
                })
            elif event.event_type == EventType.TRADE_EXECUTED:
                if "trades" not in aggregate.state:
                    aggregate.state["trades"] = []
                aggregate.state["trades"].append({
                    "trade_id": event.event_data.get("trade_id"),
                    "quantity": event.event_data.get("quantity"),
                    "price": event.event_data.get("price"),
                    "timestamp": event.timestamp.isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error applying event to aggregate: {e}")
            
    async def _execute_command(self, command: Command) -> bool:
        """Execute a command"""
        try:
            # Find command handler
            handler = self.command_handlers.get(command.command_type)
            if not handler:
                logger.warning(f"No handler found for command type: {command.command_type}")
                return False
                
            # Execute command handler
            result = await handler(command)
            
            # Store result
            command.result = result
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            command.error_message = str(e)
            return False
            
    async def _process_pending_events(self):
        """Process pending events"""
        try:
            pending_events = [
                event for event in self.events.values()
                if event.status == EventStatus.PENDING
            ]
            
            # Process events in batches
            batch_size = self.config["event_processing_batch_size"]
            for i in range(0, len(pending_events), batch_size):
                batch = pending_events[i:i + batch_size]
                await self._process_event_batch(batch)
                
        except Exception as e:
            logger.error(f"Error processing pending events: {e}")
            
    async def _process_event_batch(self, events: List[Event]):
        """Process a batch of events"""
        try:
            for event in events:
                event.status = EventStatus.PROCESSING
                
                try:
                    # Process event handlers
                    handlers = self.event_handlers.get(event.event_type, [])
                    for handler in handlers:
                        await handler(event)
                        
                    # Update projections
                    await self._update_projections_for_event(event)
                    
                    event.status = EventStatus.PROCESSED
                    event.processed_at = datetime.now()
                    
                    self.event_sourcing_stats["events_processed"] += 1
                    
                except Exception as e:
                    event.status = EventStatus.FAILED
                    event.error_message = str(e)
                    self.event_sourcing_stats["events_failed"] += 1
                    logger.error(f"Error processing event {event.event_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
            
    async def _update_projections(self):
        """Update all projections"""
        try:
            for projection in self.projections.values():
                await self._update_single_projection(projection)
                
        except Exception as e:
            logger.error(f"Error updating projections: {e}")
            
    async def _update_projections_for_event(self, event: Event):
        """Update projections for a specific event"""
        try:
            for projection in self.projections.values():
                if (projection.aggregate_type == event.aggregate_type and 
                    event.event_type in projection.event_types):
                    await self._update_single_projection(projection)
                    
        except Exception as e:
            logger.error(f"Error updating projections for event: {e}")
            
    async def _update_single_projection(self, projection: Projection):
        """Update a single projection"""
        try:
            # Find events to process
            events_to_process = []
            for event in self.events.values():
                if (event.aggregate_type == projection.aggregate_type and 
                    event.event_type in projection.event_types and
                    event.status == EventStatus.PROCESSED):
                    
                    if (not projection.last_processed_event or 
                        event.event_id > projection.last_processed_event):
                        events_to_process.append(event)
                        
            # Process events
            for event in events_to_process:
                await self._apply_event_to_projection(projection, event)
                projection.last_processed_event = event.event_id
                
            projection.updated_at = datetime.now()
            self.event_sourcing_stats["projections_updated"] += 1
            
        except Exception as e:
            logger.error(f"Error updating projection {projection.projection_id}: {e}")
            
    async def _apply_event_to_projection(self, projection: Projection, event: Event):
        """Apply event to projection"""
        try:
            # Update projection state based on event
            if event.event_type == EventType.USER_CREATED:
                if "users" not in projection.state:
                    projection.state["users"] = {}
                projection.state["users"][event.aggregate_id] = {
                    "user_id": event.event_data.get("user_id"),
                    "email": event.event_data.get("email"),
                    "name": event.event_data.get("name"),
                    "created_at": event.timestamp.isoformat()
                }
            elif event.event_type == EventType.MARKET_CREATED:
                if "markets" not in projection.state:
                    projection.state["markets"] = {}
                projection.state["markets"][event.aggregate_id] = {
                    "market_id": event.event_data.get("market_id"),
                    "name": event.event_data.get("name"),
                    "description": event.event_data.get("description"),
                    "status": "active",
                    "created_at": event.timestamp.isoformat()
                }
            elif event.event_type == EventType.TRADE_EXECUTED:
                if "trades" not in projection.state:
                    projection.state["trades"] = []
                projection.state["trades"].append({
                    "trade_id": event.event_data.get("trade_id"),
                    "market_id": event.event_data.get("market_id"),
                    "quantity": event.event_data.get("quantity"),
                    "price": event.event_data.get("price"),
                    "timestamp": event.timestamp.isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error applying event to projection: {e}")
            
    async def _create_snapshots(self):
        """Create snapshots for aggregates"""
        try:
            for aggregate_id, aggregate in self.aggregates.items():
                # Check if snapshot is needed
                if len(aggregate.events) % self.config["snapshot_interval"] == 0:
                    snapshot_data = {
                        "aggregate_id": aggregate_id,
                        "version": aggregate.version,
                        "state": aggregate.state,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Store snapshot
                    snapshot_key = f"snapshot:{aggregate_id}:{aggregate.version}"
                    await enhanced_cache.set(snapshot_key, snapshot_data, ttl=86400 * 30)  # 30 days
                    
                    self.event_sourcing_stats["snapshots_created"] += 1
                    
        except Exception as e:
            logger.error(f"Error creating snapshots: {e}")
            
    async def _cleanup_old_events(self):
        """Clean up old events"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=self.config["event_retention_days"])
            
            # Remove old events
            to_remove = []
            for event_id, event in self.events.items():
                if event.timestamp < cutoff_time:
                    to_remove.append(event_id)
                    
            for event_id in to_remove:
                del self.events[event_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")
            
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler"""
        try:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Event handler registered for: {event_type.value}")
            
        except Exception as e:
            logger.error(f"Error registering event handler: {e}")
            
    def register_command_handler(self, command_type: str, handler: Callable):
        """Register a command handler"""
        try:
            self.command_handlers[command_type] = handler
            logger.info(f"Command handler registered for: {command_type}")
            
        except Exception as e:
            logger.error(f"Error registering command handler: {e}")
            
    def get_event_sourcing_summary(self) -> Dict[str, Any]:
        """Get comprehensive event sourcing summary"""
        try:
            # Calculate event statistics
            events_by_type = defaultdict(int)
            events_by_status = defaultdict(int)
            
            for event in self.events.values():
                events_by_type[event.event_type.value] += 1
                events_by_status[event.status.value] += 1
                
            # Calculate aggregate statistics
            aggregates_by_type = defaultdict(int)
            for aggregate in self.aggregates.values():
                aggregates_by_type[aggregate.aggregate_type.value] += 1
                
            # Calculate command statistics
            commands_by_status = defaultdict(int)
            for command in self.commands.values():
                commands_by_status[command.status] += 1
                
            return {
                "timestamp": datetime.now().isoformat(),
                "event_sourcing_active": self.event_sourcing_active,
                "total_events": len(self.events),
                "events_by_type": dict(events_by_type),
                "events_by_status": dict(events_by_status),
                "total_aggregates": len(self.aggregates),
                "aggregates_by_type": dict(aggregates_by_type),
                "total_projections": len(self.projections),
                "total_commands": len(self.commands),
                "commands_by_status": dict(commands_by_status),
                "registered_event_handlers": len(self.event_handlers),
                "registered_command_handlers": len(self.command_handlers),
                "stats": self.event_sourcing_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting event sourcing summary: {e}")
            return {"error": str(e)}


# Global instance
event_sourcing_engine = EventSourcingEngine()
