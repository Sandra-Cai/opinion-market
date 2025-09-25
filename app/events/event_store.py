"""
Event Store for Event Sourcing
Stores and retrieves events for audit trails and data consistency
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data structure"""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    version: int
    causation_id: Optional[str] = None
    correlation_id: Optional[str] = None


class EventStore:
    """Event store for persisting and retrieving events"""
    
    def __init__(self):
        self.events_cache = {}
        self.snapshots_cache = {}
        
    async def append_event(self, event: Event) -> bool:
        """Append event to the event store"""
        try:
            # Store in database
            query = """
            INSERT INTO events (event_id, event_type, aggregate_id, aggregate_type, 
                              event_data, metadata, timestamp, version, causation_id, correlation_id)
            VALUES (:event_id, :event_type, :aggregate_id, :aggregate_type, 
                   :event_data, :metadata, :timestamp, :version, :causation_id, :correlation_id)
            """
            
            with engine.connect() as conn:
                conn.execute(text(query), {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "aggregate_id": event.aggregate_id,
                    "aggregate_type": event.aggregate_type,
                    "event_data": json.dumps(event.event_data),
                    "metadata": json.dumps(event.metadata),
                    "timestamp": event.timestamp,
                    "version": event.version,
                    "causation_id": event.causation_id,
                    "correlation_id": event.correlation_id
                })
                conn.commit()
            
            # Cache the event
            await self._cache_event(event)
            
            logger.info(f"Event stored: {event.event_type} for {event.aggregate_type}:{event.aggregate_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append event: {e}")
            return False
    
    async def get_events(self, aggregate_id: str, aggregate_type: str, 
                        from_version: int = 0) -> List[Event]:
        """Get events for a specific aggregate"""
        try:
            # Check cache first
            cache_key = f"events_{aggregate_type}_{aggregate_id}_{from_version}"
            cached_events = await enhanced_cache.get(cache_key)
            if cached_events:
                return [Event(**event_dict) for event_dict in cached_events]
            
            # Query database
            query = """
            SELECT event_id, event_type, aggregate_id, aggregate_type, event_data, 
                   metadata, timestamp, version, causation_id, correlation_id
            FROM events 
            WHERE aggregate_id = :aggregate_id AND aggregate_type = :aggregate_type 
            AND version >= :from_version
            ORDER BY version ASC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), {
                    "aggregate_id": aggregate_id,
                    "aggregate_type": aggregate_type,
                    "from_version": from_version
                })
                
                events = []
                for row in result:
                    event = Event(
                        event_id=row[0],
                        event_type=row[1],
                        aggregate_id=row[2],
                        aggregate_type=row[3],
                        event_data=json.loads(row[4]),
                        metadata=json.loads(row[5]),
                        timestamp=row[6],
                        version=row[7],
                        causation_id=row[8],
                        correlation_id=row[9]
                    )
                    events.append(event)
            
            # Cache the events
            await enhanced_cache.set(
                cache_key,
                [asdict(event) for event in events],
                ttl=3600,
                tags=["events", aggregate_type, aggregate_id]
            )
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events for {aggregate_type}:{aggregate_id}: {e}")
            return []
    
    async def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Event]:
        """Get events by type"""
        try:
            query = """
            SELECT event_id, event_type, aggregate_id, aggregate_type, event_data, 
                   metadata, timestamp, version, causation_id, correlation_id
            FROM events 
            WHERE event_type = :event_type
            ORDER BY timestamp DESC
            LIMIT :limit
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), {
                    "event_type": event_type,
                    "limit": limit
                })
                
                events = []
                for row in result:
                    event = Event(
                        event_id=row[0],
                        event_type=row[1],
                        aggregate_id=row[2],
                        aggregate_type=row[3],
                        event_data=json.loads(row[4]),
                        metadata=json.loads(row[5]),
                        timestamp=row[6],
                        version=row[7],
                        causation_id=row[8],
                        correlation_id=row[9]
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events by type {event_type}: {e}")
            return []
    
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[Event]:
        """Get events by correlation ID"""
        try:
            query = """
            SELECT event_id, event_type, aggregate_id, aggregate_type, event_data, 
                   metadata, timestamp, version, causation_id, correlation_id
            FROM events 
            WHERE correlation_id = :correlation_id
            ORDER BY timestamp ASC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), {"correlation_id": correlation_id})
                
                events = []
                for row in result:
                    event = Event(
                        event_id=row[0],
                        event_type=row[1],
                        aggregate_id=row[2],
                        aggregate_type=row[3],
                        event_data=json.loads(row[4]),
                        metadata=json.loads(row[5]),
                        timestamp=row[6],
                        version=row[7],
                        causation_id=row[8],
                        correlation_id=row[9]
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events by correlation ID {correlation_id}: {e}")
            return []
    
    async def save_snapshot(self, aggregate_id: str, aggregate_type: str, 
                          snapshot_data: Dict[str, Any], version: int) -> bool:
        """Save aggregate snapshot"""
        try:
            query = """
            INSERT INTO snapshots (aggregate_id, aggregate_type, snapshot_data, version, timestamp)
            VALUES (:aggregate_id, :aggregate_type, :snapshot_data, :version, :timestamp)
            ON CONFLICT (aggregate_id, aggregate_type) 
            DO UPDATE SET snapshot_data = :snapshot_data, version = :version, timestamp = :timestamp
            """
            
            with engine.connect() as conn:
                conn.execute(text(query), {
                    "aggregate_id": aggregate_id,
                    "aggregate_type": aggregate_type,
                    "snapshot_data": json.dumps(snapshot_data),
                    "version": version,
                    "timestamp": time.time()
                })
                conn.commit()
            
            # Cache the snapshot
            await enhanced_cache.set(
                f"snapshot_{aggregate_type}_{aggregate_id}",
                {"data": snapshot_data, "version": version, "timestamp": time.time()},
                ttl=7200,
                tags=["snapshot", aggregate_type, aggregate_id]
            )
            
            logger.info(f"Snapshot saved for {aggregate_type}:{aggregate_id} at version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return False
    
    async def get_snapshot(self, aggregate_id: str, aggregate_type: str) -> Optional[Dict[str, Any]]:
        """Get aggregate snapshot"""
        try:
            # Check cache first
            cached_snapshot = await enhanced_cache.get(f"snapshot_{aggregate_type}_{aggregate_id}")
            if cached_snapshot:
                return cached_snapshot
            
            # Query database
            query = """
            SELECT snapshot_data, version, timestamp
            FROM snapshots 
            WHERE aggregate_id = :aggregate_id AND aggregate_type = :aggregate_type
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), {
                    "aggregate_id": aggregate_id,
                    "aggregate_type": aggregate_type
                })
                row = result.fetchone()
                
                if row:
                    snapshot = {
                        "data": json.loads(row[0]),
                        "version": row[1],
                        "timestamp": row[2]
                    }
                    
                    # Cache the snapshot
                    await enhanced_cache.set(
                        f"snapshot_{aggregate_type}_{aggregate_id}",
                        snapshot,
                        ttl=7200,
                        tags=["snapshot", aggregate_type, aggregate_id]
                    )
                    
                    return snapshot
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get snapshot for {aggregate_type}:{aggregate_id}: {e}")
            return None
    
    async def _cache_event(self, event: Event):
        """Cache event for quick access"""
        try:
            await enhanced_cache.set(
                f"event_{event.event_id}",
                asdict(event),
                ttl=86400,  # 24 hours
                tags=["event", event.event_type, event.aggregate_type]
            )
        except Exception as e:
            logger.error(f"Failed to cache event: {e}")
    
    async def get_event_statistics(self) -> Dict[str, Any]:
        """Get event store statistics"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT aggregate_id) as total_aggregates,
                COUNT(DISTINCT event_type) as total_event_types,
                MIN(timestamp) as oldest_event,
                MAX(timestamp) as newest_event
            FROM events
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
            
            return {
                "total_events": row[0] or 0,
                "total_aggregates": row[1] or 0,
                "total_event_types": row[2] or 0,
                "oldest_event": row[3],
                "newest_event": row[4]
            }
            
        except Exception as e:
            logger.error(f"Failed to get event statistics: {e}")
            return {}


# Global event store instance
event_store = EventStore()
