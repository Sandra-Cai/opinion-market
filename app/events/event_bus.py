"""
Event Bus for Event-Driven Architecture
Handles event publishing, subscription, and routing
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict

from app.events.event_store import EventStore, Event
from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


@dataclass
class EventSubscription:
    """Event subscription information"""
    subscriber_id: str
    event_types: List[str]
    handler: Callable
    filter_func: Optional[Callable] = None


class EventBus:
    """Event bus for event-driven communication"""
    
    def __init__(self):
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.event_store = EventStore()
        self.published_events = 0
        self.failed_events = 0
        
    async def publish_event(self, event: Event) -> bool:
        """Publish an event to the event bus"""
        try:
            # Store event in event store
            await self.event_store.append_event(event)
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            # Update metrics
            self.published_events += 1
            
            logger.info(f"Event published: {event.event_type} for {event.aggregate_type}:{event.aggregate_id}")
            return True
            
        except Exception as e:
            self.failed_events += 1
            logger.error(f"Failed to publish event: {e}")
            return False
    
    async def subscribe(self, subscriber_id: str, event_types: List[str], 
                       handler: Callable, filter_func: Optional[Callable] = None):
        """Subscribe to specific event types"""
        try:
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                event_types=event_types,
                handler=handler,
                filter_func=filter_func
            )
            
            for event_type in event_types:
                self.subscriptions[event_type].append(subscription)
            
            logger.info(f"Subscriber {subscriber_id} subscribed to events: {event_types}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe {subscriber_id}: {e}")
    
    async def unsubscribe(self, subscriber_id: str, event_types: Optional[List[str]] = None):
        """Unsubscribe from event types"""
        try:
            if event_types is None:
                # Unsubscribe from all events
                for event_type in list(self.subscriptions.keys()):
                    self.subscriptions[event_type] = [
                        sub for sub in self.subscriptions[event_type]
                        if sub.subscriber_id != subscriber_id
                    ]
            else:
                # Unsubscribe from specific event types
                for event_type in event_types:
                    if event_type in self.subscriptions:
                        self.subscriptions[event_type] = [
                            sub for sub in self.subscriptions[event_type]
                            if sub.subscriber_id != subscriber_id
                        ]
            
            logger.info(f"Subscriber {subscriber_id} unsubscribed from events: {event_types or 'all'}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe {subscriber_id}: {e}")
    
    async def _notify_subscribers(self, event: Event):
        """Notify all subscribers of an event"""
        try:
            subscribers = self.subscriptions.get(event.event_type, [])
            
            if not subscribers:
                return
            
            # Create tasks for all subscribers
            tasks = []
            for subscription in subscribers:
                task = asyncio.create_task(
                    self._notify_subscriber(subscription, event)
                )
                tasks.append(task)
            
            # Wait for all notifications to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to notify subscriber {subscribers[i].subscriber_id}: {result}")
                else:
                    logger.debug(f"Successfully notified subscriber {subscribers[i].subscriber_id}")
            
        except Exception as e:
            logger.error(f"Error notifying subscribers: {e}")
    
    async def _notify_subscriber(self, subscription: EventSubscription, event: Event):
        """Notify a single subscriber"""
        try:
            # Apply filter if provided
            if subscription.filter_func and not subscription.filter_func(event):
                return
            
            # Call the handler
            await subscription.handler(event)
            
        except Exception as e:
            logger.error(f"Error in subscriber {subscription.subscriber_id}: {e}")
            raise e
    
    async def replay_events(self, event_types: List[str], from_timestamp: float = 0):
        """Replay events for subscribers"""
        try:
            logger.info(f"Starting event replay for types: {event_types}")
            
            for event_type in event_types:
                events = await self.event_store.get_events_by_type(event_type)
                
                # Filter events by timestamp
                filtered_events = [
                    event for event in events
                    if event.timestamp >= from_timestamp
                ]
                
                # Replay events
                for event in filtered_events:
                    await self._notify_subscribers(event)
            
            logger.info(f"Event replay completed for types: {event_types}")
            
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
    
    async def get_event_bus_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            "published_events": self.published_events,
            "failed_events": self.failed_events,
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "event_types": list(self.subscriptions.keys()),
            "subscribers_per_type": {
                event_type: len(subs) 
                for event_type, subs in self.subscriptions.items()
            }
        }
    
    def get_subscriptions(self) -> Dict[str, List[str]]:
        """Get current subscriptions"""
        return {
            event_type: [sub.subscriber_id for sub in subs]
            for event_type, subs in self.subscriptions.items()
        }


# Global event bus instance
event_bus = EventBus()
