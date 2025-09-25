"""
Market Microservice
Handles all market-related operations in a dedicated microservice
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.services.base_service import BaseService
from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class MarketService(BaseService):
    """Market microservice for handling market operations"""
    
    def __init__(self):
        super().__init__("market-service", "1.0.0")
        self.dependencies = ["user-service", "notification-service"]
    
    async def initialize(self):
        """Initialize market service"""
        logger.info("Initializing Market Service")
        
        # Initialize database connection
        self.db_engine = engine
        
        # Start background tasks
        asyncio.create_task(self._market_cleanup_loop())
        asyncio.create_task(self._market_analytics_loop())
        
        logger.info("Market Service initialized")
    
    async def cleanup(self):
        """Cleanup market service resources"""
        logger.info("Cleaning up Market Service")
    
    async def process_request(self, request_data: Dict[str, Any]) -> Any:
        """Process market-related requests"""
        action = request_data.get("action")
        
        if action == "create_market":
            return await self.create_market(request_data.get("data", {}))
        elif action == "get_market":
            return await self.get_market(request_data.get("market_id"))
        elif action == "list_markets":
            return await self.list_markets(request_data.get("filters", {}))
        elif action == "update_market":
            return await self.update_market(request_data.get("market_id"), request_data.get("data", {}))
        elif action == "resolve_market":
            return await self.resolve_market(request_data.get("market_id"), request_data.get("outcome"))
        elif action == "get_market_analytics":
            return await self.get_market_analytics(request_data.get("market_id"))
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def create_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new prediction market"""
        try:
            # Validate market data
            required_fields = ["title", "description", "category", "market_type", "outcomes", "end_date"]
            for field in required_fields:
                if field not in market_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create market in database
            query = """
            INSERT INTO markets (title, description, category, market_type, outcomes, 
                               end_date, created_by, status, created_at)
            VALUES (:title, :description, :category, :market_type, :outcomes, 
                   :end_date, :created_by, 'active', NOW())
            RETURNING id
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), {
                    "title": market_data["title"],
                    "description": market_data["description"],
                    "category": market_data["category"],
                    "market_type": market_data["market_type"],
                    "outcomes": json.dumps(market_data["outcomes"]),
                    "end_date": market_data["end_date"],
                    "created_by": market_data.get("created_by")
                })
                market_id = result.fetchone()[0]
                conn.commit()
            
            # Cache the new market
            await enhanced_cache.set(
                f"market_{market_id}",
                {"id": market_id, **market_data},
                ttl=3600,
                tags=["market", "new"]
            )
            
            # Broadcast market creation event
            await self._broadcast_market_event("market_created", {
                "market_id": market_id,
                "title": market_data["title"],
                "category": market_data["category"]
            })
            
            return {
                "market_id": market_id,
                "status": "created",
                "message": "Market created successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to create market: {e}")
            raise e
    
    async def get_market(self, market_id: int) -> Dict[str, Any]:
        """Get market details"""
        try:
            # Check cache first
            cached_market = await enhanced_cache.get(f"market_{market_id}")
            if cached_market:
                return cached_market
            
            # Query database
            query = """
            SELECT id, title, description, category, market_type, outcomes, 
                   end_date, created_by, status, created_at, updated_at
            FROM markets 
            WHERE id = :market_id
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), {"market_id": market_id})
                row = result.fetchone()
                
                if not row:
                    raise ValueError(f"Market not found: {market_id}")
                
                market_data = {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "category": row[3],
                    "market_type": row[4],
                    "outcomes": json.loads(row[5]) if row[5] else [],
                    "end_date": row[6].isoformat() if row[6] else None,
                    "created_by": row[7],
                    "status": row[8],
                    "created_at": row[9].isoformat() if row[9] else None,
                    "updated_at": row[10].isoformat() if row[10] else None
                }
            
            # Cache the result
            await enhanced_cache.set(
                f"market_{market_id}",
                market_data,
                ttl=1800,
                tags=["market"]
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get market {market_id}: {e}")
            raise e
    
    async def list_markets(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List markets with filters"""
        try:
            # Build query with filters
            where_conditions = []
            params = {}
            
            if filters.get("category"):
                where_conditions.append("category = :category")
                params["category"] = filters["category"]
            
            if filters.get("status"):
                where_conditions.append("status = :status")
                params["status"] = filters["status"]
            
            if filters.get("created_by"):
                where_conditions.append("created_by = :created_by")
                params["created_by"] = filters["created_by"]
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            query = f"""
            SELECT id, title, description, category, market_type, outcomes, 
                   end_date, created_by, status, created_at
            FROM markets 
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
            """
            
            params.update({
                "limit": filters.get("limit", 20),
                "offset": filters.get("offset", 0)
            })
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), params)
                markets = []
                
                for row in result:
                    market_data = {
                        "id": row[0],
                        "title": row[1],
                        "description": row[2],
                        "category": row[3],
                        "market_type": row[4],
                        "outcomes": json.loads(row[5]) if row[5] else [],
                        "end_date": row[6].isoformat() if row[6] else None,
                        "created_by": row[7],
                        "status": row[8],
                        "created_at": row[9].isoformat() if row[9] else None
                    }
                    markets.append(market_data)
            
            return markets
            
        except Exception as e:
            logger.error(f"Failed to list markets: {e}")
            raise e
    
    async def update_market(self, market_id: int, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market information"""
        try:
            # Build update query
            update_fields = []
            params = {"market_id": market_id}
            
            allowed_fields = ["title", "description", "category", "outcomes", "end_date"]
            for field in allowed_fields:
                if field in update_data:
                    update_fields.append(f"{field} = :{field}")
                    params[field] = update_data[field]
            
            if not update_fields:
                raise ValueError("No valid fields to update")
            
            query = f"""
            UPDATE markets 
            SET {', '.join(update_fields)}, updated_at = NOW()
            WHERE id = :market_id
            RETURNING id
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), params)
                if not result.fetchone():
                    raise ValueError(f"Market not found: {market_id}")
                conn.commit()
            
            # Invalidate cache
            await enhanced_cache.delete(f"market_{market_id}")
            
            # Broadcast update event
            await self._broadcast_market_event("market_updated", {
                "market_id": market_id,
                "updated_fields": list(update_data.keys())
            })
            
            return {
                "market_id": market_id,
                "status": "updated",
                "message": "Market updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to update market {market_id}: {e}")
            raise e
    
    async def resolve_market(self, market_id: int, outcome: str) -> Dict[str, Any]:
        """Resolve a market with the winning outcome"""
        try:
            # Update market status
            query = """
            UPDATE markets 
            SET status = 'resolved', winning_outcome = :outcome, resolved_at = NOW()
            WHERE id = :market_id AND status = 'active'
            RETURNING id
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), {
                    "market_id": market_id,
                    "outcome": outcome
                })
                if not result.fetchone():
                    raise ValueError(f"Market not found or already resolved: {market_id}")
                conn.commit()
            
            # Invalidate cache
            await enhanced_cache.delete(f"market_{market_id}")
            
            # Broadcast resolution event
            await self._broadcast_market_event("market_resolved", {
                "market_id": market_id,
                "winning_outcome": outcome
            })
            
            return {
                "market_id": market_id,
                "status": "resolved",
                "winning_outcome": outcome,
                "message": "Market resolved successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to resolve market {market_id}: {e}")
            raise e
    
    async def get_market_analytics(self, market_id: int) -> Dict[str, Any]:
        """Get market analytics and statistics"""
        try:
            # Get market data
            market = await self.get_market(market_id)
            
            # Get trading statistics
            query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(amount) as total_volume,
                AVG(amount) as avg_trade_size,
                COUNT(DISTINCT user_id) as unique_traders
            FROM trades 
            WHERE market_id = :market_id
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), {"market_id": market_id})
                stats = result.fetchone()
            
            analytics = {
                "market": market,
                "trading_stats": {
                    "total_trades": stats[0] or 0,
                    "total_volume": float(stats[1] or 0),
                    "avg_trade_size": float(stats[2] or 0),
                    "unique_traders": stats[3] or 0
                },
                "generated_at": time.time()
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get market analytics for {market_id}: {e}")
            raise e
    
    async def _broadcast_market_event(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast market-related events"""
        try:
            from app.services.inter_service_communication import inter_service_comm
            
            await inter_service_comm.broadcast_event(
                event_type,
                {
                    "service": "market-service",
                    "timestamp": time.time(),
                    **event_data
                },
                target_services=["notification-service", "analytics-service"]
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast market event: {e}")
    
    async def _market_cleanup_loop(self):
        """Background task to clean up expired markets"""
        while True:
            try:
                # Find markets that should be expired
                query = """
                UPDATE markets 
                SET status = 'expired'
                WHERE status = 'active' AND end_date < NOW()
                """
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(text(query))
                    expired_count = result.rowcount
                    conn.commit()
                
                if expired_count > 0:
                    logger.info(f"Expired {expired_count} markets")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in market cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _market_analytics_loop(self):
        """Background task to update market analytics"""
        while True:
            try:
                # Update market analytics cache
                query = """
                SELECT id FROM markets WHERE status = 'active'
                """
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(text(query))
                    market_ids = [row[0] for row in result]
                
                # Update analytics for active markets
                for market_id in market_ids:
                    try:
                        analytics = await self.get_market_analytics(market_id)
                        await enhanced_cache.set(
                            f"market_analytics_{market_id}",
                            analytics,
                            ttl=1800,
                            tags=["analytics", "market"]
                        )
                    except Exception as e:
                        logger.error(f"Failed to update analytics for market {market_id}: {e}")
                
                await asyncio.sleep(1800)  # Update every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in market analytics loop: {e}")
                await asyncio.sleep(1800)
    
    async def check_service_health(self) -> bool:
        """Check market service health"""
        try:
            # Check database connection
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            return True
            
        except Exception as e:
            logger.error(f"Market service health check failed: {e}")
            return False


# Global market service instance
market_service = MarketService()
