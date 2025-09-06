"""
Social Trading API Endpoints
Provides copy trading, social signals, leaderboards, and community features
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.redis_client import get_redis_client
from app.services.social_trading import (
    get_social_trading_service,
    TraderProfile,
    SocialSignal,
    CopyTrade,
    LeaderboardEntry,
    CommunityPost,
)
from app.schemas.social_trading import (
    TraderProfileRequest,
    TraderProfileResponse,
    SocialSignalRequest,
    SocialSignalResponse,
    CopyTradeRequest,
    CopyTradeResponse,
    LeaderboardResponse,
    CommunityPostRequest,
    CommunityPostResponse,
    TraderAnalyticsResponse,
    TraderFeedResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time social trading updates
websocket_connections: Dict[str, WebSocket] = {}


@router.post("/profiles/create", response_model=TraderProfileResponse)
async def create_trader_profile(
    request: TraderProfileRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a new trader profile
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        profile = await social_service.create_trader_profile(
            user_id=current_user.id,
            username=request.username,
            display_name=request.display_name,
            bio=request.bio,
            avatar_url=request.avatar_url,
            risk_level=request.risk_level,
        )

        return TraderProfileResponse(
            user_id=profile.user_id,
            username=profile.username,
            display_name=profile.display_name,
            bio=profile.bio,
            avatar_url=profile.avatar_url,
            join_date=profile.join_date,
            total_trades=profile.total_trades,
            win_rate=profile.win_rate,
            total_profit=profile.total_profit,
            followers_count=profile.followers_count,
            following_count=profile.following_count,
            is_verified=profile.is_verified,
            risk_level=profile.risk_level,
            preferred_markets=profile.preferred_markets,
            performance_rating=profile.performance_rating,
            last_active=profile.last_active,
        )

    except Exception as e:
        logger.error(f"Error creating trader profile: {e}")
        raise HTTPException(status_code=500, detail="Error creating trader profile")


@router.get("/profiles/{user_id}", response_model=TraderProfileResponse)
async def get_trader_profile(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get trader profile by user ID
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        profile = social_service.trader_profiles.get(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Trader profile not found")

        return TraderProfileResponse(
            user_id=profile.user_id,
            username=profile.username,
            display_name=profile.display_name,
            bio=profile.bio,
            avatar_url=profile.avatar_url,
            join_date=profile.join_date,
            total_trades=profile.total_trades,
            win_rate=profile.win_rate,
            total_profit=profile.total_profit,
            followers_count=profile.followers_count,
            following_count=profile.following_count,
            is_verified=profile.is_verified,
            risk_level=profile.risk_level,
            preferred_markets=profile.preferred_markets,
            performance_rating=profile.performance_rating,
            last_active=profile.last_active,
        )

    except Exception as e:
        logger.error(f"Error getting trader profile: {e}")
        raise HTTPException(status_code=500, detail="Error getting trader profile")


@router.post("/signals/create", response_model=SocialSignalResponse)
async def create_social_signal(
    request: SocialSignalRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a social trading signal
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        signal = await social_service.create_social_signal(
            trader_id=current_user.id,
            market_id=request.market_id,
            signal_type=request.signal_type,
            confidence=request.confidence,
            reasoning=request.reasoning,
            target_price=request.target_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            tags=request.tags,
        )

        return SocialSignalResponse(
            signal_id=signal.signal_id,
            trader_id=signal.trader_id,
            market_id=signal.market_id,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            created_at=signal.created_at,
            expires_at=signal.expires_at,
            followers_count=signal.followers_count,
            success_rate=signal.success_rate,
            tags=signal.tags,
        )

    except Exception as e:
        logger.error(f"Error creating social signal: {e}")
        raise HTTPException(status_code=500, detail="Error creating social signal")


@router.get("/signals/{signal_id}", response_model=SocialSignalResponse)
async def get_social_signal(
    signal_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get social signal by ID
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        signal = social_service.social_signals.get(signal_id)
        if not signal:
            raise HTTPException(status_code=404, detail="Social signal not found")

        return SocialSignalResponse(
            signal_id=signal.signal_id,
            trader_id=signal.trader_id,
            market_id=signal.market_id,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            created_at=signal.created_at,
            expires_at=signal.expires_at,
            followers_count=signal.followers_count,
            success_rate=signal.success_rate,
            tags=signal.tags,
        )

    except Exception as e:
        logger.error(f"Error getting social signal: {e}")
        raise HTTPException(status_code=500, detail="Error getting social signal")


@router.get("/signals")
async def get_social_signals(
    trader_id: Optional[int] = Query(None, description="Filter by trader ID"),
    market_id: Optional[int] = Query(None, description="Filter by market ID"),
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    limit: int = Query(20, description="Number of signals to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get social signals with optional filtering
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        signals = []
        for signal in social_service.social_signals.values():
            # Apply filters
            if trader_id and signal.trader_id != trader_id:
                continue
            if market_id and signal.market_id != market_id:
                continue
            if signal_type and signal.signal_type != signal_type:
                continue

            signals.append(
                {
                    "signal_id": signal.signal_id,
                    "trader_id": signal.trader_id,
                    "market_id": signal.market_id,
                    "signal_type": signal.signal_type,
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning,
                    "target_price": signal.target_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "created_at": signal.created_at.isoformat(),
                    "expires_at": signal.expires_at.isoformat(),
                    "followers_count": signal.followers_count,
                    "success_rate": signal.success_rate,
                    "tags": signal.tags,
                }
            )

        # Sort by creation date and limit
        signals.sort(key=lambda x: x["created_at"], reverse=True)
        signals = signals[:limit]

        return JSONResponse(
            content={
                "signals": signals,
                "count": len(signals),
                "filters": {
                    "trader_id": trader_id,
                    "market_id": market_id,
                    "signal_type": signal_type,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting social signals: {e}")
        raise HTTPException(status_code=500, detail="Error getting social signals")


@router.post("/follow/{leader_id}")
async def follow_trader(
    leader_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Follow a trader
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        success = await social_service.follow_trader(current_user.id, leader_id)

        if success:
            return JSONResponse(
                content={
                    "message": f"Successfully started following trader {leader_id}",
                    "follower_id": current_user.id,
                    "leader_id": leader_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Already following this trader or invalid request",
            )

    except Exception as e:
        logger.error(f"Error following trader: {e}")
        raise HTTPException(status_code=500, detail="Error following trader")


@router.delete("/unfollow/{leader_id}")
async def unfollow_trader(
    leader_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Unfollow a trader
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        success = await social_service.unfollow_trader(current_user.id, leader_id)

        if success:
            return JSONResponse(
                content={
                    "message": f"Successfully unfollowed trader {leader_id}",
                    "follower_id": current_user.id,
                    "leader_id": leader_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Not following this trader")

    except Exception as e:
        logger.error(f"Error unfollowing trader: {e}")
        raise HTTPException(status_code=500, detail="Error unfollowing trader")


@router.post("/copy-trades/create", response_model=CopyTradeResponse)
async def create_copy_trade(
    request: CopyTradeRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a copy trade
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        copy_trade = await social_service.create_copy_trade(
            follower_id=current_user.id,
            leader_id=request.leader_id,
            market_id=request.market_id,
            original_trade_id=request.original_trade_id,
            copy_percentage=request.copy_percentage,
        )

        return CopyTradeResponse(
            copy_trade_id=copy_trade.copy_trade_id,
            follower_id=copy_trade.follower_id,
            leader_id=copy_trade.leader_id,
            market_id=copy_trade.market_id,
            original_trade_id=copy_trade.original_trade_id,
            copied_amount=copy_trade.copied_amount,
            copy_percentage=copy_trade.copy_percentage,
            status=copy_trade.status,
            created_at=copy_trade.created_at,
            completed_at=copy_trade.completed_at,
            profit_loss=copy_trade.profit_loss,
            performance_ratio=copy_trade.performance_ratio,
        )

    except Exception as e:
        logger.error(f"Error creating copy trade: {e}")
        raise HTTPException(status_code=500, detail="Error creating copy trade")


@router.get("/copy-trades")
async def get_copy_trades(
    follower_id: Optional[int] = Query(None, description="Filter by follower ID"),
    leader_id: Optional[int] = Query(None, description="Filter by leader ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, description="Number of copy trades to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get copy trades with optional filtering
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        copy_trades = []
        for copy_trade in social_service.copy_trades.values():
            # Apply filters
            if follower_id and copy_trade.follower_id != follower_id:
                continue
            if leader_id and copy_trade.leader_id != leader_id:
                continue
            if status and copy_trade.status != status:
                continue

            copy_trades.append(
                {
                    "copy_trade_id": copy_trade.copy_trade_id,
                    "follower_id": copy_trade.follower_id,
                    "leader_id": copy_trade.leader_id,
                    "market_id": copy_trade.market_id,
                    "original_trade_id": copy_trade.original_trade_id,
                    "copied_amount": copy_trade.copied_amount,
                    "copy_percentage": copy_trade.copy_percentage,
                    "status": copy_trade.status,
                    "created_at": copy_trade.created_at.isoformat(),
                    "completed_at": (
                        copy_trade.completed_at.isoformat()
                        if copy_trade.completed_at
                        else None
                    ),
                    "profit_loss": copy_trade.profit_loss,
                    "performance_ratio": copy_trade.performance_ratio,
                }
            )

        # Sort by creation date and limit
        copy_trades.sort(key=lambda x: x["created_at"], reverse=True)
        copy_trades = copy_trades[:limit]

        return JSONResponse(
            content={
                "copy_trades": copy_trades,
                "count": len(copy_trades),
                "filters": {
                    "follower_id": follower_id,
                    "leader_id": leader_id,
                    "status": status,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting copy trades: {e}")
        raise HTTPException(status_code=500, detail="Error getting copy trades")


@router.get("/leaderboard", response_model=List[LeaderboardResponse])
async def get_leaderboard(
    period: str = Query("weekly", description="Leaderboard period"),
    limit: int = Query(50, description="Number of entries to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get leaderboard for a specific period
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        leaderboard = await social_service.get_leaderboard(period, limit)

        return [
            LeaderboardResponse(
                user_id=entry.user_id,
                username=entry.username,
                display_name=entry.display_name,
                avatar_url=entry.avatar_url,
                rank=entry.rank,
                total_profit=entry.total_profit,
                win_rate=entry.win_rate,
                total_trades=entry.total_trades,
                followers_count=entry.followers_count,
                performance_score=entry.performance_score,
                period=entry.period,
                last_updated=entry.last_updated,
            )
            for entry in leaderboard
        ]

    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Error getting leaderboard")


@router.post("/posts/create", response_model=CommunityPostResponse)
async def create_community_post(
    request: CommunityPostRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a community post
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        post = await social_service.create_community_post(
            user_id=current_user.id,
            content=request.content,
            post_type=request.post_type,
            market_id=request.market_id,
            tags=request.tags,
        )

        return CommunityPostResponse(
            post_id=post.post_id,
            user_id=post.user_id,
            username=post.username,
            display_name=post.display_name,
            avatar_url=post.avatar_url,
            content=post.content,
            post_type=post.post_type,
            market_id=post.market_id,
            tags=post.tags,
            likes_count=post.likes_count,
            comments_count=post.comments_count,
            shares_count=post.shares_count,
            created_at=post.created_at,
            updated_at=post.updated_at,
            is_verified=post.is_verified,
        )

    except Exception as e:
        logger.error(f"Error creating community post: {e}")
        raise HTTPException(status_code=500, detail="Error creating community post")


@router.get("/posts")
async def get_community_posts(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    post_type: Optional[str] = Query(None, description="Filter by post type"),
    market_id: Optional[int] = Query(None, description="Filter by market ID"),
    limit: int = Query(20, description="Number of posts to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get community posts with optional filtering
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        posts = []
        for post in social_service.community_posts.values():
            # Apply filters
            if user_id and post.user_id != user_id:
                continue
            if post_type and post.post_type != post_type:
                continue
            if market_id and post.market_id != market_id:
                continue

            posts.append(
                {
                    "post_id": post.post_id,
                    "user_id": post.user_id,
                    "username": post.username,
                    "display_name": post.display_name,
                    "avatar_url": post.avatar_url,
                    "content": post.content,
                    "post_type": post.post_type,
                    "market_id": post.market_id,
                    "tags": post.tags,
                    "likes_count": post.likes_count,
                    "comments_count": post.comments_count,
                    "shares_count": post.shares_count,
                    "created_at": post.created_at.isoformat(),
                    "updated_at": post.updated_at.isoformat(),
                    "is_verified": post.is_verified,
                }
            )

        # Sort by creation date and limit
        posts.sort(key=lambda x: x["created_at"], reverse=True)
        posts = posts[:limit]

        return JSONResponse(
            content={
                "posts": posts,
                "count": len(posts),
                "filters": {
                    "user_id": user_id,
                    "post_type": post_type,
                    "market_id": market_id,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting community posts: {e}")
        raise HTTPException(status_code=500, detail="Error getting community posts")


@router.get("/feed", response_model=TraderFeedResponse)
async def get_trader_feed(
    limit: int = Query(20, description="Number of feed items to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get personalized feed for the current trader
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        feed_items = await social_service.get_trader_feed(current_user.id, limit)

        return TraderFeedResponse(
            user_id=current_user.id,
            feed_items=feed_items,
            count=len(feed_items),
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Error getting trader feed: {e}")
        raise HTTPException(status_code=500, detail="Error getting trader feed")


@router.get("/analytics/{user_id}", response_model=TraderAnalyticsResponse)
async def get_trader_analytics(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get comprehensive analytics for a trader
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        analytics = await social_service.get_trader_analytics(user_id)

        if not analytics:
            raise HTTPException(status_code=404, detail="Trader analytics not found")

        return TraderAnalyticsResponse(
            user_id=user_id, analytics=analytics, timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Error getting trader analytics: {e}")
        raise HTTPException(status_code=500, detail="Error getting trader analytics")


@router.websocket("/ws/social-trading/{client_id}")
async def websocket_social_trading_updates(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time social trading updates
    """
    await websocket.accept()
    websocket_connections[client_id] = websocket

    try:
        # Send initial connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "type": "social_trading_connected",
                    "client_id": client_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "available_features": [
                        "trader_updates",
                        "signal_notifications",
                        "leaderboard_updates",
                        "community_posts",
                        "copy_trade_alerts",
                    ],
                }
            )
        )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "subscribe_trader_updates":
                    trader_id = message.get("trader_id")
                    # Subscribe to trader updates
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "trader_subscription_confirmed",
                                "trader_id": trader_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "subscribe_signals":
                    # Subscribe to signal notifications
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "signal_subscription_confirmed",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "get_leaderboard_updates":
                    period = message.get("period", "weekly")
                    # Get leaderboard updates
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "leaderboard_update",
                                "period": period,
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": {
                                    "update_type": "periodic",
                                    "entries_count": 50,
                                },
                            }
                        )
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info(f"Social trading client {client_id} disconnected")
    finally:
        if client_id in websocket_connections:
            del websocket_connections[client_id]


@router.get("/health/social-trading")
async def social_trading_health():
    """
    Health check for social trading services
    """
    return {
        "status": "healthy",
        "service": "social_trading",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "trader_profiles",
            "social_signals",
            "copy_trading",
            "leaderboards",
            "community_posts",
            "trader_analytics",
            "personalized_feeds",
            "websocket_updates",
        ],
        "metrics": {
            "active_traders": 0,  # Would be calculated from actual data
            "total_signals": 0,
            "active_copy_trades": 0,
            "community_posts": 0,
        },
    }


@router.get("/stats/social-trading")
async def get_social_trading_stats(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get social trading statistics
    """
    try:
        redis_client = await get_redis_client()
        social_service = await get_social_trading_service(redis_client, db)

        stats = {
            "total_traders": len(social_service.trader_profiles),
            "total_signals": len(social_service.social_signals),
            "total_copy_trades": len(social_service.copy_trades),
            "total_posts": len(social_service.community_posts),
            "active_signals": len(
                [
                    s
                    for s in social_service.social_signals.values()
                    if s.expires_at > datetime.utcnow()
                ]
            ),
            "active_copy_trades": len(
                [c for c in social_service.copy_trades.values() if c.status == "active"]
            ),
            "top_performers": [
                {
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "performance_rating": profile.performance_rating,
                }
                for profile in sorted(
                    social_service.trader_profiles.values(),
                    key=lambda p: p.performance_rating,
                    reverse=True,
                )[:5]
            ],
            "most_followed_traders": [
                {
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "followers_count": profile.followers_count,
                }
                for profile in sorted(
                    social_service.trader_profiles.values(),
                    key=lambda p: p.followers_count,
                    reverse=True,
                )[:5]
            ],
        }

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Error getting social trading stats: {e}")
        raise HTTPException(
            status_code=500, detail="Error getting social trading stats"
        )
