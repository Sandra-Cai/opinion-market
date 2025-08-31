"""
Pydantic schemas for Social Trading API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TraderProfileRequest(BaseModel):
    """Request model for creating a trader profile"""
    username: str = Field(..., description="Username")
    display_name: str = Field(..., description="Display name")
    bio: str = Field(default="", description="Trader bio")
    avatar_url: str = Field(default="", description="Avatar URL")
    risk_level: str = Field(default="moderate", description="Risk level (conservative, moderate, aggressive)")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "crypto_trader_123",
                "display_name": "Crypto Trader",
                "bio": "Experienced cryptocurrency trader with 5+ years in the market",
                "avatar_url": "https://example.com/avatar.jpg",
                "risk_level": "moderate"
            }
        }


class TraderProfileResponse(BaseModel):
    """Response model for trader profile"""
    user_id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    display_name: str = Field(..., description="Display name")
    bio: str = Field(..., description="Trader bio")
    avatar_url: str = Field(..., description="Avatar URL")
    join_date: datetime = Field(..., description="Join date")
    total_trades: int = Field(..., description="Total number of trades")
    win_rate: float = Field(..., description="Win rate percentage")
    total_profit: float = Field(..., description="Total profit/loss")
    followers_count: int = Field(..., description="Number of followers")
    following_count: int = Field(..., description="Number of traders being followed")
    is_verified: bool = Field(..., description="Verification status")
    risk_level: str = Field(..., description="Risk level")
    preferred_markets: List[str] = Field(..., description="Preferred markets")
    performance_rating: float = Field(..., description="Performance rating")
    last_active: datetime = Field(..., description="Last active timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "username": "crypto_trader_123",
                "display_name": "Crypto Trader",
                "bio": "Experienced cryptocurrency trader with 5+ years in the market",
                "avatar_url": "https://example.com/avatar.jpg",
                "join_date": "2024-01-15T10:30:00Z",
                "total_trades": 150,
                "win_rate": 0.75,
                "total_profit": 2500.0,
                "followers_count": 1250,
                "following_count": 45,
                "is_verified": True,
                "risk_level": "moderate",
                "preferred_markets": ["cryptocurrency", "stocks", "forex"],
                "performance_rating": 85.5,
                "last_active": "2024-01-15T10:30:00Z"
            }
        }


class SocialSignalRequest(BaseModel):
    """Request model for creating a social signal"""
    market_id: int = Field(..., description="Market ID")
    signal_type: str = Field(..., description="Signal type (buy, sell, hold)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    reasoning: str = Field(..., description="Signal reasoning")
    target_price: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    tags: List[str] = Field(default=[], description="Signal tags")
    
    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "signal_type": "buy",
                "confidence": 0.85,
                "reasoning": "Strong technical indicators showing bullish momentum",
                "target_price": 50000.0,
                "stop_loss": 45000.0,
                "take_profit": 55000.0,
                "tags": ["technical_analysis", "bullish", "bitcoin"]
            }
        }


class SocialSignalResponse(BaseModel):
    """Response model for social signal"""
    signal_id: str = Field(..., description="Signal ID")
    trader_id: int = Field(..., description="Trader ID")
    market_id: int = Field(..., description="Market ID")
    signal_type: str = Field(..., description="Signal type")
    confidence: float = Field(..., description="Confidence level")
    reasoning: str = Field(..., description="Signal reasoning")
    target_price: Optional[float] = Field(None, description="Target price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")
    followers_count: int = Field(..., description="Number of followers")
    success_rate: float = Field(..., description="Success rate")
    tags: List[str] = Field(..., description="Signal tags")
    
    class Config:
        schema_extra = {
            "example": {
                "signal_id": "signal_1_1_20240115_103000",
                "trader_id": 1,
                "market_id": 1,
                "signal_type": "buy",
                "confidence": 0.85,
                "reasoning": "Strong technical indicators showing bullish momentum",
                "target_price": 50000.0,
                "stop_loss": 45000.0,
                "take_profit": 55000.0,
                "created_at": "2024-01-15T10:30:00Z",
                "expires_at": "2024-01-16T10:30:00Z",
                "followers_count": 45,
                "success_rate": 0.75,
                "tags": ["technical_analysis", "bullish", "bitcoin"]
            }
        }


class CopyTradeRequest(BaseModel):
    """Request model for creating a copy trade"""
    leader_id: int = Field(..., description="Leader trader ID")
    market_id: int = Field(..., description="Market ID")
    original_trade_id: str = Field(..., description="Original trade ID")
    copy_percentage: float = Field(..., ge=1.0, le=100.0, description="Copy percentage (1-100)")
    
    class Config:
        schema_extra = {
            "example": {
                "leader_id": 1,
                "market_id": 1,
                "original_trade_id": "trade_123",
                "copy_percentage": 50.0
            }
        }


class CopyTradeResponse(BaseModel):
    """Response model for copy trade"""
    copy_trade_id: str = Field(..., description="Copy trade ID")
    follower_id: int = Field(..., description="Follower ID")
    leader_id: int = Field(..., description="Leader ID")
    market_id: int = Field(..., description="Market ID")
    original_trade_id: str = Field(..., description="Original trade ID")
    copied_amount: float = Field(..., description="Copied amount")
    copy_percentage: float = Field(..., description="Copy percentage")
    status: str = Field(..., description="Copy trade status")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    profit_loss: Optional[float] = Field(None, description="Profit/loss")
    performance_ratio: float = Field(..., description="Performance ratio")
    
    class Config:
        schema_extra = {
            "example": {
                "copy_trade_id": "copy_2_1_20240115_103000",
                "follower_id": 2,
                "leader_id": 1,
                "market_id": 1,
                "original_trade_id": "trade_123",
                "copied_amount": 500.0,
                "copy_percentage": 50.0,
                "status": "active",
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": None,
                "profit_loss": None,
                "performance_ratio": 1.0
            }
        }


class LeaderboardResponse(BaseModel):
    """Response model for leaderboard entry"""
    user_id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    display_name: str = Field(..., description="Display name")
    avatar_url: str = Field(..., description="Avatar URL")
    rank: int = Field(..., description="Rank position")
    total_profit: float = Field(..., description="Total profit")
    win_rate: float = Field(..., description="Win rate")
    total_trades: int = Field(..., description="Total trades")
    followers_count: int = Field(..., description="Followers count")
    performance_score: float = Field(..., description="Performance score")
    period: str = Field(..., description="Leaderboard period")
    last_updated: datetime = Field(..., description="Last updated timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "username": "crypto_trader_123",
                "display_name": "Crypto Trader",
                "avatar_url": "https://example.com/avatar.jpg",
                "rank": 1,
                "total_profit": 2500.0,
                "win_rate": 0.75,
                "total_trades": 150,
                "followers_count": 1250,
                "performance_score": 85.5,
                "period": "weekly",
                "last_updated": "2024-01-15T10:30:00Z"
            }
        }


class CommunityPostRequest(BaseModel):
    """Request model for creating a community post"""
    content: str = Field(..., description="Post content")
    post_type: str = Field(..., description="Post type (analysis, prediction, discussion, news)")
    market_id: Optional[int] = Field(None, description="Related market ID")
    tags: List[str] = Field(default=[], description="Post tags")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Bitcoin showing strong support at $45k level. Expecting breakout above $50k resistance.",
                "post_type": "analysis",
                "market_id": 1,
                "tags": ["bitcoin", "technical_analysis", "breakout"]
            }
        }


class CommunityPostResponse(BaseModel):
    """Response model for community post"""
    post_id: str = Field(..., description="Post ID")
    user_id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    display_name: str = Field(..., description="Display name")
    avatar_url: str = Field(..., description="Avatar URL")
    content: str = Field(..., description="Post content")
    post_type: str = Field(..., description="Post type")
    market_id: Optional[int] = Field(None, description="Related market ID")
    tags: List[str] = Field(..., description="Post tags")
    likes_count: int = Field(..., description="Likes count")
    comments_count: int = Field(..., description="Comments count")
    shares_count: int = Field(..., description="Shares count")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_verified: bool = Field(..., description="Verification status")
    
    class Config:
        schema_extra = {
            "example": {
                "post_id": "post_1_20240115_103000",
                "user_id": 1,
                "username": "crypto_trader_123",
                "display_name": "Crypto Trader",
                "avatar_url": "https://example.com/avatar.jpg",
                "content": "Bitcoin showing strong support at $45k level. Expecting breakout above $50k resistance.",
                "post_type": "analysis",
                "market_id": 1,
                "tags": ["bitcoin", "technical_analysis", "breakout"],
                "likes_count": 25,
                "comments_count": 8,
                "shares_count": 3,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "is_verified": True
            }
        }


class TraderFeedResponse(BaseModel):
    """Response model for trader feed"""
    user_id: int = Field(..., description="User ID")
    feed_items: List[Dict[str, Any]] = Field(..., description="Feed items")
    count: int = Field(..., description="Number of feed items")
    timestamp: datetime = Field(..., description="Feed timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "feed_items": [
                    {
                        "type": "signal",
                        "data": {
                            "signal_id": "signal_1_1_20240115_103000",
                            "trader_id": 1,
                            "market_id": 1,
                            "signal_type": "buy",
                            "confidence": 0.85
                        },
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                ],
                "count": 1,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class TraderAnalyticsResponse(BaseModel):
    """Response model for trader analytics"""
    user_id: int = Field(..., description="User ID")
    analytics: Dict[str, Any] = Field(..., description="Analytics data")
    timestamp: datetime = Field(..., description="Analytics timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "analytics": {
                    "profile": {
                        "user_id": 1,
                        "username": "crypto_trader_123",
                        "total_trades": 150,
                        "win_rate": 0.75,
                        "total_profit": 2500.0
                    },
                    "performance": {
                        "daily_profit": 150.0,
                        "weekly_profit": 800.0,
                        "monthly_profit": 2500.0
                    },
                    "social": {
                        "signals_created": 25,
                        "posts_created": 15,
                        "copy_trades": 5
                    }
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SocialTradingHealthResponse(BaseModel):
    """Response model for social trading health check"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    features: List[str] = Field(..., description="Available features")
    metrics: Dict[str, Any] = Field(..., description="Service metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "social_trading",
                "timestamp": "2024-01-15T10:30:00Z",
                "features": [
                    "trader_profiles",
                    "social_signals",
                    "copy_trading",
                    "leaderboards",
                    "community_posts",
                    "trader_analytics",
                    "personalized_feeds",
                    "websocket_updates"
                ],
                "metrics": {
                    "active_traders": 1500,
                    "total_signals": 2500,
                    "active_copy_trades": 800,
                    "community_posts": 1200
                }
            }
        }


class WebSocketSocialTradingMessage(BaseModel):
    """Base model for WebSocket social trading messages"""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")


class WebSocketTraderSubscription(BaseModel):
    """Request model for WebSocket trader subscription"""
    type: str = Field("subscribe_trader_updates", description="Message type")
    trader_id: int = Field(..., description="Trader ID to subscribe to")


class WebSocketSignalSubscription(BaseModel):
    """Request model for WebSocket signal subscription"""
    type: str = Field("subscribe_signals", description="Message type")


class WebSocketLeaderboardRequest(BaseModel):
    """Request model for WebSocket leaderboard updates"""
    type: str = Field("get_leaderboard_updates", description="Message type")
    period: str = Field(default="weekly", description="Leaderboard period")


class SocialTradingStatsResponse(BaseModel):
    """Response model for social trading statistics"""
    total_traders: int = Field(..., description="Total number of traders")
    total_signals: int = Field(..., description="Total number of signals")
    total_copy_trades: int = Field(..., description="Total number of copy trades")
    total_posts: int = Field(..., description="Total number of community posts")
    active_signals: int = Field(..., description="Number of active signals")
    active_copy_trades: int = Field(..., description="Number of active copy trades")
    top_performers: List[Dict[str, Any]] = Field(..., description="Top performing traders")
    most_followed_traders: List[Dict[str, Any]] = Field(..., description="Most followed traders")
    
    class Config:
        schema_extra = {
            "example": {
                "total_traders": 1500,
                "total_signals": 2500,
                "total_copy_trades": 800,
                "total_posts": 1200,
                "active_signals": 150,
                "active_copy_trades": 300,
                "top_performers": [
                    {
                        "user_id": 1,
                        "username": "crypto_trader_123",
                        "performance_rating": 85.5
                    }
                ],
                "most_followed_traders": [
                    {
                        "user_id": 1,
                        "username": "crypto_trader_123",
                        "followers_count": 1250
                    }
                ]
            }
        }


class FollowRequest(BaseModel):
    """Request model for following a trader"""
    leader_id: int = Field(..., description="Leader trader ID")
    
    class Config:
        schema_extra = {
            "example": {
                "leader_id": 1
            }
        }


class FollowResponse(BaseModel):
    """Response model for follow action"""
    message: str = Field(..., description="Response message")
    follower_id: int = Field(..., description="Follower ID")
    leader_id: int = Field(..., description="Leader ID")
    timestamp: datetime = Field(..., description="Action timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Successfully started following trader 1",
                "follower_id": 2,
                "leader_id": 1,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class SignalFilterRequest(BaseModel):
    """Request model for filtering signals"""
    trader_id: Optional[int] = Field(None, description="Filter by trader ID")
    market_id: Optional[int] = Field(None, description="Filter by market ID")
    signal_type: Optional[str] = Field(None, description="Filter by signal type")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=20, description="Number of signals to return")
    
    class Config:
        schema_extra = {
            "example": {
                "trader_id": 1,
                "market_id": 1,
                "signal_type": "buy",
                "min_confidence": 0.7,
                "tags": ["bitcoin", "technical_analysis"],
                "limit": 20
            }
        }


class CopyTradeFilterRequest(BaseModel):
    """Request model for filtering copy trades"""
    follower_id: Optional[int] = Field(None, description="Filter by follower ID")
    leader_id: Optional[int] = Field(None, description="Filter by leader ID")
    market_id: Optional[int] = Field(None, description="Filter by market ID")
    status: Optional[str] = Field(None, description="Filter by status")
    min_copy_percentage: Optional[float] = Field(None, ge=1.0, le=100.0, description="Minimum copy percentage")
    limit: int = Field(default=20, description="Number of copy trades to return")
    
    class Config:
        schema_extra = {
            "example": {
                "follower_id": 2,
                "leader_id": 1,
                "market_id": 1,
                "status": "active",
                "min_copy_percentage": 25.0,
                "limit": 20
            }
        }
