import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
import redis.asyncio as redis

from app.core.database import SessionLocal
from app.models.user import User
from app.models.market import Market
from app.models.trade import Trade
from app.services.machine_learning import get_ml_service

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Enhanced user profile with social features"""

    user_id: int
    username: str
    display_name: str
    bio: str
    avatar_url: str
    cover_image_url: str
    location: str
    website: str
    social_links: Dict[str, str]
    trading_stats: Dict[str, Any]
    reputation_score: float
    followers_count: int
    following_count: int
    is_verified: bool
    badges: List[str]
    created_at: datetime
    last_active: datetime
    metadata: Dict[str, Any]


@dataclass
class SocialPost:
    """Social post in the platform"""

    post_id: str
    user_id: int
    content: str
    post_type: str  # market_analysis, trade_alert, general, etc.
    market_id: Optional[int]
    trade_id: Optional[int]
    images: List[str]
    tags: List[str]
    likes_count: int
    comments_count: int
    shares_count: int
    is_pinned: bool
    is_edited: bool
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]


@dataclass
class Community:
    """Trading community/group"""

    community_id: str
    name: str
    description: str
    avatar_url: str
    cover_image_url: str
    owner_id: int
    moderators: List[int]
    members_count: int
    is_private: bool
    is_verified: bool
    rules: List[str]
    categories: List[str]
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class SocialAnalytics:
    """Social analytics and insights"""

    user_id: int
    influence_score: float
    engagement_rate: float
    reach_impressions: int
    follower_growth_rate: float
    content_performance: Dict[str, Any]
    audience_demographics: Dict[str, Any]
    best_performing_content: List[str]
    trending_topics: List[str]
    timestamp: datetime


class SocialFeaturesService:
    """Comprehensive social features service"""

    def __init__(self):
        self.redis_client: Optional[redis_sync.Redis] = None
        self.ml_service = get_ml_service()
        self.user_profiles = {}
        self.social_posts = {}
        self.communities = {}
        self.followers_cache = {}
        self.engagement_cache = {}

        # Social scoring weights
        self.scoring_weights = {
            "trading_performance": 0.4,
            "social_engagement": 0.3,
            "content_quality": 0.2,
            "community_contribution": 0.1,
        }

    async def initialize(self, redis_url: str):
        """Initialize the social features service"""
        self.redis_client = redis.from_url(redis_url)
        await self.redis_client.ping()
        logger.info("Social features service initialized")

    async def create_user_profile(
        self, user_id: int, profile_data: Dict[str, Any]
    ) -> Optional[UserProfile]:
        """Create or update user profile"""
        try:
            db = SessionLocal()
            user = db.query(User).filter(User.id == user_id).first()

            if not user:
                return None

            # Get trading statistics
            trading_stats = await self._calculate_trading_stats(user_id)

            # Calculate reputation score
            reputation_score = await self._calculate_reputation_score(
                user_id, trading_stats
            )

            # Create profile
            profile = UserProfile(
                user_id=user_id,
                username=user.username,
                display_name=profile_data.get("display_name", user.username),
                bio=profile_data.get("bio", ""),
                avatar_url=profile_data.get("avatar_url", ""),
                cover_image_url=profile_data.get("cover_image_url", ""),
                location=profile_data.get("location", ""),
                website=profile_data.get("website", ""),
                social_links=profile_data.get("social_links", {}),
                trading_stats=trading_stats,
                reputation_score=reputation_score,
                followers_count=await self._get_followers_count(user_id),
                following_count=await self._get_following_count(user_id),
                is_verified=profile_data.get("is_verified", False),
                badges=await self._get_user_badges(user_id),
                created_at=user.created_at,
                last_active=datetime.utcnow(),
                metadata=profile_data.get("metadata", {}),
            )

            # Cache profile
            self.user_profiles[user_id] = profile

            # Store in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    f"user_profile:{user_id}",
                    3600,  # 1 hour
                    json.dumps(
                        {
                            "display_name": profile.display_name,
                            "bio": profile.bio,
                            "avatar_url": profile.avatar_url,
                            "reputation_score": profile.reputation_score,
                            "followers_count": profile.followers_count,
                            "badges": profile.badges,
                        }
                    ),
                )

            db.close()
            return profile

        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return None

    async def create_social_post(
        self, user_id: int, post_data: Dict[str, Any]
    ) -> Optional[SocialPost]:
        """Create a social post"""
        try:
            post_id = f"post_{int(datetime.utcnow().timestamp())}_{user_id}"

            post = SocialPost(
                post_id=post_id,
                user_id=user_id,
                content=post_data["content"],
                post_type=post_data.get("post_type", "general"),
                market_id=post_data.get("market_id"),
                trade_id=post_data.get("trade_id"),
                images=post_data.get("images", []),
                tags=post_data.get("tags", []),
                likes_count=0,
                comments_count=0,
                shares_count=0,
                is_pinned=False,
                is_edited=False,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=post_data.get("metadata", {}),
            )

            # Store post
            self.social_posts[post_id] = post

            # Add to user's feed
            await self._add_to_user_feed(user_id, post_id)

            # Process hashtags and mentions
            await self._process_post_tags(post)

            # Update user activity
            await self._update_user_activity(user_id)

            logger.info(f"Social post created: {post_id}")
            return post

        except Exception as e:
            logger.error(f"Error creating social post: {e}")
            return None

    async def create_community(
        self, owner_id: int, community_data: Dict[str, Any]
    ) -> Optional[Community]:
        """Create a trading community"""
        try:
            community_id = f"community_{int(datetime.utcnow().timestamp())}_{owner_id}"

            community = Community(
                community_id=community_id,
                name=community_data["name"],
                description=community_data["description"],
                avatar_url=community_data.get("avatar_url", ""),
                cover_image_url=community_data.get("cover_image_url", ""),
                owner_id=owner_id,
                moderators=[owner_id],
                members_count=1,  # Owner is first member
                is_private=community_data.get("is_private", False),
                is_verified=False,
                rules=community_data.get("rules", []),
                categories=community_data.get("categories", []),
                created_at=datetime.utcnow(),
                metadata=community_data.get("metadata", {}),
            )

            # Store community
            self.communities[community_id] = community

            # Add owner as member
            await self._add_community_member(community_id, owner_id, "owner")

            logger.info(f"Community created: {community_id}")
            return community

        except Exception as e:
            logger.error(f"Error creating community: {e}")
            return None

    async def follow_user(self, follower_id: int, followed_id: int) -> bool:
        """Follow a user"""
        try:
            if follower_id == followed_id:
                return False

            # Check if already following
            if await self._is_following(follower_id, followed_id):
                return False

            # Add follow relationship
            await self._add_follow_relationship(follower_id, followed_id)

            # Update follower counts
            await self._update_follower_counts(followed_id, 1)
            await self._update_following_counts(follower_id, 1)

            # Add to feed
            await self._add_to_follower_feed(follower_id, followed_id)

            # Send notification
            await self._send_follow_notification(followed_id, follower_id)

            logger.info(f"User {follower_id} followed user {followed_id}")
            return True

        except Exception as e:
            logger.error(f"Error following user: {e}")
            return False

    async def unfollow_user(self, follower_id: int, followed_id: int) -> bool:
        """Unfollow a user"""
        try:
            if not await self._is_following(follower_id, followed_id):
                return False

            # Remove follow relationship
            await self._remove_follow_relationship(follower_id, followed_id)

            # Update follower counts
            await self._update_follower_counts(followed_id, -1)
            await self._update_following_counts(follower_id, -1)

            # Remove from feed
            await self._remove_from_follower_feed(follower_id, followed_id)

            logger.info(f"User {follower_id} unfollowed user {followed_id}")
            return True

        except Exception as e:
            logger.error(f"Error unfollowing user: {e}")
            return False

    async def like_post(self, user_id: int, post_id: str) -> bool:
        """Like a social post"""
        try:
            if post_id not in self.social_posts:
                return False

            # Check if already liked
            if await self._has_liked_post(user_id, post_id):
                return False

            # Add like
            await self._add_post_like(user_id, post_id)

            # Update post like count
            self.social_posts[post_id].likes_count += 1

            # Send notification
            post = self.social_posts[post_id]
            if post.user_id != user_id:
                await self._send_like_notification(post.user_id, user_id, post_id)

            logger.info(f"User {user_id} liked post {post_id}")
            return True

        except Exception as e:
            logger.error(f"Error liking post: {e}")
            return False

    async def get_user_feed(
        self, user_id: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get user's personalized feed"""
        try:
            # Get posts from followed users and communities
            feed_posts = await self._get_feed_posts(user_id, limit)

            # Add engagement data
            for post in feed_posts:
                post["user_has_liked"] = await self._has_liked_post(
                    user_id, post["post_id"]
                )
                post["user_has_shared"] = await self._has_shared_post(
                    user_id, post["post_id"]
                )

            return feed_posts

        except Exception as e:
            logger.error(f"Error getting user feed: {e}")
            return []

    async def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending topics and hashtags"""
        try:
            if not self.redis_client:
                return []

            # Get trending hashtags from Redis
            trending_hashtags = await self.redis_client.zrevrange(
                "trending_hashtags", 0, limit - 1, withscores=True
            )

            trending_topics = []
            for hashtag, score in trending_hashtags:
                trending_topics.append(
                    {
                        "hashtag": hashtag.decode(),
                        "score": score,
                        "post_count": await self._get_hashtag_post_count(
                            hashtag.decode()
                        ),
                    }
                )

            return trending_topics

        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []

    async def get_social_analytics(self, user_id: int) -> Optional[SocialAnalytics]:
        """Get comprehensive social analytics for a user"""
        try:
            # Calculate influence score
            influence_score = await self._calculate_influence_score(user_id)

            # Calculate engagement rate
            engagement_rate = await self._calculate_engagement_rate(user_id)

            # Get reach and impressions
            reach_impressions = await self._get_reach_impressions(user_id)

            # Calculate follower growth rate
            follower_growth_rate = await self._calculate_follower_growth_rate(user_id)

            # Get content performance
            content_performance = await self._get_content_performance(user_id)

            # Get audience demographics
            audience_demographics = await self._get_audience_demographics(user_id)

            # Get best performing content
            best_performing_content = await self._get_best_performing_content(user_id)

            # Get trending topics
            trending_topics = await self.get_trending_topics(5)

            analytics = SocialAnalytics(
                user_id=user_id,
                influence_score=influence_score,
                engagement_rate=engagement_rate,
                reach_impressions=reach_impressions,
                follower_growth_rate=follower_growth_rate,
                content_performance=content_performance,
                audience_demographics=audience_demographics,
                best_performing_content=best_performing_content,
                trending_topics=[topic["hashtag"] for topic in trending_topics],
                timestamp=datetime.utcnow(),
            )

            return analytics

        except Exception as e:
            logger.error(f"Error getting social analytics: {e}")
            return None

    async def _calculate_trading_stats(self, user_id: int) -> Dict[str, Any]:
        """Calculate comprehensive trading statistics"""
        try:
            db = SessionLocal()

            # Get user's trades
            trades = db.query(Trade).filter(Trade.user_id == user_id).all()

            if not trades:
                return {
                    "total_trades": 0,
                    "total_volume": 0.0,
                    "win_rate": 0.0,
                    "avg_trade_size": 0.0,
                    "profit_loss": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                }

            # Calculate statistics
            total_trades = len(trades)
            total_volume = sum(trade.total_value for trade in trades)
            avg_trade_size = total_volume / total_trades

            # Simplified win rate calculation
            win_rate = 0.6  # Placeholder

            # Calculate profit/loss (simplified)
            profit_loss = sum(
                trade.total_value * 0.1 for trade in trades
            )  # Placeholder

            # Best and worst trades
            trade_values = [trade.total_value for trade in trades]
            best_trade = max(trade_values)
            worst_trade = min(trade_values)

            db.close()

            return {
                "total_trades": total_trades,
                "total_volume": total_volume,
                "win_rate": win_rate,
                "avg_trade_size": avg_trade_size,
                "profit_loss": profit_loss,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
            }

        except Exception as e:
            logger.error(f"Error calculating trading stats: {e}")
            return {}

    async def _calculate_reputation_score(
        self, user_id: int, trading_stats: Dict[str, Any]
    ) -> float:
        """Calculate user's reputation score"""
        try:
            # Trading performance component
            trading_score = min(trading_stats.get("win_rate", 0) * 100, 100)

            # Social engagement component
            social_score = await self._calculate_social_engagement_score(user_id)

            # Content quality component
            content_score = await self._calculate_content_quality_score(user_id)

            # Community contribution component
            community_score = await self._calculate_community_contribution_score(
                user_id
            )

            # Weighted average
            reputation_score = (
                trading_score * self.scoring_weights["trading_performance"]
                + social_score * self.scoring_weights["social_engagement"]
                + content_score * self.scoring_weights["content_quality"]
                + community_score * self.scoring_weights["community_contribution"]
            )

            return min(reputation_score, 100)

        except Exception as e:
            logger.error(f"Error calculating reputation score: {e}")
            return 50.0

    async def _get_followers_count(self, user_id: int) -> int:
        """Get user's followers count"""
        try:
            if not self.redis_client:
                return 0

            count = await self.redis_client.scard(f"followers:{user_id}")
            return count

        except Exception as e:
            logger.error(f"Error getting followers count: {e}")
            return 0

    async def _get_following_count(self, user_id: int) -> int:
        """Get user's following count"""
        try:
            if not self.redis_client:
                return 0

            count = await self.redis_client.scard(f"following:{user_id}")
            return count

        except Exception as e:
            logger.error(f"Error getting following count: {e}")
            return 0

    async def _get_user_badges(self, user_id: int) -> List[str]:
        """Get user's earned badges"""
        try:
            badges = []

            # Check for various achievements
            if await self._has_achievement(user_id, "first_trade"):
                badges.append("First Trader")

            if await self._has_achievement(user_id, "winning_streak"):
                badges.append("Winning Streak")

            if await self._has_achievement(user_id, "high_volume"):
                badges.append("High Volume Trader")

            if await self._has_achievement(user_id, "community_leader"):
                badges.append("Community Leader")

            if await self._has_achievement(user_id, "verified_trader"):
                badges.append("Verified Trader")

            return badges

        except Exception as e:
            logger.error(f"Error getting user badges: {e}")
            return []

    async def _add_to_user_feed(self, user_id: int, post_id: str):
        """Add post to user's feed"""
        try:
            if not self.redis_client:
                return

            # Add to user's own feed
            await self.redis_client.lpush(f"user_feed:{user_id}", post_id)

            # Trim feed to keep only recent posts
            await self.redis_client.ltrim(f"user_feed:{user_id}", 0, 999)

        except Exception as e:
            logger.error(f"Error adding to user feed: {e}")

    async def _process_post_tags(self, post: SocialPost):
        """Process hashtags and mentions in post"""
        try:
            if not self.redis_client:
                return

            # Process hashtags
            for tag in post.tags:
                # Add to trending hashtags
                await self.redis_client.zincrby("trending_hashtags", 1, tag)

                # Add post to hashtag index
                await self.redis_client.sadd(f"hashtag:{tag}", post.post_id)

            # Process mentions (simplified)
            # In a real implementation, you would parse @mentions from content

        except Exception as e:
            logger.error(f"Error processing post tags: {e}")

    async def _update_user_activity(self, user_id: int):
        """Update user's last activity"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.setex(
                f"user_activity:{user_id}",
                86400,  # 24 hours
                datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error updating user activity: {e}")

    async def _add_follow_relationship(self, follower_id: int, followed_id: int):
        """Add follow relationship"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.sadd(f"followers:{followed_id}", follower_id)
            await self.redis_client.sadd(f"following:{follower_id}", followed_id)

        except Exception as e:
            logger.error(f"Error adding follow relationship: {e}")

    async def _remove_follow_relationship(self, follower_id: int, followed_id: int):
        """Remove follow relationship"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.srem(f"followers:{followed_id}", follower_id)
            await self.redis_client.srem(f"following:{follower_id}", followed_id)

        except Exception as e:
            logger.error(f"Error removing follow relationship: {e}")

    async def _is_following(self, follower_id: int, followed_id: int) -> bool:
        """Check if user is following another user"""
        try:
            if not self.redis_client:
                return False

            return await self.redis_client.sismember(
                f"followers:{followed_id}", follower_id
            )

        except Exception as e:
            logger.error(f"Error checking follow relationship: {e}")
            return False

    async def _update_follower_counts(self, user_id: int, delta: int):
        """Update follower counts"""
        try:
            if not self.redis_client:
                return

            # Update cached count
            current_count = await self.redis_client.get(f"follower_count:{user_id}")
            new_count = int(current_count or 0) + delta
            await self.redis_client.setex(f"follower_count:{user_id}", 3600, new_count)

        except Exception as e:
            logger.error(f"Error updating follower counts: {e}")

    async def _update_following_counts(self, user_id: int, delta: int):
        """Update following counts"""
        try:
            if not self.redis_client:
                return

            # Update cached count
            current_count = await self.redis_client.get(f"following_count:{user_id}")
            new_count = int(current_count or 0) + delta
            await self.redis_client.setex(f"following_count:{user_id}", 3600, new_count)

        except Exception as e:
            logger.error(f"Error updating following counts: {e}")

    async def _add_to_follower_feed(self, follower_id: int, followed_id: int):
        """Add followed user's posts to follower's feed"""
        try:
            if not self.redis_client:
                return

            # Get recent posts from followed user
            recent_posts = await self.redis_client.lrange(
                f"user_feed:{followed_id}", 0, 9
            )

            # Add to follower's feed
            for post_id in recent_posts:
                await self.redis_client.lpush(f"user_feed:{follower_id}", post_id)

            # Trim feed
            await self.redis_client.ltrim(f"user_feed:{follower_id}", 0, 999)

        except Exception as e:
            logger.error(f"Error adding to follower feed: {e}")

    async def _remove_from_follower_feed(self, follower_id: int, followed_id: int):
        """Remove followed user's posts from follower's feed"""
        try:
            if not self.redis_client:
                return

            # Get all posts from followed user
            user_posts = await self.redis_client.lrange(
                f"user_feed:{followed_id}", 0, -1
            )

            # Remove from follower's feed
            for post_id in user_posts:
                await self.redis_client.lrem(f"user_feed:{follower_id}", 0, post_id)

        except Exception as e:
            logger.error(f"Error removing from follower feed: {e}")

    async def _send_follow_notification(self, followed_id: int, follower_id: int):
        """Send follow notification"""
        # In a real implementation, you would send a notification
        logger.info(
            f"Follow notification sent to user {followed_id} from user {follower_id}"
        )

    async def _add_post_like(self, user_id: int, post_id: str):
        """Add like to post"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.sadd(f"post_likes:{post_id}", user_id)

        except Exception as e:
            logger.error(f"Error adding post like: {e}")

    async def _has_liked_post(self, user_id: int, post_id: str) -> bool:
        """Check if user has liked a post"""
        try:
            if not self.redis_client:
                return False

            return await self.redis_client.sismember(f"post_likes:{post_id}", user_id)

        except Exception as e:
            logger.error(f"Error checking post like: {e}")
            return False

    async def _has_shared_post(self, user_id: int, post_id: str) -> bool:
        """Check if user has shared a post"""
        try:
            if not self.redis_client:
                return False

            return await self.redis_client.sismember(f"post_shares:{post_id}", user_id)

        except Exception as e:
            logger.error(f"Error checking post share: {e}")
            return False

    async def _send_like_notification(
        self, post_user_id: int, liker_id: int, post_id: str
    ):
        """Send like notification"""
        # In a real implementation, you would send a notification
        logger.info(
            f"Like notification sent to user {post_user_id} from user {liker_id}"
        )

    async def _get_feed_posts(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Get posts for user's feed"""
        try:
            if not self.redis_client:
                return []

            # Get post IDs from feed
            post_ids = await self.redis_client.lrange(
                f"user_feed:{user_id}", 0, limit - 1
            )

            # Get post details
            feed_posts = []
            for post_id in post_ids:
                post_id_str = post_id.decode()
                if post_id_str in self.social_posts:
                    post = self.social_posts[post_id_str]
                    feed_posts.append(
                        {
                            "post_id": post.post_id,
                            "user_id": post.user_id,
                            "content": post.content,
                            "post_type": post.post_type,
                            "market_id": post.market_id,
                            "trade_id": post.trade_id,
                            "images": post.images,
                            "tags": post.tags,
                            "likes_count": post.likes_count,
                            "comments_count": post.comments_count,
                            "shares_count": post.shares_count,
                            "created_at": post.created_at.isoformat(),
                        }
                    )

            return feed_posts

        except Exception as e:
            logger.error(f"Error getting feed posts: {e}")
            return []

    async def _get_hashtag_post_count(self, hashtag: str) -> int:
        """Get number of posts with a hashtag"""
        try:
            if not self.redis_client:
                return 0

            count = await self.redis_client.scard(f"hashtag:{hashtag}")
            return count

        except Exception as e:
            logger.error(f"Error getting hashtag post count: {e}")
            return 0

    async def _calculate_influence_score(self, user_id: int) -> float:
        """Calculate user's influence score"""
        try:
            followers_count = await self._get_followers_count(user_id)
            engagement_rate = await self._calculate_engagement_rate(user_id)

            # Simple influence calculation
            influence_score = min(followers_count * engagement_rate / 100, 100)
            return influence_score

        except Exception as e:
            logger.error(f"Error calculating influence score: {e}")
            return 0.0

    async def _calculate_engagement_rate(self, user_id: int) -> float:
        """Calculate user's engagement rate"""
        try:
            # Get user's posts
            user_posts = [
                post for post in self.social_posts.values() if post.user_id == user_id
            ]

            if not user_posts:
                return 0.0

            total_engagement = sum(
                post.likes_count + post.comments_count + post.shares_count
                for post in user_posts
            )

            followers_count = await self._get_followers_count(user_id)
            if followers_count == 0:
                return 0.0

            engagement_rate = (
                (total_engagement / len(user_posts)) / followers_count * 100
            )
            return min(engagement_rate, 100)

        except Exception as e:
            logger.error(f"Error calculating engagement rate: {e}")
            return 0.0

    async def _get_reach_impressions(self, user_id: int) -> int:
        """Get user's reach and impressions"""
        try:
            # Simplified calculation
            followers_count = await self._get_followers_count(user_id)
            user_posts = [
                post for post in self.social_posts.values() if post.user_id == user_id
            ]

            # Estimate reach based on followers and post engagement
            total_reach = followers_count * len(user_posts) * 0.3  # 30% average reach
            return int(total_reach)

        except Exception as e:
            logger.error(f"Error getting reach impressions: {e}")
            return 0

    async def _calculate_follower_growth_rate(self, user_id: int) -> float:
        """Calculate follower growth rate"""
        try:
            # Simplified calculation
            current_followers = await self._get_followers_count(user_id)

            # Assume 5% monthly growth for active users
            growth_rate = 5.0 if current_followers > 10 else 0.0
            return growth_rate

        except Exception as e:
            logger.error(f"Error calculating follower growth rate: {e}")
            return 0.0

    async def _get_content_performance(self, user_id: int) -> Dict[str, Any]:
        """Get content performance metrics"""
        try:
            user_posts = [
                post for post in self.social_posts.values() if post.user_id == user_id
            ]

            if not user_posts:
                return {
                    "total_posts": 0,
                    "avg_likes": 0,
                    "avg_comments": 0,
                    "avg_shares": 0,
                    "best_performing_post": None,
                }

            total_posts = len(user_posts)
            avg_likes = sum(post.likes_count for post in user_posts) / total_posts
            avg_comments = sum(post.comments_count for post in user_posts) / total_posts
            avg_shares = sum(post.shares_count for post in user_posts) / total_posts

            # Find best performing post
            best_post = max(
                user_posts,
                key=lambda p: p.likes_count + p.comments_count + p.shares_count,
            )

            return {
                "total_posts": total_posts,
                "avg_likes": avg_likes,
                "avg_comments": avg_comments,
                "avg_shares": avg_shares,
                "best_performing_post": best_post.post_id,
            }

        except Exception as e:
            logger.error(f"Error getting content performance: {e}")
            return {}

    async def _get_audience_demographics(self, user_id: int) -> Dict[str, Any]:
        """Get audience demographics"""
        try:
            # Simplified demographics
            return {
                "age_groups": {"18-24": 25, "25-34": 40, "35-44": 20, "45+": 15},
                "locations": {
                    "United States": 45,
                    "Europe": 30,
                    "Asia": 15,
                    "Other": 10,
                },
                "interests": ["trading", "cryptocurrency", "finance", "technology"],
            }

        except Exception as e:
            logger.error(f"Error getting audience demographics: {e}")
            return {}

    async def _get_best_performing_content(self, user_id: int) -> List[str]:
        """Get best performing content IDs"""
        try:
            user_posts = [
                post for post in self.social_posts.values() if post.user_id == user_id
            ]

            # Sort by engagement (likes + comments + shares)
            sorted_posts = sorted(
                user_posts,
                key=lambda p: p.likes_count + p.comments_count + p.shares_count,
                reverse=True,
            )

            return [post.post_id for post in sorted_posts[:5]]

        except Exception as e:
            logger.error(f"Error getting best performing content: {e}")
            return []

    async def _calculate_social_engagement_score(self, user_id: int) -> float:
        """Calculate social engagement score"""
        try:
            followers_count = await self._get_followers_count(user_id)
            engagement_rate = await self._calculate_engagement_rate(user_id)

            # Normalize to 0-100 scale
            score = min(followers_count * 0.1 + engagement_rate * 10, 100)
            return score

        except Exception as e:
            logger.error(f"Error calculating social engagement score: {e}")
            return 0.0

    async def _calculate_content_quality_score(self, user_id: int) -> float:
        """Calculate content quality score"""
        try:
            user_posts = [
                post for post in self.social_posts.values() if post.user_id == user_id
            ]

            if not user_posts:
                return 0.0

            # Calculate average engagement per post
            total_engagement = sum(
                post.likes_count + post.comments_count + post.shares_count
                for post in user_posts
            )

            avg_engagement = total_engagement / len(user_posts)

            # Normalize to 0-100 scale
            score = min(avg_engagement * 2, 100)
            return score

        except Exception as e:
            logger.error(f"Error calculating content quality score: {e}")
            return 0.0

    async def _calculate_community_contribution_score(self, user_id: int) -> float:
        """Calculate community contribution score"""
        try:
            # Count user's posts and community memberships
            user_posts = len(
                [post for post in self.social_posts.values() if post.user_id == user_id]
            )

            # Simplified scoring
            score = min(user_posts * 5, 100)
            return score

        except Exception as e:
            logger.error(f"Error calculating community contribution score: {e}")
            return 0.0

    async def _has_achievement(self, user_id: int, achievement: str) -> bool:
        """Check if user has a specific achievement"""
        try:
            # Simplified achievement checking
            achievements = {
                "first_trade": True,  # Placeholder
                "winning_streak": True,  # Placeholder
                "high_volume": True,  # Placeholder
                "community_leader": False,  # Placeholder
                "verified_trader": False,  # Placeholder
            }

            return achievements.get(achievement, False)

        except Exception as e:
            logger.error(f"Error checking achievement: {e}")
            return False

    async def _add_community_member(self, community_id: str, user_id: int, role: str):
        """Add member to community"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.sadd(f"community_members:{community_id}", user_id)
            await self.redis_client.hset(
                f"community_roles:{community_id}", user_id, role
            )

        except Exception as e:
            logger.error(f"Error adding community member: {e}")


# Global social features instance
social_features = SocialFeaturesService()


def get_social_features() -> SocialFeaturesService:
    """Get the global social features instance"""
    return social_features
