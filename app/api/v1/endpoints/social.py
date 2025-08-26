from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import asyncio

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.services.social_features import get_social_features

router = APIRouter()

@router.post("/profile")
def create_user_profile(
    profile_data: Dict = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Create or update user profile"""
    social_features = get_social_features()
    
    # Create profile
    profile = asyncio.run(social_features.create_user_profile(current_user.id, profile_data))
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create user profile"
        )
    
    return {
        "user_id": profile.user_id,
        "username": profile.username,
        "display_name": profile.display_name,
        "bio": profile.bio,
        "avatar_url": profile.avatar_url,
        "cover_image_url": profile.cover_image_url,
        "location": profile.location,
        "website": profile.website,
        "social_links": profile.social_links,
        "trading_stats": profile.trading_stats,
        "reputation_score": profile.reputation_score,
        "followers_count": profile.followers_count,
        "following_count": profile.following_count,
        "is_verified": profile.is_verified,
        "badges": profile.badges,
        "created_at": profile.created_at.isoformat(),
        "last_active": profile.last_active.isoformat()
    }

@router.get("/profile/{user_id}")
def get_user_profile(
    user_id: int,
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get user profile"""
    social_features = get_social_features()
    
    # Get profile from cache or create if not exists
    if user_id in social_features.user_profiles:
        profile = social_features.user_profiles[user_id]
    else:
        # Create basic profile
        profile = asyncio.run(social_features.create_user_profile(user_id, {}))
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found"
        )
    
    return {
        "user_id": profile.user_id,
        "username": profile.username,
        "display_name": profile.display_name,
        "bio": profile.bio,
        "avatar_url": profile.avatar_url,
        "cover_image_url": profile.cover_image_url,
        "location": profile.location,
        "website": profile.website,
        "social_links": profile.social_links,
        "trading_stats": profile.trading_stats,
        "reputation_score": profile.reputation_score,
        "followers_count": profile.followers_count,
        "following_count": profile.following_count,
        "is_verified": profile.is_verified,
        "badges": profile.badges,
        "created_at": profile.created_at.isoformat(),
        "last_active": profile.last_active.isoformat()
    }

@router.post("/posts")
def create_social_post(
    post_data: Dict = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Create a social post"""
    social_features = get_social_features()
    
    # Validate required fields
    if 'content' not in post_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Content is required"
        )
    
    # Create post
    post = asyncio.run(social_features.create_social_post(current_user.id, post_data))
    
    if not post:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create social post"
        )
    
    return {
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
        "is_pinned": post.is_pinned,
        "is_edited": post.is_edited,
        "created_at": post.created_at.isoformat(),
        "updated_at": post.updated_at.isoformat()
    }

@router.get("/posts")
def get_social_posts(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    post_type: Optional[str] = Query(None, description="Filter by post type"),
    limit: int = Query(20, description="Number of posts to return"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get social posts with optional filtering"""
    social_features = get_social_features()
    
    # Filter posts
    posts = []
    for post in social_features.social_posts.values():
        if user_id and post.user_id != user_id:
            continue
        if post_type and post.post_type != post_type:
            continue
        
        posts.append({
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
            "is_pinned": post.is_pinned,
            "is_edited": post.is_edited,
            "created_at": post.created_at.isoformat(),
            "updated_at": post.updated_at.isoformat()
        })
    
    # Sort by creation date (newest first)
    posts.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Limit results
    posts = posts[:limit]
    
    return {
        "posts": posts,
        "total": len(posts),
        "filters": {
            "user_id": user_id,
            "post_type": post_type
        }
    }

@router.post("/posts/{post_id}/like")
def like_post(
    post_id: str,
    current_user: User = Depends(get_current_user)
):
    """Like a social post"""
    social_features = get_social_features()
    
    success = asyncio.run(social_features.like_post(current_user.id, post_id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to like post"
        )
    
    return {
        "message": "Post liked successfully",
        "post_id": post_id,
        "user_id": current_user.id
    }

@router.get("/feed")
def get_user_feed(
    limit: int = Query(20, description="Number of posts to return"),
    current_user: User = Depends(get_current_user)
):
    """Get user's personalized feed"""
    social_features = get_social_features()
    
    feed_posts = asyncio.run(social_features.get_user_feed(current_user.id, limit))
    
    return {
        "posts": feed_posts,
        "total": len(feed_posts),
        "user_id": current_user.id
    }

@router.post("/follow/{user_id}")
def follow_user(
    user_id: int,
    current_user: User = Depends(get_current_user)
):
    """Follow a user"""
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot follow yourself"
        )
    
    social_features = get_social_features()
    
    success = asyncio.run(social_features.follow_user(current_user.id, user_id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to follow user"
        )
    
    return {
        "message": "User followed successfully",
        "follower_id": current_user.id,
        "followed_id": user_id
    }

@router.delete("/follow/{user_id}")
def unfollow_user(
    user_id: int,
    current_user: User = Depends(get_current_user)
):
    """Unfollow a user"""
    social_features = get_social_features()
    
    success = asyncio.run(social_features.unfollow_user(current_user.id, user_id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to unfollow user"
        )
    
    return {
        "message": "User unfollowed successfully",
        "follower_id": current_user.id,
        "followed_id": user_id
    }

@router.post("/communities")
def create_community(
    community_data: Dict = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Create a trading community"""
    social_features = get_social_features()
    
    # Validate required fields
    if 'name' not in community_data or 'description' not in community_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name and description are required"
        )
    
    # Create community
    community = asyncio.run(social_features.create_community(current_user.id, community_data))
    
    if not community:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create community"
        )
    
    return {
        "community_id": community.community_id,
        "name": community.name,
        "description": community.description,
        "avatar_url": community.avatar_url,
        "cover_image_url": community.cover_image_url,
        "owner_id": community.owner_id,
        "moderators": community.moderators,
        "members_count": community.members_count,
        "is_private": community.is_private,
        "is_verified": community.is_verified,
        "rules": community.rules,
        "categories": community.categories,
        "created_at": community.created_at.isoformat()
    }

@router.get("/communities")
def get_communities(
    owner_id: Optional[int] = Query(None, description="Filter by owner ID"),
    is_private: Optional[bool] = Query(None, description="Filter by privacy setting"),
    limit: int = Query(20, description="Number of communities to return"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get communities with optional filtering"""
    social_features = get_social_features()
    
    # Filter communities
    communities = []
    for community in social_features.communities.values():
        if owner_id and community.owner_id != owner_id:
            continue
        if is_private is not None and community.is_private != is_private:
            continue
        
        communities.append({
            "community_id": community.community_id,
            "name": community.name,
            "description": community.description,
            "avatar_url": community.avatar_url,
            "cover_image_url": community.cover_image_url,
            "owner_id": community.owner_id,
            "moderators": community.moderators,
            "members_count": community.members_count,
            "is_private": community.is_private,
            "is_verified": community.is_verified,
            "rules": community.rules,
            "categories": community.categories,
            "created_at": community.created_at.isoformat()
        })
    
    # Sort by member count (most popular first)
    communities.sort(key=lambda x: x["members_count"], reverse=True)
    
    # Limit results
    communities = communities[:limit]
    
    return {
        "communities": communities,
        "total": len(communities),
        "filters": {
            "owner_id": owner_id,
            "is_private": is_private
        }
    }

@router.get("/trending")
def get_trending_topics(
    limit: int = Query(10, description="Number of trending topics to return"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get trending topics and hashtags"""
    social_features = get_social_features()
    
    trending_topics = asyncio.run(social_features.get_trending_topics(limit))
    
    return {
        "trending_topics": trending_topics,
        "total": len(trending_topics)
    }

@router.get("/analytics")
def get_social_analytics(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive social analytics"""
    social_features = get_social_features()
    
    analytics = asyncio.run(social_features.get_social_analytics(current_user.id))
    
    if not analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Social analytics not available"
        )
    
    return {
        "user_id": analytics.user_id,
        "influence_score": analytics.influence_score,
        "engagement_rate": analytics.engagement_rate,
        "reach_impressions": analytics.reach_impressions,
        "follower_growth_rate": analytics.follower_growth_rate,
        "content_performance": analytics.content_performance,
        "audience_demographics": analytics.audience_demographics,
        "best_performing_content": analytics.best_performing_content,
        "trending_topics": analytics.trending_topics,
        "timestamp": analytics.timestamp.isoformat()
    }

@router.get("/users/{user_id}/followers")
def get_user_followers(
    user_id: int,
    limit: int = Query(50, description="Number of followers to return"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get user's followers"""
    social_features = get_social_features()
    
    # Get followers from Redis
    if not social_features.redis_client:
        return {"followers": [], "total": 0}
    
    followers = asyncio.run(social_features.redis_client.smembers(f"followers:{user_id}"))
    
    # Convert to list and limit
    followers_list = [int(follower.decode()) for follower in followers][:limit]
    
    return {
        "followers": followers_list,
        "total": len(followers_list),
        "user_id": user_id
    }

@router.get("/users/{user_id}/following")
def get_user_following(
    user_id: int,
    limit: int = Query(50, description="Number of following to return"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get users that a user is following"""
    social_features = get_social_features()
    
    # Get following from Redis
    if not social_features.redis_client:
        return {"following": [], "total": 0}
    
    following = asyncio.run(social_features.redis_client.smembers(f"following:{user_id}"))
    
    # Convert to list and limit
    following_list = [int(follow.decode()) for follow in following][:limit]
    
    return {
        "following": following_list,
        "total": len(following_list),
        "user_id": user_id
    }

@router.get("/posts/{post_id}")
def get_social_post(
    post_id: str,
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get a specific social post"""
    social_features = get_social_features()
    
    if post_id not in social_features.social_posts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found"
        )
    
    post = social_features.social_posts[post_id]
    
    # Check if current user has liked the post
    user_has_liked = False
    if current_user:
        user_has_liked = asyncio.run(social_features._has_liked_post(current_user.id, post_id))
    
    return {
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
        "is_pinned": post.is_pinned,
        "is_edited": post.is_edited,
        "created_at": post.created_at.isoformat(),
        "updated_at": post.updated_at.isoformat(),
        "user_has_liked": user_has_liked
    }

@router.get("/communities/{community_id}")
def get_community(
    community_id: str,
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get a specific community"""
    social_features = get_social_features()
    
    if community_id not in social_features.communities:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Community not found"
        )
    
    community = social_features.communities[community_id]
    
    # Check if current user is a member
    is_member = False
    if current_user and social_features.redis_client:
        is_member = asyncio.run(social_features.redis_client.sismember(f"community_members:{community_id}", current_user.id))
    
    return {
        "community_id": community.community_id,
        "name": community.name,
        "description": community.description,
        "avatar_url": community.avatar_url,
        "cover_image_url": community.cover_image_url,
        "owner_id": community.owner_id,
        "moderators": community.moderators,
        "members_count": community.members_count,
        "is_private": community.is_private,
        "is_verified": community.is_verified,
        "rules": community.rules,
        "categories": community.categories,
        "created_at": community.created_at.isoformat(),
        "is_member": is_member
    }

@router.get("/search")
def search_social_content(
    query: str = Query(..., description="Search query"),
    search_type: str = Query("all", description="Search type: all, users, posts, communities"),
    limit: int = Query(20, description="Number of results to return"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Search social content"""
    social_features = get_social_features()
    
    results = {
        "users": [],
        "posts": [],
        "communities": [],
        "total": 0
    }
    
    query_lower = query.lower()
    
    # Search users
    if search_type in ["all", "users"]:
        for profile in social_features.user_profiles.values():
            if (query_lower in profile.display_name.lower() or 
                query_lower in profile.bio.lower() or
                query_lower in profile.username.lower()):
                results["users"].append({
                    "user_id": profile.user_id,
                    "username": profile.username,
                    "display_name": profile.display_name,
                    "bio": profile.bio,
                    "avatar_url": profile.avatar_url,
                    "reputation_score": profile.reputation_score,
                    "followers_count": profile.followers_count
                })
    
    # Search posts
    if search_type in ["all", "posts"]:
        for post in social_features.social_posts.values():
            if query_lower in post.content.lower():
                results["posts"].append({
                    "post_id": post.post_id,
                    "user_id": post.user_id,
                    "content": post.content,
                    "post_type": post.post_type,
                    "tags": post.tags,
                    "likes_count": post.likes_count,
                    "created_at": post.created_at.isoformat()
                })
    
    # Search communities
    if search_type in ["all", "communities"]:
        for community in social_features.communities.values():
            if (query_lower in community.name.lower() or 
                query_lower in community.description.lower()):
                results["communities"].append({
                    "community_id": community.community_id,
                    "name": community.name,
                    "description": community.description,
                    "avatar_url": community.avatar_url,
                    "members_count": community.members_count,
                    "is_private": community.is_private
                })
    
    # Calculate total
    results["total"] = len(results["users"]) + len(results["posts"]) + len(results["communities"])
    
    # Limit results
    results["users"] = results["users"][:limit]
    results["posts"] = results["posts"][:limit]
    results["communities"] = results["communities"][:limit]
    
    return {
        "query": query,
        "search_type": search_type,
        "results": results
    }
