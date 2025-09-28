"""
Search API Endpoints
Provides advanced search and filtering capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.search.search_engine import search_engine, SearchType, SortOrder

router = APIRouter()


@router.get("/")
async def search(
    q: str = Query(..., description="Search query"),
    type: str = Query("full_text", description="Search type (full_text, fuzzy, filtered)"),
    filters: Optional[str] = Query(None, description="JSON string of filters to apply"),
    sort: str = Query("relevance", description="Sort order (relevance, date_asc, date_desc, price_asc, price_desc, popularity, rating)"),
    limit: int = Query(20, description="Maximum number of results to return"),
    offset: int = Query(0, description="Number of results to skip"),
    current_user: User = Depends(get_current_user)
):
    """Perform a search"""
    try:
        # Parse search type
        try:
            search_type = SearchType(type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid search type: {type}")
        
        # Parse sort order
        try:
            sort_order = SortOrder(sort)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid sort order: {sort}")
        
        # Parse filters
        parsed_filters = {}
        if filters:
            try:
                import json
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filters JSON")
        
        # Perform search
        results = await search_engine.search(
            query_text=q,
            search_type=search_type,
            filters=parsed_filters,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
            user_id=str(current_user.id)
        )
        
        return {
            "success": True,
            "data": results,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Query text for suggestions"),
    limit: int = Query(10, description="Maximum number of suggestions"),
    current_user: User = Depends(get_current_user)
):
    """Get search suggestions"""
    try:
        suggestions = await search_engine.get_search_suggestions(q, limit)
        
        return {
            "success": True,
            "data": {
                "query": q,
                "suggestions": suggestions
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/filters")
async def get_available_filters(current_user: User = Depends(get_current_user)):
    """Get available search filters"""
    try:
        return {
            "success": True,
            "data": {
                "filters": search_engine.filter_definitions
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get filters: {str(e)}")


@router.get("/analytics")
async def get_search_analytics(
    user_id: Optional[str] = Query(None, description="User ID to get analytics for"),
    time_range: int = Query(86400, description="Time range in seconds"),
    current_user: User = Depends(get_current_user)
):
    """Get search analytics"""
    try:
        analytics = await search_engine.get_search_analytics(user_id, time_range)
        
        return {
            "success": True,
            "data": analytics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search analytics: {str(e)}")


@router.get("/stats")
async def get_search_stats(current_user: User = Depends(get_current_user)):
    """Get search engine statistics"""
    try:
        stats = search_engine.get_search_stats()
        
        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search stats: {str(e)}")


@router.post("/index/rebuild")
async def rebuild_search_index(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Rebuild the search index"""
    try:
        # Add rebuild task to background
        background_tasks.add_task(search_engine._build_search_index)
        
        return {
            "success": True,
            "message": "Search index rebuild started",
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


@router.get("/popular")
async def get_popular_searches(
    limit: int = Query(20, description="Maximum number of popular searches"),
    current_user: User = Depends(get_current_user)
):
    """Get popular search terms"""
    try:
        # Get popular terms from analytics
        analytics_summary = await search_engine.enhanced_cache.get("search_analytics_summary")
        
        popular_terms = []
        if analytics_summary and "popular_terms" in analytics_summary:
            popular_terms = analytics_summary["popular_terms"][:limit]
        
        return {
            "success": True,
            "data": {
                "popular_searches": popular_terms
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get popular searches: {str(e)}")


@router.post("/click")
async def record_search_click(
    click_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Record a search result click for analytics"""
    try:
        query_id = click_data.get("query_id")
        result_id = click_data.get("result_id")
        position = click_data.get("position", 0)
        
        if not query_id or not result_id:
            raise HTTPException(status_code=400, detail="query_id and result_id are required")
        
        # Record click analytics
        click_analytics = {
            "query_id": query_id,
            "result_id": result_id,
            "position": position,
            "user_id": str(current_user.id),
            "timestamp": time.time()
        }
        
        # Store click analytics
        await search_engine.enhanced_cache.set(
            f"search_click_{query_id}_{result_id}",
            click_analytics,
            ttl=86400,
            tags=["search_analytics", "clicks"]
        )
        
        return {
            "success": True,
            "message": "Search click recorded",
            "data": click_analytics,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record click: {str(e)}")


@router.get("/trending")
async def get_trending_searches(
    time_range: int = Query(3600, description="Time range in seconds"),
    limit: int = Query(10, description="Maximum number of trending searches"),
    current_user: User = Depends(get_current_user)
):
    """Get trending search terms"""
    try:
        # Calculate trending terms based on recent searches
        cutoff_time = time.time() - time_range
        trending_terms = []
        
        # Get recent searches from all users
        recent_searches = []
        for user_history in search_engine.search_history.values():
            recent_searches.extend([
                s for s in user_history 
                if s.timestamp >= cutoff_time
            ])
        
        # Count search terms
        from collections import Counter
        term_counts = Counter()
        
        for search in recent_searches:
            terms = search_engine._tokenize_text(search.query_text)
            for term in terms:
                term_counts[term] += 1
        
        # Get trending terms
        trending_terms = [
            {"term": term, "count": count}
            for term, count in term_counts.most_common(limit)
        ]
        
        return {
            "success": True,
            "data": {
                "trending_searches": trending_terms,
                "time_range": time_range
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trending searches: {str(e)}")


@router.get("/health")
async def search_health_check(current_user: User = Depends(get_current_user)):
    """Get search engine health status"""
    try:
        stats = search_engine.get_search_stats()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if stats["index_size"] == 0:
            health_status = "degraded"
            issues.append("Search index is empty")
        
        if stats["searches_performed"] == 0:
            health_status = "degraded"
            issues.append("No searches performed")
        
        return {
            "success": True,
            "data": {
                "status": health_status,
                "issues": issues,
                "stats": stats
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get search health: {str(e)}")
