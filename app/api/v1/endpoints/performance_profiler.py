"""
Performance Profiler API Endpoints
Provides access to performance profiling data and controls
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging

from app.core.performance_profiler import (
    performance_profiler,
    start_profile,
    end_profile,
    get_performance_summary,
    get_slowest_functions,
    get_most_called_functions,
    ProfilerType
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/summary")
async def get_profiler_summary():
    """Get comprehensive performance profiling summary"""
    try:
        summary = get_performance_summary()
        return {
            "success": True,
            "data": summary
        }
    except Exception as e:
        logger.error(f"Error getting profiler summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/functions/slowest")
async def get_slowest_functions_endpoint(limit: int = 10):
    """Get the slowest functions by average execution time"""
    if limit > 100:
        limit = 100
    
    try:
        functions = get_slowest_functions(limit)
        return {
            "success": True,
            "data": {
                "functions": functions,
                "count": len(functions)
            }
        }
    except Exception as e:
        logger.error(f"Error getting slowest functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/functions/most-called")
async def get_most_called_functions_endpoint(limit: int = 10):
    """Get the most frequently called functions"""
    if limit > 100:
        limit = 100
    
    try:
        functions = get_most_called_functions(limit)
        return {
            "success": True,
            "data": {
                "functions": functions,
                "count": len(functions)
            }
        }
    except Exception as e:
        logger.error(f"Error getting most called functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory")
async def get_memory_profile():
    """Get current memory profile"""
    try:
        memory_profile = performance_profiler.get_memory_profile()
        return {
            "success": True,
            "data": {
                "timestamp": memory_profile.timestamp.isoformat(),
                "current_memory": memory_profile.current_memory,
                "peak_memory": memory_profile.peak_memory,
                "memory_growth": memory_profile.memory_growth,
                "top_allocations": memory_profile.top_allocations,
                "memory_leaks": memory_profile.memory_leaks
            }
        }
    except Exception as e:
        logger.error(f"Error getting memory profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system")
async def get_system_metrics():
    """Get current system performance metrics"""
    try:
        metrics = performance_profiler.get_system_metrics()
        return {
            "success": True,
            "data": metrics
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/profile/start")
async def start_named_profile(name: str, profile_type: str = "function"):
    """Start a named performance profile"""
    try:
        # Validate profile type
        try:
            profiler_type = ProfilerType(profile_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid profile type: {profile_type}")
        
        profile_id = start_profile(name, profiler_type)
        return {
            "success": True,
            "data": {
                "profile_id": profile_id,
                "name": name,
                "type": profile_type
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/profile/{profile_id}/end")
async def end_named_profile(profile_id: str):
    """End a named performance profile"""
    try:
        result = end_profile(profile_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return {
            "success": True,
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_profiler_status():
    """Get profiler status and configuration"""
    try:
        return {
            "success": True,
            "data": {
                "profiling_enabled": performance_profiler.profiling_enabled,
                "memory_tracking_enabled": performance_profiler.memory_tracking_enabled,
                "active_profiles": len(performance_profiler.active_profiles),
                "function_profiles": len(performance_profiler.function_profiles),
                "memory_snapshots": len(performance_profiler.memory_snapshots),
                "uptime_seconds": performance_profiler._start_time
            }
        }
    except Exception as e:
        logger.error(f"Error getting profiler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enable")
async def enable_profiling():
    """Enable performance profiling"""
    try:
        performance_profiler.enable_profiling()
        return {
            "success": True,
            "message": "Performance profiling enabled"
        }
    except Exception as e:
        logger.error(f"Error enabling profiling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disable")
async def disable_profiling():
    """Disable performance profiling"""
    try:
        performance_profiler.disable_profiling()
        return {
            "success": True,
            "message": "Performance profiling disabled"
        }
    except Exception as e:
        logger.error(f"Error disabling profiling: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_profiles():
    """Clear all profiling data"""
    try:
        performance_profiler.clear_profiles()
        return {
            "success": True,
            "message": "All profiling data cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations")
async def get_performance_recommendations():
    """Get performance optimization recommendations"""
    try:
        summary = get_performance_summary()
        recommendations = summary.get("recommendations", [])
        
        # Categorize recommendations by severity
        categorized = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        for rec in recommendations:
            severity = rec.get("severity", "low")
            if severity in categorized:
                categorized[severity].append(rec)
        
        return {
            "success": True,
            "data": {
                "recommendations": recommendations,
                "categorized": categorized,
                "total": len(recommendations)
            }
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/functions/{function_name}")
async def get_function_profile(function_name: str):
    """Get detailed profile for a specific function"""
    try:
        profile_key = None
        for key in performance_profiler.function_profiles.keys():
            if function_name in key:
                profile_key = key
                break
        
        if not profile_key:
            raise HTTPException(status_code=404, detail="Function not found")
        
        profile = performance_profiler.function_profiles[profile_key]
        
        return {
            "success": True,
            "data": {
                "function_name": profile.function_name,
                "module_name": profile.module_name,
                "total_calls": profile.total_calls,
                "total_time": profile.total_time,
                "avg_time": profile.avg_time,
                "min_time": profile.min_time,
                "max_time": profile.max_time,
                "memory_usage": profile.memory_usage,
                "last_called": profile.last_called.isoformat(),
                "call_stack": profile.call_stack
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting function profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export")
async def export_profiling_data():
    """Export all profiling data"""
    try:
        summary = get_performance_summary()
        slowest = get_slowest_functions(50)
        most_called = get_most_called_functions(50)
        memory_profile = performance_profiler.get_memory_profile()
        system_metrics = performance_profiler.get_system_metrics()
        
        export_data = {
            "export_timestamp": performance_profiler._start_time,
            "summary": summary,
            "slowest_functions": slowest,
            "most_called_functions": most_called,
            "memory_profile": {
                "timestamp": memory_profile.timestamp.isoformat(),
                "current_memory": memory_profile.current_memory,
                "peak_memory": memory_profile.peak_memory,
                "memory_growth": memory_profile.memory_growth,
                "top_allocations": memory_profile.top_allocations
            },
            "system_metrics": system_metrics
        }
        
        return {
            "success": True,
            "data": export_data
        }
    except Exception as e:
        logger.error(f"Error exporting profiling data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
