"""
Advanced Performance Profiling System
Provides comprehensive performance monitoring, profiling, and optimization recommendations
"""

import asyncio
import time
import threading
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import functools
import inspect
import linecache
import sys

logger = logging.getLogger(__name__)

class ProfilerType(Enum):
    """Types of profilers"""
    FUNCTION = "function"
    MEMORY = "memory"
    CPU = "cpu"
    I_O = "io"
    DATABASE = "database"
    API = "api"

class PerformanceLevel(Enum):
    """Performance levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    level: PerformanceLevel
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FunctionProfile:
    """Profiles a function's performance"""
    function_name: str
    module_name: str
    total_calls: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    memory_usage: float
    last_called: datetime
    call_stack: List[str] = field(default_factory=list)

@dataclass
class MemoryProfile:
    """Memory usage profile"""
    timestamp: datetime
    current_memory: int
    peak_memory: int
    memory_growth: int
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    memory_leaks: List[Dict[str, Any]] = field(default_factory=list)

class PerformanceProfiler:
    """Advanced performance profiler"""
    
    def __init__(self):
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.memory_snapshots: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=1000)
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.profiling_enabled = True
        self.memory_tracking_enabled = False
        self._lock = threading.RLock()
        self._start_time = time.time()
        
        # Start memory tracking if available
        if hasattr(tracemalloc, 'start'):
            try:
                tracemalloc.start()
                self.memory_tracking_enabled = True
                logger.info("Memory tracking enabled")
            except Exception as e:
                logger.warning(f"Could not enable memory tracking: {e}")
    
    def profile_function(self, func: Callable = None, *, 
                        track_memory: bool = False,
                        track_calls: bool = True,
                        min_time_threshold: float = 0.001):
        """Decorator to profile function performance"""
        def decorator(f):
            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return await f(*args, **kwargs)
                
                profile_key = f"{f.__module__}.{f.__name__}"
                start_time = time.perf_counter()
                start_memory = 0
                
                if track_memory and self.memory_tracking_enabled:
                    start_memory = tracemalloc.get_traced_memory()[0]
                
                try:
                    result = await f(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    if execution_time >= min_time_threshold:
                        end_memory = 0
                        if track_memory and self.memory_tracking_enabled:
                            end_memory = tracemalloc.get_traced_memory()[0]
                        
                        self._record_function_call(
                            profile_key, f.__name__, f.__module__,
                            execution_time, end_memory - start_memory,
                            track_calls
                        )
            
            @functools.wraps(f)
            def sync_wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return f(*args, **kwargs)
                
                profile_key = f"{f.__module__}.{f.__name__}"
                start_time = time.perf_counter()
                start_memory = 0
                
                if track_memory and self.memory_tracking_enabled:
                    start_memory = tracemalloc.get_traced_memory()[0]
                
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    if execution_time >= min_time_threshold:
                        end_memory = 0
                        if track_memory and self.memory_tracking_enabled:
                            end_memory = tracemalloc.get_traced_memory()[0]
                        
                        self._record_function_call(
                            profile_key, f.__name__, f.__module__,
                            execution_time, end_memory - start_memory,
                            track_calls
                        )
            
            return async_wrapper if asyncio.iscoroutinefunction(f) else sync_wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def _record_function_call(self, profile_key: str, function_name: str, 
                            module_name: str, execution_time: float, 
                            memory_usage: float, track_calls: bool):
        """Record a function call in the profile"""
        with self._lock:
            if profile_key not in self.function_profiles:
                self.function_profiles[profile_key] = FunctionProfile(
                    function_name=function_name,
                    module_name=module_name,
                    total_calls=0,
                    total_time=0.0,
                    avg_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    memory_usage=0.0,
                    last_called=datetime.utcnow()
                )
            
            profile = self.function_profiles[profile_key]
            
            if track_calls:
                profile.total_calls += 1
                profile.total_time += execution_time
                profile.avg_time = profile.total_time / profile.total_calls
                profile.min_time = min(profile.min_time, execution_time)
                profile.max_time = max(profile.max_time, execution_time)
                profile.memory_usage += memory_usage
                profile.last_called = datetime.utcnow()
    
    def start_profile(self, name: str, profile_type: ProfilerType = ProfilerType.FUNCTION) -> str:
        """Start a named profile"""
        profile_id = f"{name}_{int(time.time() * 1000)}"
        
        with self._lock:
            self.active_profiles[profile_id] = {
                "name": name,
                "type": profile_type,
                "start_time": time.perf_counter(),
                "start_memory": tracemalloc.get_traced_memory()[0] if self.memory_tracking_enabled else 0,
                "start_cpu": psutil.Process().cpu_percent()
            }
        
        return profile_id
    
    def end_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """End a named profile and return results"""
        with self._lock:
            if profile_id not in self.active_profiles:
                return None
            
            profile_data = self.active_profiles[profile_id]
            end_time = time.perf_counter()
            end_memory = tracemalloc.get_traced_memory()[0] if self.memory_tracking_enabled else 0
            end_cpu = psutil.Process().cpu_percent()
            
            duration = end_time - profile_data["start_time"]
            memory_delta = end_memory - profile_data["start_memory"]
            cpu_delta = end_cpu - profile_data["start_cpu"]
            
            result = {
                "profile_id": profile_id,
                "name": profile_data["name"],
                "type": profile_data["type"].value,
                "duration": duration,
                "memory_delta": memory_delta,
                "cpu_delta": cpu_delta,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            del self.active_profiles[profile_id]
            return result
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get current memory profile"""
        current_memory = psutil.Process().memory_info().rss
        peak_memory = psutil.Process().memory_info().peak_wss if hasattr(psutil.Process().memory_info(), 'peak_wss') else current_memory
        
        # Calculate memory growth
        memory_growth = 0
        if self.memory_snapshots:
            last_snapshot = self.memory_snapshots[-1]
            memory_growth = current_memory - last_snapshot.current_memory
        
        # Get top memory allocations
        top_allocations = []
        if self.memory_tracking_enabled:
            try:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                top_allocations = [
                    {
                        "filename": stat.traceback.format()[0],
                        "size": stat.size,
                        "count": stat.count
                    }
                    for stat in top_stats[:10]
                ]
            except Exception as e:
                logger.error(f"Error getting memory allocations: {e}")
        
        profile = MemoryProfile(
            timestamp=datetime.utcnow(),
            current_memory=current_memory,
            peak_memory=peak_memory,
            memory_growth=memory_growth,
            top_allocations=top_allocations
        )
        
        self.memory_snapshots.append(profile)
        return profile
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        process = psutil.Process()
        
        # CPU metrics
        cpu_percent = process.cpu_percent()
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # I/O metrics
        io_counters = process.io_counters()
        
        # Thread metrics
        num_threads = process.num_threads()
        
        # File descriptor metrics
        num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count
            },
            "memory": {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": memory_percent
            },
            "io": {
                "read_count": io_counters.read_count,
                "write_count": io_counters.write_count,
                "read_bytes": io_counters.read_bytes,
                "write_bytes": io_counters.write_bytes
            },
            "threads": num_threads,
            "file_descriptors": num_fds
        }
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            # Function performance summary
            function_summary = {}
            for profile_key, profile in self.function_profiles.items():
                function_summary[profile_key] = {
                    "total_calls": profile.total_calls,
                    "total_time": profile.total_time,
                    "avg_time": profile.avg_time,
                    "min_time": profile.min_time,
                    "max_time": profile.max_time,
                    "memory_usage": profile.memory_usage,
                    "last_called": profile.last_called.isoformat()
                }
            
            # Memory profile
            memory_profile = self.get_memory_profile()
            
            # System metrics
            system_metrics = self.get_system_metrics()
            
            # Performance recommendations
            recommendations = self._generate_recommendations()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - self._start_time,
                "function_profiles": function_summary,
                "memory_profile": {
                    "current_memory": memory_profile.current_memory,
                    "peak_memory": memory_profile.peak_memory,
                    "memory_growth": memory_profile.memory_growth,
                    "top_allocations": memory_profile.top_allocations
                },
                "system_metrics": system_metrics,
                "recommendations": recommendations,
                "profiling_enabled": self.profiling_enabled,
                "memory_tracking_enabled": self.memory_tracking_enabled
            }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        with self._lock:
            # Analyze function performance
            for profile_key, profile in self.function_profiles.items():
                if profile.total_calls > 100 and profile.avg_time > 0.1:
                    recommendations.append({
                        "type": "slow_function",
                        "severity": "medium",
                        "function": profile_key,
                        "avg_time": profile.avg_time,
                        "total_calls": profile.total_calls,
                        "recommendation": f"Function {profile_key} is slow (avg {profile.avg_time:.3f}s). Consider optimization or caching."
                    })
                
                if profile.memory_usage > 1024 * 1024:  # 1MB
                    recommendations.append({
                        "type": "high_memory_usage",
                        "severity": "high",
                        "function": profile_key,
                        "memory_usage": profile.memory_usage,
                        "recommendation": f"Function {profile_key} uses {profile.memory_usage / 1024 / 1024:.2f}MB. Check for memory leaks."
                    })
        
        # Analyze memory profile
        if self.memory_snapshots:
            recent_snapshots = list(self.memory_snapshots)[-10:]
            if len(recent_snapshots) > 1:
                memory_growth = recent_snapshots[-1].current_memory - recent_snapshots[0].current_memory
                if memory_growth > 50 * 1024 * 1024:  # 50MB growth
                    recommendations.append({
                        "type": "memory_growth",
                        "severity": "high",
                        "memory_growth": memory_growth,
                        "recommendation": f"Memory usage increased by {memory_growth / 1024 / 1024:.2f}MB. Check for memory leaks."
                    })
        
        return recommendations
    
    def get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest functions by average execution time"""
        with self._lock:
            sorted_functions = sorted(
                self.function_profiles.items(),
                key=lambda x: x[1].avg_time,
                reverse=True
            )
            
            return [
                {
                    "function": profile_key,
                    "avg_time": profile.avg_time,
                    "total_calls": profile.total_calls,
                    "total_time": profile.total_time,
                    "memory_usage": profile.memory_usage
                }
                for profile_key, profile in sorted_functions[:limit]
            ]
    
    def get_most_called_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently called functions"""
        with self._lock:
            sorted_functions = sorted(
                self.function_profiles.items(),
                key=lambda x: x[1].total_calls,
                reverse=True
            )
            
            return [
                {
                    "function": profile_key,
                    "total_calls": profile.total_calls,
                    "avg_time": profile.avg_time,
                    "total_time": profile.total_time
                }
                for profile_key, profile in sorted_functions[:limit]
            ]
    
    def clear_profiles(self):
        """Clear all profiling data"""
        with self._lock:
            self.function_profiles.clear()
            self.memory_snapshots.clear()
            self.performance_history.clear()
            self.active_profiles.clear()
    
    def enable_profiling(self):
        """Enable profiling"""
        self.profiling_enabled = True
        logger.info("Performance profiling enabled")
    
    def disable_profiling(self):
        """Disable profiling"""
        self.profiling_enabled = False
        logger.info("Performance profiling disabled")

# Global profiler instance
performance_profiler = PerformanceProfiler()

# Convenience decorator
def profile_function(func: Callable = None, *, 
                    track_memory: bool = False,
                    track_calls: bool = True,
                    min_time_threshold: float = 0.001):
    """Convenience decorator for function profiling"""
    return performance_profiler.profile_function(
        func, 
        track_memory=track_memory,
        track_calls=track_calls,
        min_time_threshold=min_time_threshold
    )

# Convenience functions
def start_profile(name: str, profile_type: ProfilerType = ProfilerType.FUNCTION) -> str:
    """Start a named profile"""
    return performance_profiler.start_profile(name, profile_type)

def end_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    """End a named profile"""
    return performance_profiler.end_profile(profile_id)

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary"""
    return performance_profiler.get_performance_summary()

def get_slowest_functions(limit: int = 10) -> List[Dict[str, Any]]:
    """Get slowest functions"""
    return performance_profiler.get_slowest_functions(limit)

def get_most_called_functions(limit: int = 10) -> List[Dict[str, Any]]:
    """Get most called functions"""
    return performance_profiler.get_most_called_functions(limit)
