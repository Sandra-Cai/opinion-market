"""
Advanced Test Utilities
Comprehensive testing utilities for Opinion Market platform
"""

import asyncio
import time
import json
import random
import string
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.enhanced_cache import enhanced_cache
from app.core.security_manager import security_manager
from app.services.service_registry import service_registry
from app.events.event_store import event_store
from app.events.event_bus import event_bus


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_user_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate test user data"""
        users = []
        for i in range(count):
            users.append({
                "username": f"testuser_{i}_{random.randint(1000, 9999)}",
                "email": f"test_{i}_{random.randint(1000, 9999)}@example.com",
                "password": f"password_{i}_{random.randint(1000, 9999)}",
                "full_name": f"Test User {i}"
            })
        return users
    
    @staticmethod
    def generate_market_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate test market data"""
        markets = []
        categories = ["politics", "sports", "technology", "economics", "entertainment"]
        market_types = ["binary", "multiple_choice"]
        
        for i in range(count):
            markets.append({
                "title": f"Test Market {i} - {random.choice(categories).title()}",
                "description": f"This is a test market for {random.choice(categories)}",
                "category": random.choice(categories),
                "market_type": random.choice(market_types),
                "outcomes": ["Yes", "No"] if random.choice(market_types) == "binary" else ["Option A", "Option B", "Option C"],
                "end_date": (datetime.now() + timedelta(days=random.randint(1, 365))).isoformat()
            })
        return markets
    
    @staticmethod
    def generate_trade_data(count: int = 1, market_ids: List[int] = None) -> List[Dict[str, Any]]:
        """Generate test trade data"""
        trades = []
        outcomes = ["Yes", "No", "Option A", "Option B", "Option C"]
        order_types = ["market", "limit", "stop"]
        sides = ["buy", "sell"]
        
        for i in range(count):
            trades.append({
                "market_id": random.choice(market_ids) if market_ids else random.randint(1, 100),
                "outcome": random.choice(outcomes),
                "amount": round(random.uniform(10, 1000), 2),
                "order_type": random.choice(order_types),
                "side": random.choice(sides),
                "limit_price": round(random.uniform(0.1, 0.9), 2) if random.choice(order_types) == "limit" else None
            })
        return trades
    
    @staticmethod
    def generate_event_data(count: int = 1) -> List[Dict[str, Any]]:
        """Generate test event data"""
        events = []
        event_types = ["user_created", "market_created", "trade_executed", "market_resolved"]
        aggregate_types = ["user", "market", "trade", "position"]
        
        for i in range(count):
            events.append({
                "event_type": random.choice(event_types),
                "aggregate_id": f"aggregate_{i}_{random.randint(1000, 9999)}",
                "aggregate_type": random.choice(aggregate_types),
                "event_data": {
                    "test_field": f"test_value_{i}",
                    "timestamp": time.time(),
                    "random_data": random.randint(1, 1000)
                },
                "metadata": {
                    "source": "test_generator",
                    "version": "1.0.0"
                }
            })
        return events


class PerformanceTestRunner:
    """Run performance tests with comprehensive metrics"""
    
    def __init__(self):
        self.results = []
        self.metrics = {}
    
    async def run_concurrent_test(self, test_func: Callable, concurrent_users: int, 
                                test_duration: int) -> Dict[str, Any]:
        """Run concurrent performance test"""
        start_time = time.time()
        tasks = []
        
        # Create concurrent tasks
        for i in range(concurrent_users):
            task = asyncio.create_task(self._run_user_test(test_func, i))
            tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(test_duration)
        
        # Cancel remaining tasks
        for task in tasks:
            task.cancel()
        
        # Collect results
        results = []
        for task in tasks:
            try:
                result = await task
                if result:
                    results.append(result)
            except asyncio.CancelledError:
                pass
        
        end_time = time.time()
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, start_time, end_time)
        
        return {
            "total_requests": len(results),
            "concurrent_users": concurrent_users,
            "test_duration": test_duration,
            "metrics": metrics,
            "results": results
        }
    
    async def _run_user_test(self, test_func: Callable, user_id: int) -> Optional[Dict[str, Any]]:
        """Run test for a single user"""
        try:
            start_time = time.time()
            result = await test_func(user_id)
            end_time = time.time()
            
            return {
                "user_id": user_id,
                "success": True,
                "response_time": end_time - start_time,
                "result": result
            }
        except Exception as e:
            return {
                "user_id": user_id,
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    def _calculate_metrics(self, results: List[Dict[str, Any]], 
                          start_time: float, end_time: float) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        response_times = [r["response_time"] for r in successful_results]
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "requests_per_second": len(results) / (end_time - start_time),
            "average_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "p95_response_time": self._percentile(response_times, 95) if response_times else 0,
            "p99_response_time": self._percentile(response_times, 99) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class LoadTestRunner:
    """Run load tests with gradual ramp-up"""
    
    def __init__(self):
        self.results = []
    
    async def run_load_test(self, test_func: Callable, max_users: int, 
                          ramp_up_time: int, test_duration: int) -> Dict[str, Any]:
        """Run load test with gradual ramp-up"""
        start_time = time.time()
        all_tasks = []
        
        # Gradual ramp-up
        ramp_up_interval = ramp_up_time / max_users if max_users > 0 else 0
        
        for i in range(max_users):
            # Wait for ramp-up interval
            if i > 0:
                await asyncio.sleep(ramp_up_interval)
            
            # Start user test
            task = asyncio.create_task(self._run_load_user_test(test_func, i))
            all_tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(test_duration)
        
        # Cancel all tasks
        for task in all_tasks:
            task.cancel()
        
        # Collect results
        results = []
        for task in all_tasks:
            try:
                result = await task
                if result:
                    results.append(result)
            except asyncio.CancelledError:
                pass
        
        end_time = time.time()
        
        return {
            "max_users": max_users,
            "ramp_up_time": ramp_up_time,
            "test_duration": test_duration,
            "total_duration": end_time - start_time,
            "results": results
        }
    
    async def _run_load_user_test(self, test_func: Callable, user_id: int) -> Optional[Dict[str, Any]]:
        """Run load test for a single user"""
        try:
            start_time = time.time()
            result = await test_func(user_id)
            end_time = time.time()
            
            return {
                "user_id": user_id,
                "success": True,
                "response_time": end_time - start_time,
                "result": result
            }
        except Exception as e:
            return {
                "user_id": user_id,
                "success": False,
                "error": str(e),
                "response_time": 0
            }


class TestAssertions:
    """Advanced test assertions for Opinion Market platform"""
    
    @staticmethod
    def assert_response_time(response_time: float, max_time: float):
        """Assert response time is within acceptable limits"""
        assert response_time <= max_time, f"Response time {response_time:.3f}s exceeds maximum {max_time:.3f}s"
    
    @staticmethod
    def assert_success_rate(successful_requests: int, total_requests: int, min_rate: float):
        """Assert success rate meets minimum requirements"""
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        assert success_rate >= min_rate, f"Success rate {success_rate:.2%} below minimum {min_rate:.2%}"
    
    @staticmethod
    def assert_cache_hit_rate(cache_hits: int, total_requests: int, min_rate: float):
        """Assert cache hit rate meets minimum requirements"""
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        assert hit_rate >= min_rate, f"Cache hit rate {hit_rate:.2%} below minimum {min_rate:.2%}"
    
    @staticmethod
    def assert_memory_usage(memory_usage: float, max_memory: float):
        """Assert memory usage is within acceptable limits"""
        assert memory_usage <= max_memory, f"Memory usage {memory_usage:.2f}MB exceeds maximum {max_memory:.2f}MB"
    
    @staticmethod
    def assert_database_connections(active_connections: int, max_connections: int):
        """Assert database connections are within limits"""
        assert active_connections <= max_connections, f"Active connections {active_connections} exceed maximum {max_connections}"


class TestDataCleanup:
    """Clean up test data after tests"""
    
    @staticmethod
    async def cleanup_cache():
        """Clean up test cache data"""
        await enhanced_cache.clear()
    
    @staticmethod
    async def cleanup_security_manager():
        """Clean up security manager test data"""
        security_manager.security_events.clear()
        security_manager.blocked_ips.clear()
        security_manager.rate_limits.clear()
    
    @staticmethod
    async def cleanup_service_registry():
        """Clean up service registry test data"""
        service_registry.services.clear()
    
    @staticmethod
    async def cleanup_event_store():
        """Clean up event store test data"""
        event_store.events_cache.clear()
        event_store.snapshots_cache.clear()
    
    @staticmethod
    async def cleanup_event_bus():
        """Clean up event bus test data"""
        event_bus.subscriptions.clear()
        event_bus.published_events = 0
        event_bus.failed_events = 0
    
    @staticmethod
    async def cleanup_all():
        """Clean up all test data"""
        await TestDataCleanup.cleanup_cache()
        await TestDataCleanup.cleanup_security_manager()
        await TestDataCleanup.cleanup_service_registry()
        await TestDataCleanup.cleanup_event_store()
        await TestDataCleanup.cleanup_event_bus()


class TestReportGenerator:
    """Generate comprehensive test reports"""
    
    @staticmethod
    def generate_performance_report(test_results: List[TestResult]) -> Dict[str, Any]:
        """Generate performance test report"""
        if not test_results:
            return {}
        
        successful_tests = [r for r in test_results if r.success]
        failed_tests = [r for r in test_results if not r.success]
        
        durations = [r.duration for r in test_results]
        
        return {
            "summary": {
                "total_tests": len(test_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(test_results)
            },
            "performance": {
                "average_duration": statistics.mean(durations),
                "median_duration": statistics.median(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations)
            },
            "failed_tests": [
                {
                    "name": r.test_name,
                    "error": r.error_message
                }
                for r in failed_tests
            ]
        }
    
    @staticmethod
    def generate_coverage_report(coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test coverage report"""
        return {
            "overall_coverage": coverage_data.get("overall_coverage", 0),
            "line_coverage": coverage_data.get("line_coverage", 0),
            "branch_coverage": coverage_data.get("branch_coverage", 0),
            "function_coverage": coverage_data.get("function_coverage", 0),
            "files": coverage_data.get("files", {})
        }
