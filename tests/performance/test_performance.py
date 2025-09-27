"""
Performance Tests for Opinion Market Platform
Comprehensive performance testing suite
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any

from app.core.enhanced_cache import EnhancedCache, EvictionPolicy, CompressionLevel
from app.core.security_manager import security_manager
from app.services.service_registry import service_registry
from app.services.inter_service_communication import inter_service_comm
from tests.test_utils import (
    PerformanceTestRunner, LoadTestRunner, TestDataGenerator, 
    TestAssertions, TestResult
)


class TestCachePerformance:
    """Test cache performance under various loads"""
    
    @pytest.fixture
    async def performance_cache(self):
        """Create cache optimized for performance testing"""
        cache = EnhancedCache(
            max_size=1000,
            default_ttl=3600,
            eviction_policy=EvictionPolicy.LRU,
            compression_level=CompressionLevel.MEDIUM,
            enable_analytics=True,
            max_memory_mb=100
        )
        yield cache
        await cache.stop_cleanup()
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_cache_throughput(self, performance_cache):
        """Test cache throughput under high load"""
        runner = PerformanceTestRunner()
        
        async def cache_operation(user_id: int):
            """Simulate cache operations for a user"""
            key = f"user_{user_id}_data"
            value = f"user_data_{user_id}" * 100  # Larger value for realistic testing
            
            # Set operation
            await performance_cache.set(key, value)
            
            # Get operation
            retrieved_value = await performance_cache.get(key)
            
            return retrieved_value == value
        
        # Run performance test
        results = await runner.run_concurrent_test(
            test_func=cache_operation,
            concurrent_users=100,
            test_duration=30
        )
        
        # Assertions
        TestAssertions.assert_success_rate(
            results["metrics"]["successful_requests"],
            results["metrics"]["total_requests"],
            0.95  # 95% success rate
        )
        
        TestAssertions.assert_response_time(
            results["metrics"]["average_response_time"],
            0.1  # 100ms max average response time
        )
        
        # Should handle at least 1000 requests per second
        assert results["metrics"]["requests_per_second"] >= 1000
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_cache_memory_efficiency(self, performance_cache):
        """Test cache memory efficiency under load"""
        # Add large amounts of data
        large_data = []
        for i in range(500):
            data = {
                "id": i,
                "content": "x" * 1000,  # 1KB per entry
                "metadata": {"created": time.time(), "user": f"user_{i}"}
            }
            large_data.append(data)
            await performance_cache.set(f"large_data_{i}", data)
        
        # Check memory usage
        memory_info = performance_cache.get_memory_usage()
        
        # Should use reasonable amount of memory
        TestAssertions.assert_memory_usage(
            memory_info["memory_usage_mb"],
            50  # Should not exceed 50MB
        )
        
        # Compression should be effective
        assert memory_info["compression_ratio"] < 0.8  # At least 20% compression
    
    @pytest.mark.performance
    async def test_cache_eviction_performance(self, performance_cache):
        """Test cache eviction performance"""
        # Fill cache to capacity
        for i in range(1000):
            await performance_cache.set(f"key_{i}", f"value_{i}")
        
        # Measure eviction performance
        start_time = time.time()
        
        # Trigger evictions by adding more data
        for i in range(1000, 1100):
            await performance_cache.set(f"key_{i}", f"value_{i}")
        
        end_time = time.time()
        eviction_time = end_time - start_time
        
        # Eviction should be fast
        TestAssertions.assert_response_time(eviction_time, 1.0)  # 1 second max
        
        # Cache should still be functional
        stats = performance_cache.get_stats()
        assert stats["entry_count"] <= 1000  # Should not exceed max size


class TestSecurityPerformance:
    """Test security system performance"""
    
    @pytest.mark.performance
    async def test_rate_limiting_performance(self):
        """Test rate limiting performance under load"""
        runner = PerformanceTestRunner()
        
        async def rate_limit_operation(user_id: int):
            """Simulate rate limit checking"""
            client_ip = f"192.168.1.{user_id % 255}"
            allowed, _ = await security_manager.check_rate_limit(client_ip, "api")
            return allowed
        
        # Run performance test
        results = await runner.run_concurrent_test(
            test_func=rate_limit_operation,
            concurrent_users=200,
            test_duration=20
        )
        
        # Rate limiting should be fast
        TestAssertions.assert_response_time(
            results["metrics"]["average_response_time"],
            0.01  # 10ms max average response time
        )
        
        # Should handle high throughput
        assert results["metrics"]["requests_per_second"] >= 5000
    
    @pytest.mark.performance
    async def test_threat_detection_performance(self):
        """Test threat detection performance"""
        runner = PerformanceTestRunner()
        
        # Generate test data with various patterns
        test_data_generator = TestDataGenerator()
        
        async def threat_detection_operation(user_id: int):
            """Simulate threat detection"""
            # Mix of safe and potentially malicious data
            if user_id % 10 == 0:
                # Potentially malicious data
                request_data = {
                    "source_ip": f"192.168.1.{user_id % 255}",
                    "user_id": f"user_{user_id}",
                    "query": "SELECT * FROM users; DROP TABLE users;",
                    "script": "<script>alert('xss')</script>"
                }
            else:
                # Safe data
                request_data = {
                    "source_ip": f"192.168.1.{user_id % 255}",
                    "user_id": f"user_{user_id}",
                    "query": "SELECT name FROM users WHERE id = 1",
                    "content": "This is safe content"
                }
            
            threats = await security_manager.detect_threats(request_data)
            return len(threats)
        
        # Run performance test
        results = await runner.run_concurrent_test(
            test_func=threat_detection_operation,
            concurrent_users=100,
            test_duration=15
        )
        
        # Threat detection should be reasonably fast
        TestAssertions.assert_response_time(
            results["metrics"]["average_response_time"],
            0.05  # 50ms max average response time
        )


class TestServiceRegistryPerformance:
    """Test service registry performance"""
    
    @pytest.mark.performance
    async def test_service_discovery_performance(self):
        """Test service discovery performance"""
        # Register multiple services
        for i in range(50):
            await service_registry.register_service(
                service_name="test-service",
                instance_id=f"instance_{i}",
                host=f"192.168.1.{i % 255}",
                port=8000 + i,
                version="1.0.0"
            )
        
        runner = PerformanceTestRunner()
        
        async def discovery_operation(user_id: int):
            """Simulate service discovery"""
            instance = await service_registry.get_service_instance("test-service")
            return instance is not None
        
        # Run performance test
        results = await runner.run_concurrent_test(
            test_func=discovery_operation,
            concurrent_users=100,
            test_duration=20
        )
        
        # Service discovery should be fast
        TestAssertions.assert_response_time(
            results["metrics"]["average_response_time"],
            0.01  # 10ms max average response time
        )
        
        # Should have high success rate
        TestAssertions.assert_success_rate(
            results["metrics"]["successful_requests"],
            results["metrics"]["total_requests"],
            0.99  # 99% success rate
        )


class TestInterServiceCommunicationPerformance:
    """Test inter-service communication performance"""
    
    @pytest.fixture
    async def comm_setup(self):
        """Setup inter-service communication for testing"""
        await inter_service_comm.initialize()
        yield inter_service_comm
        await inter_service_comm.cleanup()
    
    @pytest.mark.performance
    async def test_communication_throughput(self, comm_setup):
        """Test communication throughput"""
        runner = PerformanceTestRunner()
        
        async def communication_operation(user_id: int):
            """Simulate inter-service communication"""
            # Mock service call (since we don't have real services in tests)
            try:
                # Simulate the communication overhead
                await asyncio.sleep(0.001)  # 1ms simulated network delay
                return {"status": "success", "user_id": user_id}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        # Run performance test
        results = await runner.run_concurrent_test(
            test_func=communication_operation,
            concurrent_users=50,
            test_duration=15
        )
        
        # Communication should be reasonably fast
        TestAssertions.assert_response_time(
            results["metrics"]["average_response_time"],
            0.02  # 20ms max average response time
        )


class TestLoadScenarios:
    """Test various load scenarios"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_gradual_load_increase(self):
        """Test system behavior under gradually increasing load"""
        cache = EnhancedCache(max_size=1000, enable_analytics=True)
        
        runner = LoadTestRunner()
        
        async def load_operation(user_id: int):
            """Simulate user operations"""
            # Simulate typical user behavior
            await cache.set(f"user_{user_id}_session", {"user_id": user_id, "timestamp": time.time()})
            await cache.get(f"user_{user_id}_session")
            await cache.set(f"user_{user_id}_preferences", {"theme": "dark", "notifications": True})
            
            return True
        
        # Run load test with gradual ramp-up
        results = await runner.run_load_test(
            test_func=load_operation,
            max_users=200,
            ramp_up_time=30,  # 30 seconds ramp-up
            test_duration=60  # 60 seconds test
        )
        
        # System should handle gradual load increase
        assert len(results["results"]) > 0
        
        # Check cache performance under load
        stats = cache.get_stats()
        assert stats["hit_rate"] > 0.8  # Should maintain good hit rate
        
        await cache.stop_cleanup()
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_burst_load(self):
        """Test system behavior under burst load"""
        cache = EnhancedCache(max_size=500, enable_analytics=True)
        
        runner = PerformanceTestRunner()
        
        async def burst_operation(user_id: int):
            """Simulate burst operations"""
            # Rapid operations
            for i in range(10):
                await cache.set(f"burst_{user_id}_{i}", f"data_{i}")
                await cache.get(f"burst_{user_id}_{i}")
            
            return True
        
        # Run burst test
        results = await runner.run_concurrent_test(
            test_func=burst_operation,
            concurrent_users=100,
            test_duration=10  # Short burst
        )
        
        # System should handle burst load
        TestAssertions.assert_success_rate(
            results["metrics"]["successful_requests"],
            results["metrics"]["total_requests"],
            0.9  # 90% success rate under burst
        )
        
        await cache.stop_cleanup()


class TestMemoryPerformance:
    """Test memory performance and leak detection"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_memory_usage_under_load(self):
        """Test memory usage under sustained load"""
        cache = EnhancedCache(
            max_size=1000,
            enable_analytics=True,
            max_memory_mb=50
        )
        
        # Record initial memory
        initial_memory = cache.get_memory_usage()
        
        # Run sustained load
        for cycle in range(10):
            # Add data
            for i in range(100):
                await cache.set(f"cycle_{cycle}_key_{i}", f"data_{i}" * 100)
            
            # Access data
            for i in range(100):
                await cache.get(f"cycle_{cycle}_key_{i}")
            
            # Check memory usage
            current_memory = cache.get_memory_usage()
            
            # Memory should not grow unbounded
            TestAssertions.assert_memory_usage(
                current_memory["memory_usage_mb"],
                50  # Should not exceed 50MB
            )
        
        # Final memory check
        final_memory = cache.get_memory_usage()
        
        # Memory should be reasonable
        TestAssertions.assert_memory_usage(
            final_memory["memory_usage_mb"],
            50
        )
        
        await cache.stop_cleanup()
    
    @pytest.mark.performance
    async def test_cache_cleanup_performance(self):
        """Test cache cleanup performance"""
        cache = EnhancedCache(max_size=1000, enable_analytics=True)
        
        # Fill cache
        for i in range(1000):
            await cache.set(f"cleanup_key_{i}", f"data_{i}")
        
        # Measure cleanup time
        start_time = time.time()
        await cache.clear()
        end_time = time.time()
        
        cleanup_time = end_time - start_time
        
        # Cleanup should be fast
        TestAssertions.assert_response_time(cleanup_time, 0.5)  # 500ms max
        
        # Verify cache is empty
        stats = cache.get_stats()
        assert stats["entry_count"] == 0
        
        await cache.stop_cleanup()
