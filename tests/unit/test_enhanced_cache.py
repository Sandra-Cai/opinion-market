"""
Unit Tests for Enhanced Cache System
Comprehensive unit tests for cache functionality
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from app.core.enhanced_cache import (
    EnhancedCache, EvictionPolicy, CompressionLevel, 
    CacheError, CacheCompressionError, CacheSerializationError
)
from tests.test_utils import TestDataGenerator, TestAssertions


class TestEnhancedCache:
    """Test enhanced cache functionality"""
    
    @pytest.fixture
    async def cache(self):
        """Create test cache instance"""
        cache = EnhancedCache(
            max_size=10,
            default_ttl=60,
            eviction_policy=EvictionPolicy.LRU,
            compression_level=CompressionLevel.MEDIUM,
            enable_analytics=True,
            max_memory_mb=10
        )
        yield cache
        await cache.stop_cleanup()
    
    @pytest.mark.unit
    async def test_cache_initialization(self, cache):
        """Test cache initialization"""
        assert cache.max_size == 10
        assert cache.default_ttl == 60
        assert cache.eviction_policy == EvictionPolicy.LRU
        assert cache.compression_level == CompressionLevel.MEDIUM
        assert cache.enable_analytics is True
        assert cache.max_memory_mb == 10
    
    @pytest.mark.unit
    async def test_set_and_get(self, cache):
        """Test basic set and get operations"""
        # Set a value
        success = await cache.set("test_key", "test_value")
        assert success is True
        
        # Get the value
        value = await cache.get("test_key")
        assert value == "test_value"
        
        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["entry_count"] == 1
    
    @pytest.mark.unit
    async def test_get_nonexistent_key(self, cache):
        """Test getting non-existent key"""
        value = await cache.get("nonexistent_key")
        assert value is None
        
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
    
    @pytest.mark.unit
    async def test_ttl_expiration(self, cache):
        """Test TTL expiration"""
        # Set value with short TTL
        await cache.set("expiring_key", "expiring_value", ttl=1)
        
        # Value should be available immediately
        value = await cache.get("expiring_key")
        assert value == "expiring_value"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Value should be expired
        value = await cache.get("expiring_key")
        assert value is None
    
    @pytest.mark.unit
    async def test_eviction_policy_lru(self, cache):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        for i in range(10):
            await cache.set(f"key_{i}", f"value_{i}")
        
        # Access first key to make it recently used
        await cache.get("key_0")
        
        # Add one more key to trigger eviction
        await cache.set("key_10", "value_10")
        
        # key_1 should be evicted (least recently used)
        value = await cache.get("key_1")
        assert value is None
        
        # key_0 should still be available
        value = await cache.get("key_0")
        assert value == "value_0"
    
    @pytest.mark.unit
    async def test_eviction_policy_lfu(self):
        """Test LFU eviction policy"""
        cache = EnhancedCache(
            max_size=3,
            eviction_policy=EvictionPolicy.LFU,
            enable_analytics=True
        )
        
        # Add keys
        await cache.set("key_0", "value_0")
        await cache.set("key_1", "value_1")
        await cache.set("key_2", "value_2")
        
        # Access key_0 multiple times
        await cache.get("key_0")
        await cache.get("key_0")
        await cache.get("key_0")
        
        # Access key_1 once
        await cache.get("key_1")
        
        # Add new key to trigger eviction
        await cache.set("key_3", "value_3")
        
        # key_2 should be evicted (least frequently used)
        value = await cache.get("key_2")
        assert value is None
        
        # key_0 should still be available (most frequently used)
        value = await cache.get("key_0")
        assert value == "value_0"
        
        await cache.stop_cleanup()
    
    @pytest.mark.unit
    async def test_compression(self):
        """Test cache compression"""
        cache = EnhancedCache(
            max_size=10,
            compression_level=CompressionLevel.MAX,
            enable_analytics=True
        )
        
        # Set large value
        large_value = "x" * 1000
        await cache.set("large_key", large_value)
        
        # Get value
        retrieved_value = await cache.get("large_key")
        assert retrieved_value == large_value
        
        # Check compression stats
        stats = cache.get_stats()
        assert stats["compression_ratio"] < 1.0  # Should be compressed
        
        await cache.stop_cleanup()
    
    @pytest.mark.unit
    async def test_tags(self, cache):
        """Test cache tags functionality"""
        # Set values with tags
        await cache.set("key_1", "value_1", tags=["tag1", "tag2"])
        await cache.set("key_2", "value_2", tags=["tag2", "tag3"])
        await cache.set("key_3", "value_3", tags=["tag1"])
        
        # Get entries by tag
        tag1_entries = cache.get_entries_by_tag("tag1")
        assert len(tag1_entries) == 2
        
        tag2_entries = cache.get_entries_by_tag("tag2")
        assert len(tag2_entries) == 2
        
        # Delete by tags
        deleted_count = await cache.delete_by_tags(["tag1"])
        assert deleted_count == 2
        
        # Check remaining entries
        value = await cache.get("key_2")
        assert value == "value_2"  # Should still exist
        
        value = await cache.get("key_1")
        assert value is None  # Should be deleted
    
    @pytest.mark.unit
    async def test_priority_and_cost(self, cache):
        """Test priority and cost-based caching"""
        # Set values with different priorities and costs
        await cache.set("high_priority", "value_1", priority=10, cost=1.0)
        await cache.set("low_priority", "value_2", priority=1, cost=5.0)
        await cache.set("medium_priority", "value_3", priority=5, cost=2.0)
        
        # All should be available
        assert await cache.get("high_priority") == "value_1"
        assert await cache.get("low_priority") == "value_2"
        assert await cache.get("medium_priority") == "value_3"
    
    @pytest.mark.unit
    async def test_analytics(self, cache):
        """Test cache analytics"""
        # Perform various operations
        await cache.set("key_1", "value_1")
        await cache.set("key_2", "value_2")
        await cache.get("key_1")
        await cache.get("key_1")
        await cache.get("nonexistent")
        
        # Get analytics
        analytics = cache.get_analytics()
        assert analytics is not None
        assert analytics.hit_rate > 0
        assert analytics.miss_rate > 0
        assert analytics.total_requests > 0
    
    @pytest.mark.unit
    async def test_benchmark_performance(self, cache):
        """Test performance benchmarking"""
        benchmark_results = await cache.benchmark_performance(100)
        
        assert "set_operations" in benchmark_results
        assert "get_operations" in benchmark_results
        assert "average_set_time" in benchmark_results
        assert "average_get_time" in benchmark_results
        
        # Check that operations completed successfully
        assert benchmark_results["set_operations"]["successful"] > 0
        assert benchmark_results["get_operations"]["successful"] > 0
    
    @pytest.mark.unit
    async def test_health_check(self, cache):
        """Test cache health check"""
        health_status = await cache.health_check()
        
        assert "status" in health_status
        assert "checks" in health_status
        assert "timestamp" in health_status
        
        # Should be healthy for basic operations
        assert health_status["status"] == "healthy"
    
    @pytest.mark.unit
    async def test_memory_usage(self, cache):
        """Test memory usage tracking"""
        # Add some data
        for i in range(5):
            await cache.set(f"key_{i}", f"value_{i}")
        
        memory_info = cache.get_memory_usage()
        
        assert "total_size_bytes" in memory_info
        assert "compressed_size_bytes" in memory_info
        assert "memory_usage_mb" in memory_info
        assert "compression_ratio" in memory_info
        
        # Should have some memory usage
        assert memory_info["total_size_bytes"] > 0
    
    @pytest.mark.unit
    async def test_export_cache_data(self, cache):
        """Test cache data export"""
        # Add some data
        await cache.set("export_key", "export_value", tags=["export"])
        
        export_data = cache.export_cache_data()
        
        assert "cache_entries" in export_data
        assert "statistics" in export_data
        assert "analytics" in export_data
        assert "timestamp" in export_data
        
        # Should contain our test data
        assert len(export_data["cache_entries"]) > 0
    
    @pytest.mark.unit
    async def test_error_handling(self, cache):
        """Test error handling"""
        # Test with invalid data that might cause serialization errors
        with pytest.raises(CacheError):
            await cache.set("invalid_key", object())  # Non-serializable object
    
    @pytest.mark.unit
    async def test_concurrent_operations(self, cache):
        """Test concurrent cache operations"""
        async def set_operation(key_suffix):
            for i in range(10):
                await cache.set(f"concurrent_key_{key_suffix}_{i}", f"value_{i}")
        
        async def get_operation(key_suffix):
            for i in range(10):
                await cache.get(f"concurrent_key_{key_suffix}_{i}")
        
        # Run concurrent operations
        tasks = []
        for i in range(5):
            tasks.append(set_operation(i))
            tasks.append(get_operation(i))
        
        await asyncio.gather(*tasks)
        
        # Verify some operations succeeded
        stats = cache.get_stats()
        assert stats["entry_count"] > 0
        assert stats["hits"] > 0 or stats["misses"] > 0
    
    @pytest.mark.unit
    async def test_clear_cache(self, cache):
        """Test cache clearing"""
        # Add some data
        await cache.set("key_1", "value_1")
        await cache.set("key_2", "value_2")
        
        # Verify data exists
        assert await cache.get("key_1") == "value_1"
        assert await cache.get("key_2") == "value_2"
        
        # Clear cache
        await cache.clear()
        
        # Verify data is gone
        assert await cache.get("key_1") is None
        assert await cache.get("key_2") is None
        
        # Check stats
        stats = cache.get_stats()
        assert stats["entry_count"] == 0
