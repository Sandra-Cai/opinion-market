"""
Comprehensive tests for the Enhanced Cache System
"""

import pytest
import asyncio
import time
import json
from unittest.mock import patch, MagicMock

from app.core.enhanced_cache import (
    EnhancedCache, 
    CacheEntry, 
    CacheStats, 
    CacheAnalytics,
    EvictionPolicy, 
    CompressionLevel
)


class TestEnhancedCache:
    """Test suite for EnhancedCache class"""
    
    @pytest.fixture
    async def cache(self):
        """Create a test cache instance"""
        cache = EnhancedCache(
            max_size=100,
            default_ttl=60,
            eviction_policy=EvictionPolicy.LRU,
            compression_level=CompressionLevel.DEFAULT,
            enable_analytics=True,
            max_memory_mb=10
        )
        await cache.start_cleanup()
        yield cache
        await cache.stop_cleanup()
    
    @pytest.mark.asyncio
    async def test_basic_set_get(self, cache):
        """Test basic set and get operations"""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Set value
        success = await cache.set(key, value)
        assert success is True
        
        # Get value
        retrieved_value = await cache.get(key)
        assert retrieved_value == value
        
        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["entry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_compression(self, cache):
        """Test compression functionality"""
        # Large data that should be compressed
        large_data = "x" * 10000
        key = "large_data"
        
        success = await cache.set(key, large_data)
        assert success is True
        
        # Check if compression was applied
        retrieved_value = await cache.get(key)
        assert retrieved_value == large_data
        
        # Check compression stats
        memory_info = cache.get_memory_usage()
        assert memory_info["compression_ratio"] < 1.0  # Should be compressed
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache):
        """Test TTL expiration"""
        key = "expiring_key"
        value = "expiring_value"
        
        # Set with short TTL
        success = await cache.set(key, value, ttl=1)
        assert success is True
        
        # Should be available immediately
        retrieved_value = await cache.get(key)
        assert retrieved_value == value
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        retrieved_value = await cache.get(key)
        assert retrieved_value is None
    
    @pytest.mark.asyncio
    async def test_tag_based_operations(self, cache):
        """Test tag-based operations"""
        # Set multiple entries with tags
        await cache.set("key1", "value1", tags=["tag1", "tag2"])
        await cache.set("key2", "value2", tags=["tag2", "tag3"])
        await cache.set("key3", "value3", tags=["tag1"])
        
        # Get entries by tag
        tag1_entries = cache.get_entries_by_tag("tag1")
        assert len(tag1_entries) == 2
        
        tag2_entries = cache.get_entries_by_tag("tag2")
        assert len(tag2_entries) == 2
        
        # Delete by tags
        deleted_count = await cache.delete_by_tags(["tag1"])
        assert deleted_count == 2
        
        # Check remaining entries
        remaining_entries = cache.get_entries_by_tag("tag2")
        assert len(remaining_entries) == 1
    
    @pytest.mark.asyncio
    async def test_eviction_policies(self, cache):
        """Test different eviction policies"""
        # Test LRU eviction
        cache.eviction_policy = EvictionPolicy.LRU
        
        # Fill cache beyond max_size
        for i in range(150):  # More than max_size (100)
            await cache.set(f"key_{i}", f"value_{i}")
        
        # Check that cache size is within limits
        stats = cache.get_stats()
        assert stats["entry_count"] <= cache.max_size
        
        # Test LFU eviction
        cache.eviction_policy = EvictionPolicy.LFU
        
        # Access some keys multiple times
        for _ in range(5):
            await cache.get("key_0")
            await cache.get("key_1")
        
        # Add more entries to trigger eviction
        for i in range(150, 200):
            await cache.set(f"key_{i}", f"value_{i}")
        
        # Frequently accessed keys should still be there
        assert await cache.get("key_0") is not None
        assert await cache.get("key_1") is not None
    
    @pytest.mark.asyncio
    async def test_analytics(self, cache):
        """Test analytics functionality"""
        # Perform various operations
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.get("key1")
        await cache.get("key1")  # Hit
        await cache.get("nonexistent")  # Miss
        
        # Get analytics
        analytics = cache.get_analytics()
        assert analytics is not None
        assert analytics.total_requests > 0
        assert analytics.hit_rate > 0
        assert "key1" in analytics.access_patterns
    
    @pytest.mark.asyncio
    async def test_benchmark_performance(self, cache):
        """Test performance benchmarking"""
        benchmark_results = await cache.benchmark_performance(iterations=100)
        
        assert "set_operations" in benchmark_results
        assert "get_operations" in benchmark_results
        
        set_ops = benchmark_results["set_operations"]
        get_ops = benchmark_results["get_operations"]
        
        assert set_ops["count"] == 100
        assert get_ops["count"] == 100
        assert set_ops["average_time_ms"] > 0
        assert get_ops["average_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_memory_management(self, cache):
        """Test memory management features"""
        # Set memory limit
        cache.max_memory_mb = 1  # 1MB limit
        
        # Add large entries to trigger memory-based eviction
        large_data = "x" * 100000  # 100KB per entry
        
        for i in range(20):  # Should exceed 1MB
            await cache.set(f"large_key_{i}", large_data)
        
        # Check memory usage
        memory_info = cache.get_memory_usage()
        assert memory_info["memory_usage_mb"] <= cache.max_memory_mb * 1.1  # Allow 10% tolerance
    
    @pytest.mark.asyncio
    async def test_warm_up(self, cache):
        """Test cache warm-up functionality"""
        warm_up_data = {
            "warm_key1": "warm_value1",
            "warm_key2": {"nested": "data"},
            "warm_key3": [1, 2, 3, 4, 5]
        }
        
        results = await cache.warm_up(warm_up_data)
        
        assert results["success"] == 3
        assert results["failed"] == 0
        assert results["total_size"] > 0
        
        # Verify all entries are in cache
        for key in warm_up_data:
            value = await cache.get(key)
            assert value == warm_up_data[key]
    
    @pytest.mark.asyncio
    async def test_export_import(self, cache):
        """Test cache data export"""
        # Add some test data
        await cache.set("export_key1", "export_value1", tags=["export"])
        await cache.set("export_key2", {"data": "export_value2"}, priority=5)
        
        # Export data
        export_data = cache.export_cache_data()
        
        assert "cache_entries" in export_data
        assert "statistics" in export_data
        assert "analytics" in export_data
        assert "memory_usage" in export_data
        
        # Check exported entries
        entries = export_data["cache_entries"]
        assert "export_key1" in entries
        assert "export_key2" in entries
        assert entries["export_key2"]["priority"] == 5
    
    @pytest.mark.asyncio
    async def test_decorator_functionality(self, cache):
        """Test cache decorator"""
        call_count = 0
        
        @cache.cache_decorator(ttl=60, tags=["decorator_test"])
        async def expensive_function(param):
            nonlocal call_count
            call_count += 1
            return f"result_for_{param}"
        
        # First call should execute function
        result1 = await expensive_function("test")
        assert result1 == "result_for_test"
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_function("test")
        assert result2 == "result_for_test"
        assert call_count == 1  # Should not increment
        
        # Different parameter should execute function again
        result3 = await expensive_function("different")
        assert result3 == "result_for_different"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cache):
        """Test error handling"""
        # Test with invalid data
        success = await cache.set("invalid_key", None)
        assert success is True  # Should handle None values
        
        # Test with very large data
        huge_data = "x" * (10 * 1024 * 1024)  # 10MB
        success = await cache.set("huge_key", huge_data)
        assert success is True
        
        # Test deletion of non-existent key
        success = await cache.delete("nonexistent_key")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cache):
        """Test concurrent operations"""
        async def set_operation(i):
            await cache.set(f"concurrent_key_{i}", f"value_{i}")
        
        async def get_operation(i):
            return await cache.get(f"concurrent_key_{i}")
        
        # Run concurrent set operations
        tasks = [set_operation(i) for i in range(50)]
        await asyncio.gather(*tasks)
        
        # Run concurrent get operations
        tasks = [get_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        # All gets should succeed
        assert all(result is not None for result in results)
        
        # Check final stats
        stats = cache.get_stats()
        assert stats["entry_count"] == 50


class TestCacheEntry:
    """Test suite for CacheEntry class"""
    
    def test_cache_entry_creation(self):
        """Test CacheEntry creation"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            priority=5,
            cost=10.0
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.priority == 5
        assert entry.cost == 10.0
        assert entry.access_count == 0
        assert entry.tags == []
        assert entry.metadata == {}


class TestEvictionPolicies:
    """Test suite for eviction policies"""
    
    @pytest.mark.asyncio
    async def test_all_eviction_policies(self):
        """Test all eviction policies"""
        policies = [
            EvictionPolicy.LRU,
            EvictionPolicy.LFU,
            EvictionPolicy.SIZE,
            EvictionPolicy.HYBRID
        ]
        
        for policy in policies:
            cache = EnhancedCache(
                max_size=10,
                eviction_policy=policy,
                enable_analytics=False
            )
            
            # Fill cache
            for i in range(15):
                await cache.set(f"key_{i}", f"value_{i}")
            
            # Check size limit
            stats = cache.get_stats()
            assert stats["entry_count"] <= 10
            
            await cache.stop_cleanup()


class TestCompressionLevels:
    """Test suite for compression levels"""
    
    @pytest.mark.asyncio
    async def test_all_compression_levels(self):
        """Test all compression levels"""
        levels = [
            CompressionLevel.NONE,
            CompressionLevel.FAST,
            CompressionLevel.DEFAULT,
            CompressionLevel.MAX
        ]
        
        test_data = "x" * 5000  # 5KB of data
        
        for level in levels:
            cache = EnhancedCache(
                compression_level=level,
                enable_analytics=False
            )
            
            await cache.set("test_key", test_data)
            retrieved_value = await cache.get("test_key")
            
            assert retrieved_value == test_data
            
            await cache.stop_cleanup()


if __name__ == "__main__":
    pytest.main([__file__])
