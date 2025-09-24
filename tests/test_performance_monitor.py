"""
Performance Monitoring Test Suite
Tests the enhanced performance monitoring system
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from app.core.performance_monitor import performance_monitor, PerformanceMetric, PerformanceAlert, PerformanceRecommendation
from app.core.enhanced_cache import enhanced_cache, CacheEntry


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Clear any existing data
        performance_monitor.metrics_history.clear()
        performance_monitor.alerts.clear()
        performance_monitor.recommendations.clear()
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        assert performance_monitor.max_size == 1000
        assert performance_monitor.default_ttl == 3600
        assert not performance_monitor.monitoring_active
        assert performance_monitor.monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping performance monitoring"""
        # Start monitoring
        await performance_monitor.start_monitoring(interval=1)
        assert performance_monitor.monitoring_active
        assert performance_monitor.monitoring_task is not None
        
        # Wait a bit for monitoring to collect some data
        await asyncio.sleep(2)
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active
        assert performance_monitor.monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_record_metric(self):
        """Test recording performance metrics"""
        await performance_monitor._record_metric("test_metric", 75.5, "percent")
        
        assert "test_metric" in performance_monitor.metrics_history
        assert len(performance_monitor.metrics_history["test_metric"]) == 1
        
        metric = performance_monitor.metrics_history["test_metric"][0]
        assert metric.name == "test_metric"
        assert metric.value == 75.5
        assert metric.unit == "percent"
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self):
        """Test system metrics collection"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net:
            
            # Mock memory object
            mock_memory.return_value.percent = 60.0
            mock_memory.return_value.available = 2 * 1024**3  # 2GB
            
            # Mock disk object
            mock_disk.return_value.used = 50 * 1024**3  # 50GB
            mock_disk.return_value.total = 100 * 1024**3  # 100GB
            mock_disk.return_value.free = 50 * 1024**3  # 50GB
            
            # Mock network object
            mock_net.return_value.bytes_sent = 1024**2  # 1MB
            mock_net.return_value.bytes_recv = 2 * 1024**2  # 2MB
            
            await performance_monitor._collect_system_metrics()
            
            # Check that metrics were recorded
            assert "cpu_usage" in performance_monitor.metrics_history
            assert "memory_usage" in performance_monitor.metrics_history
            assert "disk_usage" in performance_monitor.metrics_history
    
    @pytest.mark.asyncio
    async def test_threshold_checking(self):
        """Test threshold checking and alert generation"""
        # Record a high CPU usage metric
        await performance_monitor._record_metric("cpu_usage", 95.0, "percent")
        await performance_monitor._record_metric("cpu_usage", 96.0, "percent")
        await performance_monitor._record_metric("cpu_usage", 97.0, "percent")
        
        # Check thresholds
        await performance_monitor._check_thresholds()
        
        # Should generate an alert for high CPU usage
        assert len(performance_monitor.alerts) > 0
        alert = performance_monitor.alerts[0]
        assert alert.metric_name == "cpu_usage"
        assert alert.current_value > 90.0
        assert alert.severity in ["high", "critical"]
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self):
        """Test performance recommendation generation"""
        # Record high CPU usage
        for i in range(10):
            await performance_monitor._record_metric("cpu_usage", 85.0, "percent")
        
        await performance_monitor._generate_recommendations()
        
        # Should generate a recommendation for high CPU usage
        assert len(performance_monitor.recommendations) > 0
        recommendation = performance_monitor.recommendations[0]
        assert "CPU" in recommendation.title
        assert recommendation.priority == "high"
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        # Add some test metrics
        performance_monitor.metrics_history["test_metric"] = [
            PerformanceMetric("test_metric", 50.0, "percent", datetime.now()),
            PerformanceMetric("test_metric", 60.0, "percent", datetime.now()),
            PerformanceMetric("test_metric", 70.0, "percent", datetime.now())
        ]
        
        summary = performance_monitor.get_metrics_summary()
        
        assert "test_metric" in summary
        assert summary["test_metric"]["current"] == 70.0
        assert summary["test_metric"]["average"] == 60.0
        assert summary["test_metric"]["min"] == 50.0
        assert summary["test_metric"]["max"] == 70.0
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        # Add a test alert
        alert = PerformanceAlert(
            metric_name="test_metric",
            threshold=80.0,
            current_value=90.0,
            severity="high",
            message="Test alert",
            timestamp=datetime.now()
        )
        performance_monitor.alerts.append(alert)
        
        active_alerts = performance_monitor.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0]["metric_name"] == "test_metric"
        assert active_alerts[0]["severity"] == "high"
    
    def test_get_recommendations(self):
        """Test getting recommendations"""
        # Add a test recommendation
        recommendation = PerformanceRecommendation(
            category="system",
            priority="high",
            title="Test Recommendation",
            description="Test description",
            impact="High impact",
            effort="medium",
            implementation="Test implementation",
            created_at=datetime.now()
        )
        performance_monitor.recommendations.append(recommendation)
        
        recommendations = performance_monitor.get_recommendations("high")
        
        assert len(recommendations) == 1
        assert recommendations[0]["title"] == "Test Recommendation"
        assert recommendations[0]["priority"] == "high"
    
    @pytest.mark.asyncio
    async def test_measure_execution_time(self):
        """Test execution time measurement context manager"""
        async with performance_monitor.measure_execution_time("test_operation"):
            await asyncio.sleep(0.1)  # Simulate some work
        
        # Check that execution time was recorded
        assert "execution_time_test_operation" in performance_monitor.metrics_history
        metrics = performance_monitor.metrics_history["execution_time_test_operation"]
        assert len(metrics) == 1
        assert metrics[0].value >= 100  # Should be at least 100ms


class TestEnhancedCache:
    """Test enhanced caching functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        enhanced_cache.cache.clear()
        enhanced_cache.stats.hits = 0
        enhanced_cache.stats.misses = 0
        enhanced_cache.stats.evictions = 0
        enhanced_cache.stats.total_size = 0
        enhanced_cache.stats.entry_count = 0
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self):
        """Test basic cache set and get operations"""
        # Set a value
        success = await enhanced_cache.set("test_key", "test_value", ttl=60)
        assert success
        
        # Get the value
        value = await enhanced_cache.get("test_key")
        assert value == "test_value"
        
        # Check stats
        assert enhanced_cache.stats.hits == 1
        assert enhanced_cache.stats.misses == 0
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss behavior"""
        # Try to get non-existent key
        value = await enhanced_cache.get("non_existent_key")
        assert value is None
        
        # Check stats
        assert enhanced_cache.stats.hits == 0
        assert enhanced_cache.stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration"""
        # Set a value with short TTL
        await enhanced_cache.set("expiring_key", "expiring_value", ttl=1)
        
        # Value should be available immediately
        value = await enhanced_cache.get("expiring_key")
        assert value == "expiring_value"
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Value should be expired
        value = await enhanced_cache.get("expiring_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test cache deletion"""
        # Set a value
        await enhanced_cache.set("delete_key", "delete_value")
        
        # Delete the value
        success = await enhanced_cache.delete("delete_key")
        assert success
        
        # Value should be gone
        value = await enhanced_cache.get("delete_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete_by_tags(self):
        """Test cache deletion by tags"""
        # Set values with tags
        await enhanced_cache.set("tagged_key1", "value1", tags=["test", "tag1"])
        await enhanced_cache.set("tagged_key2", "value2", tags=["test", "tag2"])
        await enhanced_cache.set("untagged_key", "value3")
        
        # Delete by tag
        deleted_count = await enhanced_cache.delete_by_tags(["test"])
        assert deleted_count == 2
        
        # Check that tagged values are gone but untagged remains
        assert await enhanced_cache.get("tagged_key1") is None
        assert await enhanced_cache.get("tagged_key2") is None
        assert await enhanced_cache.get("untagged_key") == "value3"
    
    def test_cache_stats(self):
        """Test cache statistics"""
        stats = enhanced_cache.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "evictions" in stats
        assert "hit_rate" in stats
        assert "total_size" in stats
        assert "entry_count" in stats
        assert "max_size" in stats
        assert "utilization" in stats
    
    def test_cache_decorator(self):
        """Test cache decorator functionality"""
        call_count = 0
        
        @enhanced_cache.cache_decorator(ttl=60, key_prefix="test")
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = test_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = test_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        # Set max size to 2 for testing
        enhanced_cache.max_size = 2
        
        # Add 3 items
        await enhanced_cache.set("key1", "value1")
        await enhanced_cache.set("key2", "value2")
        await enhanced_cache.set("key3", "value3")
        
        # key1 should be evicted (least recently used)
        assert await enhanced_cache.get("key1") is None
        assert await enhanced_cache.get("key2") == "value2"
        assert await enhanced_cache.get("key3") == "value3"
        
        # Check eviction stats
        assert enhanced_cache.stats.evictions > 0


class TestPerformanceIntegration:
    """Integration tests for performance monitoring"""
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test integration between performance monitor and cache"""
        # Start monitoring
        await performance_monitor.start_monitoring(interval=1)
        
        # Use cache to generate some metrics
        await enhanced_cache.set("integration_test", "test_value")
        await enhanced_cache.get("integration_test")
        await enhanced_cache.get("non_existent")  # Cache miss
        
        # Wait for monitoring to collect data
        await asyncio.sleep(2)
        
        # Get metrics summary
        summary = performance_monitor.get_metrics_summary()
        
        # Should have cache statistics
        assert "cache" in summary
        cache_stats = summary["cache"]
        assert "hits" in cache_stats
        assert "misses" in cache_stats
        assert "hit_rate" in cache_stats
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_performance_alerting_integration(self):
        """Test performance alerting integration"""
        # Record high CPU usage to trigger alerts
        for i in range(5):
            await performance_monitor._record_metric("cpu_usage", 95.0, "percent")
        
        # Check thresholds
        await performance_monitor._check_thresholds()
        
        # Should have alerts
        alerts = performance_monitor.get_active_alerts()
        assert len(alerts) > 0
        
        # Should have recommendations
        recommendations = performance_monitor.get_recommendations()
        assert len(recommendations) > 0
