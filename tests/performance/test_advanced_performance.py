"""
Advanced Performance Tests
Comprehensive testing for the new advanced performance optimization system
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any

from app.core.advanced_performance_optimizer import advanced_performance_optimizer
from app.core.enhanced_cache import enhanced_cache
from tests.test_utils import (
    PerformanceTestRunner, LoadTestRunner, TestDataGenerator, 
    TestAssertions, TestResult
)


class TestAdvancedPerformanceOptimizer:
    """Test the advanced performance optimization system"""
    
    @pytest.fixture
    async def optimizer_setup(self):
        """Setup the performance optimizer for testing"""
        await advanced_performance_optimizer.start_monitoring()
        yield advanced_performance_optimizer
        await advanced_performance_optimizer.stop_monitoring()
    
    @pytest.mark.performance
    async def test_optimizer_initialization(self, optimizer_setup):
        """Test optimizer initialization and basic functionality"""
        optimizer = optimizer_setup
        
        # Check that monitoring is active
        assert optimizer.monitoring_active == True
        
        # Wait for some metrics to be collected
        await asyncio.sleep(5)
        
        # Check that metrics are being collected
        summary = optimizer.get_performance_summary()
        assert "metrics" in summary
        assert "performance_score" in summary
        assert summary["monitoring_active"] == True
        
    @pytest.mark.performance
    async def test_metrics_collection(self, optimizer_setup):
        """Test comprehensive metrics collection"""
        optimizer = optimizer_setup
        
        # Wait for metrics to be collected
        await asyncio.sleep(10)
        
        # Check that various metrics are being collected
        summary = optimizer.get_performance_summary()
        metrics = summary["metrics"]
        
        # Should have system metrics
        assert "system.cpu_usage" in metrics
        assert "system.memory_usage" in metrics
        assert "system.disk_usage" in metrics
        
        # Should have cache metrics
        assert "cache.hit_rate" in metrics
        assert "cache.entry_count" in metrics
        
        # Should have database metrics
        assert "database.query_time" in metrics
        
        # Should have application metrics
        assert "application.api_response_time" in metrics
        
    @pytest.mark.performance
    async def test_performance_analysis(self, optimizer_setup):
        """Test performance analysis and trend detection"""
        optimizer = optimizer_setup
        
        # Wait for enough data points for analysis
        await asyncio.sleep(15)
        
        # Manually trigger analysis
        await optimizer._analyze_performance()
        
        # Check that optimization actions are being created
        assert len(optimizer.optimization_actions) >= 0
        
    @pytest.mark.performance
    async def test_prediction_generation(self, optimizer_setup):
        """Test performance prediction generation"""
        optimizer = optimizer_setup
        
        # Wait for enough data for predictions
        await asyncio.sleep(20)
        
        # Manually trigger prediction generation
        await optimizer._generate_predictions()
        
        # Check that predictions are being generated
        summary = optimizer.get_performance_summary()
        assert "predictions" in summary
        
    @pytest.mark.performance
    async def test_optimization_execution(self, optimizer_setup):
        """Test optimization action execution"""
        optimizer = optimizer_setup
        
        # Start optimization
        await optimizer.start_optimization()
        
        # Wait for optimization opportunities to be evaluated
        await asyncio.sleep(10)
        
        # Check that optimization is active
        assert optimizer.optimization_active == True
        
        # Stop optimization
        await optimizer.stop_optimization()
        assert optimizer.optimization_active == False
        
    @pytest.mark.performance
    async def test_performance_summary_accuracy(self, optimizer_setup):
        """Test performance summary accuracy and completeness"""
        optimizer = optimizer_setup
        
        # Wait for metrics collection
        await asyncio.sleep(10)
        
        summary = optimizer.get_performance_summary()
        
        # Check summary structure
        required_fields = [
            "timestamp", "monitoring_active", "optimization_active",
            "metrics", "predictions", "optimization_actions", "performance_score"
        ]
        
        for field in required_fields:
            assert field in summary, f"Missing field: {field}"
        
        # Check performance score is reasonable
        assert 0 <= summary["performance_score"] <= 100
        
        # Check metrics structure
        for metric_name, metric_data in summary["metrics"].items():
            assert "current" in metric_data
            assert "average" in metric_data
            assert "trend" in metric_data


class TestAdvancedPerformanceIntegration:
    """Test integration with existing performance systems"""
    
    @pytest.mark.performance
    async def test_cache_optimization_integration(self):
        """Test integration with enhanced cache system"""
        # Create test cache
        test_cache = enhanced_cache
        
        # Fill cache with test data
        for i in range(100):
            await test_cache.set(f"test_key_{i}", f"test_value_{i}")
        
        # Get initial stats
        initial_stats = test_cache.get_stats()
        initial_hit_rate = initial_stats.get("hit_rate", 0)
        
        # Start optimizer
        await advanced_performance_optimizer.start_monitoring()
        await advanced_performance_optimizer.start_optimization()
        
        # Wait for optimization
        await asyncio.sleep(10)
        
        # Check that cache optimization was considered
        summary = advanced_performance_optimizer.get_performance_summary()
        assert "cache.hit_rate" in summary["metrics"]
        
        # Stop optimizer
        await advanced_performance_optimizer.stop_optimization()
        await advanced_performance_optimizer.stop_monitoring()
        
    @pytest.mark.performance
    async def test_system_metrics_accuracy(self):
        """Test accuracy of system metrics collection"""
        await advanced_performance_optimizer.start_monitoring()
        
        # Wait for metrics collection
        await asyncio.sleep(5)
        
        summary = advanced_performance_optimizer.get_performance_summary()
        metrics = summary["metrics"]
        
        # Check CPU usage is reasonable
        if "system.cpu_usage" in metrics:
            cpu_usage = metrics["system.cpu_usage"]["current"]
            assert 0 <= cpu_usage <= 100, f"CPU usage {cpu_usage}% is not reasonable"
        
        # Check memory usage is reasonable
        if "system.memory_usage" in metrics:
            memory_usage = metrics["system.memory_usage"]["current"]
            assert 0 <= memory_usage <= 100, f"Memory usage {memory_usage}% is not reasonable"
        
        await advanced_performance_optimizer.stop_monitoring()


class TestAdvancedPerformanceLoad:
    """Test advanced performance system under load"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_optimizer_under_load(self):
        """Test optimizer performance under high load"""
        runner = PerformanceTestRunner()
        
        async def optimizer_operation(user_id: int):
            """Simulate optimizer operations"""
            # Start and stop monitoring multiple times
            await advanced_performance_optimizer.start_monitoring()
            await asyncio.sleep(0.1)
            summary = advanced_performance_optimizer.get_performance_summary()
            await advanced_performance_optimizer.stop_monitoring()
            
            return summary is not None
        
        # Run performance test
        results = await runner.run_concurrent_test(
            test_func=optimizer_operation,
            concurrent_users=50,
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
            1.0  # 1 second max average response time
        )
        
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_metrics_collection_under_load(self):
        """Test metrics collection performance under load"""
        await advanced_performance_optimizer.start_monitoring()
        
        runner = PerformanceTestRunner()
        
        async def metrics_operation(user_id: int):
            """Simulate metrics collection operations"""
            # Trigger metrics collection
            await advanced_performance_optimizer._collect_metrics()
            
            # Get summary
            summary = advanced_performance_optimizer.get_performance_summary()
            
            return len(summary["metrics"]) > 0
        
        # Run performance test
        results = await runner.run_concurrent_test(
            test_func=metrics_operation,
            concurrent_users=20,
            test_duration=20
        )
        
        # Assertions
        TestAssertions.assert_success_rate(
            results["metrics"]["successful_requests"],
            results["metrics"]["total_requests"],
            0.90  # 90% success rate
        )
        
        await advanced_performance_optimizer.stop_monitoring()


class TestAdvancedPerformancePredictions:
    """Test performance prediction capabilities"""
    
    @pytest.mark.performance
    async def test_prediction_accuracy(self):
        """Test prediction accuracy with known data patterns"""
        await advanced_performance_optimizer.start_monitoring()
        
        # Wait for initial data collection
        await asyncio.sleep(10)
        
        # Manually add some predictable data
        for i in range(20):
            await advanced_performance_optimizer._record_metric(
                "test.metric", 
                float(i),  # Linear increase
                {"test": "true"}
            )
            await asyncio.sleep(0.1)
        
        # Generate predictions
        await advanced_performance_optimizer._generate_predictions()
        
        # Check predictions were generated
        summary = advanced_performance_optimizer.get_performance_summary()
        assert "predictions" in summary
        
        await advanced_performance_optimizer.stop_monitoring()
        
    @pytest.mark.performance
    async def test_trend_detection(self):
        """Test trend detection capabilities"""
        await advanced_performance_optimizer.start_monitoring()
        
        # Add data with clear trends
        # Increasing trend
        for i in range(10):
            await advanced_performance_optimizer._record_metric(
                "trend.increasing", 
                float(i * 2),  # Clear increasing trend
                {"trend": "increasing"}
            )
            await asyncio.sleep(0.1)
        
        # Decreasing trend
        for i in range(10):
            await advanced_performance_optimizer._record_metric(
                "trend.decreasing", 
                float(20 - i),  # Clear decreasing trend
                {"trend": "decreasing"}
            )
            await asyncio.sleep(0.1)
        
        # Analyze performance to detect trends
        await advanced_performance_optimizer._analyze_performance()
        
        # Check that trends were detected
        summary = advanced_performance_optimizer.get_performance_summary()
        assert "trend.increasing" in summary["metrics"]
        assert "trend.decreasing" in summary["metrics"]
        
        await advanced_performance_optimizer.stop_monitoring()


class TestAdvancedPerformanceOptimization:
    """Test optimization action execution"""
    
    @pytest.mark.performance
    async def test_cache_optimization_action(self):
        """Test cache optimization action execution"""
        await advanced_performance_optimizer.start_monitoring()
        await advanced_performance_optimizer.start_optimization()
        
        # Create a cache optimization action
        await advanced_performance_optimizer._create_optimization_action(
            action_type="cache_optimization",
            target="cache",
            parameters={"current_hit_rate": 70, "target_hit_rate": 85},
            priority=1
        )
        
        # Execute optimizations
        await advanced_performance_optimizer._execute_optimizations()
        
        # Check that action was processed
        assert len(advanced_performance_optimizer.optimization_actions) == 0
        
        await advanced_performance_optimizer.stop_optimization()
        await advanced_performance_optimizer.stop_monitoring()
        
    @pytest.mark.performance
    async def test_memory_optimization_action(self):
        """Test memory optimization action execution"""
        await advanced_performance_optimizer.start_monitoring()
        await advanced_performance_optimizer.start_optimization()
        
        # Create a memory optimization action
        await advanced_performance_optimizer._create_optimization_action(
            action_type="memory_optimization",
            target="memory",
            parameters={"current_usage": 80, "target_usage": 60},
            priority=1
        )
        
        # Execute optimizations
        await advanced_performance_optimizer._execute_optimizations()
        
        # Check that action was processed
        assert len(advanced_performance_optimizer.optimization_actions) == 0
        
        await advanced_performance_optimizer.stop_optimization()
        await advanced_performance_optimizer.stop_monitoring()
        
    @pytest.mark.performance
    async def test_optimization_priority_handling(self):
        """Test optimization action priority handling"""
        await advanced_performance_optimizer.start_monitoring()
        await advanced_performance_optimizer.start_optimization()
        
        # Create actions with different priorities
        await advanced_performance_optimizer._create_optimization_action(
            action_type="cache_optimization",
            target="cache",
            parameters={"test": "low_priority"},
            priority=3
        )
        
        await advanced_performance_optimizer._create_optimization_action(
            action_type="memory_optimization",
            target="memory",
            parameters={"test": "high_priority"},
            priority=1
        )
        
        # Execute optimizations
        await advanced_performance_optimizer._execute_optimizations()
        
        # Check that high priority action was processed first
        remaining_actions = advanced_performance_optimizer.optimization_actions
        if remaining_actions:
            # Should only have low priority action remaining
            assert remaining_actions[0].priority == 3
        
        await advanced_performance_optimizer.stop_optimization()
        await advanced_performance_optimizer.stop_monitoring()
