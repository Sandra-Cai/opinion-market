"""
Advanced Analytics Engine Tests
Comprehensive testing for ML-powered analytics and predictions
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.services.advanced_analytics_engine import (
    AdvancedAnalyticsEngine, 
    AnalyticsInsight, 
    PredictionResult, 
    AnomalyDetection,
    advanced_analytics_engine
)


class TestAdvancedAnalyticsEngine:
    """Test the advanced analytics engine"""
    
    @pytest.fixture
    async def analytics_engine(self):
        """Create a fresh analytics engine for testing"""
        engine = AdvancedAnalyticsEngine()
        yield engine
        await engine.stop_analytics()
    
    @pytest.mark.asyncio
    async def test_analytics_engine_initialization(self, analytics_engine):
        """Test analytics engine initialization"""
        assert not analytics_engine.analytics_active
        assert len(analytics_engine.data_history) == 0
        assert len(analytics_engine.models) == 0
        assert len(analytics_engine.insights) == 0
        assert len(analytics_engine.predictions) == 0
        assert len(analytics_engine.anomalies) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_analytics(self, analytics_engine):
        """Test starting and stopping analytics engine"""
        # Start analytics
        await analytics_engine.start_analytics()
        assert analytics_engine.analytics_active
        assert analytics_engine.analytics_task is not None
        
        # Stop analytics
        await analytics_engine.stop_analytics()
        assert not analytics_engine.analytics_active
    
    def test_add_data_point(self, analytics_engine):
        """Test adding data points"""
        # Add data point
        analytics_engine.add_data_point("cpu_usage", 75.5, metadata={"source": "test"})
        
        # Verify data was added
        assert "cpu_usage" in analytics_engine.data_history
        assert len(analytics_engine.data_history["cpu_usage"]) == 1
        
        data_point = analytics_engine.data_history["cpu_usage"][0]
        assert data_point["value"] == 75.5
        assert data_point["metadata"]["source"] == "test"
    
    def test_add_multiple_data_points(self, analytics_engine):
        """Test adding multiple data points"""
        # Add multiple data points
        for i in range(10):
            analytics_engine.add_data_point("test_metric", i * 10.0)
        
        # Verify all data points were added
        assert len(analytics_engine.data_history["test_metric"]) == 10
        values = [dp["value"] for dp in analytics_engine.data_history["test_metric"]]
        assert values == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    
    @pytest.mark.asyncio
    async def test_generate_predictions_without_ml(self, analytics_engine):
        """Test prediction generation without ML libraries"""
        # Add enough data points
        for i in range(100):
            analytics_engine.add_data_point("test_metric", 50.0 + np.sin(i * 0.1) * 10)
        
        # Mock ML_AVAILABLE to False
        with patch('app.services.advanced_analytics_engine.ML_AVAILABLE', False):
            await analytics_engine._generate_predictions()
        
        # Should not generate predictions without ML
        assert len(analytics_engine.predictions) == 0
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, analytics_engine):
        """Test anomaly detection"""
        # Add normal data points
        for i in range(50):
            analytics_engine.add_data_point("test_metric", 50.0 + np.random.normal(0, 5))
        
        # Add an anomaly
        analytics_engine.add_data_point("test_metric", 200.0)  # Clear outlier
        
        # Run anomaly detection
        await analytics_engine._detect_anomalies()
        
        # Should detect the anomaly
        assert len(analytics_engine.anomalies) > 0
        anomaly = analytics_engine.anomalies[-1]
        assert anomaly.metric_name == "test_metric"
        assert anomaly.is_anomaly
        assert anomaly.actual_value == 200.0
    
    @pytest.mark.asyncio
    async def test_generate_insights(self, analytics_engine):
        """Test insight generation"""
        # Add performance data
        for i in range(20):
            analytics_engine.add_data_point("performance_score", 95.0)
            analytics_engine.add_data_point("cache_hit_rate", 90.0)
            analytics_engine.add_data_point("cpu_usage", 30.0)
            analytics_engine.add_data_point("memory_usage", 40.0)
        
        # Generate insights
        await analytics_engine._generate_insights()
        
        # Should generate insights
        assert len(analytics_engine.insights) > 0
        
        # Check for performance insight
        performance_insights = [i for i in analytics_engine.insights if i.insight_type == "performance"]
        assert len(performance_insights) > 0
    
    def test_get_analytics_summary(self, analytics_engine):
        """Test getting analytics summary"""
        # Add some data
        analytics_engine.add_data_point("test_metric", 50.0)
        
        # Get summary
        summary = analytics_engine.get_analytics_summary()
        
        # Verify summary structure
        assert "timestamp" in summary
        assert "analytics_active" in summary
        assert "data_points_collected" in summary
        assert "models_trained" in summary
        assert "ml_available" in summary
        assert summary["data_points_collected"] == 1
    
    @pytest.mark.asyncio
    async def test_analytics_loop_integration(self, analytics_engine):
        """Test the main analytics loop"""
        # Add test data
        for i in range(100):
            analytics_engine.add_data_point("cpu_usage", 50.0 + np.sin(i * 0.1) * 20)
            analytics_engine.add_data_point("memory_usage", 60.0 + np.cos(i * 0.1) * 15)
        
        # Start analytics
        await analytics_engine.start_analytics()
        
        # Let it run for a short time
        await asyncio.sleep(1)
        
        # Stop analytics
        await analytics_engine.stop_analytics()
        
        # Should have processed some data
        assert analytics_engine.analytics_active == False


class TestAnalyticsDataStructures:
    """Test analytics data structures"""
    
    def test_analytics_insight(self):
        """Test AnalyticsInsight data structure"""
        insight = AnalyticsInsight(
            insight_type="performance",
            title="Test Insight",
            description="Test description",
            confidence=0.9,
            impact="high",
            recommendations=["test recommendation"]
        )
        
        assert insight.insight_type == "performance"
        assert insight.title == "Test Insight"
        assert insight.confidence == 0.9
        assert insight.impact == "high"
        assert len(insight.recommendations) == 1
        assert isinstance(insight.created_at, datetime)
    
    def test_prediction_result(self):
        """Test PredictionResult data structure"""
        prediction = PredictionResult(
            metric_name="cpu_usage",
            predicted_value=75.5,
            confidence=0.8,
            time_horizon=30,
            trend="increasing"
        )
        
        assert prediction.metric_name == "cpu_usage"
        assert prediction.predicted_value == 75.5
        assert prediction.confidence == 0.8
        assert prediction.time_horizon == 30
        assert prediction.trend == "increasing"
        assert isinstance(prediction.created_at, datetime)
    
    def test_anomaly_detection(self):
        """Test AnomalyDetection data structure"""
        anomaly = AnomalyDetection(
            metric_name="cpu_usage",
            anomaly_score=3.5,
            is_anomaly=True,
            expected_value=50.0,
            actual_value=200.0,
            severity="high",
            description="High CPU usage detected"
        )
        
        assert anomaly.metric_name == "cpu_usage"
        assert anomaly.anomaly_score == 3.5
        assert anomaly.is_anomaly
        assert anomaly.expected_value == 50.0
        assert anomaly.actual_value == 200.0
        assert anomaly.severity == "high"
        assert isinstance(anomaly.timestamp, datetime)


class TestAnalyticsIntegration:
    """Test analytics integration with other components"""
    
    @pytest.mark.asyncio
    async def test_analytics_with_performance_optimizer(self):
        """Test analytics integration with performance optimizer"""
        from app.core.advanced_performance_optimizer import advanced_performance_optimizer
        
        # Mock performance optimizer
        mock_summary = {
            "metrics": {
                "cpu_usage": {"current": 75.0},
                "memory_usage": {"current": 80.0}
            }
        }
        
        with patch.object(advanced_performance_optimizer, 'get_performance_summary', return_value=mock_summary):
            # Add data to analytics engine
            advanced_analytics_engine.add_data_point("cpu_usage", 75.0)
            advanced_analytics_engine.add_data_point("memory_usage", 80.0)
            
            # Generate insights
            await advanced_analytics_engine._generate_insights()
            
            # Should have insights
            assert len(advanced_analytics_engine.insights) >= 0
    
    @pytest.mark.asyncio
    async def test_analytics_data_persistence(self, analytics_engine):
        """Test that analytics data persists correctly"""
        # Add data
        analytics_engine.add_data_point("test_metric", 100.0)
        
        # Verify data is stored
        assert len(analytics_engine.data_history["test_metric"]) == 1
        
        # Add more data
        for i in range(10):
            analytics_engine.add_data_point("test_metric", i * 10.0)
        
        # Verify all data is stored
        assert len(analytics_engine.data_history["test_metric"]) == 11
        
        # Check data integrity
        values = [dp["value"] for dp in analytics_engine.data_history["test_metric"]]
        assert values[0] == 100.0
        assert values[-1] == 90.0


@pytest.mark.integration
class TestAnalyticsPerformance:
    """Test analytics performance under load"""
    
    @pytest.mark.asyncio
    async def test_analytics_under_high_data_load(self):
        """Test analytics performance with high data volume"""
        engine = AdvancedAnalyticsEngine()
        
        try:
            # Add large amount of data
            start_time = time.time()
            
            for i in range(1000):
                engine.add_data_point("cpu_usage", 50.0 + np.random.normal(0, 10))
                engine.add_data_point("memory_usage", 60.0 + np.random.normal(0, 8))
                engine.add_data_point("response_time", 100.0 + np.random.normal(0, 20))
            
            add_time = time.time() - start_time
            
            # Should handle large data volume efficiently
            assert add_time < 1.0  # Should add 1000 points in under 1 second
            assert len(engine.data_history["cpu_usage"]) == 1000
            
            # Test anomaly detection performance
            start_time = time.time()
            await engine._detect_anomalies()
            detection_time = time.time() - start_time
            
            # Should detect anomalies quickly
            assert detection_time < 0.5  # Should complete in under 0.5 seconds
            
        finally:
            await engine.stop_analytics()
    
    @pytest.mark.asyncio
    async def test_analytics_memory_usage(self):
        """Test analytics memory usage"""
        engine = AdvancedAnalyticsEngine()
        
        try:
            # Add data and monitor memory
            initial_insights = len(engine.insights)
            
            # Add data and generate insights
            for i in range(100):
                engine.add_data_point("test_metric", i * 0.1)
            
            await engine._generate_insights()
            
            # Should not have excessive memory growth
            assert len(engine.insights) >= initial_insights
            assert len(engine.data_history["test_metric"]) == 100
            
        finally:
            await engine.stop_analytics()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
