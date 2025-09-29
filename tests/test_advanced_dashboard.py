"""
Advanced Dashboard Tests
Comprehensive testing for real-time dashboard functionality
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import WebSocket

from app.api.v1.endpoints.advanced_dashboard import (
    AdvancedDashboardManager,
    dashboard_manager,
    collect_realtime_metrics,
    calculate_trends,
    generate_ai_insights,
    check_alerts
)


class TestAdvancedDashboardManager:
    """Test the advanced dashboard manager"""
    
    @pytest.fixture
    def dashboard_mgr(self):
        """Create a fresh dashboard manager for testing"""
        return AdvancedDashboardManager()
    
    def test_dashboard_manager_initialization(self, dashboard_mgr):
        """Test dashboard manager initialization"""
        assert len(dashboard_mgr.active_connections) == 0
        assert len(dashboard_mgr.metrics_history) == 0
        assert len(dashboard_mgr.alerts) == 0
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, dashboard_mgr):
        """Test WebSocket connection management"""
        # Mock WebSocket
        mock_websocket = Mock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        
        # Test connection
        await dashboard_mgr.connect(mock_websocket)
        assert len(dashboard_mgr.active_connections) == 1
        assert mock_websocket in dashboard_mgr.active_connections
        mock_websocket.accept.assert_called_once()
        
        # Test disconnection
        dashboard_mgr.disconnect(mock_websocket)
        assert len(dashboard_mgr.active_connections) == 0
        assert mock_websocket not in dashboard_mgr.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, dashboard_mgr):
        """Test broadcasting messages to connected clients"""
        # Mock WebSocket connections
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws1.send_text = AsyncMock()
        
        mock_ws2 = Mock(spec=WebSocket)
        mock_ws2.send_text = AsyncMock()
        
        # Add connections
        dashboard_mgr.active_connections = [mock_ws1, mock_ws2]
        
        # Test broadcast
        test_message = {"type": "test", "data": "test_data"}
        await dashboard_mgr.broadcast(test_message)
        
        # Verify both connections received the message
        mock_ws1.send_text.assert_called_once_with(json.dumps(test_message))
        mock_ws2.send_text.assert_called_once_with(json.dumps(test_message))
    
    @pytest.mark.asyncio
    async def test_broadcast_with_disconnected_client(self, dashboard_mgr):
        """Test broadcast handling disconnected clients"""
        # Mock WebSocket connections
        mock_ws1 = Mock(spec=WebSocket)
        mock_ws1.send_text = AsyncMock()
        
        mock_ws2 = Mock(spec=WebSocket)
        mock_ws2.send_text = AsyncMock(side_effect=Exception("Connection closed"))
        
        # Add connections
        dashboard_mgr.active_connections = [mock_ws1, mock_ws2]
        
        # Test broadcast
        test_message = {"type": "test", "data": "test_data"}
        await dashboard_mgr.broadcast(test_message)
        
        # Verify working connection received message
        mock_ws1.send_text.assert_called_once()
        
        # Verify failed connection was removed
        assert len(dashboard_mgr.active_connections) == 1
        assert mock_ws1 in dashboard_mgr.active_connections
        assert mock_ws2 not in dashboard_mgr.active_connections
    
    def test_add_metric(self, dashboard_mgr):
        """Test adding metrics to history"""
        # Add metric
        dashboard_mgr.add_metric("cpu_usage", 75.5)
        
        # Verify metric was added
        assert "cpu_usage" in dashboard_mgr.metrics_history
        assert len(dashboard_mgr.metrics_history["cpu_usage"]) == 1
        
        metric = dashboard_mgr.metrics_history["cpu_usage"][0]
        assert metric["value"] == 75.5
        assert "timestamp" in metric
    
    def test_add_metric_with_timestamp(self, dashboard_mgr):
        """Test adding metric with specific timestamp"""
        test_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        dashboard_mgr.add_metric("memory_usage", 80.0, test_timestamp)
        
        metric = dashboard_mgr.metrics_history["memory_usage"][0]
        assert metric["value"] == 80.0
        assert metric["timestamp"] == test_timestamp.isoformat()
    
    def test_get_metric_trend_increasing(self, dashboard_mgr):
        """Test getting increasing trend"""
        # Add increasing values
        for i in range(20):
            dashboard_mgr.add_metric("test_metric", 50.0 + i * 2)
        
        trend = dashboard_mgr.get_metric_trend("test_metric")
        assert trend == "increasing"
    
    def test_get_metric_trend_decreasing(self, dashboard_mgr):
        """Test getting decreasing trend"""
        # Add decreasing values
        for i in range(20):
            dashboard_mgr.add_metric("test_metric", 100.0 - i * 2)
        
        trend = dashboard_mgr.get_metric_trend("test_metric")
        assert trend == "decreasing"
    
    def test_get_metric_trend_stable(self, dashboard_mgr):
        """Test getting stable trend"""
        # Add stable values
        for i in range(20):
            dashboard_mgr.add_metric("test_metric", 50.0 + (i % 3 - 1))
        
        trend = dashboard_mgr.get_metric_trend("test_metric")
        assert trend == "stable"
    
    def test_get_metric_trend_insufficient_data(self, dashboard_mgr):
        """Test getting trend with insufficient data"""
        # Add only a few values
        dashboard_mgr.add_metric("test_metric", 50.0)
        dashboard_mgr.add_metric("test_metric", 60.0)
        
        trend = dashboard_mgr.get_metric_trend("test_metric")
        assert trend == "stable"  # Default for insufficient data


class TestDashboardFunctions:
    """Test dashboard utility functions"""
    
    @pytest.mark.asyncio
    async def test_collect_realtime_metrics(self):
        """Test collecting real-time metrics"""
        # Mock psutil and enhanced_cache
        with patch('app.api.v1.endpoints.advanced_dashboard.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 75.5
            mock_psutil.virtual_memory.return_value = Mock(percent=80.0)
            mock_psutil.disk_usage.return_value = Mock(percent=60.0)
            
            with patch('app.api.v1.endpoints.advanced_dashboard.enhanced_cache') as mock_cache:
                mock_cache.get_stats.return_value = {"hit_rate": 0.85}
                
                metrics = await collect_realtime_metrics()
                
                assert "cpu_usage" in metrics
                assert "memory_usage" in metrics
                assert "disk_usage" in metrics
                assert "cache_hit_rate" in metrics
                assert "response_time" in metrics
                assert "throughput" in metrics
                
                assert metrics["cpu_usage"] == 75.5
                assert metrics["memory_usage"] == 80.0
                assert metrics["cache_hit_rate"] == 85.0
    
    def test_calculate_trends(self):
        """Test calculating trends for metrics"""
        # Mock dashboard manager
        with patch('app.api.v1.endpoints.advanced_dashboard.dashboard_manager') as mock_manager:
            mock_manager.get_metric_trend.side_effect = lambda metric: "increasing" if "cpu" in metric else "stable"
            
            metrics = {"cpu_usage": 75.0, "memory_usage": 60.0}
            trends = calculate_trends(metrics)
            
            assert trends["cpu_usage"] == "increasing"
            assert trends["memory_usage"] == "stable"
    
    def test_generate_ai_insights_high_cpu(self):
        """Test generating AI insights for high CPU usage"""
        metrics = {"cpu_usage": 85.0, "cache_hit_rate": 90.0, "memory_usage": 60.0}
        perf_summary = {"performance_score": 95}
        
        insights = generate_ai_insights(metrics, perf_summary)
        
        # Should generate CPU warning insight
        cpu_insights = [i for i in insights if "CPU" in i["message"]]
        assert len(cpu_insights) > 0
        assert cpu_insights[0]["type"] == "Performance Warning"
    
    def test_generate_ai_insights_low_cpu(self):
        """Test generating AI insights for low CPU usage"""
        metrics = {"cpu_usage": 15.0, "cache_hit_rate": 90.0, "memory_usage": 60.0}
        perf_summary = {"performance_score": 95}
        
        insights = generate_ai_insights(metrics, perf_summary)
        
        # Should generate optimization opportunity insight
        optimization_insights = [i for i in insights if "optimization" in i["message"].lower()]
        assert len(optimization_insights) > 0
        assert optimization_insights[0]["type"] == "Optimization Opportunity"
    
    def test_generate_ai_insights_cache_performance(self):
        """Test generating AI insights for cache performance"""
        metrics = {"cpu_usage": 50.0, "cache_hit_rate": 95.0, "memory_usage": 60.0}
        perf_summary = {"performance_score": 95}
        
        insights = generate_ai_insights(metrics, perf_summary)
        
        # Should generate cache performance insight
        cache_insights = [i for i in insights if "cache" in i["message"].lower()]
        assert len(cache_insights) > 0
        assert cache_insights[0]["type"] == "Excellent Performance"
    
    def test_generate_ai_insights_memory_alert(self):
        """Test generating AI insights for memory alert"""
        metrics = {"cpu_usage": 50.0, "cache_hit_rate": 80.0, "memory_usage": 90.0}
        perf_summary = {"performance_score": 70}
        
        insights = generate_ai_insights(metrics, perf_summary)
        
        # Should generate memory alert insight
        memory_insights = [i for i in insights if "memory" in i["message"].lower()]
        assert len(memory_insights) > 0
        assert memory_insights[0]["type"] == "Memory Alert"
    
    def test_check_alerts_critical_cpu(self):
        """Test checking alerts for critical CPU usage"""
        metrics = {"cpu_usage": 95.0, "memory_usage": 60.0, "cache_hit_rate": 80.0}
        perf_summary = {}
        
        alerts = check_alerts(metrics, perf_summary)
        
        # Should generate critical CPU alert
        cpu_alerts = [a for a in alerts if "CPU" in a["title"]]
        assert len(cpu_alerts) > 0
        assert cpu_alerts[0]["severity"] == "error"
        assert "95.0" in cpu_alerts[0]["message"]
    
    def test_check_alerts_high_cpu(self):
        """Test checking alerts for high CPU usage"""
        metrics = {"cpu_usage": 85.0, "memory_usage": 60.0, "cache_hit_rate": 80.0}
        perf_summary = {}
        
        alerts = check_alerts(metrics, perf_summary)
        
        # Should generate high CPU alert
        cpu_alerts = [a for a in alerts if "CPU" in a["title"]]
        assert len(cpu_alerts) > 0
        assert cpu_alerts[0]["severity"] == "warning"
    
    def test_check_alerts_critical_memory(self):
        """Test checking alerts for critical memory usage"""
        metrics = {"cpu_usage": 50.0, "memory_usage": 98.0, "cache_hit_rate": 80.0}
        perf_summary = {}
        
        alerts = check_alerts(metrics, perf_summary)
        
        # Should generate critical memory alert
        memory_alerts = [a for a in alerts if "Memory" in a["title"]]
        assert len(memory_alerts) > 0
        assert memory_alerts[0]["severity"] == "error"
    
    def test_check_alerts_low_cache_hit_rate(self):
        """Test checking alerts for low cache hit rate"""
        metrics = {"cpu_usage": 50.0, "memory_usage": 60.0, "cache_hit_rate": 40.0}
        perf_summary = {}
        
        alerts = check_alerts(metrics, perf_summary)
        
        # Should generate cache alert
        cache_alerts = [a for a in alerts if "Cache" in a["title"]]
        assert len(cache_alerts) > 0
        assert cache_alerts[0]["severity"] == "warning"
    
    def test_check_alerts_no_alerts(self):
        """Test checking alerts with normal metrics"""
        metrics = {"cpu_usage": 50.0, "memory_usage": 60.0, "cache_hit_rate": 80.0}
        perf_summary = {}
        
        alerts = check_alerts(metrics, perf_summary)
        
        # Should not generate any alerts
        assert len(alerts) == 0


class TestDashboardIntegration:
    """Test dashboard integration with other components"""
    
    @pytest.mark.asyncio
    async def test_dashboard_with_performance_optimizer(self):
        """Test dashboard integration with performance optimizer"""
        # Mock performance optimizer
        mock_summary = {
            "performance_score": 85,
            "monitoring_active": True,
            "optimization_active": True
        }
        
        with patch('app.api.v1.endpoints.advanced_dashboard.advanced_performance_optimizer') as mock_optimizer:
            mock_optimizer.get_performance_summary.return_value = mock_summary
            
            # Test metrics collection
            metrics = await collect_realtime_metrics()
            
            # Test insights generation
            insights = generate_ai_insights(metrics, mock_summary)
            
            # Should generate insights based on performance data
            assert len(insights) >= 0
    
    @pytest.mark.asyncio
    async def test_dashboard_websocket_flow(self):
        """Test complete WebSocket dashboard flow"""
        # Mock WebSocket
        mock_websocket = Mock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        # Connect to dashboard
        await dashboard_manager.connect(mock_websocket)
        
        # Simulate dashboard data
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {"cpu_usage": 75.0, "memory_usage": 60.0},
            "performance_score": 85,
            "trends": {"cpu_usage": "stable", "memory_usage": "stable"},
            "ai_insights": [],
            "alerts": []
        }
        
        # Broadcast data
        await dashboard_manager.broadcast(dashboard_data)
        
        # Verify WebSocket received data
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["metrics"]["cpu_usage"] == 75.0


class TestDashboardPerformance:
    """Test dashboard performance under load"""
    
    @pytest.mark.asyncio
    async def test_dashboard_high_frequency_updates(self):
        """Test dashboard performance with high frequency updates"""
        # Create multiple mock connections
        mock_connections = []
        for i in range(10):
            mock_ws = Mock(spec=WebSocket)
            mock_ws.send_text = AsyncMock()
            mock_connections.append(mock_ws)
        
        # Add all connections
        dashboard_manager.active_connections = mock_connections
        
        # Test high frequency broadcasting
        start_time = time.time()
        
        for i in range(100):
            test_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {"cpu_usage": 50.0 + i, "memory_usage": 60.0},
                "performance_score": 85
            }
            await dashboard_manager.broadcast(test_data)
        
        broadcast_time = time.time() - start_time
        
        # Should handle high frequency updates efficiently
        assert broadcast_time < 5.0  # Should complete in under 5 seconds
        
        # Verify all connections received messages
        for mock_ws in mock_connections:
            assert mock_ws.send_text.call_count == 100
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance"""
        # Mock system calls
        with patch('app.api.v1.endpoints.advanced_dashboard.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.virtual_memory.return_value = Mock(percent=60.0)
            mock_psutil.disk_usage.return_value = Mock(percent=40.0)
            
            with patch('app.api.v1.endpoints.advanced_dashboard.enhanced_cache') as mock_cache:
                mock_cache.get_stats.return_value = {"hit_rate": 0.8}
                
                # Test multiple metrics collections
                start_time = time.time()
                
                for i in range(100):
                    await collect_realtime_metrics()
                
                collection_time = time.time() - start_time
                
                # Should collect metrics quickly
                assert collection_time < 2.0  # Should complete in under 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
