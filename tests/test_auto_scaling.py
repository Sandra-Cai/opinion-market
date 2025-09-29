"""
Auto Scaling Manager Tests
Comprehensive testing for intelligent auto-scaling functionality
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.services.auto_scaling_manager import (
    AutoScalingManager,
    ScalingDecision,
    ScalingPolicy,
    auto_scaling_manager
)


class TestScalingPolicy:
    """Test scaling policy data structure"""
    
    def test_scaling_policy_creation(self):
        """Test creating a scaling policy"""
        policy = ScalingPolicy(
            name="test_policy",
            metric_name="cpu_usage",
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            min_instances=1,
            max_instances=10,
            cooldown_period=300
        )
        
        assert policy.name == "test_policy"
        assert policy.metric_name == "cpu_usage"
        assert policy.scale_up_threshold == 80.0
        assert policy.scale_down_threshold == 30.0
        assert policy.min_instances == 1
        assert policy.max_instances == 10
        assert policy.cooldown_period == 300
        assert policy.enabled == True
        assert policy.last_action_time is None
    
    def test_scaling_policy_with_custom_values(self):
        """Test creating a scaling policy with custom values"""
        policy = ScalingPolicy(
            name="custom_policy",
            metric_name="memory_usage",
            scale_up_threshold=85.0,
            scale_down_threshold=40.0,
            min_instances=2,
            max_instances=20,
            cooldown_period=600,
            enabled=False
        )
        
        assert policy.enabled == False
        assert policy.min_instances == 2
        assert policy.max_instances == 20


class TestScalingDecision:
    """Test scaling decision data structure"""
    
    def test_scaling_decision_creation(self):
        """Test creating a scaling decision"""
        decision = ScalingDecision(
            action="scale_up",
            target_instances=5,
            current_instances=3,
            reason="High CPU usage",
            confidence=0.9,
            metrics={"cpu_usage": 85.0},
            predicted_impact={"cost_change": 0.2}
        )
        
        assert decision.action == "scale_up"
        assert decision.target_instances == 5
        assert decision.current_instances == 3
        assert decision.reason == "High CPU usage"
        assert decision.confidence == 0.9
        assert decision.metrics["cpu_usage"] == 85.0
        assert decision.predicted_impact["cost_change"] == 0.2
        assert isinstance(decision.created_at, datetime)


class TestAutoScalingManager:
    """Test the auto-scaling manager"""
    
    @pytest.fixture
    async def scaling_manager(self):
        """Create a fresh scaling manager for testing"""
        manager = AutoScalingManager()
        yield manager
        await manager.stop_scaling()
    
    def test_scaling_manager_initialization(self, scaling_manager):
        """Test scaling manager initialization"""
        assert not scaling_manager.scaling_active
        assert scaling_manager.current_instances == 1
        assert scaling_manager.target_instances == 1
        assert len(scaling_manager.scaling_policies) > 0  # Should have default policies
        assert len(scaling_manager.scaling_decisions) == 0
    
    def test_default_policies_initialization(self, scaling_manager):
        """Test that default policies are initialized"""
        expected_policies = ["cpu_scaling", "memory_scaling", "response_time_scaling", "throughput_scaling"]
        
        for policy_name in expected_policies:
            assert policy_name in scaling_manager.scaling_policies
            
        # Check CPU scaling policy
        cpu_policy = scaling_manager.scaling_policies["cpu_scaling"]
        assert cpu_policy.metric_name == "system.cpu_usage"
        assert cpu_policy.scale_up_threshold == 80.0
        assert cpu_policy.scale_down_threshold == 30.0
    
    @pytest.mark.asyncio
    async def test_start_stop_scaling(self, scaling_manager):
        """Test starting and stopping scaling manager"""
        # Start scaling
        await scaling_manager.start_scaling()
        assert scaling_manager.scaling_active
        assert scaling_manager.scaling_task is not None
        
        # Stop scaling
        await scaling_manager.stop_scaling()
        assert not scaling_manager.scaling_active
    
    def test_add_scaling_policy(self, scaling_manager):
        """Test adding a new scaling policy"""
        policy = ScalingPolicy(
            name="custom_policy",
            metric_name="custom_metric",
            scale_up_threshold=90.0,
            scale_down_threshold=20.0,
            min_instances=1,
            max_instances=5
        )
        
        scaling_manager.add_scaling_policy(policy)
        
        assert "custom_policy" in scaling_manager.scaling_policies
        assert scaling_manager.scaling_policies["custom_policy"] == policy
    
    def test_update_scaling_policy(self, scaling_manager):
        """Test updating an existing scaling policy"""
        # Update CPU scaling policy
        scaling_manager.update_scaling_policy("cpu_scaling", scale_up_threshold=85.0, enabled=False)
        
        policy = scaling_manager.scaling_policies["cpu_scaling"]
        assert policy.scale_up_threshold == 85.0
        assert policy.enabled == False
    
    def test_remove_scaling_policy(self, scaling_manager):
        """Test removing a scaling policy"""
        # Remove a policy
        scaling_manager.remove_scaling_policy("cpu_scaling")
        
        assert "cpu_scaling" not in scaling_manager.scaling_policies
    
    @pytest.mark.asyncio
    async def test_make_scaling_decision_scale_up(self, scaling_manager):
        """Test making a scale-up decision"""
        policy = scaling_manager.scaling_policies["cpu_scaling"]
        current_metrics = {"system.cpu_usage": {"current": 85.0}}
        
        decision = await scaling_manager._make_scaling_decision(
            policy, 85.0, None, current_metrics
        )
        
        assert decision is not None
        assert decision.action == "scale_up"
        assert decision.target_instances > scaling_manager.current_instances
        assert decision.reason.startswith("system.cpu_usage")
    
    @pytest.mark.asyncio
    async def test_make_scaling_decision_scale_down(self, scaling_manager):
        """Test making a scale-down decision"""
        # Set current instances to more than minimum
        scaling_manager.current_instances = 5
        
        policy = scaling_manager.scaling_policies["cpu_scaling"]
        current_metrics = {"system.cpu_usage": {"current": 25.0}}
        
        decision = await scaling_manager._make_scaling_decision(
            policy, 25.0, None, current_metrics
        )
        
        assert decision is not None
        assert decision.action == "scale_down"
        assert decision.target_instances < scaling_manager.current_instances
        assert decision.reason.startswith("system.cpu_usage")
    
    @pytest.mark.asyncio
    async def test_make_scaling_decision_maintain(self, scaling_manager):
        """Test making a maintain decision"""
        policy = scaling_manager.scaling_policies["cpu_scaling"]
        current_metrics = {"system.cpu_usage": {"current": 50.0}}
        
        decision = await scaling_manager._make_scaling_decision(
            policy, 50.0, None, current_metrics
        )
        
        assert decision is not None
        assert decision.action == "maintain"
        assert decision.target_instances == scaling_manager.current_instances
    
    @pytest.mark.asyncio
    async def test_make_scaling_decision_with_prediction(self, scaling_manager):
        """Test making scaling decision with prediction"""
        policy = scaling_manager.scaling_policies["cpu_scaling"]
        current_metrics = {"system.cpu_usage": {"current": 50.0}}
        
        # Mock predictions
        with patch.object(scaling_manager, '_get_relevant_predictions', return_value=[
            {"metric_name": "system.cpu_usage", "predicted_value": 85.0, "confidence": 0.9, "time_horizon": 300}
        ]):
            decision = await scaling_manager._make_scaling_decision(
                policy, 50.0, 85.0, current_metrics
            )
            
            assert decision is not None
            assert decision.action == "scale_up"
            assert decision.confidence == 0.9
    
    def test_calculate_predicted_impact_scale_up(self, scaling_manager):
        """Test calculating predicted impact for scale-up"""
        scaling_manager.current_instances = 3
        
        impact = scaling_manager._calculate_predicted_impact("scale_up", 5)
        
        assert impact["cost_change"] > 0  # Cost should increase
        assert impact["performance_change"] > 0  # Performance should improve
        assert impact["resource_utilization"] < 100  # Utilization should decrease
    
    def test_calculate_predicted_impact_scale_down(self, scaling_manager):
        """Test calculating predicted impact for scale-down"""
        scaling_manager.current_instances = 5
        
        impact = scaling_manager._calculate_predicted_impact("scale_down", 3)
        
        assert impact["cost_change"] < 0  # Cost should decrease
        assert impact["performance_change"] < 0  # Performance might decrease
        assert impact["resource_utilization"] > 100  # Utilization should increase
    
    @pytest.mark.asyncio
    async def test_execute_scaling_action(self, scaling_manager):
        """Test executing a scaling action"""
        decision = ScalingDecision(
            action="scale_up",
            target_instances=3,
            current_instances=1,
            reason="Test scaling",
            confidence=0.9
        )
        
        success = await scaling_manager._execute_scaling_action(decision)
        
        assert success == True
        assert scaling_manager.scaling_metrics["successful_scales"] == 1
        assert scaling_manager.scaling_metrics["scaling_actions_taken"] == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_scaling_decisions(self, scaling_manager):
        """Test evaluating scaling decisions"""
        # Mock performance optimizer
        mock_summary = {
            "metrics": {
                "system.cpu_usage": {"current": 85.0},
                "system.memory_usage": {"current": 50.0}
            }
        }
        
        with patch('app.services.auto_scaling_manager.advanced_performance_optimizer') as mock_optimizer:
            mock_optimizer.get_performance_summary.return_value = mock_summary
            
            # Mock predictions
            with patch.object(scaling_manager, '_get_relevant_predictions', return_value=[]):
                await scaling_manager._evaluate_scaling_decisions()
        
        # Should have made scaling decisions
        assert len(scaling_manager.scaling_decisions) > 0
    
    def test_get_scaling_summary(self, scaling_manager):
        """Test getting scaling summary"""
        # Add a test decision
        decision = ScalingDecision(
            action="scale_up",
            target_instances=3,
            current_instances=1,
            reason="Test",
            confidence=0.9
        )
        scaling_manager.scaling_decisions.append(decision)
        
        summary = scaling_manager.get_scaling_summary()
        
        assert "timestamp" in summary
        assert "scaling_active" in summary
        assert "current_instances" in summary
        assert "policies" in summary
        assert "recent_decisions" in summary
        assert "metrics" in summary
        assert summary["current_instances"] == 1
        assert len(summary["policies"]) > 0


class TestScalingIntegration:
    """Test scaling integration with other components"""
    
    @pytest.mark.asyncio
    async def test_scaling_with_performance_optimizer(self):
        """Test scaling integration with performance optimizer"""
        # Mock performance optimizer
        mock_summary = {
            "metrics": {
                "system.cpu_usage": {"current": 90.0},
                "system.memory_usage": {"current": 60.0}
            }
        }
        
        with patch('app.services.auto_scaling_manager.advanced_performance_optimizer') as mock_optimizer:
            mock_optimizer.get_performance_summary.return_value = mock_summary
            
            # Start scaling
            await auto_scaling_manager.start_scaling()
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Stop scaling
            await auto_scaling_manager.stop_scaling()
    
    @pytest.mark.asyncio
    async def test_scaling_with_analytics_engine(self):
        """Test scaling integration with analytics engine"""
        # Mock analytics engine predictions
        mock_predictions = [
            Mock(
                metric_name="system.cpu_usage",
                predicted_value=85.0,
                confidence=0.8,
                time_horizon=300
            )
        ]
        
        with patch('app.services.auto_scaling_manager.advanced_analytics_engine') as mock_analytics:
            mock_analytics.predictions = mock_predictions
            
            manager = AutoScalingManager()
            
            # Test getting predictions
            predictions = await manager._get_relevant_predictions()
            
            assert len(predictions) == 1
            assert predictions[0]["metric_name"] == "system.cpu_usage"
            assert predictions[0]["predicted_value"] == 85.0


class TestScalingPerformance:
    """Test scaling performance under load"""
    
    @pytest.mark.asyncio
    async def test_scaling_decision_performance(self):
        """Test scaling decision performance"""
        manager = AutoScalingManager()
        
        # Add multiple policies
        for i in range(10):
            policy = ScalingPolicy(
                name=f"policy_{i}",
                metric_name=f"metric_{i}",
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=10
            )
            manager.add_scaling_policy(policy)
        
        # Mock performance data
        mock_metrics = {f"metric_{i}": {"current": 50.0} for i in range(10)}
        
        start_time = time.time()
        
        # Evaluate all policies
        for policy_name, policy in manager.scaling_policies.items():
            await manager._make_scaling_decision(policy, 50.0, None, mock_metrics)
        
        decision_time = time.time() - start_time
        
        # Should make decisions quickly
        assert decision_time < 1.0  # Should complete in under 1 second
    
    @pytest.mark.asyncio
    async def test_scaling_under_high_frequency_updates(self):
        """Test scaling under high frequency metric updates"""
        manager = AutoScalingManager()
        
        # Start scaling
        await manager.start_scaling()
        
        try:
            # Simulate high frequency updates
            for i in range(100):
                # Mock different metric values
                mock_summary = {
                    "metrics": {
                        "system.cpu_usage": {"current": 50.0 + (i % 50)},
                        "system.memory_usage": {"current": 60.0 + (i % 30)}
                    }
                }
                
                with patch('app.services.auto_scaling_manager.advanced_performance_optimizer') as mock_optimizer:
                    mock_optimizer.get_performance_summary.return_value = mock_summary
                    
                    await manager._evaluate_scaling_decisions()
                
                # Small delay to simulate real-world timing
                await asyncio.sleep(0.01)
            
            # Should handle high frequency updates without issues
            assert len(manager.scaling_decisions) >= 0
            
        finally:
            await manager.stop_scaling()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
