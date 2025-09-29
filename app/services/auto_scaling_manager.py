"""
Auto Scaling Manager
Intelligent scaling based on performance metrics and predictions
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

from app.core.advanced_performance_optimizer import advanced_performance_optimizer
from app.services.advanced_analytics_engine import advanced_analytics_engine

logger = logging.getLogger(__name__)


@dataclass
class ScalingDecision:
    """Scaling decision data structure"""
    action: str  # 'scale_up', 'scale_down', 'maintain'
    target_instances: int
    current_instances: int
    reason: str
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    predicted_impact: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingPolicy:
    """Scaling policy configuration"""
    name: str
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int
    max_instances: int
    cooldown_period: int  # seconds
    enabled: bool = True
    last_action_time: Optional[datetime] = None


class AutoScalingManager:
    """Intelligent auto-scaling manager"""
    
    def __init__(self):
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.scaling_decisions: List[ScalingDecision] = []
        self.current_instances = 1
        self.target_instances = 1
        
        # Scaling configuration
        self.config = {
            "default_min_instances": 1,
            "default_max_instances": 10,
            "default_cooldown": 300,  # 5 minutes
            "scaling_factor": 0.5,  # Scale by 50% of current instances
            "prediction_horizon": 300,  # 5 minutes
            "confidence_threshold": 0.7
        }
        
        # Performance tracking
        self.scaling_metrics = {
            "scaling_actions_taken": 0,
            "successful_scales": 0,
            "failed_scales": 0,
            "average_scale_time": 0,
            "cost_savings": 0
        }
        
        self.scaling_active = False
        self.scaling_task: Optional[asyncio.Task] = None
        
        # Initialize default policies
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default scaling policies"""
        default_policies = [
            ScalingPolicy(
                name="cpu_scaling",
                metric_name="system.cpu_usage",
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=10,
                cooldown_period=300
            ),
            ScalingPolicy(
                name="memory_scaling",
                metric_name="system.memory_usage",
                scale_up_threshold=85.0,
                scale_down_threshold=40.0,
                min_instances=1,
                max_instances=10,
                cooldown_period=300
            ),
            ScalingPolicy(
                name="response_time_scaling",
                metric_name="application.api_response_time",
                scale_up_threshold=100.0,  # ms
                scale_down_threshold=50.0,  # ms
                min_instances=1,
                max_instances=10,
                cooldown_period=300
            ),
            ScalingPolicy(
                name="throughput_scaling",
                metric_name="application.requests_per_second",
                scale_up_threshold=1000.0,
                scale_down_threshold=500.0,
                min_instances=1,
                max_instances=10,
                cooldown_period=300
            )
        ]
        
        for policy in default_policies:
            self.scaling_policies[policy.name] = policy
            
    async def start_scaling(self):
        """Start the auto-scaling manager"""
        if self.scaling_active:
            logger.warning("Auto-scaling already active")
            return
            
        self.scaling_active = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaling manager started")
        
    async def stop_scaling(self):
        """Stop the auto-scaling manager"""
        self.scaling_active = False
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-scaling manager stopped")
        
    async def _scaling_loop(self):
        """Main scaling decision loop"""
        while self.scaling_active:
            try:
                # Evaluate scaling decisions
                await self._evaluate_scaling_decisions()
                
                # Execute pending scaling actions
                await self._execute_scaling_actions()
                
                # Wait before next evaluation
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(120)  # Wait longer on error
                
    async def _evaluate_scaling_decisions(self):
        """Evaluate scaling decisions based on policies and predictions"""
        try:
            # Get current performance metrics
            perf_summary = advanced_performance_optimizer.get_performance_summary()
            current_metrics = perf_summary.get("metrics", {})
            
            # Get predictions from analytics engine
            predictions = await self._get_relevant_predictions()
            
            # Evaluate each scaling policy
            for policy_name, policy in self.scaling_policies.items():
                if not policy.enabled:
                    continue
                    
                # Check cooldown period
                if policy.last_action_time:
                    time_since_last_action = (datetime.now() - policy.last_action_time).total_seconds()
                    if time_since_last_action < policy.cooldown_period:
                        continue
                        
                # Get current metric value
                current_value = current_metrics.get(policy.metric_name, {}).get("current", 0)
                
                # Get predicted value
                predicted_value = self._get_predicted_value(predictions, policy.metric_name)
                
                # Make scaling decision
                decision = await self._make_scaling_decision(
                    policy, current_value, predicted_value, current_metrics
                )
                
                if decision:
                    self.scaling_decisions.append(decision)
                    policy.last_action_time = datetime.now()
                    
        except Exception as e:
            logger.error(f"Error evaluating scaling decisions: {e}")
            
    async def _get_relevant_predictions(self) -> List[Dict[str, Any]]:
        """Get relevant predictions from analytics engine"""
        try:
            # Get predictions for the next 5 minutes
            relevant_predictions = []
            
            for prediction in advanced_analytics_engine.predictions:
                if prediction.time_horizon <= self.config["prediction_horizon"]:
                    relevant_predictions.append({
                        "metric_name": prediction.metric_name,
                        "predicted_value": prediction.predicted_value,
                        "confidence": prediction.confidence,
                        "time_horizon": prediction.time_horizon
                    })
                    
            return relevant_predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
            
    def _get_predicted_value(self, predictions: List[Dict[str, Any]], metric_name: str) -> Optional[float]:
        """Get predicted value for a specific metric"""
        for prediction in predictions:
            if prediction["metric_name"] == metric_name:
                return prediction["predicted_value"]
        return None
        
    async def _make_scaling_decision(self, policy: ScalingPolicy, current_value: float, 
                                   predicted_value: Optional[float], 
                                   current_metrics: Dict[str, Any]) -> Optional[ScalingDecision]:
        """Make a scaling decision based on policy and metrics"""
        try:
            # Use predicted value if available and confident
            value_to_use = current_value
            confidence = 0.5
            
            if predicted_value is not None:
                # Find the prediction confidence
                for prediction in advanced_analytics_engine.predictions:
                    if (prediction.metric_name == policy.metric_name and 
                        prediction.time_horizon <= self.config["prediction_horizon"]):
                        confidence = prediction.confidence
                        if confidence >= self.config["confidence_threshold"]:
                            value_to_use = predicted_value
                        break
                        
            # Determine scaling action
            action = None
            target_instances = self.current_instances
            reason = ""
            
            if value_to_use > policy.scale_up_threshold:
                # Scale up
                if self.current_instances < policy.max_instances:
                    action = "scale_up"
                    target_instances = min(
                        policy.max_instances,
                        int(self.current_instances * (1 + self.config["scaling_factor"]))
                    )
                    reason = f"{policy.metric_name} at {value_to_use:.1f} exceeds threshold {policy.scale_up_threshold}"
                else:
                    reason = f"Already at maximum instances ({policy.max_instances})"
                    
            elif value_to_use < policy.scale_down_threshold:
                # Scale down
                if self.current_instances > policy.min_instances:
                    action = "scale_down"
                    target_instances = max(
                        policy.min_instances,
                        int(self.current_instances * (1 - self.config["scaling_factor"]))
                    )
                    reason = f"{policy.metric_name} at {value_to_use:.1f} below threshold {policy.scale_down_threshold}"
                else:
                    reason = f"Already at minimum instances ({policy.min_instances})"
                    
            else:
                # Maintain current level
                action = "maintain"
                reason = f"{policy.metric_name} at {value_to_use:.1f} within normal range"
                
            # Create scaling decision
            if action and target_instances != self.current_instances:
                decision = ScalingDecision(
                    action=action,
                    target_instances=target_instances,
                    current_instances=self.current_instances,
                    reason=reason,
                    confidence=confidence,
                    metrics=current_metrics,
                    predicted_impact=self._calculate_predicted_impact(action, target_instances)
                )
                
                return decision
                
        except Exception as e:
            logger.error(f"Error making scaling decision: {e}")
            
        return None
        
    def _calculate_predicted_impact(self, action: str, target_instances: int) -> Dict[str, Any]:
        """Calculate predicted impact of scaling action"""
        try:
            impact = {
                "cost_change": 0,
                "performance_change": 0,
                "resource_utilization": 0
            }
            
            instance_change = target_instances - self.current_instances
            
            if action == "scale_up":
                # Estimate cost increase
                impact["cost_change"] = instance_change * 0.1  # $0.10 per instance per hour
                impact["performance_change"] = min(50, instance_change * 10)  # Up to 50% improvement
                impact["resource_utilization"] = max(0, 100 - (instance_change * 15))  # Reduce utilization
                
            elif action == "scale_down":
                # Estimate cost savings
                impact["cost_change"] = -abs(instance_change) * 0.1  # Cost savings
                impact["performance_change"] = max(-20, -abs(instance_change) * 5)  # Up to 20% reduction
                impact["resource_utilization"] = min(100, 100 + (abs(instance_change) * 15))  # Increase utilization
                
            return impact
            
        except Exception as e:
            logger.error(f"Error calculating predicted impact: {e}")
            return {}
            
    async def _execute_scaling_actions(self):
        """Execute pending scaling actions"""
        try:
            # Get the most recent scaling decision
            if not self.scaling_decisions:
                return
                
            latest_decision = self.scaling_decisions[-1]
            
            # Check if we need to execute this decision
            if latest_decision.target_instances == self.target_instances:
                return  # Already executed
                
            # Execute scaling action
            success = await self._execute_scaling_action(latest_decision)
            
            if success:
                self.current_instances = latest_decision.target_instances
                self.target_instances = latest_decision.target_instances
                self.scaling_metrics["successful_scales"] += 1
                self.scaling_metrics["scaling_actions_taken"] += 1
                
                logger.info(f"Successfully scaled to {latest_decision.target_instances} instances")
            else:
                self.scaling_metrics["failed_scales"] += 1
                logger.error(f"Failed to scale to {latest_decision.target_instances} instances")
                
        except Exception as e:
            logger.error(f"Error executing scaling actions: {e}")
            
    async def _execute_scaling_action(self, decision: ScalingDecision) -> bool:
        """Execute a specific scaling action"""
        try:
            # Simulate scaling action (in real implementation, this would call cloud provider APIs)
            logger.info(f"Executing scaling action: {decision.action} to {decision.target_instances} instances")
            
            # Simulate scaling time
            scaling_time = abs(decision.target_instances - decision.current_instances) * 2  # 2 seconds per instance
            await asyncio.sleep(min(scaling_time, 10))  # Cap at 10 seconds for demo
            
            # Update scaling metrics
            self.scaling_metrics["average_scale_time"] = (
                (self.scaling_metrics["average_scale_time"] * (self.scaling_metrics["scaling_actions_taken"] - 1) + scaling_time) /
                self.scaling_metrics["scaling_actions_taken"]
            )
            
            # Calculate cost savings
            if decision.action == "scale_down":
                cost_savings = abs(decision.target_instances - decision.current_instances) * 0.1
                self.scaling_metrics["cost_savings"] += cost_savings
                
            return True
            
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            return False
            
    def add_scaling_policy(self, policy: ScalingPolicy):
        """Add a new scaling policy"""
        self.scaling_policies[policy.name] = policy
        logger.info(f"Added scaling policy: {policy.name}")
        
    def update_scaling_policy(self, policy_name: str, **kwargs):
        """Update an existing scaling policy"""
        if policy_name in self.scaling_policies:
            policy = self.scaling_policies[policy_name]
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            logger.info(f"Updated scaling policy: {policy_name}")
        else:
            logger.warning(f"Scaling policy not found: {policy_name}")
            
    def remove_scaling_policy(self, policy_name: str):
        """Remove a scaling policy"""
        if policy_name in self.scaling_policies:
            del self.scaling_policies[policy_name]
            logger.info(f"Removed scaling policy: {policy_name}")
        else:
            logger.warning(f"Scaling policy not found: {policy_name}")
            
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get comprehensive scaling summary"""
        try:
            # Get recent decisions
            recent_decisions = [d for d in self.scaling_decisions if 
                              (datetime.now() - d.created_at).total_seconds() < 3600]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "scaling_active": self.scaling_active,
                "current_instances": self.current_instances,
                "target_instances": self.target_instances,
                "policies": {
                    name: {
                        "enabled": policy.enabled,
                        "metric_name": policy.metric_name,
                        "scale_up_threshold": policy.scale_up_threshold,
                        "scale_down_threshold": policy.scale_down_threshold,
                        "min_instances": policy.min_instances,
                        "max_instances": policy.max_instances,
                        "cooldown_period": policy.cooldown_period,
                        "last_action_time": policy.last_action_time.isoformat() if policy.last_action_time else None
                    }
                    for name, policy in self.scaling_policies.items()
                },
                "recent_decisions": [
                    {
                        "action": d.action,
                        "target_instances": d.target_instances,
                        "current_instances": d.current_instances,
                        "reason": d.reason,
                        "confidence": d.confidence,
                        "created_at": d.created_at.isoformat()
                    }
                    for d in recent_decisions
                ],
                "metrics": self.scaling_metrics,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling summary: {e}")
            return {"error": str(e)}


# Global instance
auto_scaling_manager = AutoScalingManager()


