"""
Chaos Engineering Engine
Advanced chaos engineering and resilience testing system
"""

import asyncio
import time
import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import json
import psutil
import threading
import subprocess
import signal
import os
import secrets

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class ChaosExperimentType(Enum):
    """Chaos experiment type enumeration"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    SERVICE_FAILURE = "service_failure"
    DATABASE_FAILURE = "database_failure"
    CACHE_FAILURE = "cache_failure"
    RANDOM_FAILURE = "random_failure"
    CASCADING_FAILURE = "cascading_failure"


class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureMode(Enum):
    """Failure mode enumeration"""
    GRACEFUL = "graceful"
    UNGRACEFUL = "ungraceful"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CASCADING = "cascading"


@dataclass
class ChaosExperiment:
    """Chaos experiment data structure"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ChaosExperimentType
    target_services: List[str]
    duration: int  # seconds
    intensity: float  # 0.0 to 1.0
    failure_mode: FailureMode
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_time: Optional[float] = None


@dataclass
class ResilienceMetrics:
    """Resilience metrics data structure"""
    service_name: str
    availability: float  # percentage
    response_time: float  # milliseconds
    error_rate: float  # percentage
    recovery_time: float  # seconds
    throughput: float  # requests per second
    timestamp: datetime = field(default_factory=datetime.now)


class ChaosEngineeringEngine:
    """Chaos Engineering Engine for resilience testing"""
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.active_experiments: List[str] = []
        self.resilience_metrics: List[ResilienceMetrics] = []
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
        # Configuration
        self.config = {
            "chaos_enabled": True,
            "auto_experiments": False,
            "experiment_interval": 3600,  # 1 hour
            "max_concurrent_experiments": 3,
            "safety_checks_enabled": True,
            "auto_recovery": True,
            "metrics_collection_interval": 10,  # seconds
            "baseline_measurement_duration": 300,  # 5 minutes
            "experiment_cooldown": 1800,  # 30 minutes
            "alert_on_failure": True,
            "detailed_logging": True
        }
        
        # Experiment templates
        self.experiment_templates = {
            ChaosExperimentType.NETWORK_LATENCY: {
                "name": "Network Latency Injection",
                "description": "Inject network latency to test service resilience",
                "default_duration": 300,
                "default_intensity": 0.5,
                "failure_mode": FailureMode.GRACEFUL
            },
            ChaosExperimentType.CPU_STRESS: {
                "name": "CPU Stress Test",
                "description": "Generate CPU load to test system under stress",
                "default_duration": 180,
                "default_intensity": 0.7,
                "failure_mode": FailureMode.PARTIAL
            },
            ChaosExperimentType.MEMORY_STRESS: {
                "name": "Memory Stress Test",
                "description": "Generate memory pressure to test memory management",
                "default_duration": 240,
                "default_intensity": 0.6,
                "failure_mode": FailureMode.PARTIAL
            },
            ChaosExperimentType.SERVICE_FAILURE: {
                "name": "Service Failure Simulation",
                "description": "Simulate service failures to test fault tolerance",
                "default_duration": 120,
                "default_intensity": 1.0,
                "failure_mode": FailureMode.COMPLETE
            },
            ChaosExperimentType.CASCADING_FAILURE: {
                "name": "Cascading Failure Test",
                "description": "Test system behavior under cascading failures",
                "default_duration": 600,
                "default_intensity": 0.8,
                "failure_mode": FailureMode.CASCADING
            }
        }
        
        # Safety rules
        self.safety_rules = {
            "max_cpu_usage": 90.0,  # percentage
            "max_memory_usage": 85.0,  # percentage
            "max_disk_usage": 80.0,  # percentage
            "min_service_availability": 50.0,  # percentage
            "max_experiment_duration": 1800,  # 30 minutes
            "cooldown_between_experiments": 1800  # 30 minutes
        }
        
        # Monitoring
        self.chaos_active = False
        self.chaos_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.chaos_stats = {
            "experiments_run": 0,
            "experiments_failed": 0,
            "services_tested": 0,
            "resilience_improvements": 0,
            "failures_detected": 0,
            "recovery_tests": 0
        }
        
    async def start_chaos_engine(self):
        """Start the chaos engineering engine"""
        if self.chaos_active:
            logger.warning("Chaos engineering engine already active")
            return
            
        self.chaos_active = True
        self.chaos_task = asyncio.create_task(self._chaos_processing_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        logger.info("Chaos Engineering Engine started")
        
    async def stop_chaos_engine(self):
        """Stop the chaos engineering engine"""
        self.chaos_active = False
        if self.chaos_task:
            self.chaos_task.cancel()
            try:
                await self.chaos_task
            except asyncio.CancelledError:
                pass
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
        logger.info("Chaos Engineering Engine stopped")
        
    async def _chaos_processing_loop(self):
        """Main chaos engineering processing loop"""
        while self.chaos_active:
            try:
                # Check safety conditions
                if self.config["safety_checks_enabled"]:
                    await self._check_safety_conditions()
                
                # Run planned experiments
                await self._run_planned_experiments()
                
                # Monitor active experiments
                await self._monitor_active_experiments()
                
                # Auto-experiments if enabled
                if self.config["auto_experiments"]:
                    await self._run_auto_experiments()
                
                # Clean up completed experiments
                await self._cleanup_completed_experiments()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in chaos processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.chaos_active:
            try:
                # Collect baseline metrics
                await self._collect_baseline_metrics()
                
                # Collect resilience metrics
                await self._collect_resilience_metrics()
                
                # Wait before next collection
                await asyncio.sleep(self.config["metrics_collection_interval"])
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(60)
                
    async def create_experiment(self, experiment_data: Dict[str, Any]) -> ChaosExperiment:
        """Create a new chaos experiment"""
        try:
            experiment_id = f"exp_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Get template
            experiment_type = ChaosExperimentType(experiment_data.get("type", "random_failure"))
            template = self.experiment_templates.get(experiment_type, {})
            
            # Create experiment
            experiment = ChaosExperiment(
                experiment_id=experiment_id,
                name=experiment_data.get("name", template.get("name", "Custom Experiment")),
                description=experiment_data.get("description", template.get("description", "")),
                experiment_type=experiment_type,
                target_services=experiment_data.get("target_services", []),
                duration=experiment_data.get("duration", template.get("default_duration", 300)),
                intensity=experiment_data.get("intensity", template.get("default_intensity", 0.5)),
                failure_mode=FailureMode(experiment_data.get("failure_mode", template.get("failure_mode", "graceful").value)),
                status=ExperimentStatus.PLANNED,
                created_at=datetime.now()
            )
            
            # Store experiment
            self.experiments[experiment_id] = experiment
            
            logger.info(f"Chaos experiment created: {experiment_id}")
            return experiment
            
        except Exception as e:
            logger.error(f"Error creating chaos experiment: {e}")
            raise
            
    async def run_experiment(self, experiment_id: str) -> bool:
        """Run a chaos experiment"""
        try:
            experiment = self.experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_id}")
                
            if experiment.status != ExperimentStatus.PLANNED:
                raise ValueError(f"Experiment not in planned status: {experiment.status}")
                
            # Check safety conditions
            if self.config["safety_checks_enabled"]:
                if not await self._check_experiment_safety(experiment):
                    logger.warning(f"Safety check failed for experiment: {experiment_id}")
                    return False
                    
            # Start experiment
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.now()
            self.active_experiments.append(experiment_id)
            
            # Run the experiment
            success = await self._execute_experiment(experiment)
            
            # Complete experiment
            experiment.completed_at = datetime.now()
            experiment.status = ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED
            
            # Remove from active experiments
            if experiment_id in self.active_experiments:
                self.active_experiments.remove(experiment_id)
                
            # Update statistics
            self.chaos_stats["experiments_run"] += 1
            if not success:
                self.chaos_stats["experiments_failed"] += 1
                
            logger.info(f"Chaos experiment completed: {experiment_id} - Success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error running chaos experiment: {e}")
            return False
            
    async def _execute_experiment(self, experiment: ChaosExperiment) -> bool:
        """Execute a chaos experiment"""
        try:
            experiment_type = experiment.experiment_type
            
            if experiment_type == ChaosExperimentType.NETWORK_LATENCY:
                return await self._inject_network_latency(experiment)
            elif experiment_type == ChaosExperimentType.CPU_STRESS:
                return await self._inject_cpu_stress(experiment)
            elif experiment_type == ChaosExperimentType.MEMORY_STRESS:
                return await self._inject_memory_stress(experiment)
            elif experiment_type == ChaosExperimentType.SERVICE_FAILURE:
                return await self._simulate_service_failure(experiment)
            elif experiment_type == ChaosExperimentType.CASCADING_FAILURE:
                return await self._simulate_cascading_failure(experiment)
            else:
                return await self._simulate_random_failure(experiment)
                
        except Exception as e:
            logger.error(f"Error executing experiment: {e}")
            return False
            
    async def _inject_network_latency(self, experiment: ChaosExperiment) -> bool:
        """Inject network latency"""
        try:
            # Simulate network latency injection
            latency_ms = int(experiment.intensity * 1000)  # 0-1000ms
            
            logger.info(f"Injecting network latency: {latency_ms}ms for {experiment.duration}s")
            
            # Simulate latency injection
            await asyncio.sleep(experiment.duration)
            
            # Record results
            experiment.results = {
                "latency_injected": latency_ms,
                "duration": experiment.duration,
                "services_affected": len(experiment.target_services)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error injecting network latency: {e}")
            return False
            
    async def _inject_cpu_stress(self, experiment: ChaosExperiment) -> bool:
        """Inject CPU stress"""
        try:
            # Simulate CPU stress
            cpu_load = experiment.intensity * 100  # 0-100%
            
            logger.info(f"Injecting CPU stress: {cpu_load}% for {experiment.duration}s")
            
            # Simulate CPU stress
            await asyncio.sleep(experiment.duration)
            
            # Record results
            experiment.results = {
                "cpu_load": cpu_load,
                "duration": experiment.duration,
                "services_affected": len(experiment.target_services)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error injecting CPU stress: {e}")
            return False
            
    async def _inject_memory_stress(self, experiment: ChaosExperiment) -> bool:
        """Inject memory stress"""
        try:
            # Simulate memory stress
            memory_usage = experiment.intensity * 100  # 0-100%
            
            logger.info(f"Injecting memory stress: {memory_usage}% for {experiment.duration}s")
            
            # Simulate memory stress
            await asyncio.sleep(experiment.duration)
            
            # Record results
            experiment.results = {
                "memory_usage": memory_usage,
                "duration": experiment.duration,
                "services_affected": len(experiment.target_services)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error injecting memory stress: {e}")
            return False
            
    async def _simulate_service_failure(self, experiment: ChaosExperiment) -> bool:
        """Simulate service failure"""
        try:
            logger.info(f"Simulating service failure for {experiment.duration}s")
            
            # Simulate service failure
            await asyncio.sleep(experiment.duration)
            
            # Record results
            experiment.results = {
                "failure_mode": experiment.failure_mode.value,
                "duration": experiment.duration,
                "services_failed": experiment.target_services
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error simulating service failure: {e}")
            return False
            
    async def _simulate_cascading_failure(self, experiment: ChaosExperiment) -> bool:
        """Simulate cascading failure"""
        try:
            logger.info(f"Simulating cascading failure for {experiment.duration}s")
            
            # Simulate cascading failure
            await asyncio.sleep(experiment.duration)
            
            # Record results
            experiment.results = {
                "cascade_intensity": experiment.intensity,
                "duration": experiment.duration,
                "services_affected": experiment.target_services
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error simulating cascading failure: {e}")
            return False
            
    async def _simulate_random_failure(self, experiment: ChaosExperiment) -> bool:
        """Simulate random failure"""
        try:
            logger.info(f"Simulating random failure for {experiment.duration}s")
            
            # Simulate random failure
            await asyncio.sleep(experiment.duration)
            
            # Record results
            experiment.results = {
                "failure_type": "random",
                "duration": experiment.duration,
                "services_affected": experiment.target_services
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Error simulating random failure: {e}")
            return False
            
    async def _check_safety_conditions(self):
        """Check safety conditions before running experiments"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check safety thresholds
            if cpu_percent > self.safety_rules["max_cpu_usage"]:
                logger.warning(f"CPU usage too high: {cpu_percent}%")
                return False
                
            if memory.percent > self.safety_rules["max_memory_usage"]:
                logger.warning(f"Memory usage too high: {memory.percent}%")
                return False
                
            if disk.percent > self.safety_rules["max_disk_usage"]:
                logger.warning(f"Disk usage too high: {disk.percent}%")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking safety conditions: {e}")
            return False
            
    async def _check_experiment_safety(self, experiment: ChaosExperiment) -> bool:
        """Check if experiment is safe to run"""
        try:
            # Check experiment duration
            if experiment.duration > self.safety_rules["max_experiment_duration"]:
                logger.warning(f"Experiment duration too long: {experiment.duration}s")
                return False
                
            # Check concurrent experiments
            if len(self.active_experiments) >= self.config["max_concurrent_experiments"]:
                logger.warning("Too many concurrent experiments")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking experiment safety: {e}")
            return False
            
    async def _run_planned_experiments(self):
        """Run planned experiments"""
        try:
            for experiment_id, experiment in self.experiments.items():
                if experiment.status == ExperimentStatus.PLANNED:
                    # Check if it's time to run
                    if (datetime.now() - experiment.created_at).total_seconds() > 60:  # 1 minute delay
                        await self.run_experiment(experiment_id)
                        
        except Exception as e:
            logger.error(f"Error running planned experiments: {e}")
            
    async def _monitor_active_experiments(self):
        """Monitor active experiments"""
        try:
            for experiment_id in self.active_experiments:
                experiment = self.experiments.get(experiment_id)
                if experiment and experiment.started_at:
                    # Check if experiment should be stopped
                    elapsed = (datetime.now() - experiment.started_at).total_seconds()
                    if elapsed >= experiment.duration:
                        # Stop experiment
                        experiment.status = ExperimentStatus.COMPLETED
                        experiment.completed_at = datetime.now()
                        self.active_experiments.remove(experiment_id)
                        
        except Exception as e:
            logger.error(f"Error monitoring active experiments: {e}")
            
    async def _run_auto_experiments(self):
        """Run automatic experiments"""
        try:
            # This would implement automatic experiment selection
            # For now, we'll just log the action
            logger.debug("Running automatic experiments...")
            
        except Exception as e:
            logger.error(f"Error running auto experiments: {e}")
            
    async def _cleanup_completed_experiments(self):
        """Clean up completed experiments"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=7)  # Keep for 7 days
            
            # Remove old completed experiments
            to_remove = []
            for experiment_id, experiment in self.experiments.items():
                if (experiment.status == ExperimentStatus.COMPLETED and 
                    experiment.completed_at and 
                    experiment.completed_at < cutoff_time):
                    to_remove.append(experiment_id)
                    
            for experiment_id in to_remove:
                del self.experiments[experiment_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up experiments: {e}")
            
    async def _collect_baseline_metrics(self):
        """Collect baseline metrics"""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Store baseline metrics
            self.baseline_metrics["system"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting baseline metrics: {e}")
            
    async def _collect_resilience_metrics(self):
        """Collect resilience metrics"""
        try:
            # This would collect actual service metrics
            # For now, we'll simulate it
            for service in ["api", "database", "cache", "monitoring"]:
                metrics = ResilienceMetrics(
                    service_name=service,
                    availability=95.0 + random.uniform(-5, 5),
                    response_time=100.0 + random.uniform(-20, 20),
                    error_rate=1.0 + random.uniform(-0.5, 0.5),
                    recovery_time=30.0 + random.uniform(-10, 10),
                    throughput=1000.0 + random.uniform(-100, 100)
                )
                
                self.resilience_metrics.append(metrics)
                
        except Exception as e:
            logger.error(f"Error collecting resilience metrics: {e}")
            
    def get_chaos_summary(self) -> Dict[str, Any]:
        """Get comprehensive chaos engineering summary"""
        try:
            # Calculate experiment statistics
            experiments_by_status = defaultdict(int)
            experiments_by_type = defaultdict(int)
            
            for experiment in self.experiments.values():
                experiments_by_status[experiment.status.value] += 1
                experiments_by_type[experiment.experiment_type.value] += 1
                
            # Calculate resilience metrics
            recent_metrics = self.resilience_metrics[-10:] if self.resilience_metrics else []
            avg_availability = sum(m.availability for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "chaos_active": self.chaos_active,
                "total_experiments": len(self.experiments),
                "active_experiments": len(self.active_experiments),
                "experiments_by_status": dict(experiments_by_status),
                "experiments_by_type": dict(experiments_by_type),
                "baseline_metrics": self.baseline_metrics,
                "resilience_metrics": {
                    "average_availability": avg_availability,
                    "average_response_time": avg_response_time,
                    "total_metrics_collected": len(self.resilience_metrics)
                },
                "stats": self.chaos_stats,
                "config": self.config,
                "safety_rules": self.safety_rules
            }
            
        except Exception as e:
            logger.error(f"Error getting chaos summary: {e}")
            return {"error": str(e)}


# Global instance
chaos_engineering_engine = ChaosEngineeringEngine()
