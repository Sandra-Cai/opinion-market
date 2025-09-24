"""
AI-Powered System Optimization Engine
Provides intelligent system optimization and recommendations
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import numpy as np

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation data structure"""
    category: str
    priority: int  # 1-10, higher is more important
    title: str
    description: str
    impact: str  # "high", "medium", "low"
    effort: str  # "low", "medium", "high"
    estimated_improvement: float  # Percentage improvement
    parameters: Dict[str, Any]
    confidence: float  # 0-1, confidence in recommendation


@dataclass
class PerformancePrediction:
    """Performance prediction data structure"""
    metric: str
    current_value: float
    predicted_value: float
    confidence: float
    time_horizon: int  # Hours
    trend: str  # "improving", "stable", "declining"


class AIOptimizationEngine:
    """AI-powered system optimization engine"""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.recommendations_cache = {}
        self.performance_predictions = {}
        self.optimization_history = []
        
        # Optimization parameters
        self.optimization_config = {
            "cache": {
                "target_hit_rate": 90.0,
                "max_memory_usage": 80.0,
                "min_compression_ratio": 0.7
            },
            "performance": {
                "target_response_time": 100.0,  # ms
                "max_cpu_usage": 70.0,
                "max_memory_usage": 80.0
            }
        }
    
    async def start_optimization_monitoring(self):
        """Start continuous optimization monitoring"""
        logger.info("Starting AI optimization monitoring")
        asyncio.create_task(self._optimization_loop())
    
    async def stop_optimization_monitoring(self):
        """Stop optimization monitoring"""
        logger.info("Stopping AI optimization monitoring")
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        while True:
            try:
                await self._collect_optimization_metrics()
                recommendations = await self._generate_optimization_recommendations()
                await self._apply_automatic_optimizations(recommendations)
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(300)
    
    async def _collect_optimization_metrics(self):
        """Collect metrics for optimization analysis"""
        try:
            system_metrics = await self._get_system_metrics()
            self.metrics_history["system"].append(system_metrics)
            
            cache_metrics = await self._get_cache_metrics()
            self.metrics_history["cache"].append(cache_metrics)
        except Exception as e:
            logger.error(f"Error collecting optimization metrics: {e}")
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        import psutil
        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    
    async def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        try:
            stats = enhanced_cache.get_stats()
            analytics = enhanced_cache.get_analytics()
            return {
                "timestamp": time.time(),
                "hit_rate": stats.get("hit_rate", 0),
                "memory_usage_mb": stats.get("memory_usage_mb", 0),
                "compression_ratio": stats.get("compression_ratio", 1.0),
                "entry_count": stats.get("entry_count", 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache metrics: {e}")
            return {"timestamp": time.time(), "error": str(e)}
    
    async def _generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate AI-powered optimization recommendations"""
        recommendations = []
        
        try:
            cache_recs = await self._analyze_cache_optimization()
            recommendations.extend(cache_recs)
            
            perf_recs = await self._analyze_performance_optimization()
            recommendations.extend(perf_recs)
            
            recommendations.sort(key=lambda x: (x.priority, x.estimated_improvement), reverse=True)
            
            self.recommendations_cache = {
                "recommendations": recommendations,
                "generated_at": time.time(),
                "count": len(recommendations)
            }
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    async def _analyze_cache_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze cache performance and generate recommendations"""
        recommendations = []
        
        try:
            if not self.metrics_history["cache"]:
                return recommendations
            
            latest_cache = list(self.metrics_history["cache"])[-1]
            hit_rate = latest_cache.get("hit_rate", 0)
            
            if hit_rate < self.optimization_config["cache"]["target_hit_rate"]:
                recommendations.append(OptimizationRecommendation(
                    category="cache",
                    priority=8,
                    title="Improve Cache Hit Rate",
                    description=f"Current hit rate is {hit_rate:.1f}%, below target",
                    impact="high",
                    effort="medium",
                    estimated_improvement=15.0,
                    parameters={"current_hit_rate": hit_rate},
                    confidence=0.85
                ))
        except Exception as e:
            logger.error(f"Error analyzing cache optimization: {e}")
        
        return recommendations
    
    async def _analyze_performance_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze system performance and generate recommendations"""
        recommendations = []
        
        try:
            if not self.metrics_history["system"]:
                return recommendations
            
            latest_system = list(self.metrics_history["system"])[-1]
            cpu_percent = latest_system.get("cpu_percent", 0)
            
            if cpu_percent > self.optimization_config["performance"]["max_cpu_usage"]:
                recommendations.append(OptimizationRecommendation(
                    category="performance",
                    priority=9,
                    title="Reduce CPU Usage",
                    description=f"CPU usage is {cpu_percent:.1f}%, above threshold",
                    impact="high",
                    effort="high",
                    estimated_improvement=30.0,
                    parameters={"current_cpu_percent": cpu_percent},
                    confidence=0.90
                ))
        except Exception as e:
            logger.error(f"Error analyzing performance optimization: {e}")
        
        return recommendations
    
    async def _apply_automatic_optimizations(self, recommendations: List[OptimizationRecommendation]):
        """Apply automatic optimizations based on recommendations"""
        try:
            for recommendation in recommendations:
                if (recommendation.effort == "low" and 
                    recommendation.confidence > 0.8 and 
                    recommendation.impact in ["high", "medium"]):
                    
                    await self._apply_optimization(recommendation)
        except Exception as e:
            logger.error(f"Error applying automatic optimizations: {e}")
    
    async def _apply_optimization(self, recommendation: OptimizationRecommendation):
        """Apply a specific optimization"""
        try:
            if recommendation.category == "cache":
                await self._apply_cache_optimization(recommendation)
            
            self.optimization_history.append({
                "recommendation": recommendation,
                "applied_at": time.time(),
                "status": "applied"
            })
            
            logger.info(f"Applied optimization: {recommendation.title}")
        except Exception as e:
            logger.error(f"Error applying optimization {recommendation.title}: {e}")
    
    async def _apply_cache_optimization(self, recommendation: OptimizationRecommendation):
        """Apply cache-specific optimizations"""
        try:
            if "compression" in recommendation.title.lower():
                enhanced_cache.compression_level = enhanced_cache.compression_level.__class__.MAX
            elif "memory" in recommendation.title.lower():
                enhanced_cache.max_memory_mb = min(enhanced_cache.max_memory_mb * 1.2, 200)
            elif "hit rate" in recommendation.title.lower():
                enhanced_cache.max_size = min(enhanced_cache.max_size * 1.5, 5000)
        except Exception as e:
            logger.error(f"Error applying cache optimization: {e}")
    
    def get_current_recommendations(self) -> Dict[str, Any]:
        """Get current optimization recommendations"""
        return self.recommendations_cache
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        return self.optimization_history[-50:]


# Global AI optimizer instance
ai_optimizer = AIOptimizationEngine()