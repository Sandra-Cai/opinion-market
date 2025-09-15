"""
AI-powered optimization and intelligent features
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)


class AIOptimizer:
    """AI-powered system optimization"""

    def __init__(self):
        self.performance_history: List[Dict[str, Any]] = []
        self.optimization_rules: List[Dict[str, Any]] = []
        self.learning_enabled = True
        self.optimization_threshold = 0.1  # 10% improvement threshold

    async def analyze_performance_patterns(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance patterns using AI"""
        try:
            # Store current metrics
            self.performance_history.append(
                {"timestamp": datetime.now().isoformat(), "metrics": metrics}
            )

            # Keep only last 1000 entries
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

            # Analyze patterns
            patterns = await self._detect_patterns()

            # Generate recommendations
            recommendations = await self._generate_ai_recommendations(patterns, metrics)

            return {
                "patterns_detected": patterns,
                "recommendations": recommendations,
                "confidence_score": self._calculate_confidence_score(patterns),
                "optimization_potential": self._calculate_optimization_potential(
                    metrics
                ),
            }

        except Exception as e:
            logger.error(f"AI optimization analysis failed: {e}")
            return {"error": str(e)}

    async def _detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect performance patterns using machine learning"""
        if len(self.performance_history) < 10:
            return []

        patterns = []

        # Analyze response time patterns
        response_times = [
            entry["metrics"]
            .get("timers", {})
            .get("http_request_duration", {})
            .get("avg", 0)
            for entry in self.performance_history[-50:]  # Last 50 entries
        ]

        if response_times:
            # Detect trend
            trend = self._calculate_trend(response_times)
            if abs(trend) > 0.1:  # Significant trend
                patterns.append(
                    {
                        "type": "response_time_trend",
                        "trend": "increasing" if trend > 0 else "decreasing",
                        "magnitude": abs(trend),
                        "confidence": min(0.9, abs(trend) * 2),
                    }
                )

            # Detect anomalies
            anomalies = self._detect_anomalies(response_times)
            if anomalies:
                patterns.append(
                    {
                        "type": "response_time_anomaly",
                        "anomaly_count": len(anomalies),
                        "severity": "high" if len(anomalies) > 3 else "medium",
                        "confidence": 0.8,
                    }
                )

        # Analyze error patterns
        error_rates = [
            self._calculate_error_rate(entry["metrics"])
            for entry in self.performance_history[-50:]
        ]

        if error_rates:
            avg_error_rate = np.mean(error_rates)
            if avg_error_rate > 5:  # 5% error rate threshold
                patterns.append(
                    {
                        "type": "high_error_rate",
                        "average_error_rate": avg_error_rate,
                        "severity": "critical" if avg_error_rate > 10 else "high",
                        "confidence": 0.9,
                    }
                )

        return patterns

    async def _generate_ai_recommendations(
        self, patterns: List[Dict[str, Any]], current_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered optimization recommendations"""
        recommendations = []

        for pattern in patterns:
            if (
                pattern["type"] == "response_time_trend"
                and pattern["trend"] == "increasing"
            ):
                recommendations.append(
                    {
                        "type": "performance",
                        "priority": "high",
                        "title": "Response Time Degradation Detected",
                        "description": f"Response times are increasing with magnitude {pattern['magnitude']:.2f}",
                        "actions": [
                            "Implement caching for frequently accessed data",
                            "Optimize database queries",
                            "Consider horizontal scaling",
                            "Review and optimize slow endpoints",
                        ],
                        "confidence": pattern["confidence"],
                    }
                )

            elif pattern["type"] == "response_time_anomaly":
                recommendations.append(
                    {
                        "type": "reliability",
                        "priority": "medium",
                        "title": "Response Time Anomalies Detected",
                        "description": f"Found {pattern['anomaly_count']} response time anomalies",
                        "actions": [
                            "Investigate specific time periods with anomalies",
                            "Check for resource contention",
                            "Monitor external dependencies",
                            "Implement circuit breakers",
                        ],
                        "confidence": pattern["confidence"],
                    }
                )

            elif pattern["type"] == "high_error_rate":
                recommendations.append(
                    {
                        "type": "reliability",
                        "priority": "critical",
                        "title": "High Error Rate Detected",
                        "description": f"Average error rate is {pattern['average_error_rate']:.1f}%",
                        "actions": [
                            "Immediately investigate error sources",
                            "Implement proper error handling",
                            "Add monitoring and alerting",
                            "Consider rollback if recent deployment",
                        ],
                        "confidence": pattern["confidence"],
                    }
                )

        # Generate proactive recommendations based on current state
        current_response_time = (
            current_metrics.get("timers", {})
            .get("http_request_duration", {})
            .get("avg", 0)
        )
        if current_response_time > 0.5:  # 500ms
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "medium",
                    "title": "Current Response Time Optimization",
                    "description": f"Current response time is {current_response_time:.3f}s",
                    "actions": [
                        "Enable response compression",
                        "Implement request batching",
                        "Optimize serialization",
                        "Add response caching",
                    ],
                    "confidence": 0.7,
                }
            )

        return recommendations

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def _detect_anomalies(self, values: List[float]) -> List[int]:
        """Detect anomalies using statistical methods"""
        if len(values) < 3:
            return []

        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)

        # Z-score method
        z_scores = np.abs((values_array - mean) / std)
        threshold = 2.0  # 2 standard deviations

        anomalies = []
        for i, z_score in enumerate(z_scores):
            if z_score > threshold:
                anomalies.append(i)

        return anomalies

    def _calculate_error_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate error rate from metrics"""
        counters = metrics.get("counters", {})
        total_requests = counters.get("http_requests_total", 0)
        error_requests = counters.get("http_requests_500", 0)

        if total_requests == 0:
            return 0.0

        return (error_requests / total_requests) * 100

    def _calculate_confidence_score(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for AI analysis"""
        if not patterns:
            return 0.0

        confidence_scores = [pattern.get("confidence", 0.0) for pattern in patterns]
        return np.mean(confidence_scores)

    def _calculate_optimization_potential(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate potential optimization gains"""
        current_response_time = (
            metrics.get("timers", {}).get("http_request_duration", {}).get("avg", 0)
        )
        current_error_rate = self._calculate_error_rate(metrics)

        # Estimate optimization potential
        response_time_potential = (
            max(0, (current_response_time - 0.1) / current_response_time * 100)
            if current_response_time > 0.1
            else 0
        )
        error_rate_potential = (
            max(0, (current_error_rate - 1) / current_error_rate * 100)
            if current_error_rate > 1
            else 0
        )

        return {
            "response_time_improvement_potential": min(
                50, response_time_potential
            ),  # Cap at 50%
            "error_rate_improvement_potential": min(
                90, error_rate_potential
            ),  # Cap at 90%
            "overall_optimization_potential": (
                response_time_potential + error_rate_potential
            )
            / 2,
        }

    async def optimize_system_parameters(
        self, current_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """AI-powered system parameter optimization"""
        try:
            # Analyze current performance
            performance_analysis = await self.analyze_performance_patterns(
                current_config
            )

            # Generate optimized parameters
            optimized_params = {}

            # Optimize cache settings
            if "cache" in current_config:
                cache_config = current_config["cache"]
                optimized_params["cache"] = {
                    "ttl": self._optimize_cache_ttl(cache_config.get("ttl", 300)),
                    "max_size": self._optimize_cache_size(
                        cache_config.get("max_size", 10000)
                    ),
                }

            # Optimize database pool settings
            if "database" in current_config:
                db_config = current_config["database"]
                optimized_params["database"] = {
                    "pool_size": self._optimize_pool_size(
                        db_config.get("pool_size", 20)
                    ),
                    "max_overflow": self._optimize_max_overflow(
                        db_config.get("max_overflow", 30)
                    ),
                }

            return {
                "optimized_parameters": optimized_params,
                "expected_improvements": performance_analysis.get(
                    "optimization_potential", {}
                ),
                "confidence": performance_analysis.get("confidence_score", 0.0),
            }

        except Exception as e:
            logger.error(f"System parameter optimization failed: {e}")
            return {"error": str(e)}

    def _optimize_cache_ttl(self, current_ttl: int) -> int:
        """Optimize cache TTL based on usage patterns"""
        # Simple heuristic: increase TTL if system is stable, decrease if volatile
        if len(self.performance_history) < 10:
            return current_ttl

        recent_variance = self._calculate_performance_variance()
        if recent_variance < 0.1:  # Low variance, stable system
            return min(3600, current_ttl * 1.2)  # Increase TTL
        elif recent_variance > 0.3:  # High variance, volatile system
            return max(60, current_ttl * 0.8)  # Decrease TTL

        return current_ttl

    def _optimize_cache_size(self, current_size: int) -> int:
        """Optimize cache size based on usage patterns"""
        # Simple heuristic based on request volume
        if len(self.performance_history) < 5:
            return current_size

        recent_requests = [
            entry["metrics"].get("counters", {}).get("http_requests_total", 0)
            for entry in self.performance_history[-10:]
        ]

        avg_requests = np.mean(recent_requests)
        if avg_requests > 1000:  # High traffic
            return min(50000, int(current_size * 1.5))
        elif avg_requests < 100:  # Low traffic
            return max(1000, int(current_size * 0.7))

        return current_size

    def _optimize_pool_size(self, current_size: int) -> int:
        """Optimize database pool size"""
        if len(self.performance_history) < 5:
            return current_size

        # Analyze database response times
        db_response_times = [
            entry["metrics"]
            .get("timers", {})
            .get("database_query_duration", {})
            .get("avg", 0)
            for entry in self.performance_history[-10:]
        ]

        avg_db_time = np.mean(db_response_times)
        if avg_db_time > 0.1:  # Slow database queries
            return min(50, int(current_size * 1.3))
        elif avg_db_time < 0.01:  # Fast database queries
            return max(5, int(current_size * 0.8))

        return current_size

    def _optimize_max_overflow(self, current_overflow: int) -> int:
        """Optimize database max overflow"""
        if len(self.performance_history) < 5:
            return current_overflow

        # Analyze connection usage patterns
        recent_connections = [
            entry["metrics"].get("database", {}).get("active_connections", 0)
            for entry in self.performance_history[-10:]
        ]

        max_connections = max(recent_connections) if recent_connections else 0
        if max_connections > current_overflow * 0.8:  # High connection usage
            return min(100, int(current_overflow * 1.5))
        elif max_connections < current_overflow * 0.3:  # Low connection usage
            return max(5, int(current_overflow * 0.7))

        return current_overflow

    def _calculate_performance_variance(self) -> float:
        """Calculate performance variance from recent history"""
        if len(self.performance_history) < 5:
            return 0.0

        recent_response_times = [
            entry["metrics"]
            .get("timers", {})
            .get("http_request_duration", {})
            .get("avg", 0)
            for entry in self.performance_history[-10:]
        ]

        if not recent_response_times:
            return 0.0

        return np.var(recent_response_times)


# Global AI optimizer instance
ai_optimizer = AIOptimizer()
