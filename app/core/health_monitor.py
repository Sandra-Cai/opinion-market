"""
Advanced health monitoring and system diagnostics
"""

import time
import psutil
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Comprehensive health monitoring system"""

    def __init__(self):
        self.start_time = time.time()
        self.health_checks: List[Dict[str, Any]] = []

    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "status": "healthy",
            "checks": {},
        }

        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        health_status["checks"]["cpu"] = {
            "status": (
                "healthy"
                if cpu_percent < 80
                else "warning" if cpu_percent < 95 else "critical"
            ),
            "value": cpu_percent,
            "threshold": 80,
        }

        # Memory check
        memory = psutil.virtual_memory()
        health_status["checks"]["memory"] = {
            "status": (
                "healthy"
                if memory.percent < 80
                else "warning" if memory.percent < 95 else "critical"
            ),
            "value": memory.percent,
            "used_gb": round(memory.used / (1024**3), 2),
            "total_gb": round(memory.total / (1024**3), 2),
            "threshold": 80,
        }

        # Disk check
        disk = psutil.disk_usage("/")
        health_status["checks"]["disk"] = {
            "status": (
                "healthy"
                if disk.percent < 80
                else "warning" if disk.percent < 95 else "critical"
            ),
            "value": disk.percent,
            "used_gb": round(disk.used / (1024**9), 2),
            "total_gb": round(disk.total / (1024**9), 2),
            "threshold": 80,
        }

        # Determine overall status
        critical_checks = [
            check
            for check in health_status["checks"].values()
            if check["status"] == "critical"
        ]
        warning_checks = [
            check
            for check in health_status["checks"].values()
            if check["status"] == "warning"
        ]

        if critical_checks:
            health_status["status"] = "critical"
        elif warning_checks:
            health_status["status"] = "warning"

        return health_status

    async def check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health metrics"""
        try:
            from app.main_simple import app
            from fastapi.testclient import TestClient

            client = TestClient(app)

            # Test critical endpoints
            endpoints = {
                "/": "root",
                "/health": "health",
                "/ready": "readiness",
                "/metrics": "metrics",
            }

            app_health = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "endpoints": {},
            }

            for endpoint, name in endpoints.items():
                try:
                    start_time = time.time()
                    response = client.get(endpoint)
                    response_time = (time.time() - start_time) * 1000

                    app_health["endpoints"][name] = {
                        "status": (
                            "healthy" if response.status_code == 200 else "unhealthy"
                        ),
                        "status_code": response.status_code,
                        "response_time_ms": round(response_time, 2),
                    }
                except Exception as e:
                    app_health["endpoints"][name] = {"status": "error", "error": str(e)}

            # Determine overall application status
            unhealthy_endpoints = [
                ep
                for ep in app_health["endpoints"].values()
                if ep["status"] in ["unhealthy", "error"]
            ]

            if unhealthy_endpoints:
                app_health["status"] = "unhealthy"

            return app_health

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
            }

    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        system_health = await self.check_system_health()
        app_health = await self.check_application_health()

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "system": system_health,
            "application": app_health,
            "summary": {
                "system_status": system_health["status"],
                "app_status": app_health["status"],
                "uptime_seconds": system_health["uptime"],
            },
        }


# Global health monitor instance
health_monitor = HealthMonitor()
