"""
System Monitoring Dashboard
Provides comprehensive system monitoring and metrics
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional
import psutil
import time
import asyncio
import os
import sys
from datetime import datetime, timedelta
from collections import deque
import json

from app.core.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import text

router = APIRouter()

# Global metrics storage
system_metrics_history = deque(maxlen=1000)
performance_metrics_history = deque(maxlen=1000)

class SystemMonitor:
    """System monitoring and metrics collection"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_metrics_update = time.time()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # System uptime
            uptime = time.time() - self.start_time
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime,
                "uptime_human": str(timedelta(seconds=int(uptime))),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                "memory": {
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "percent": memory.percent,
                    "swap_total_bytes": swap.total,
                    "swap_used_bytes": swap.used,
                    "swap_percent": swap.percent
                },
                "disk": {
                    "total_bytes": disk.total,
                    "used_bytes": disk.used,
                    "free_bytes": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                    "read_bytes": disk_io.read_bytes if disk_io else 0,
                    "write_bytes": disk_io.write_bytes if disk_io else 0
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                "process": {
                    "pid": process.pid,
                    "memory_rss_bytes": process_memory.rss,
                    "memory_vms_bytes": process_memory.vms,
                    "cpu_percent": process_cpu,
                    "num_threads": process.num_threads(),
                    "create_time": process.create_time()
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "platform": sys.platform
                }
            }
            
            # Store in history
            system_metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error collecting system metrics: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get application performance metrics"""
        try:
            current_time = time.time()
            time_since_last_update = current_time - self.last_metrics_update
            
            # Calculate rates
            request_rate = self.request_count / time_since_last_update if time_since_last_update > 0 else 0
            error_rate = self.error_count / time_since_last_update if time_since_last_update > 0 else 0
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "requests": {
                    "total": self.request_count,
                    "rate_per_second": request_rate,
                    "errors": self.error_count,
                    "error_rate_per_second": error_rate,
                    "success_rate": ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count > 0 else 100
                },
                "response_times": {
                    "avg_ms": self._calculate_avg_response_time(),
                    "min_ms": self._calculate_min_response_time(),
                    "max_ms": self._calculate_max_response_time(),
                    "p95_ms": self._calculate_p95_response_time()
                },
                "memory_usage": {
                    "current_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                    "peak_mb": self._get_peak_memory_usage()
                }
            }
            
            # Store in history
            performance_metrics_history.append(metrics)
            
            # Reset counters
            self.request_count = 0
            self.error_count = 0
            self.last_metrics_update = current_time
            
            return metrics
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error collecting performance metrics: {str(e)}")
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from history"""
        if not performance_metrics_history:
            return 0.0
        
        total_time = sum(metric.get("response_times", {}).get("avg_ms", 0) for metric in performance_metrics_history)
        return total_time / len(performance_metrics_history)
    
    def _calculate_min_response_time(self) -> float:
        """Calculate minimum response time from history"""
        if not performance_metrics_history:
            return 0.0
        
        min_times = [metric.get("response_times", {}).get("min_ms", 0) for metric in performance_metrics_history]
        return min(min_times) if min_times else 0.0
    
    def _calculate_max_response_time(self) -> float:
        """Calculate maximum response time from history"""
        if not performance_metrics_history:
            return 0.0
        
        max_times = [metric.get("response_times", {}).get("max_ms", 0) for metric in performance_metrics_history]
        return max(max_times) if max_times else 0.0
    
    def _calculate_p95_response_time(self) -> float:
        """Calculate 95th percentile response time"""
        if not performance_metrics_history:
            return 0.0
        
        response_times = [metric.get("response_times", {}).get("avg_ms", 0) for metric in performance_metrics_history]
        response_times.sort()
        p95_index = int(len(response_times) * 0.95)
        return response_times[p95_index] if p95_index < len(response_times) else response_times[-1]
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage"""
        if not system_metrics_history:
            return 0.0
        
        memory_usages = [metric.get("process", {}).get("memory_rss_bytes", 0) for metric in system_metrics_history]
        return max(memory_usages) / 1024 / 1024 if memory_usages else 0.0
    
    def increment_request_count(self):
        """Increment request counter"""
        self.request_count += 1
    
    def increment_error_count(self):
        """Increment error counter"""
        self.error_count += 1

# Global system monitor instance
system_monitor = SystemMonitor()

@router.get("/metrics/system")
async def get_system_metrics():
    """Get current system metrics"""
    return system_monitor.get_system_metrics()

@router.get("/metrics/performance")
async def get_performance_metrics():
    """Get application performance metrics"""
    return system_monitor.get_performance_metrics()

@router.get("/metrics/history")
async def get_metrics_history(limit: int = 100):
    """Get historical metrics"""
    if limit > 1000:
        limit = 1000
    
    return {
        "system_metrics": list(system_metrics_history)[-limit:],
        "performance_metrics": list(performance_metrics_history)[-limit:]
    }

@router.get("/health/detailed")
async def get_detailed_health(db: Session = Depends(get_db)):
    """Get detailed system health information"""
    try:
        # System metrics
        system_metrics = system_monitor.get_system_metrics()
        
        # Database health
        db_health = await check_database_health(db)
        
        # Application health
        app_health = {
            "status": "healthy",
            "uptime_seconds": system_metrics["uptime_seconds"],
            "uptime_human": system_metrics["uptime_human"],
            "request_count": system_monitor.request_count,
            "error_count": system_monitor.error_count
        }
        
        # Overall health status
        overall_status = "healthy"
        if system_metrics["cpu"]["percent"] > 90:
            overall_status = "warning"
        if system_metrics["memory"]["percent"] > 90:
            overall_status = "warning"
        if db_health["status"] != "healthy":
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "system": system_metrics,
            "database": db_health,
            "application": app_health
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting detailed health: {str(e)}")

@router.get("/alerts")
async def get_system_alerts():
    """Get current system alerts"""
    alerts = []
    
    try:
        system_metrics = system_monitor.get_system_metrics()
        
        # CPU alerts
        if system_metrics["cpu"]["percent"] > 90:
            alerts.append({
                "type": "warning",
                "component": "cpu",
                "message": f"High CPU usage: {system_metrics['cpu']['percent']:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif system_metrics["cpu"]["percent"] > 80:
            alerts.append({
                "type": "info",
                "component": "cpu",
                "message": f"Elevated CPU usage: {system_metrics['cpu']['percent']:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Memory alerts
        if system_metrics["memory"]["percent"] > 90:
            alerts.append({
                "type": "warning",
                "component": "memory",
                "message": f"High memory usage: {system_metrics['memory']['percent']:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif system_metrics["memory"]["percent"] > 80:
            alerts.append({
                "type": "info",
                "component": "memory",
                "message": f"Elevated memory usage: {system_metrics['memory']['percent']:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Disk alerts
        if system_metrics["disk"]["percent"] > 90:
            alerts.append({
                "type": "warning",
                "component": "disk",
                "message": f"High disk usage: {system_metrics['disk']['percent']:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif system_metrics["disk"]["percent"] > 80:
            alerts.append({
                "type": "info",
                "component": "disk",
                "message": f"Elevated disk usage: {system_metrics['disk']['percent']:.1f}%",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Error rate alerts
        if system_monitor.error_count > 0:
            error_rate = (system_monitor.error_count / system_monitor.request_count * 100) if system_monitor.request_count > 0 else 0
            if error_rate > 10:
                alerts.append({
                    "type": "warning",
                    "component": "application",
                    "message": f"High error rate: {error_rate:.1f}%",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@router.get("/dashboard", response_class=HTMLResponse)
async def get_monitoring_dashboard():
    """Get HTML monitoring dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>System Monitoring Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
            .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
            .metric-unit { font-size: 14px; color: #7f8c8d; }
            .status-healthy { color: #27ae60; }
            .status-warning { color: #f39c12; }
            .status-error { color: #e74c3c; }
            .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            .refresh-btn:hover { background: #2980b9; }
            .alerts { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin-top: 20px; }
            .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
            .alert-warning { background: #fff3cd; border-left: 4px solid #f39c12; }
            .alert-info { background: #d1ecf1; border-left: 4px solid #17a2b8; }
        </style>
        <script>
            function refreshMetrics() {
                location.reload();
            }
            
            // Auto-refresh every 30 seconds
            setInterval(refreshMetrics, 30000);
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 System Monitoring Dashboard</h1>
                <p>Real-time system metrics and health monitoring</p>
                <button class="refresh-btn" onclick="refreshMetrics()">🔄 Refresh</button>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">CPU Usage</div>
                    <div class="metric-value" id="cpu-usage">Loading...</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Memory Usage</div>
                    <div class="metric-value" id="memory-usage">Loading...</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Disk Usage</div>
                    <div class="metric-value" id="disk-usage">Loading...</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">System Uptime</div>
                    <div class="metric-value" id="uptime">Loading...</div>
                    <div class="metric-unit">hours</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Request Rate</div>
                    <div class="metric-value" id="request-rate">Loading...</div>
                    <div class="metric-unit">req/sec</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Error Rate</div>
                    <div class="metric-value" id="error-rate">Loading...</div>
                    <div class="metric-unit">%</div>
                </div>
            </div>
            
            <div class="alerts" id="alerts">
                <h3>🚨 System Alerts</h3>
                <div id="alerts-content">Loading alerts...</div>
            </div>
        </div>
        
        <script>
            async function loadMetrics() {
                try {
                    // Load system metrics
                    const systemResponse = await fetch('/api/v1/system-monitor/metrics/system');
                    const systemMetrics = await systemResponse.json();
                    
                    // Load performance metrics
                    const perfResponse = await fetch('/api/v1/system-monitor/metrics/performance');
                    const perfMetrics = await perfResponse.json();
                    
                    // Load alerts
                    const alertsResponse = await fetch('/api/v1/system-monitor/alerts');
                    const alerts = await alertsResponse.json();
                    
                    // Update UI
                    document.getElementById('cpu-usage').textContent = systemMetrics.cpu.percent.toFixed(1);
                    document.getElementById('memory-usage').textContent = systemMetrics.memory.percent.toFixed(1);
                    document.getElementById('disk-usage').textContent = systemMetrics.disk.percent.toFixed(1);
                    document.getElementById('uptime').textContent = (systemMetrics.uptime_seconds / 3600).toFixed(1);
                    document.getElementById('request-rate').textContent = perfMetrics.requests.rate_per_second.toFixed(2);
                    document.getElementById('error-rate').textContent = perfMetrics.requests.success_rate.toFixed(1);
                    
                    // Update alerts
                    const alertsContent = document.getElementById('alerts-content');
                    if (alerts.alerts.length === 0) {
                        alertsContent.innerHTML = '<div class="alert alert-info">✅ No alerts - System is healthy</div>';
                    } else {
                        alertsContent.innerHTML = alerts.alerts.map(alert => 
                            `<div class="alert alert-${alert.type}">${alert.message}</div>`
                        ).join('');
                    }
                    
                } catch (error) {
                    console.error('Error loading metrics:', error);
                }
            }
            
            // Load metrics on page load
            loadMetrics();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

async def check_database_health(db: Session) -> Dict[str, Any]:
    """Check database health"""
    try:
        # Test database connection
        result = db.execute(text("SELECT 1 as test")).fetchone()
        
        if result and result[0] == 1:
            # Get table count
            result = db.execute(text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")).fetchone()
            table_count = result[0] if result else 0
            
            return {
                "status": "healthy",
                "connection": "ok",
                "table_count": table_count,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "unhealthy",
                "connection": "failed",
                "error": "Database query failed",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Middleware to track requests
@router.middleware("http")
async def track_requests(request, call_next):
    """Middleware to track request metrics"""
    system_monitor.increment_request_count()
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        system_monitor.increment_error_count()
        raise e

