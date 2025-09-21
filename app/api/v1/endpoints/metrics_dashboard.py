"""
Real-time Metrics Dashboard
Provides live system metrics and performance data with WebSocket support
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, List, Any, Optional
import asyncio
import json
import psutil
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from app.core.database import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter()

# Global metrics storage
metrics_history = deque(maxlen=1000)
active_connections: List[WebSocket] = []
metrics_cache: Dict[str, Any] = {}
last_metrics_update = time.time()

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Alert:
    """Represents a system alert"""
    id: str
    timestamp: datetime
    level: AlertLevel
    metric: str
    value: float
    threshold: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class MetricThreshold:
    """Defines thresholds for a metric"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    enabled: bool = True

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.alert_listeners: List[Callable[[Alert], None]] = []
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default metric thresholds"""
        self.thresholds = {
            "cpu_percent": MetricThreshold("cpu_percent", 70.0, 85.0, 95.0),
            "memory_percent": MetricThreshold("memory_percent", 80.0, 90.0, 95.0),
            "disk_percent": MetricThreshold("disk_percent", 85.0, 95.0, 98.0),
            "error_rate": MetricThreshold("error_rate", 5.0, 10.0, 20.0),
            "response_time": MetricThreshold("response_time", 1000.0, 2000.0, 5000.0),
            "request_rate": MetricThreshold("request_rate", 1000.0, 2000.0, 5000.0),
        }
    
    def add_alert_listener(self, listener: Callable[[Alert], None]):
        """Add an alert listener"""
        self.alert_listeners.append(listener)
    
    def check_metrics(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        current_time = datetime.utcnow()
        
        # Check CPU usage
        if "system" in metrics and "cpu" in metrics["system"]:
            cpu_percent = metrics["system"]["cpu"]["percent"]
            self._check_threshold("cpu_percent", cpu_percent, current_time)
        
        # Check memory usage
        if "system" in metrics and "memory" in metrics["system"]:
            memory_percent = metrics["system"]["memory"]["percent"]
            self._check_threshold("memory_percent", memory_percent, current_time)
        
        # Check disk usage
        if "system" in metrics and "disk" in metrics["system"]:
            disk_percent = metrics["system"]["disk"]["percent"]
            self._check_threshold("disk_percent", disk_percent, current_time)
        
        # Check error rate
        if "application" in metrics and "requests" in metrics["application"]:
            error_rate = 100 - metrics["application"]["requests"]["success_rate"]
            self._check_threshold("error_rate", error_rate, current_time)
        
        # Check response time
        if "application" in metrics and "response_times" in metrics["application"]:
            response_time = metrics["application"]["response_times"]["avg_ms"]
            self._check_threshold("response_time", response_time, current_time)
    
    def _check_threshold(self, metric_name: str, value: float, timestamp: datetime):
        """Check a single metric against its thresholds"""
        if metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_name]
        if not threshold.enabled:
            return
        
        alert_level = None
        threshold_value = None
        
        if value >= threshold.emergency_threshold:
            alert_level = AlertLevel.EMERGENCY
            threshold_value = threshold.emergency_threshold
        elif value >= threshold.critical_threshold:
            alert_level = AlertLevel.CRITICAL
            threshold_value = threshold.critical_threshold
        elif value >= threshold.warning_threshold:
            alert_level = AlertLevel.WARNING
            threshold_value = threshold.warning_threshold
        
        if alert_level:
            alert_id = f"{metric_name}_{alert_level.value}"
            
            # Check if alert already exists
            if alert_id in self.active_alerts:
                return  # Alert already active
            
            # Create new alert
            alert = Alert(
                id=alert_id,
                timestamp=timestamp,
                level=alert_level,
                metric=metric_name,
                value=value,
                threshold=threshold_value,
                message=f"{metric_name} is {value:.1f}, exceeding {alert_level.value} threshold of {threshold_value}"
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Notify listeners
            for listener in self.alert_listeners:
                try:
                    listener(alert)
                except Exception as e:
                    logger.error(f"Error in alert listener: {e}")
            
            logger.warning(f"Alert triggered: {alert.message}")
        else:
            # Check if we need to resolve existing alerts
            alert_id = f"{metric_name}_"
            for level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                full_alert_id = f"{metric_name}_{level.value}"
                if full_alert_id in self.active_alerts:
                    alert = self.active_alerts[full_alert_id]
                    alert.resolved = True
                    alert.resolved_at = timestamp
                    del self.active_alerts[full_alert_id]
                    logger.info(f"Alert resolved: {alert.message}")
                    break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return self.alert_history[-limit:]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            logger.info(f"Alert manually resolved: {alert.message}")
            return True
        return False

# Global alert manager
alert_manager = AlertManager()

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.response_times = deque(maxlen=100)
        self.endpoint_metrics = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
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
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime,
                "uptime_human": str(timedelta(seconds=int(uptime))),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None,
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
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
                }
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        current_time = time.time()
        time_since_start = current_time - self.start_time
        
        # Calculate rates
        request_rate = self.request_count / time_since_start if time_since_start > 0 else 0
        error_rate = self.error_count / time_since_start if time_since_start > 0 else 0
        
        # Calculate response time statistics
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)
            p95_response_time = sorted(self.response_times)[int(len(self.response_times) * 0.95)]
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "requests": {
                "total": self.request_count,
                "rate_per_second": request_rate,
                "errors": self.error_count,
                "error_rate_per_second": error_rate,
                "success_rate": ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count > 0 else 100
            },
            "response_times": {
                "avg_ms": avg_response_time,
                "min_ms": min_response_time,
                "max_ms": max_response_time,
                "p95_ms": p95_response_time
            },
            "endpoints": dict(self.endpoint_metrics)
        }
    
    def record_request(self, endpoint: str, response_time: float, is_error: bool = False):
        """Record a request metric"""
        self.request_count += 1
        if is_error:
            self.error_count += 1
        
        self.response_times.append(response_time)
        
        # Update endpoint metrics
        self.endpoint_metrics[endpoint]["count"] += 1
        self.endpoint_metrics[endpoint]["total_time"] += response_time
        if is_error:
            self.endpoint_metrics[endpoint]["errors"] += 1

# Global metrics collector
metrics_collector = MetricsCollector()

@router.get("/metrics/current")
async def get_current_metrics():
    """Get current system and application metrics"""
    system_metrics = metrics_collector.get_system_metrics()
    app_metrics = metrics_collector.get_application_metrics()
    
    return {
        "system": system_metrics,
        "application": app_metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics/history")
async def get_metrics_history(limit: int = 100):
    """Get historical metrics"""
    if limit > 1000:
        limit = 1000
    
    return {
        "metrics": list(metrics_history)[-limit:],
        "count": len(metrics_history),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics/endpoints")
async def get_endpoint_metrics():
    """Get endpoint-specific metrics"""
    endpoint_metrics = {}
    
    for endpoint, metrics in metrics_collector.endpoint_metrics.items():
        endpoint_metrics[endpoint] = {
            "request_count": metrics["count"],
            "avg_response_time": metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0,
            "error_count": metrics["errors"],
            "error_rate": (metrics["errors"] / metrics["count"] * 100) if metrics["count"] > 0 else 0
        }
    
    return {
        "endpoints": endpoint_metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/alerts/active")
async def get_active_alerts():
    """Get all active alerts"""
    alerts = alert_manager.get_active_alerts()
    return {
        "alerts": [
            {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "message": alert.message,
                "resolved": alert.resolved
            }
            for alert in alerts
        ],
        "count": len(alerts),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/alerts/history")
async def get_alert_history(limit: int = 100):
    """Get alert history"""
    if limit > 1000:
        limit = 1000
    
    alerts = alert_manager.get_alert_history(limit)
    return {
        "alerts": [
            {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold,
                "message": alert.message,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ],
        "count": len(alerts),
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Manually resolve an alert"""
    success = alert_manager.resolve_alert(alert_id)
    if success:
        return {"message": f"Alert {alert_id} resolved successfully"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@router.get("/thresholds")
async def get_thresholds():
    """Get current metric thresholds"""
    thresholds = {}
    for metric_name, threshold in alert_manager.thresholds.items():
        thresholds[metric_name] = {
            "warning_threshold": threshold.warning_threshold,
            "critical_threshold": threshold.critical_threshold,
            "emergency_threshold": threshold.emergency_threshold,
            "enabled": threshold.enabled
        }
    
    return {
        "thresholds": thresholds,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/thresholds/{metric_name}")
async def update_threshold(
    metric_name: str,
    warning_threshold: Optional[float] = None,
    critical_threshold: Optional[float] = None,
    emergency_threshold: Optional[float] = None,
    enabled: Optional[bool] = None
):
    """Update metric thresholds"""
    if metric_name not in alert_manager.thresholds:
        raise HTTPException(status_code=404, detail="Metric not found")
    
    threshold = alert_manager.thresholds[metric_name]
    
    if warning_threshold is not None:
        threshold.warning_threshold = warning_threshold
    if critical_threshold is not None:
        threshold.critical_threshold = critical_threshold
    if emergency_threshold is not None:
        threshold.emergency_threshold = emergency_threshold
    if enabled is not None:
        threshold.enabled = enabled
    
    return {
        "message": f"Thresholds updated for {metric_name}",
        "threshold": {
            "warning_threshold": threshold.warning_threshold,
            "critical_threshold": threshold.critical_threshold,
            "emergency_threshold": threshold.emergency_threshold,
            "enabled": threshold.enabled
        }
    }

@router.websocket("/metrics/ws")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Get current metrics
            system_metrics = metrics_collector.get_system_metrics()
            app_metrics = metrics_collector.get_application_metrics()
            
            metrics_data = {
                "type": "metrics_update",
                "system": system_metrics,
                "application": app_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send metrics to client
            await websocket.send_text(json.dumps(metrics_data))
            
            # Wait before next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@router.get("/dashboard", response_class=HTMLResponse)
async def get_metrics_dashboard():
    """Get HTML metrics dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Metrics Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
            .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
            .metric-unit { font-size: 14px; color: #7f8c8d; }
            .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-healthy { background-color: #27ae60; }
            .status-warning { background-color: #f39c12; }
            .status-error { background-color: #e74c3c; }
            .connection-status { position: fixed; top: 20px; right: 20px; padding: 10px; border-radius: 5px; }
            .connected { background-color: #27ae60; color: white; }
            .disconnected { background-color: #e74c3c; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Real-time Metrics Dashboard</h1>
                <p>Live system and application metrics</p>
                <div id="connection-status" class="connection-status disconnected">
                    <span id="connection-indicator" class="status-indicator"></span>
                    <span id="connection-text">Disconnected</span>
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">CPU Usage</div>
                    <div class="metric-value" id="cpu-usage">--</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Memory Usage</div>
                    <div class="metric-value" id="memory-usage">--</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Disk Usage</div>
                    <div class="metric-value" id="disk-usage">--</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Request Rate</div>
                    <div class="metric-value" id="request-rate">--</div>
                    <div class="metric-unit">req/sec</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Error Rate</div>
                    <div class="metric-value" id="error-rate">--</div>
                    <div class="metric-unit">%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Response Time</div>
                    <div class="metric-value" id="response-time">--</div>
                    <div class="metric-unit">ms</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>CPU Usage Over Time</h3>
                <canvas id="cpu-chart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Memory Usage Over Time</h3>
                <canvas id="memory-chart"></canvas>
            </div>
            
            <div class="chart-container">
                <h3>Request Rate Over Time</h3>
                <canvas id="request-chart"></canvas>
            </div>
        </div>
        
        <script>
            // WebSocket connection
            let ws = null;
            let reconnectInterval = null;
            
            // Chart data
            const chartData = {
                cpu: { labels: [], data: [] },
                memory: { labels: [], data: [] },
                requests: { labels: [], data: [] }
            };
            
            // Initialize charts
            const cpuChart = new Chart(document.getElementById('cpu-chart'), {
                type: 'line',
                data: {
                    labels: chartData.cpu.labels,
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: chartData.cpu.data,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, max: 100 }
                    }
                }
            });
            
            const memoryChart = new Chart(document.getElementById('memory-chart'), {
                type: 'line',
                data: {
                    labels: chartData.memory.labels,
                    datasets: [{
                        label: 'Memory Usage (%)',
                        data: chartData.memory.data,
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, max: 100 }
                    }
                }
            });
            
            const requestChart = new Chart(document.getElementById('request-chart'), {
                type: 'line',
                data: {
                    labels: chartData.requests.labels,
                    datasets: [{
                        label: 'Request Rate (req/sec)',
                        data: chartData.requests.data,
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
            
            function connectWebSocket() {
                ws = new WebSocket(`ws://${window.location.host}/api/v1/metrics-dashboard/metrics/ws`);
                
                ws.onopen = function() {
                    updateConnectionStatus(true);
                    if (reconnectInterval) {
                        clearInterval(reconnectInterval);
                        reconnectInterval = null;
                    }
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateMetrics(data);
                };
                
                ws.onclose = function() {
                    updateConnectionStatus(false);
                    if (!reconnectInterval) {
                        reconnectInterval = setInterval(connectWebSocket, 5000);
                    }
                };
                
                ws.onerror = function() {
                    updateConnectionStatus(false);
                };
            }
            
            function updateConnectionStatus(connected) {
                const status = document.getElementById('connection-status');
                const indicator = document.getElementById('connection-indicator');
                const text = document.getElementById('connection-text');
                
                if (connected) {
                    status.className = 'connection-status connected';
                    indicator.className = 'status-indicator status-healthy';
                    text.textContent = 'Connected';
                } else {
                    status.className = 'connection-status disconnected';
                    indicator.className = 'status-indicator status-error';
                    text.textContent = 'Disconnected';
                }
            }
            
            function updateMetrics(data) {
                // Update metric cards
                document.getElementById('cpu-usage').textContent = data.system.cpu.percent.toFixed(1);
                document.getElementById('memory-usage').textContent = data.system.memory.percent.toFixed(1);
                document.getElementById('disk-usage').textContent = data.system.disk.percent.toFixed(1);
                document.getElementById('request-rate').textContent = data.application.requests.rate_per_second.toFixed(2);
                document.getElementById('error-rate').textContent = (100 - data.application.requests.success_rate).toFixed(1);
                document.getElementById('response-time').textContent = data.application.response_times.avg_ms.toFixed(1);
                
                // Update charts
                const timestamp = new Date(data.timestamp).toLocaleTimeString();
                
                // CPU chart
                chartData.cpu.labels.push(timestamp);
                chartData.cpu.data.push(data.system.cpu.percent);
                if (chartData.cpu.labels.length > 20) {
                    chartData.cpu.labels.shift();
                    chartData.cpu.data.shift();
                }
                cpuChart.update();
                
                // Memory chart
                chartData.memory.labels.push(timestamp);
                chartData.memory.data.push(data.system.memory.percent);
                if (chartData.memory.labels.length > 20) {
                    chartData.memory.labels.shift();
                    chartData.memory.data.shift();
                }
                memoryChart.update();
                
                // Request chart
                chartData.requests.labels.push(timestamp);
                chartData.requests.data.push(data.application.requests.rate_per_second);
                if (chartData.requests.labels.length > 20) {
                    chartData.requests.labels.shift();
                    chartData.requests.data.shift();
                }
                requestChart.update();
            }
            
            // Connect on page load
            connectWebSocket();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Background task to collect metrics
async def collect_metrics_background():
    """Background task to collect and store metrics"""
    while True:
        try:
            system_metrics = metrics_collector.get_system_metrics()
            app_metrics = metrics_collector.get_application_metrics()
            
            combined_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": system_metrics,
                "application": app_metrics
            }
            
            # Check for alerts
            alert_manager.check_metrics(combined_metrics)
            
            metrics_history.append(combined_metrics)
            
            # Broadcast to WebSocket connections
            if active_connections:
                # Get active alerts for the broadcast
                active_alerts = alert_manager.get_active_alerts()
                
                metrics_data = {
                    "type": "metrics_update",
                    **combined_metrics,
                    "alerts": [
                        {
                            "id": alert.id,
                            "level": alert.level.value,
                            "metric": alert.metric,
                            "value": alert.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in active_alerts
                    ]
                }
                
                disconnected = []
                for websocket in active_connections:
                    try:
                        await websocket.send_text(json.dumps(metrics_data))
                    except:
                        disconnected.append(websocket)
                
                # Remove disconnected clients
                for websocket in disconnected:
                    active_connections.remove(websocket)
            
            await asyncio.sleep(5)  # Collect metrics every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            await asyncio.sleep(5)

# Background task will be started when the module is imported in the main application
# asyncio.create_task(collect_metrics_background())
