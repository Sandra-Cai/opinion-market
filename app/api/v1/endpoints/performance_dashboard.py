"""
Advanced Performance Dashboard API
Provides real-time performance monitoring and analytics
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional
import asyncio
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

from app.core.auth import get_current_user
from app.models.user import User
from app.core.enhanced_cache import enhanced_cache
from app.core.performance_monitor import performance_monitor
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter()

# Global storage for real-time metrics
real_time_metrics = {
    "system": deque(maxlen=1000),
    "cache": deque(maxlen=1000),
    "database": deque(maxlen=1000),
    "api": deque(maxlen=1000),
    "errors": deque(maxlen=500)
}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []


class PerformanceCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.is_collecting = False
        self.collection_task = None
    
    async def start_collection(self):
        """Start collecting performance metrics"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_task = asyncio.create_task(self._collect_metrics_loop())
    
    async def stop_collection(self):
        """Stop collecting performance metrics"""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.is_collecting:
            try:
                await self._collect_system_metrics()
                await self._collect_cache_metrics()
                await self._collect_database_metrics()
                await self._collect_api_metrics()
                
                # Broadcast to all connected clients
                await self._broadcast_metrics()
                
                await asyncio.sleep(1)  # Collect every second
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            system_metrics = {
                "timestamp": time.time(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }
            
            real_time_metrics["system"].append(system_metrics)
            self.metrics_history["system"].append(system_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}", exc_info=True)
    
    async def _collect_cache_metrics(self):
        """Collect cache performance metrics"""
        try:
            cache_stats = enhanced_cache.get_stats()
            cache_analytics = enhanced_cache.get_analytics()
            memory_info = enhanced_cache.get_memory_usage()
            
            cache_metrics = {
                "timestamp": time.time(),
                "stats": cache_stats,
                "analytics": cache_analytics.__dict__ if cache_analytics else None,
                "memory": memory_info,
                "health": await enhanced_cache.health_check()
            }
            
            real_time_metrics["cache"].append(cache_metrics)
            self.metrics_history["cache"].append(cache_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}", exc_info=True)
    
    async def _collect_database_metrics(self):
        """Collect database performance metrics"""
        try:
            # Get database connection info
            pool = engine.pool
            db_metrics = {
                "timestamp": time.time(),
                "pool": {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                }
            }
            
            # Get some basic query performance
            start_time = time.time()
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                db_metrics["query_time"] = (time.time() - start_time) * 1000  # ms
            except Exception as e:
                db_metrics["query_time"] = -1
                db_metrics["error"] = str(e)
            
            real_time_metrics["database"].append(db_metrics)
            self.metrics_history["database"].append(db_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}", exc_info=True)
    
    async def _collect_api_metrics(self):
        """Collect API performance metrics"""
        try:
            # This would typically come from middleware or request tracking
            # For now, we'll simulate some metrics
            api_metrics = {
                "timestamp": time.time(),
                "requests_per_second": 0,  # Would be calculated from actual request data
                "average_response_time": 0,
                "error_rate": 0,
                "active_connections": len(active_connections)
            }
            
            real_time_metrics["api"].append(api_metrics)
            self.metrics_history["api"].append(api_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}", exc_info=True)
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all connected WebSocket clients"""
        if not active_connections:
            return
        
        # Get latest metrics
        latest_metrics = {
            "system": list(real_time_metrics["system"])[-1] if real_time_metrics["system"] else None,
            "cache": list(real_time_metrics["cache"])[-1] if real_time_metrics["cache"] else None,
            "database": list(real_time_metrics["database"])[-1] if real_time_metrics["database"] else None,
            "api": list(real_time_metrics["api"])[-1] if real_time_metrics["api"] else None,
            "timestamp": time.time()
        }
        
        # Send to all connected clients
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(latest_metrics))
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)


# Global performance collector
performance_collector = PerformanceCollector()


@router.get("/dashboard")
async def get_dashboard_html():
    """Serve the performance dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Opinion Market - Performance Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .dashboard {
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
            }
            .metric-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #2c3e50;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }
            .metric-subtitle {
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 5px;
            }
            .chart-container {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            .status-healthy { background-color: #27ae60; }
            .status-warning { background-color: #f39c12; }
            .status-error { background-color: #e74c3c; }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="connection-status" id="connectionStatus">
            <span class="status-indicator status-error"></span>
            Connecting...
        </div>
        
        <div class="dashboard">
            <div class="header">
                <h1>🚀 Opinion Market Performance Dashboard</h1>
                <p>Real-time system monitoring and analytics</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">CPU Usage</div>
                    <div class="metric-value" id="cpuUsage">0%</div>
                    <div class="metric-subtitle">System CPU utilization</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Memory Usage</div>
                    <div class="metric-value" id="memoryUsage">0%</div>
                    <div class="metric-subtitle">System memory utilization</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Cache Hit Rate</div>
                    <div class="metric-value" id="cacheHitRate">0%</div>
                    <div class="metric-subtitle">Cache performance</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Database Pool</div>
                    <div class="metric-value" id="dbPool">0/0</div>
                    <div class="metric-subtitle">Active connections</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Response Time</div>
                    <div class="metric-value" id="responseTime">0ms</div>
                    <div class="metric-subtitle">Average API response</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Active Connections</div>
                    <div class="metric-value" id="activeConnections">0</div>
                    <div class="metric-subtitle">WebSocket connections</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="metric-title">System Performance Over Time</div>
                <canvas id="systemChart" width="400" height="200"></canvas>
            </div>
            
            <div class="chart-container">
                <div class="metric-title">Cache Performance</div>
                <canvas id="cacheChart" width="400" height="200"></canvas>
            </div>
        </div>
        
        <script>
            // WebSocket connection
            let ws;
            let systemChart, cacheChart;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/v1/performance-dashboard/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    document.getElementById('connectionStatus').innerHTML = 
                        '<span class="status-indicator status-healthy"></span>Connected';
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                ws.onclose = function() {
                    document.getElementById('connectionStatus').innerHTML = 
                        '<span class="status-indicator status-error"></span>Disconnected';
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onerror = function() {
                    document.getElementById('connectionStatus').innerHTML = 
                        '<span class="status-indicator status-error"></span>Error';
                };
            }
            
            function updateDashboard(data) {
                // Update metric cards
                if (data.system) {
                    document.getElementById('cpuUsage').textContent = data.system.cpu.percent.toFixed(1) + '%';
                    document.getElementById('memoryUsage').textContent = data.system.memory.percent.toFixed(1) + '%';
                }
                
                if (data.cache && data.cache.stats) {
                    document.getElementById('cacheHitRate').textContent = data.cache.stats.hit_rate.toFixed(1) + '%';
                }
                
                if (data.database && data.database.pool) {
                    const pool = data.database.pool;
                    document.getElementById('dbPool').textContent = `${pool.checked_out}/${pool.size}`;
                }
                
                if (data.api) {
                    document.getElementById('responseTime').textContent = data.api.average_response_time.toFixed(1) + 'ms';
                    document.getElementById('activeConnections').textContent = data.api.active_connections;
                }
                
                // Update charts
                updateCharts(data);
            }
            
            function updateCharts(data) {
                const now = new Date();
                
                // System chart
                if (systemChart && data.system) {
                    systemChart.data.labels.push(now.toLocaleTimeString());
                    systemChart.data.datasets[0].data.push(data.system.cpu.percent);
                    systemChart.data.datasets[1].data.push(data.system.memory.percent);
                    
                    if (systemChart.data.labels.length > 50) {
                        systemChart.data.labels.shift();
                        systemChart.data.datasets[0].data.shift();
                        systemChart.data.datasets[1].data.shift();
                    }
                    
                    systemChart.update('none');
                }
                
                // Cache chart
                if (cacheChart && data.cache && data.cache.stats) {
                    cacheChart.data.labels.push(now.toLocaleTimeString());
                    cacheChart.data.datasets[0].data.push(data.cache.stats.hit_rate);
                    cacheChart.data.datasets[1].data.push(data.cache.stats.memory_usage_mb);
                    
                    if (cacheChart.data.labels.length > 50) {
                        cacheChart.data.labels.shift();
                        cacheChart.data.datasets[0].data.shift();
                        cacheChart.data.datasets[1].data.shift();
                    }
                    
                    cacheChart.update('none');
                }
            }
            
            // Initialize charts
            function initCharts() {
                // System performance chart
                const systemCtx = document.getElementById('systemChart').getContext('2d');
                systemChart = new Chart(systemCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'CPU %',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }, {
                            label: 'Memory %',
                            data: [],
                            borderColor: 'rgb(255, 99, 132)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
                
                // Cache performance chart
                const cacheCtx = document.getElementById('cacheChart').getContext('2d');
                cacheChart = new Chart(cacheCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Hit Rate %',
                            data: [],
                            borderColor: 'rgb(54, 162, 235)',
                            tension: 0.1
                        }, {
                            label: 'Memory Usage (MB)',
                            data: [],
                            borderColor: 'rgb(255, 205, 86)',
                            tension: 0.1,
                            yAxisID: 'y1'
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                beginAtZero: true,
                                max: 100
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                beginAtZero: true
                            }
                        }
                    }
                });
            }
            
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                initCharts();
                connectWebSocket();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Start metrics collection if not already running
        if not performance_collector.is_collecting:
            await performance_collector.start_collection()
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        if websocket in active_connections:
            active_connections.remove(websocket)


@router.get("/metrics/current")
async def get_current_metrics(current_user: User = Depends(get_current_user)):
    """Get current performance metrics"""
    try:
        return {
            "success": True,
            "data": {
                "system": list(real_time_metrics["system"])[-1] if real_time_metrics["system"] else None,
                "cache": list(real_time_metrics["cache"])[-1] if real_time_metrics["cache"] else None,
                "database": list(real_time_metrics["database"])[-1] if real_time_metrics["database"] else None,
                "api": list(real_time_metrics["api"])[-1] if real_time_metrics["api"] else None,
                "timestamp": time.time()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/metrics/history")
async def get_metrics_history(
    hours: int = 1,
    current_user: User = Depends(get_current_user)
):
    """Get historical performance metrics"""
    try:
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_metrics = {}
        for metric_type, data in real_time_metrics.items():
            filtered_metrics[metric_type] = [
                item for item in data 
                if item.get("timestamp", 0) >= cutoff_time
            ]
        
        return {
            "success": True,
            "data": filtered_metrics,
            "hours": hours,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics history: {str(e)}")


@router.get("/alerts")
async def get_performance_alerts(current_user: User = Depends(get_current_user)):
    """Get current performance alerts"""
    try:
        alerts = []
        
        # Check system metrics
        if real_time_metrics["system"]:
            latest_system = list(real_time_metrics["system"])[-1]
            
            if latest_system["cpu"]["percent"] > 80:
                alerts.append({
                    "type": "warning",
                    "category": "system",
                    "message": f"High CPU usage: {latest_system['cpu']['percent']:.1f}%",
                    "timestamp": latest_system["timestamp"]
                })
            
            if latest_system["memory"]["percent"] > 85:
                alerts.append({
                    "type": "critical",
                    "category": "system",
                    "message": f"High memory usage: {latest_system['memory']['percent']:.1f}%",
                    "timestamp": latest_system["timestamp"]
                })
        
        # Check cache metrics
        if real_time_metrics["cache"]:
            latest_cache = list(real_time_metrics["cache"])[-1]
            
            if latest_cache["stats"]["hit_rate"] < 70:
                alerts.append({
                    "type": "warning",
                    "category": "cache",
                    "message": f"Low cache hit rate: {latest_cache['stats']['hit_rate']:.1f}%",
                    "timestamp": latest_cache["timestamp"]
                })
        
        return {
            "success": True,
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post("/start-monitoring")
async def start_monitoring(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Start performance monitoring"""
    try:
        if not performance_collector.is_collecting:
            await performance_collector.start_collection()
            return {
                "success": True,
                "message": "Performance monitoring started",
                "timestamp": time.time()
            }
        else:
            return {
                "success": True,
                "message": "Performance monitoring already running",
                "timestamp": time.time()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/stop-monitoring")
async def stop_monitoring(current_user: User = Depends(get_current_user)):
    """Stop performance monitoring"""
    try:
        if performance_collector.is_collecting:
            await performance_collector.stop_collection()
            return {
                "success": True,
                "message": "Performance monitoring stopped",
                "timestamp": time.time()
            }
        else:
            return {
                "success": True,
                "message": "Performance monitoring not running",
                "timestamp": time.time()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/summary")
async def get_performance_summary(current_user: User = Depends(get_current_user)):
    """Get performance summary with key metrics"""
    try:
        summary = {
            "timestamp": time.time(),
            "system": {},
            "cache": {},
            "database": {},
            "api": {},
            "health_score": 0
        }
        
        # System summary
        if real_time_metrics["system"]:
            system_data = list(real_time_metrics["system"])[-1]
            summary["system"] = {
                "cpu_usage": system_data["cpu"]["percent"],
                "memory_usage": system_data["memory"]["percent"],
                "disk_usage": system_data["disk"]["percent"]
            }
        
        # Cache summary
        if real_time_metrics["cache"]:
            cache_data = list(real_time_metrics["cache"])[-1]
            summary["cache"] = {
                "hit_rate": cache_data["stats"]["hit_rate"],
                "memory_usage_mb": cache_data["stats"]["memory_usage_mb"],
                "entry_count": cache_data["stats"]["entry_count"]
            }
        
        # Database summary
        if real_time_metrics["database"]:
            db_data = list(real_time_metrics["database"])[-1]
            summary["database"] = {
                "pool_utilization": (db_data["pool"]["checked_out"] / db_data["pool"]["size"]) * 100,
                "query_time_ms": db_data.get("query_time", 0)
            }
        
        # Calculate health score
        health_factors = []
        if summary["system"]:
            health_factors.append(100 - summary["system"]["cpu_usage"])
            health_factors.append(100 - summary["system"]["memory_usage"])
        if summary["cache"]:
            health_factors.append(summary["cache"]["hit_rate"])
        
        summary["health_score"] = sum(health_factors) / len(health_factors) if health_factors else 0
        
        return {
            "success": True,
            "data": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")
