"""
Advanced Real-time Performance Dashboard
Interactive dashboard with AI insights and predictive analytics
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional
import asyncio
import json
import time
import psutil
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

from app.core.auth import get_current_user
from app.models.user import User
from app.core.advanced_performance_optimizer import advanced_performance_optimizer
from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connection manager
class AdvancedDashboardManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alerts: List[Dict[str, Any]] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Dashboard disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return
            
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}", exc_info=True)
                disconnected.append(connection)
                
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
            
    def add_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Add metric to history"""
        if timestamp is None:
            timestamp = datetime.now()
        self.metrics_history[metric_name].append({
            "value": value,
            "timestamp": timestamp.isoformat()
        })
        
    def get_metric_trend(self, metric_name: str, window: int = 10) -> str:
        """Get trend for a metric"""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < window:
            return "stable"
            
        recent_values = [m["value"] for m in list(self.metrics_history[metric_name])[-window:]]
        older_values = [m["value"] for m in list(self.metrics_history[metric_name])[-window*2:-window]] if len(self.metrics_history[metric_name]) >= window*2 else recent_values
        
        recent_avg = statistics.mean(recent_values)
        older_avg = statistics.mean(older_values)
        
        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"

dashboard_manager = AdvancedDashboardManager()


@router.get("/dashboard/advanced", response_class=HTMLResponse)
async def get_advanced_dashboard():
    """Get advanced real-time performance dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 Advanced Performance Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); min-height: 100vh; color: #333; }
            .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
            .header { background: rgba(255,255,255,0.95); padding: 25px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            .header h1 { color: #2c3e50; font-size: 2.2em; margin-bottom: 10px; }
            .header p { color: #7f8c8d; font-size: 1.1em; }
            .status-bar { display: flex; justify-content: space-between; align-items: center; margin: 20px 0; }
            .status-item { display: flex; align-items: center; padding: 10px 15px; background: rgba(255,255,255,0.9); border-radius: 8px; margin: 0 5px; }
            .status-indicator { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-healthy { background: #27ae60; }
            .status-warning { background: #f39c12; }
            .status-error { background: #e74c3c; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .metric-card { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); transition: transform 0.3s; }
            .metric-card:hover { transform: translateY(-5px); }
            .metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
            .metric-title { font-size: 1.1em; font-weight: 600; color: #2c3e50; }
            .metric-trend { padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
            .trend-increasing { background: #e8f5e8; color: #27ae60; }
            .trend-decreasing { background: #fdeaea; color: #e74c3c; }
            .trend-stable { background: #e8f4fd; color: #3498db; }
            .metric-value { font-size: 2.5em; font-weight: bold; color: #2c3e50; margin: 10px 0; }
            .metric-unit { font-size: 0.9em; color: #7f8c8d; }
            .metric-change { font-size: 0.9em; margin-top: 5px; }
            .positive { color: #27ae60; }
            .negative { color: #e74c3c; }
            .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 30px; }
            .chart-container { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }
            .chart-title { font-size: 1.3em; font-weight: 600; color: #2c3e50; margin-bottom: 20px; }
            .alerts-panel { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 25px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); }
            .alert-item { padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid; }
            .alert-info { background: #e8f4fd; border-left-color: #3498db; }
            .alert-warning { background: #fff3cd; border-left-color: #ffc107; }
            .alert-error { background: #f8d7da; border-left-color: #dc3545; }
            .connection-status { position: fixed; top: 20px; right: 20px; padding: 12px 20px; border-radius: 25px; font-weight: bold; z-index: 1000; }
            .connected { background: #27ae60; color: white; }
            .disconnected { background: #e74c3c; color: white; }
            .ai-insights { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; padding: 25px; margin: 20px 0; }
            .insight-item { background: rgba(255,255,255,0.1); padding: 15px; margin: 10px 0; border-radius: 8px; }
            .loading { text-align: center; padding: 40px; color: #7f8c8d; }
            @media (max-width: 768px) {
                .charts-grid { grid-template-columns: 1fr; }
                .metrics-grid { grid-template-columns: 1fr; }
                .status-bar { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Advanced Performance Dashboard</h1>
                <p>Real-time system monitoring with AI-powered insights and predictive analytics</p>
                <div class="status-bar">
                    <div class="status-item">
                        <span class="status-indicator status-healthy" id="system-status"></span>
                        <span>System Status</span>
                    </div>
                    <div class="status-item">
                        <span class="status-indicator status-healthy" id="monitoring-status"></span>
                        <span>Monitoring Active</span>
                    </div>
                    <div class="status-item">
                        <span class="status-indicator status-healthy" id="optimization-status"></span>
                        <span>Optimization Active</span>
                    </div>
                    <div class="status-item">
                        <span id="last-update">Last Update: --</span>
                    </div>
                </div>
            </div>

            <div class="connection-status disconnected" id="connection-status">
                <span id="connection-text">Disconnected</span>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">CPU Usage</div>
                        <div class="metric-trend trend-stable" id="cpu-trend">Stable</div>
                    </div>
                    <div class="metric-value" id="cpu-usage">--</div>
                    <div class="metric-unit">%</div>
                    <div class="metric-change" id="cpu-change">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Memory Usage</div>
                        <div class="metric-trend trend-stable" id="memory-trend">Stable</div>
                    </div>
                    <div class="metric-value" id="memory-usage">--</div>
                    <div class="metric-unit">%</div>
                    <div class="metric-change" id="memory-change">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Cache Hit Rate</div>
                        <div class="metric-trend trend-stable" id="cache-trend">Stable</div>
                    </div>
                    <div class="metric-value" id="cache-hit-rate">--</div>
                    <div class="metric-unit">%</div>
                    <div class="metric-change" id="cache-change">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Response Time</div>
                        <div class="metric-trend trend-stable" id="response-trend">Stable</div>
                    </div>
                    <div class="metric-value" id="response-time">--</div>
                    <div class="metric-unit">ms</div>
                    <div class="metric-change" id="response-change">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Throughput</div>
                        <div class="metric-trend trend-stable" id="throughput-trend">Stable</div>
                    </div>
                    <div class="metric-value" id="throughput">--</div>
                    <div class="metric-unit">req/s</div>
                    <div class="metric-change" id="throughput-change">--</div>
                </div>

                <div class="metric-card">
                    <div class="metric-header">
                        <div class="metric-title">Health Score</div>
                        <div class="metric-trend trend-stable" id="health-trend">Stable</div>
                    </div>
                    <div class="metric-value" id="health-score">--</div>
                    <div class="metric-unit">/100</div>
                    <div class="metric-change" id="health-change">--</div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-title">📊 System Performance Over Time</div>
                    <canvas id="system-chart"></canvas>
                </div>
                <div class="chart-container">
                    <div class="chart-title">⚡ Cache Performance</div>
                    <canvas id="cache-chart"></canvas>
                </div>
            </div>

            <div class="ai-insights">
                <h3>🤖 AI Performance Insights</h3>
                <div id="ai-insights-content">
                    <div class="loading">Loading AI insights...</div>
                </div>
            </div>

            <div class="alerts-panel">
                <h3>🚨 System Alerts</h3>
                <div id="alerts-content">
                    <div class="loading">No alerts at this time</div>
                </div>
            </div>
        </div>

        <script>
            // Global variables
            let ws = null;
            let reconnectInterval = null;
            let charts = {};
            let previousMetrics = {};

            // Initialize charts
            function initCharts() {
                // System performance chart
                const systemCtx = document.getElementById('system-chart').getContext('2d');
                charts.system = new Chart(systemCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'CPU Usage (%)',
                            data: [],
                            borderColor: '#e74c3c',
                            backgroundColor: 'rgba(231, 76, 60, 0.1)',
                            tension: 0.4
                        }, {
                            label: 'Memory Usage (%)',
                            data: [],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true, max: 100 }
                        },
                        plugins: {
                            legend: { position: 'top' }
                        }
                    }
                });

                // Cache performance chart
                const cacheCtx = document.getElementById('cache-chart').getContext('2d');
                charts.cache = new Chart(cacheCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Hit Rate (%)',
                            data: [],
                            borderColor: '#27ae60',
                            backgroundColor: 'rgba(39, 174, 96, 0.1)',
                            tension: 0.4
                        }, {
                            label: 'Response Time (ms)',
                            data: [],
                            borderColor: '#f39c12',
                            backgroundColor: 'rgba(243, 156, 18, 0.1)',
                            tension: 0.4,
                            yAxisID: 'y1'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true, max: 100 },
                            y1: { type: 'linear', display: true, position: 'right' }
                        },
                        plugins: {
                            legend: { position: 'top' }
                        }
                    }
                });
            }

            // WebSocket connection
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/v1/advanced-dashboard/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    updateConnectionStatus(true);
                    if (reconnectInterval) {
                        clearInterval(reconnectInterval);
                        reconnectInterval = null;
                    }
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
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

            // Update connection status
            function updateConnectionStatus(connected) {
                const status = document.getElementById('connection-status');
                const text = document.getElementById('connection-text');
                
                if (connected) {
                    status.className = 'connection-status connected';
                    text.textContent = 'Connected';
                } else {
                    status.className = 'connection-status disconnected';
                    text.textContent = 'Disconnected';
                }
            }

            // Update dashboard with new data
            function updateDashboard(data) {
                // Update timestamp
                document.getElementById('last-update').textContent = 
                    `Last Update: ${new Date().toLocaleTimeString()}`;

                // Update metrics
                updateMetric('cpu-usage', data.metrics.cpu_usage, '%');
                updateMetric('memory-usage', data.metrics.memory_usage, '%');
                updateMetric('cache-hit-rate', data.metrics.cache_hit_rate, '%');
                updateMetric('response-time', data.metrics.response_time, 'ms');
                updateMetric('throughput', data.metrics.throughput, 'req/s');
                updateMetric('health-score', data.performance_score, '/100');

                // Update trends
                updateTrend('cpu-trend', data.trends.cpu);
                updateTrend('memory-trend', data.trends.memory);
                updateTrend('cache-trend', data.trends.cache);
                updateTrend('response-trend', data.trends.response_time);
                updateTrend('throughput-trend', data.trends.throughput);
                updateTrend('health-trend', data.trends.health_score);

                // Update charts
                updateCharts(data);

                // Update AI insights
                updateAIInsights(data.ai_insights);

                // Update alerts
                updateAlerts(data.alerts);

                // Store previous metrics for change calculation
                previousMetrics = { ...data.metrics };
            }

            // Update individual metric
            function updateMetric(elementId, value, unit) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = value.toFixed(1);
                }
            }

            // Update trend indicator
            function updateTrend(elementId, trend) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = trend;
                    element.className = `metric-trend trend-${trend}`;
                }
            }

            // Update charts
            function updateCharts(data) {
                const timestamp = new Date().toLocaleTimeString();
                
                // System chart
                charts.system.data.labels.push(timestamp);
                charts.system.data.datasets[0].data.push(data.metrics.cpu_usage);
                charts.system.data.datasets[1].data.push(data.metrics.memory_usage);
                
                if (charts.system.data.labels.length > 20) {
                    charts.system.data.labels.shift();
                    charts.system.data.datasets[0].data.shift();
                    charts.system.data.datasets[1].data.shift();
                }
                charts.system.update('none');

                // Cache chart
                charts.cache.data.labels.push(timestamp);
                charts.cache.data.datasets[0].data.push(data.metrics.cache_hit_rate);
                charts.cache.data.datasets[1].data.push(data.metrics.response_time);
                
                if (charts.cache.data.labels.length > 20) {
                    charts.cache.data.labels.shift();
                    charts.cache.data.datasets[0].data.shift();
                    charts.cache.data.datasets[1].data.shift();
                }
                charts.cache.update('none');
            }

            // Update AI insights
            function updateAIInsights(insights) {
                const container = document.getElementById('ai-insights-content');
                if (!insights || insights.length === 0) {
                    container.innerHTML = '<div class="loading">No insights available</div>';
                    return;
                }

                container.innerHTML = insights.map(insight => 
                    `<div class="insight-item">
                        <strong>${insight.type}:</strong> ${insight.message}
                        <br><small>Confidence: ${(insight.confidence * 100).toFixed(1)}%</small>
                    </div>`
                ).join('');
            }

            // Update alerts
            function updateAlerts(alerts) {
                const container = document.getElementById('alerts-content');
                if (!alerts || alerts.length === 0) {
                    container.innerHTML = '<div class="loading">No alerts at this time</div>';
                    return;
                }

                container.innerHTML = alerts.map(alert => 
                    `<div class="alert-item alert-${alert.severity}">
                        <strong>${alert.title}</strong><br>
                        ${alert.message}<br>
                        <small>${new Date(alert.timestamp).toLocaleString()}</small>
                    </div>`
                ).join('');
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
async def websocket_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await dashboard_manager.connect(websocket)
    
    try:
        while True:
            # Collect real-time metrics
            metrics = await collect_realtime_metrics()
            
            # Get performance summary
            perf_summary = advanced_performance_optimizer.get_performance_summary()
            
            # Calculate trends
            trends = calculate_trends(metrics)
            
            # Generate AI insights
            ai_insights = generate_ai_insights(metrics, perf_summary)
            
            # Check for alerts
            alerts = check_alerts(metrics, perf_summary)
            
            # Prepare dashboard data
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "performance_score": perf_summary.get("performance_score", 100),
                "trends": trends,
                "ai_insights": ai_insights,
                "alerts": alerts,
                "monitoring_active": perf_summary.get("monitoring_active", False),
                "optimization_active": perf_summary.get("optimization_active", False)
            }
            
            # Broadcast to all connected clients
            await dashboard_manager.broadcast(dashboard_data)
            
            # Wait before next update
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        dashboard_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        dashboard_manager.disconnect(websocket)


async def collect_realtime_metrics() -> Dict[str, float]:
    """Collect real-time system metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Cache metrics
        cache_stats = enhanced_cache.get_stats()
        
        # Performance metrics (simulated for demo)
        response_time = 1.44  # ms
        throughput = 1500.0   # req/s
        
        metrics = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "cache_hit_rate": cache_stats.get("hit_rate", 0) * 100,
            "response_time": response_time,
            "throughput": throughput
        }
        
        # Store metrics in dashboard manager
        for metric_name, value in metrics.items():
            dashboard_manager.add_metric(metric_name, value)
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}", exc_info=True)
        return {}


def calculate_trends(metrics: Dict[str, float]) -> Dict[str, str]:
    """Calculate trends for metrics"""
    trends = {}
    for metric_name in metrics.keys():
        trends[metric_name] = dashboard_manager.get_metric_trend(metric_name)
    return trends


def generate_ai_insights(metrics: Dict[str, float], perf_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate AI-powered insights"""
    insights = []
    
    # CPU insights
    if metrics.get("cpu_usage", 0) > 80:
        insights.append({
            "type": "Performance Warning",
            "message": "High CPU usage detected. Consider scaling or optimization.",
            "confidence": 0.9
        })
    elif metrics.get("cpu_usage", 0) < 20:
        insights.append({
            "type": "Optimization Opportunity",
            "message": "Low CPU usage. System can handle more load.",
            "confidence": 0.8
        })
    
    # Cache insights
    cache_hit_rate = metrics.get("cache_hit_rate", 0)
    if cache_hit_rate > 90:
        insights.append({
            "type": "Excellent Performance",
            "message": "Cache hit rate is excellent. System is well-optimized.",
            "confidence": 0.95
        })
    elif cache_hit_rate < 70:
        insights.append({
            "type": "Cache Optimization",
            "message": "Cache hit rate could be improved. Consider increasing cache size.",
            "confidence": 0.85
        })
    
    # Memory insights
    if metrics.get("memory_usage", 0) > 85:
        insights.append({
            "type": "Memory Alert",
            "message": "High memory usage. Monitor for potential memory leaks.",
            "confidence": 0.9
        })
    
    # Performance score insights
    perf_score = perf_summary.get("performance_score", 100)
    if perf_score > 90:
        insights.append({
            "type": "System Health",
            "message": "System is performing excellently with optimal health score.",
            "confidence": 0.95
        })
    elif perf_score < 70:
        insights.append({
            "type": "Performance Concern",
            "message": "Performance score indicates system needs attention.",
            "confidence": 0.9
        })
    
    return insights


def check_alerts(metrics: Dict[str, float], perf_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check for system alerts"""
    alerts = []
    current_time = datetime.now().isoformat()
    
    # CPU alert
    if metrics.get("cpu_usage", 0) > 90:
        alerts.append({
            "severity": "error",
            "title": "Critical CPU Usage",
            "message": f"CPU usage is at {metrics['cpu_usage']:.1f}%",
            "timestamp": current_time
        })
    elif metrics.get("cpu_usage", 0) > 80:
        alerts.append({
            "severity": "warning",
            "title": "High CPU Usage",
            "message": f"CPU usage is at {metrics['cpu_usage']:.1f}%",
            "timestamp": current_time
        })
    
    # Memory alert
    if metrics.get("memory_usage", 0) > 95:
        alerts.append({
            "severity": "error",
            "title": "Critical Memory Usage",
            "message": f"Memory usage is at {metrics['memory_usage']:.1f}%",
            "timestamp": current_time
        })
    elif metrics.get("memory_usage", 0) > 85:
        alerts.append({
            "severity": "warning",
            "title": "High Memory Usage",
            "message": f"Memory usage is at {metrics['memory_usage']:.1f}%",
            "timestamp": current_time
        })
    
    # Cache alert
    if metrics.get("cache_hit_rate", 0) < 50:
        alerts.append({
            "severity": "warning",
            "title": "Low Cache Hit Rate",
            "message": f"Cache hit rate is at {metrics['cache_hit_rate']:.1f}%",
            "timestamp": current_time
        })
    
    return alerts


@router.get("/dashboard/summary")
async def get_dashboard_summary(current_user: User = Depends(get_current_user)):
    """Get dashboard summary data"""
    try:
        metrics = await collect_realtime_metrics()
        perf_summary = advanced_performance_optimizer.get_performance_summary()
        trends = calculate_trends(metrics)
        ai_insights = generate_ai_insights(metrics, perf_summary)
        alerts = check_alerts(metrics, perf_summary)
        
        return {
            "success": True,
            "data": {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "performance_score": perf_summary.get("performance_score", 100),
                "trends": trends,
                "ai_insights": ai_insights,
                "alerts": alerts,
                "active_connections": len(dashboard_manager.active_connections)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard summary: {str(e)}")


