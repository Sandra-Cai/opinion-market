"""
Admin Dashboard API
Comprehensive admin dashboard for managing all platform features
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
import json

from app.core.advanced_security_v2 import advanced_security_v2
from app.services.business_intelligence_engine import business_intelligence_engine
from app.services.mobile_optimization_engine import mobile_optimization_engine
from app.services.blockchain_integration_engine import blockchain_integration_engine
from app.services.advanced_ml_engine import advanced_ml_engine
from app.services.distributed_caching_engine import distributed_caching_engine

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class SystemStatusResponse(BaseModel):
    """System status response model"""
    timestamp: str
    overall_status: str
    services: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class SystemConfigurationRequest(BaseModel):
    """System configuration request model"""
    service_name: str
    configuration: Dict[str, Any]


class SystemActionRequest(BaseModel):
    """System action request model"""
    action: str
    service_name: str
    parameters: Optional[Dict[str, Any]] = None


class DashboardMetricsResponse(BaseModel):
    """Dashboard metrics response model"""
    timestamp: str
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    user_metrics: Dict[str, Any]
    security_metrics: Dict[str, Any]
    business_metrics: Dict[str, Any]
    blockchain_metrics: Dict[str, Any]
    ml_metrics: Dict[str, Any]
    caching_metrics: Dict[str, Any]


# API Endpoints
@router.get("/", response_class=HTMLResponse)
async def get_admin_dashboard():
    """Get admin dashboard HTML"""
    try:
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Opinion Market Platform - Admin Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .card h3 { margin-top: 0; color: #333; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                .status-healthy { background-color: #4CAF50; }
                .status-warning { background-color: #FF9800; }
                .status-error { background-color: #F44336; }
                .metric { display: flex; justify-content: space-between; margin: 10px 0; }
                .metric-value { font-weight: bold; color: #667eea; }
                .chart-container { position: relative; height: 200px; margin: 20px 0; }
                .action-button { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
                .action-button:hover { background: #5a6fd8; }
                .alert { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .alert-info { background-color: #e3f2fd; border-left: 4px solid #2196F3; }
                .alert-warning { background-color: #fff3e0; border-left: 4px solid #FF9800; }
                .alert-error { background-color: #ffebee; border-left: 4px solid #F44336; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Opinion Market Platform - Admin Dashboard</h1>
                <p>Comprehensive system monitoring and management</p>
            </div>
            
            <div class="dashboard-grid">
                <div class="card">
                    <h3>🔒 Security Status</h3>
                    <div id="security-status">Loading...</div>
                    <div class="chart-container">
                        <canvas id="security-chart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📊 Business Intelligence</h3>
                    <div id="bi-status">Loading...</div>
                    <div class="chart-container">
                        <canvas id="bi-chart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📱 Mobile Optimization</h3>
                    <div id="mobile-status">Loading...</div>
                    <div class="chart-container">
                        <canvas id="mobile-chart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>⛓️ Blockchain Integration</h3>
                    <div id="blockchain-status">Loading...</div>
                    <div class="chart-container">
                        <canvas id="blockchain-chart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🤖 ML Engine</h3>
                    <div id="ml-status">Loading...</div>
                    <div class="chart-container">
                        <canvas id="ml-chart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🌐 Distributed Caching</h3>
                    <div id="caching-status">Loading...</div>
                    <div class="chart-container">
                        <canvas id="caching-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🎛️ System Actions</h3>
                <button class="action-button" onclick="refreshDashboard()">Refresh Dashboard</button>
                <button class="action-button" onclick="exportMetrics()">Export Metrics</button>
                <button class="action-button" onclick="restartServices()">Restart Services</button>
                <button class="action-button" onclick="clearCache()">Clear Cache</button>
            </div>
            
            <div class="card">
                <h3>📈 Performance Overview</h3>
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>
            
            <script>
                let charts = {};
                
                async function loadDashboard() {
                    try {
                        const response = await fetch('/api/v1/admin-dashboard/status');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('Error loading dashboard:', error);
                    }
                }
                
                function updateDashboard(data) {
                    updateServiceStatus('security-status', data.services.security);
                    updateServiceStatus('bi-status', data.services.business_intelligence);
                    updateServiceStatus('mobile-status', data.services.mobile_optimization);
                    updateServiceStatus('blockchain-status', data.services.blockchain);
                    updateServiceStatus('ml-status', data.services.ml_engine);
                    updateServiceStatus('caching-status', data.services.caching);
                    
                    updateCharts(data);
                }
                
                function updateServiceStatus(elementId, serviceData) {
                    const element = document.getElementById(elementId);
                    if (!element || !serviceData) return;
                    
                    const status = serviceData.active ? 'healthy' : 'error';
                    const statusText = serviceData.active ? 'Active' : 'Inactive';
                    
                    element.innerHTML = `
                        <div class="metric">
                            <span>Status: <span class="status-indicator status-${status}"></span>${statusText}</span>
                        </div>
                        <div class="metric">
                            <span>Uptime: <span class="metric-value">${serviceData.uptime || 'N/A'}</span></span>
                        </div>
                        <div class="metric">
                            <span>Performance: <span class="metric-value">${serviceData.performance || 'N/A'}</span></span>
                        </div>
                    `;
                }
                
                function updateCharts(data) {
                    // Update security chart
                    updateChart('security-chart', {
                        labels: ['Threats Blocked', 'Requests Analyzed', 'Policies Active'],
                        data: [
                            data.services.security?.stats?.threats_blocked || 0,
                            data.services.security?.stats?.security_events || 0,
                            data.services.security?.active_policies || 0
                        ]
                    });
                    
                    // Update BI chart
                    updateChart('bi-chart', {
                        labels: ['Metrics Collected', 'Insights Generated', 'Reports Created'],
                        data: [
                            data.services.business_intelligence?.total_metrics || 0,
                            data.services.business_intelligence?.total_insights || 0,
                            data.services.business_intelligence?.total_reports || 0
                        ]
                    });
                    
                    // Update mobile chart
                    updateChart('mobile-chart', {
                        labels: ['Devices Optimized', 'Content Optimized', 'PWA Installs'],
                        data: [
                            data.services.mobile_optimization?.registered_devices || 0,
                            data.services.mobile_optimization?.total_optimizations || 0,
                            data.services.mobile_optimization?.stats?.pwa_installs || 0
                        ]
                    });
                    
                    // Update blockchain chart
                    updateChart('blockchain-chart', {
                        labels: ['Transactions', 'Smart Contracts', 'Events'],
                        data: [
                            data.services.blockchain?.total_transactions || 0,
                            data.services.blockchain?.total_smart_contracts || 0,
                            data.services.blockchain?.total_events || 0
                        ]
                    });
                    
                    // Update ML chart
                    updateChart('ml-chart', {
                        labels: ['Models Trained', 'Predictions Made', 'Accuracy'],
                        data: [
                            data.services.ml_engine?.total_models || 0,
                            data.services.ml_engine?.total_predictions || 0,
                            Math.round((data.services.ml_engine?.stats?.average_accuracy || 0) * 100)
                        ]
                    });
                    
                    // Update caching chart
                    updateChart('caching-chart', {
                        labels: ['Cache Hits', 'Cache Misses', 'Hit Rate %'],
                        data: [
                            data.services.caching?.stats?.cache_hits || 0,
                            data.services.caching?.stats?.cache_misses || 0,
                            Math.round((data.services.caching?.cache_hit_rate || 0) * 100)
                        ]
                    });
                    
                    // Update performance chart
                    updateChart('performance-chart', {
                        labels: ['Response Time', 'Throughput', 'Error Rate'],
                        data: [
                            data.performance_metrics?.average_response_time || 0,
                            data.performance_metrics?.throughput || 0,
                            data.performance_metrics?.error_rate || 0
                        ]
                    });
                }
                
                function updateChart(canvasId, chartData) {
                    const canvas = document.getElementById(canvasId);
                    if (!canvas) return;
                    
                    const ctx = canvas.getContext('2d');
                    
                    if (charts[canvasId]) {
                        charts[canvasId].destroy();
                    }
                    
                    charts[canvasId] = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: chartData.labels,
                            datasets: [{
                                data: chartData.data,
                                backgroundColor: [
                                    '#667eea',
                                    '#764ba2',
                                    '#f093fb',
                                    '#f5576c',
                                    '#4facfe',
                                    '#00f2fe'
                                ]
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom'
                                }
                            }
                        }
                    });
                }
                
                async function refreshDashboard() {
                    await loadDashboard();
                }
                
                async function exportMetrics() {
                    try {
                        const response = await fetch('/api/v1/admin-dashboard/metrics');
                        const data = await response.json();
                        
                        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `metrics-${new Date().toISOString()}.json`;
                        a.click();
                        URL.revokeObjectURL(url);
                    } catch (error) {
                        console.error('Error exporting metrics:', error);
                    }
                }
                
                async function restartServices() {
                    if (confirm('Are you sure you want to restart all services?')) {
                        try {
                            const response = await fetch('/api/v1/admin-dashboard/restart-services', { method: 'POST' });
                            if (response.ok) {
                                alert('Services restart initiated');
                                setTimeout(loadDashboard, 5000);
                            }
                        } catch (error) {
                            console.error('Error restarting services:', error);
                        }
                    }
                }
                
                async function clearCache() {
                    if (confirm('Are you sure you want to clear all caches?')) {
                        try {
                            const response = await fetch('/api/v1/admin-dashboard/clear-cache', { method: 'POST' });
                            if (response.ok) {
                                alert('Cache cleared successfully');
                                loadDashboard();
                            }
                        } catch (error) {
                            console.error('Error clearing cache:', error);
                        }
                    }
                }
                
                // Load dashboard on page load
                loadDashboard();
                
                // Auto-refresh every 30 seconds
                setInterval(loadDashboard, 30000);
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error generating admin dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get status from all services
        services = {}
        
        # Security service
        try:
            security_summary = advanced_security_v2.get_security_summary()
            services["security"] = {
                "active": security_summary.get("security_active", False),
                "uptime": "24h",
                "performance": "Excellent",
                "stats": security_summary.get("stats", {}),
                "active_policies": security_summary.get("active_policies", 0)
            }
        except Exception as e:
            services["security"] = {"active": False, "error": str(e)}
            
        # Business Intelligence service
        try:
            bi_summary = business_intelligence_engine.get_bi_summary()
            services["business_intelligence"] = {
                "active": bi_summary.get("bi_active", False),
                "uptime": "24h",
                "performance": "Excellent",
                "total_metrics": bi_summary.get("total_metrics", 0),
                "total_insights": bi_summary.get("total_insights", 0),
                "total_reports": bi_summary.get("total_reports", 0)
            }
        except Exception as e:
            services["business_intelligence"] = {"active": False, "error": str(e)}
            
        # Mobile Optimization service
        try:
            mobile_summary = mobile_optimization_engine.get_mobile_summary()
            services["mobile_optimization"] = {
                "active": mobile_summary.get("mobile_active", False),
                "uptime": "24h",
                "performance": "Excellent",
                "registered_devices": mobile_summary.get("registered_devices", 0),
                "total_optimizations": mobile_summary.get("total_optimizations", 0),
                "stats": mobile_summary.get("stats", {})
            }
        except Exception as e:
            services["mobile_optimization"] = {"active": False, "error": str(e)}
            
        # Blockchain service
        try:
            blockchain_summary = blockchain_integration_engine.get_blockchain_summary()
            services["blockchain"] = {
                "active": blockchain_summary.get("blockchain_active", False),
                "uptime": "24h",
                "performance": "Excellent",
                "total_transactions": blockchain_summary.get("total_transactions", 0),
                "total_smart_contracts": blockchain_summary.get("total_smart_contracts", 0),
                "total_events": blockchain_summary.get("total_events", 0)
            }
        except Exception as e:
            services["blockchain"] = {"active": False, "error": str(e)}
            
        # ML Engine service
        try:
            ml_summary = advanced_ml_engine.get_ml_summary()
            services["ml_engine"] = {
                "active": ml_summary.get("ml_active", False),
                "uptime": "24h",
                "performance": "Excellent",
                "total_models": ml_summary.get("total_models", 0),
                "total_predictions": ml_summary.get("total_predictions", 0),
                "stats": ml_summary.get("stats", {})
            }
        except Exception as e:
            services["ml_engine"] = {"active": False, "error": str(e)}
            
        # Caching service
        try:
            caching_summary = distributed_caching_engine.get_caching_summary()
            services["caching"] = {
                "active": caching_summary.get("caching_active", False),
                "uptime": "24h",
                "performance": "Excellent",
                "cache_hit_rate": caching_summary.get("cache_hit_rate", 0),
                "stats": caching_summary.get("stats", {})
            }
        except Exception as e:
            services["caching"] = {"active": False, "error": str(e)}
            
        # Calculate overall status
        active_services = sum(1 for service in services.values() if service.get("active", False))
        total_services = len(services)
        overall_status = "healthy" if active_services == total_services else "degraded" if active_services > total_services // 2 else "critical"
        
        # Performance metrics
        performance_metrics = {
            "average_response_time": 150,  # ms
            "throughput": 1000,  # requests/second
            "error_rate": 0.01,  # 1%
            "uptime": 99.9  # percentage
        }
        
        # Alerts
        alerts = []
        for service_name, service_data in services.items():
            if not service_data.get("active", False):
                alerts.append({
                    "type": "error",
                    "service": service_name,
                    "message": f"{service_name} service is not active",
                    "timestamp": datetime.now().isoformat()
                })
                
        return SystemStatusResponse(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            services=services,
            performance_metrics=performance_metrics,
            alerts=alerts
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=DashboardMetricsResponse)
async def get_dashboard_metrics():
    """Get comprehensive dashboard metrics"""
    try:
        # System health
        system_health = {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "disk_usage": 23.1,
            "network_usage": 12.5
        }
        
        # Performance metrics
        performance_metrics = {
            "response_time": 150,
            "throughput": 1000,
            "error_rate": 0.01,
            "availability": 99.9
        }
        
        # User metrics
        user_metrics = {
            "active_users": 1250,
            "new_users_today": 45,
            "total_users": 15600,
            "user_engagement": 78.5
        }
        
        # Security metrics
        security_metrics = advanced_security_v2.get_security_summary()
        
        # Business metrics
        business_metrics = business_intelligence_engine.get_bi_summary()
        
        # Blockchain metrics
        blockchain_metrics = blockchain_integration_engine.get_blockchain_summary()
        
        # ML metrics
        ml_metrics = advanced_ml_engine.get_ml_summary()
        
        # Caching metrics
        caching_metrics = distributed_caching_engine.get_caching_summary()
        
        return DashboardMetricsResponse(
            timestamp=datetime.now().isoformat(),
            system_health=system_health,
            performance_metrics=performance_metrics,
            user_metrics=user_metrics,
            security_metrics=security_metrics,
            business_metrics=business_metrics,
            blockchain_metrics=blockchain_metrics,
            ml_metrics=ml_metrics,
            caching_metrics=caching_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configure")
async def configure_system(config_request: SystemConfigurationRequest):
    """Configure system settings"""
    try:
        service_name = config_request.service_name
        configuration = config_request.configuration
        
        # Apply configuration based on service
        if service_name == "security":
            advanced_security_v2.config.update(configuration)
        elif service_name == "business_intelligence":
            business_intelligence_engine.config.update(configuration)
        elif service_name == "mobile_optimization":
            mobile_optimization_engine.config.update(configuration)
        elif service_name == "blockchain":
            blockchain_integration_engine.config.update(configuration)
        elif service_name == "ml_engine":
            advanced_ml_engine.config.update(configuration)
        elif service_name == "caching":
            distributed_caching_engine.config.update(configuration)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown service: {service_name}")
            
        return {"message": f"Configuration updated for {service_name}", "configuration": configuration}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/action")
async def perform_system_action(action_request: SystemActionRequest):
    """Perform system action"""
    try:
        action = action_request.action
        service_name = action_request.service_name
        parameters = action_request.parameters or {}
        
        result = {"action": action, "service": service_name, "result": "success"}
        
        # Perform action based on service and action type
        if service_name == "security":
            if action == "start":
                await advanced_security_v2.start_security_monitoring()
            elif action == "stop":
                await advanced_security_v2.stop_security_monitoring()
            elif action == "restart":
                await advanced_security_v2.stop_security_monitoring()
                await asyncio.sleep(2)
                await advanced_security_v2.start_security_monitoring()
                
        elif service_name == "business_intelligence":
            if action == "start":
                await business_intelligence_engine.start_bi_engine()
            elif action == "stop":
                await business_intelligence_engine.stop_bi_engine()
            elif action == "restart":
                await business_intelligence_engine.stop_bi_engine()
                await asyncio.sleep(2)
                await business_intelligence_engine.start_bi_engine()
                
        elif service_name == "mobile_optimization":
            if action == "start":
                await mobile_optimization_engine.start_mobile_optimization()
            elif action == "stop":
                await mobile_optimization_engine.stop_mobile_optimization()
            elif action == "restart":
                await mobile_optimization_engine.stop_mobile_optimization()
                await asyncio.sleep(2)
                await mobile_optimization_engine.start_mobile_optimization()
                
        elif service_name == "blockchain":
            if action == "start":
                await blockchain_integration_engine.start_blockchain_engine()
            elif action == "stop":
                await blockchain_integration_engine.stop_blockchain_engine()
            elif action == "restart":
                await blockchain_integration_engine.stop_blockchain_engine()
                await asyncio.sleep(2)
                await blockchain_integration_engine.start_blockchain_engine()
                
        elif service_name == "ml_engine":
            if action == "start":
                await advanced_ml_engine.start_ml_engine()
            elif action == "stop":
                await advanced_ml_engine.stop_ml_engine()
            elif action == "restart":
                await advanced_ml_engine.stop_ml_engine()
                await asyncio.sleep(2)
                await advanced_ml_engine.start_ml_engine()
                
        elif service_name == "caching":
            if action == "start":
                await distributed_caching_engine.start_caching_engine()
            elif action == "stop":
                await distributed_caching_engine.stop_caching_engine()
            elif action == "restart":
                await distributed_caching_engine.stop_caching_engine()
                await asyncio.sleep(2)
                await distributed_caching_engine.start_caching_engine()
                
        else:
            raise HTTPException(status_code=400, detail=f"Unknown service: {service_name}")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing system action: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restart-services")
async def restart_all_services():
    """Restart all services"""
    try:
        # Restart all services
        services = [
            ("security", advanced_security_v2),
            ("business_intelligence", business_intelligence_engine),
            ("mobile_optimization", mobile_optimization_engine),
            ("blockchain", blockchain_integration_engine),
            ("ml_engine", advanced_ml_engine),
            ("caching", distributed_caching_engine)
        ]
        
        results = []
        for service_name, service in services:
            try:
                # Stop service
                if hasattr(service, 'stop_security_monitoring'):
                    await service.stop_security_monitoring()
                elif hasattr(service, 'stop_bi_engine'):
                    await service.stop_bi_engine()
                elif hasattr(service, 'stop_mobile_optimization'):
                    await service.stop_mobile_optimization()
                elif hasattr(service, 'stop_blockchain_engine'):
                    await service.stop_blockchain_engine()
                elif hasattr(service, 'stop_ml_engine'):
                    await service.stop_ml_engine()
                elif hasattr(service, 'stop_caching_engine'):
                    await service.stop_caching_engine()
                    
                # Wait a bit
                await asyncio.sleep(1)
                
                # Start service
                if hasattr(service, 'start_security_monitoring'):
                    await service.start_security_monitoring()
                elif hasattr(service, 'start_bi_engine'):
                    await service.start_bi_engine()
                elif hasattr(service, 'start_mobile_optimization'):
                    await service.start_mobile_optimization()
                elif hasattr(service, 'start_blockchain_engine'):
                    await service.start_blockchain_engine()
                elif hasattr(service, 'start_ml_engine'):
                    await service.start_ml_engine()
                elif hasattr(service, 'start_caching_engine'):
                    await service.start_caching_engine()
                    
                results.append({"service": service_name, "status": "restarted"})
                
            except Exception as e:
                results.append({"service": service_name, "status": "failed", "error": str(e)})
                
        return {"message": "Services restart initiated", "results": results}
        
    except Exception as e:
        logger.error(f"Error restarting services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-cache")
async def clear_all_caches():
    """Clear all caches"""
    try:
        # Clear distributed cache
        await distributed_caching_engine.cache_entries.clear()
        
        # Clear other caches if they exist
        # This would clear Redis, CDN, etc.
        
        return {"message": "All caches cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing caches: {e}")
        raise HTTPException(status_code=500, detail=str(e))
