"""
Business Intelligence and Advanced Analytics API
Provides comprehensive business insights and data analysis
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import pandas as pd
import numpy as np

from app.core.auth import get_current_user
from app.models.user import User
from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

router = APIRouter()


class BusinessIntelligenceEngine:
    """Advanced business intelligence and analytics engine"""
    
    def __init__(self):
        self.analytics_cache = {}
        self.report_cache = {}
        self.trend_analysis = {}
    
    async def generate_market_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive market analytics"""
        try:
            # Get market data from database
            query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as markets_created,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_markets,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) as resolved_markets,
                AVG(CASE WHEN status = 'resolved' THEN volume ELSE NULL END) as avg_volume
            FROM markets 
            WHERE created_at >= NOW() - INTERVAL %s DAY
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), (days,))
                market_data = result.fetchall()
            
            # Process market trends
            trends = self._analyze_market_trends(market_data)
            
            # Get trading analytics
            trading_analytics = await self._get_trading_analytics(days)
            
            # Get user engagement metrics
            user_metrics = await self._get_user_engagement_metrics(days)
            
            return {
                "market_trends": trends,
                "trading_analytics": trading_analytics,
                "user_metrics": user_metrics,
                "period_days": days,
                "generated_at": time.time()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate market analytics: {str(e)}")
    
    def _analyze_market_trends(self, market_data: List) -> Dict[str, Any]:
        """Analyze market creation and activity trends"""
        if not market_data:
            return {"trend": "no_data", "growth_rate": 0, "volatility": 0}
        
        # Extract data
        dates = [row[0] for row in market_data]
        markets_created = [row[1] for row in market_data]
        active_markets = [row[2] for row in market_data]
        resolved_markets = [row[3] for row in market_data]
        
        # Calculate trends
        total_markets = sum(markets_created)
        avg_daily_creation = statistics.mean(markets_created) if markets_created else 0
        
        # Growth rate calculation
        if len(markets_created) > 1:
            growth_rate = ((markets_created[0] - markets_created[-1]) / markets_created[-1]) * 100
        else:
            growth_rate = 0
        
        # Volatility calculation
        volatility = statistics.stdev(markets_created) if len(markets_created) > 1 else 0
        
        # Trend direction
        if growth_rate > 5:
            trend = "growing"
        elif growth_rate < -5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "growth_rate": round(growth_rate, 2),
            "volatility": round(volatility, 2),
            "total_markets": total_markets,
            "avg_daily_creation": round(avg_daily_creation, 2),
            "active_markets": sum(active_markets),
            "resolved_markets": sum(resolved_markets),
            "resolution_rate": round((sum(resolved_markets) / max(sum(active_markets) + sum(resolved_markets), 1)) * 100, 2)
        }
    
    async def _get_trading_analytics(self, days: int) -> Dict[str, Any]:
        """Get comprehensive trading analytics"""
        try:
            # Get trading volume data
            volume_query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as trade_count,
                SUM(amount) as total_volume,
                AVG(amount) as avg_trade_size,
                COUNT(DISTINCT user_id) as unique_traders
            FROM trades 
            WHERE created_at >= NOW() - INTERVAL %s DAY
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(volume_query), (days,))
                volume_data = result.fetchall()
            
            if not volume_data:
                return {"total_volume": 0, "avg_daily_volume": 0, "unique_traders": 0}
            
            # Calculate trading metrics
            total_volume = sum(row[2] for row in volume_data)
            avg_daily_volume = statistics.mean([row[2] for row in volume_data])
            total_trades = sum(row[1] for row in volume_data)
            avg_trade_size = statistics.mean([row[3] for row in volume_data])
            unique_traders = len(set(row[4] for row in volume_data))
            
            # Get top traders
            top_traders_query = """
            SELECT 
                u.username,
                COUNT(t.id) as trade_count,
                SUM(t.amount) as total_volume,
                AVG(t.amount) as avg_trade_size
            FROM trades t
            JOIN users u ON t.user_id = u.id
            WHERE t.created_at >= NOW() - INTERVAL %s DAY
            GROUP BY u.id, u.username
            ORDER BY total_volume DESC
            LIMIT 10
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(top_traders_query), (days,))
                top_traders = result.fetchall()
            
            return {
                "total_volume": total_volume,
                "avg_daily_volume": round(avg_daily_volume, 2),
                "total_trades": total_trades,
                "avg_trade_size": round(avg_trade_size, 2),
                "unique_traders": unique_traders,
                "top_traders": [
                    {
                        "username": trader[0],
                        "trade_count": trader[1],
                        "total_volume": trader[2],
                        "avg_trade_size": round(trader[3], 2)
                    }
                    for trader in top_traders
                ]
            }
            
        except Exception as e:
            return {"error": f"Failed to get trading analytics: {str(e)}"}
    
    async def _get_user_engagement_metrics(self, days: int) -> Dict[str, Any]:
        """Get user engagement and activity metrics"""
        try:
            # Get user activity data
            activity_query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(DISTINCT user_id) as active_users,
                COUNT(*) as total_actions
            FROM (
                SELECT user_id, created_at FROM trades WHERE created_at >= NOW() - INTERVAL %s DAY
                UNION ALL
                SELECT user_id, created_at FROM votes WHERE created_at >= NOW() - INTERVAL %s DAY
                UNION ALL
                SELECT user_id, created_at FROM markets WHERE created_at >= NOW() - INTERVAL %s DAY
            ) as user_activity
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(activity_query), (days, days, days))
                activity_data = result.fetchall()
            
            if not activity_data:
                return {"total_active_users": 0, "avg_daily_active_users": 0}
            
            # Calculate engagement metrics
            total_active_users = len(set(row[1] for row in activity_data))
            avg_daily_active_users = statistics.mean([row[1] for row in activity_data])
            total_actions = sum(row[2] for row in activity_data)
            
            # Get user retention data
            retention_query = """
            SELECT 
                COUNT(DISTINCT user_id) as total_users,
                COUNT(DISTINCT CASE WHEN last_login >= NOW() - INTERVAL 7 DAY THEN user_id END) as weekly_active,
                COUNT(DISTINCT CASE WHEN last_login >= NOW() - INTERVAL 30 DAY THEN user_id END) as monthly_active
            FROM users
            WHERE created_at >= NOW() - INTERVAL %s DAY
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(retention_query), (days,))
                retention_data = result.fetchone()
            
            return {
                "total_active_users": total_active_users,
                "avg_daily_active_users": round(avg_daily_active_users, 2),
                "total_actions": total_actions,
                "total_users": retention_data[0] if retention_data else 0,
                "weekly_active_users": retention_data[1] if retention_data else 0,
                "monthly_active_users": retention_data[2] if retention_data else 0,
                "retention_rate": round((retention_data[1] / max(retention_data[0], 1)) * 100, 2) if retention_data else 0
            }
            
        except Exception as e:
            return {"error": f"Failed to get user engagement metrics: {str(e)}"}
    
    async def generate_revenue_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Generate revenue and financial analytics"""
        try:
            # Get fee data (assuming fees are stored in trades or a separate table)
            revenue_query = """
            SELECT 
                DATE(created_at) as date,
                SUM(fee_amount) as daily_fees,
                COUNT(*) as fee_transactions
            FROM trades 
            WHERE created_at >= NOW() - INTERVAL %s DAY
            AND fee_amount > 0
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(revenue_query), (days,))
                revenue_data = result.fetchall()
            
            if not revenue_data:
                return {"total_revenue": 0, "avg_daily_revenue": 0, "revenue_trend": "no_data"}
            
            # Calculate revenue metrics
            total_revenue = sum(row[1] for row in revenue_data)
            avg_daily_revenue = statistics.mean([row[1] for row in revenue_data])
            total_fee_transactions = sum(row[2] for row in revenue_data)
            
            # Revenue trend analysis
            if len(revenue_data) > 1:
                recent_revenue = revenue_data[0][1]
                older_revenue = revenue_data[-1][1]
                revenue_growth = ((recent_revenue - older_revenue) / max(older_revenue, 1)) * 100
                
                if revenue_growth > 10:
                    revenue_trend = "growing"
                elif revenue_growth < -10:
                    revenue_trend = "declining"
                else:
                    revenue_trend = "stable"
            else:
                revenue_trend = "insufficient_data"
            
            return {
                "total_revenue": total_revenue,
                "avg_daily_revenue": round(avg_daily_revenue, 2),
                "total_fee_transactions": total_fee_transactions,
                "revenue_trend": revenue_trend,
                "revenue_growth_rate": round(revenue_growth, 2) if 'revenue_growth' in locals() else 0,
                "period_days": days
            }
            
        except Exception as e:
            return {"error": f"Failed to generate revenue analytics: {str(e)}"}
    
    async def generate_predictive_insights(self) -> Dict[str, Any]:
        """Generate predictive insights using historical data"""
        try:
            # Get historical market data for prediction
            prediction_query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as markets_created,
                SUM(volume) as total_volume,
                AVG(volume) as avg_volume
            FROM markets 
            WHERE created_at >= NOW() - INTERVAL 90 DAY
            GROUP BY DATE(created_at)
            ORDER BY date ASC
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(prediction_query))
                historical_data = result.fetchall()
            
            if len(historical_data) < 30:
                return {"predictions": "insufficient_data"}
            
            # Simple trend-based predictions
            dates = [row[0] for row in historical_data]
            markets_created = [row[1] for row in historical_data]
            volumes = [row[2] for row in historical_data]
            
            # Calculate moving averages
            ma_7 = self._calculate_moving_average(markets_created, 7)
            ma_30 = self._calculate_moving_average(markets_created, 30)
            
            # Predict next 7 days
            recent_trend = ma_7[-1] - ma_30[-1] if len(ma_7) > 0 and len(ma_30) > 0 else 0
            predicted_daily_markets = max(0, ma_7[-1] + recent_trend) if len(ma_7) > 0 else 0
            
            # Volume predictions
            recent_volume_trend = statistics.mean(volumes[-7:]) if len(volumes) >= 7 else 0
            predicted_weekly_volume = recent_volume_trend * 7
            
            return {
                "market_creation_prediction": {
                    "next_7_days": round(predicted_daily_markets * 7, 0),
                    "daily_average": round(predicted_daily_markets, 2),
                    "confidence": "medium" if len(historical_data) > 60 else "low"
                },
                "volume_prediction": {
                    "next_7_days": round(predicted_weekly_volume, 2),
                    "daily_average": round(recent_volume_trend, 2),
                    "confidence": "medium" if len(historical_data) > 60 else "low"
                },
                "trend_analysis": {
                    "market_creation_trend": "increasing" if recent_trend > 0 else "decreasing",
                    "volume_trend": "stable",
                    "seasonality_detected": False
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to generate predictive insights: {str(e)}"}
    
    def _calculate_moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average for trend analysis"""
        if len(data) < window:
            return []
        
        moving_avg = []
        for i in range(window - 1, len(data)):
            avg = statistics.mean(data[i - window + 1:i + 1])
            moving_avg.append(avg)
        
        return moving_avg


# Global BI engine instance
bi_engine = BusinessIntelligenceEngine()


@router.get("/market-analytics")
async def get_market_analytics(
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive market analytics"""
    try:
        # Check cache first
        cache_key = f"market_analytics_{days}"
        cached_result = await enhanced_cache.get(cache_key)
        if cached_result:
            return {
                "success": True,
                "data": cached_result,
                "cached": True,
                "timestamp": time.time()
            }
        
        # Generate analytics
        analytics = await bi_engine.generate_market_analytics(days)
        
        # Cache the result
        await enhanced_cache.set(cache_key, analytics, ttl=300, tags=["analytics", "market"])
        
        return {
            "success": True,
            "data": analytics,
            "cached": False,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market analytics: {str(e)}")


@router.get("/revenue-analytics")
async def get_revenue_analytics(
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Get revenue and financial analytics"""
    try:
        # Check cache first
        cache_key = f"revenue_analytics_{days}"
        cached_result = await enhanced_cache.get(cache_key)
        if cached_result:
            return {
                "success": True,
                "data": cached_result,
                "cached": True,
                "timestamp": time.time()
            }
        
        # Generate analytics
        analytics = await bi_engine.generate_revenue_analytics(days)
        
        # Cache the result
        await enhanced_cache.set(cache_key, analytics, ttl=600, tags=["analytics", "revenue"])
        
        return {
            "success": True,
            "data": analytics,
            "cached": False,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get revenue analytics: {str(e)}")


@router.get("/predictive-insights")
async def get_predictive_insights(current_user: User = Depends(get_current_user)):
    """Get predictive insights and forecasts"""
    try:
        # Check cache first
        cache_key = "predictive_insights"
        cached_result = await enhanced_cache.get(cache_key)
        if cached_result:
            return {
                "success": True,
                "data": cached_result,
                "cached": True,
                "timestamp": time.time()
            }
        
        # Generate insights
        insights = await bi_engine.generate_predictive_insights()
        
        # Cache the result
        await enhanced_cache.set(cache_key, insights, ttl=1800, tags=["analytics", "predictions"])
        
        return {
            "success": True,
            "data": insights,
            "cached": False,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictive insights: {str(e)}")


@router.get("/dashboard-summary")
async def get_dashboard_summary(current_user: User = Depends(get_current_user)):
    """Get comprehensive dashboard summary"""
    try:
        # Check cache first
        cache_key = "dashboard_summary"
        cached_result = await enhanced_cache.get(cache_key)
        if cached_result:
            return {
                "success": True,
                "data": cached_result,
                "cached": True,
                "timestamp": time.time()
            }
        
        # Generate all analytics in parallel
        market_analytics, revenue_analytics, predictive_insights = await asyncio.gather(
            bi_engine.generate_market_analytics(30),
            bi_engine.generate_revenue_analytics(30),
            bi_engine.generate_predictive_insights()
        )
        
        # Combine into summary
        summary = {
            "market_analytics": market_analytics,
            "revenue_analytics": revenue_analytics,
            "predictive_insights": predictive_insights,
            "generated_at": time.time(),
            "period": "30_days"
        }
        
        # Cache the result
        await enhanced_cache.set(cache_key, summary, ttl=300, tags=["analytics", "dashboard", "summary"])
        
        return {
            "success": True,
            "data": summary,
            "cached": False,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard summary: {str(e)}")


@router.post("/generate-report")
async def generate_custom_report(
    report_config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Generate custom analytics report"""
    try:
        report_id = f"report_{int(time.time())}"
        
        # Validate report configuration
        required_fields = ["type", "period_days", "metrics"]
        for field in required_fields:
            if field not in report_config:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Generate report in background
        background_tasks.add_task(
            _generate_report_background,
            report_id,
            report_config,
            current_user.id
        )
        
        return {
            "success": True,
            "report_id": report_id,
            "message": "Report generation started",
            "estimated_completion": "2-5 minutes",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start report generation: {str(e)}")


async def _generate_report_background(report_id: str, config: Dict[str, Any], user_id: int):
    """Background task to generate custom report"""
    try:
        # Generate report based on configuration
        report_data = {
            "report_id": report_id,
            "config": config,
            "generated_at": time.time(),
            "generated_by": user_id,
            "status": "completed"
        }
        
        # Add specific analytics based on type
        if config["type"] == "market_analysis":
            report_data["data"] = await bi_engine.generate_market_analytics(config["period_days"])
        elif config["type"] == "revenue_analysis":
            report_data["data"] = await bi_engine.generate_revenue_analytics(config["period_days"])
        elif config["type"] == "comprehensive":
            market_analytics, revenue_analytics, predictive_insights = await asyncio.gather(
                bi_engine.generate_market_analytics(config["period_days"]),
                bi_engine.generate_revenue_analytics(config["period_days"]),
                bi_engine.generate_predictive_insights()
            )
            report_data["data"] = {
                "market_analytics": market_analytics,
                "revenue_analytics": revenue_analytics,
                "predictive_insights": predictive_insights
            }
        
        # Store report
        await enhanced_cache.set(
            f"report_{report_id}",
            report_data,
            ttl=3600,
            tags=["report", "custom", f"user_{user_id}"]
        )
        
    except Exception as e:
        # Store error in report
        error_report = {
            "report_id": report_id,
            "config": config,
            "generated_at": time.time(),
            "generated_by": user_id,
            "status": "failed",
            "error": str(e)
        }
        await enhanced_cache.set(f"report_{report_id}", error_report, ttl=3600)


@router.get("/reports/{report_id}")
async def get_report(report_id: str, current_user: User = Depends(get_current_user)):
    """Get generated report by ID"""
    try:
        report_data = await enhanced_cache.get(f"report_{report_id}")
        
        if not report_data:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Check if user has access to this report
        if report_data.get("generated_by") != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "success": True,
            "data": report_data,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@router.get("/kpi-summary")
async def get_kpi_summary(current_user: User = Depends(get_current_user)):
    """Get key performance indicators summary"""
    try:
        # Check cache first
        cache_key = "kpi_summary"
        cached_result = await enhanced_cache.get(cache_key)
        if cached_result:
            return {
                "success": True,
                "data": cached_result,
                "cached": True,
                "timestamp": time.time()
            }
        
        # Get current KPIs
        market_analytics = await bi_engine.generate_market_analytics(30)
        revenue_analytics = await bi_engine.generate_revenue_analytics(30)
        
        # Calculate KPIs
        kpis = {
            "market_creation_rate": market_analytics.get("market_trends", {}).get("avg_daily_creation", 0),
            "trading_volume": market_analytics.get("trading_analytics", {}).get("total_volume", 0),
            "user_engagement": market_analytics.get("user_metrics", {}).get("avg_daily_active_users", 0),
            "revenue": revenue_analytics.get("total_revenue", 0),
            "market_resolution_rate": market_analytics.get("market_trends", {}).get("resolution_rate", 0),
            "user_retention": market_analytics.get("user_metrics", {}).get("retention_rate", 0)
        }
        
        # Cache the result
        await enhanced_cache.set(cache_key, kpis, ttl=600, tags=["analytics", "kpi", "summary"])
        
        return {
            "success": True,
            "data": kpis,
            "cached": False,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get KPI summary: {str(e)}")
