"""
AI Insights API
API endpoints for AI-powered insights and recommendations
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.ai_insights_engine import ai_insights_engine, InsightType, RecommendationType, ConfidenceLevel

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class InsightRequest(BaseModel):
    """Insight request model"""
    insight_type: str
    data: Dict[str, Any]
    user_id: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Recommendation request model"""
    recommendation_type: str
    data: Dict[str, Any]
    user_id: Optional[str] = None


class UserProfileRequest(BaseModel):
    """User profile request model"""
    user_id: str
    preferences: Dict[str, Any]
    trading_history: Optional[List[Dict[str, Any]]] = None


# API Endpoints
@router.get("/status")
async def get_ai_insights_status():
    """Get AI insights system status"""
    try:
        summary = ai_insights_engine.get_ai_insights_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting AI insights status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insights")
async def generate_insight(insight_request: InsightRequest):
    """Generate AI-powered insight"""
    try:
        insight_type = InsightType(insight_request.insight_type)
        
        insight = await ai_insights_engine.generate_insight(
            insight_type=insight_type,
            data=insight_request.data,
            user_id=insight_request.user_id
        )
        
        return JSONResponse(content={
            "message": "AI insight generated successfully",
            "insight_id": insight.insight_id,
            "title": insight.title,
            "description": insight.description,
            "confidence_level": insight.confidence_level.value,
            "confidence_score": insight.confidence_score,
            "impact_score": insight.impact_score,
            "generated_at": insight.generated_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating insight: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_insights(user_id: Optional[str] = None, insight_type: Optional[str] = None):
    """Get AI insights"""
    try:
        insights = []
        for insight_id, insight in ai_insights_engine.insights.items():
            # Filter by user if specified
            if user_id and user_id not in ai_insights_engine.user_profiles:
                continue
                
            # Filter by type if specified
            if insight_type and insight.insight_type.value != insight_type:
                continue
                
            insights.append({
                "insight_id": insight.insight_id,
                "insight_type": insight.insight_type.value,
                "title": insight.title,
                "description": insight.description,
                "confidence_level": insight.confidence_level.value,
                "confidence_score": insight.confidence_score,
                "impact_score": insight.impact_score,
                "data_points": insight.data_points,
                "generated_at": insight.generated_at.isoformat(),
                "expires_at": insight.expires_at.isoformat(),
                "tags": insight.tags,
                "metadata": insight.metadata
            })
            
        return JSONResponse(content={"insights": insights})
        
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{insight_id}")
async def get_insight(insight_id: str):
    """Get specific AI insight"""
    try:
        insight = ai_insights_engine.insights.get(insight_id)
        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")
            
        return JSONResponse(content={
            "insight_id": insight.insight_id,
            "insight_type": insight.insight_type.value,
            "title": insight.title,
            "description": insight.description,
            "confidence_level": insight.confidence_level.value,
            "confidence_score": insight.confidence_score,
            "impact_score": insight.impact_score,
            "data_points": insight.data_points,
            "generated_at": insight.generated_at.isoformat(),
            "expires_at": insight.expires_at.isoformat(),
            "tags": insight.tags,
            "metadata": insight.metadata
        })
        
    except Exception as e:
        logger.error(f"Error getting insight: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations")
async def generate_recommendation(recommendation_request: RecommendationRequest):
    """Generate AI-powered recommendation"""
    try:
        recommendation_type = RecommendationType(recommendation_request.recommendation_type)
        
        recommendation = await ai_insights_engine.generate_recommendation(
            recommendation_type=recommendation_type,
            data=recommendation_request.data,
            user_id=recommendation_request.user_id
        )
        
        return JSONResponse(content={
            "message": "AI recommendation generated successfully",
            "recommendation_id": recommendation.recommendation_id,
            "title": recommendation.title,
            "description": recommendation.description,
            "action": recommendation.action,
            "confidence_level": recommendation.confidence_level.value,
            "confidence_score": recommendation.confidence_score,
            "expected_return": recommendation.expected_return,
            "risk_level": recommendation.risk_level,
            "time_horizon": recommendation.time_horizon,
            "generated_at": recommendation.generated_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(user_id: Optional[str] = None, recommendation_type: Optional[str] = None):
    """Get AI recommendations"""
    try:
        recommendations = []
        for rec_id, rec in ai_insights_engine.recommendations.items():
            # Filter by user if specified
            if user_id and user_id not in ai_insights_engine.user_profiles:
                continue
                
            # Filter by type if specified
            if recommendation_type and rec.recommendation_type.value != recommendation_type:
                continue
                
            recommendations.append({
                "recommendation_id": rec.recommendation_id,
                "recommendation_type": rec.recommendation_type.value,
                "title": rec.title,
                "description": rec.description,
                "action": rec.action,
                "confidence_level": rec.confidence_level.value,
                "confidence_score": rec.confidence_score,
                "expected_return": rec.expected_return,
                "risk_level": rec.risk_level,
                "time_horizon": rec.time_horizon,
                "generated_at": rec.generated_at.isoformat(),
                "expires_at": rec.expires_at.isoformat(),
                "supporting_insights": rec.supporting_insights,
                "prerequisites": rec.prerequisites,
                "metadata": rec.metadata
            })
            
        return JSONResponse(content={"recommendations": recommendations})
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{recommendation_id}")
async def get_recommendation(recommendation_id: str):
    """Get specific AI recommendation"""
    try:
        recommendation = ai_insights_engine.recommendations.get(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
            
        return JSONResponse(content={
            "recommendation_id": recommendation.recommendation_id,
            "recommendation_type": recommendation.recommendation_type.value,
            "title": recommendation.title,
            "description": recommendation.description,
            "action": recommendation.action,
            "confidence_level": recommendation.confidence_level.value,
            "confidence_score": recommendation.confidence_score,
            "expected_return": recommendation.expected_return,
            "risk_level": recommendation.risk_level,
            "time_horizon": recommendation.time_horizon,
            "generated_at": recommendation.generated_at.isoformat(),
            "expires_at": recommendation.expires_at.isoformat(),
            "supporting_insights": recommendation.supporting_insights,
            "prerequisites": recommendation.prerequisites,
            "metadata": recommendation.metadata
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_ai_models():
    """Get AI models information"""
    try:
        models = []
        for model_id, model in ai_insights_engine.ai_models.items():
            models.append({
                "model_id": model.model_id,
                "model_name": model.model_name,
                "model_type": model.model_type,
                "version": model.version,
                "accuracy": model.accuracy,
                "last_trained": model.last_trained.isoformat(),
                "training_data_size": model.training_data_size,
                "features": model.features,
                "hyperparameters": model.hyperparameters,
                "performance_metrics": model.performance_metrics,
                "status": model.status
            })
            
        return JSONResponse(content={"models": models})
        
    except Exception as e:
        logger.error(f"Error getting AI models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-profiles")
async def get_user_profiles():
    """Get user profiles for personalization"""
    try:
        profiles = []
        for user_id, profile in ai_insights_engine.user_profiles.items():
            profiles.append({
                "user_id": profile["user_id"],
                "insights_count": len(profile.get("insights", [])),
                "recommendations_count": len(profile.get("recommendations", [])),
                "last_insight": profile.get("last_insight"),
                "last_recommendation": profile.get("last_recommendation"),
                "created_at": profile.get("created_at"),
                "last_updated": profile.get("last_updated")
            })
            
        return JSONResponse(content={"user_profiles": profiles})
        
    except Exception as e:
        logger.error(f"Error getting user profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user-profiles")
async def update_user_profile(profile_request: UserProfileRequest):
    """Update user profile for personalization"""
    try:
        user_id = profile_request.user_id
        
        if user_id not in ai_insights_engine.user_profiles:
            ai_insights_engine.user_profiles[user_id] = {
                "user_id": user_id,
                "insights": [],
                "recommendations": [],
                "preferences": {},
                "created_at": datetime.now().isoformat()
            }
            
        profile = ai_insights_engine.user_profiles[user_id]
        profile["preferences"].update(profile_request.preferences)
        profile["last_updated"] = datetime.now().isoformat()
        
        if profile_request.trading_history:
            profile["trading_history"] = profile_request.trading_history
            
        return JSONResponse(content={
            "message": "User profile updated successfully",
            "user_id": user_id,
            "updated_at": profile["last_updated"]
        })
        
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_ai_insights_dashboard():
    """Get AI insights dashboard data"""
    try:
        summary = ai_insights_engine.get_ai_insights_summary()
        
        # Get recent insights
        recent_insights = list(ai_insights_engine.insights.values())[-5:]
        
        # Get recent recommendations
        recent_recommendations = list(ai_insights_engine.recommendations.values())[-5:]
        
        dashboard_data = {
            "summary": summary,
            "recent_insights": [
                {
                    "insight_id": insight.insight_id,
                    "insight_type": insight.insight_type.value,
                    "title": insight.title,
                    "confidence_score": insight.confidence_score,
                    "impact_score": insight.impact_score,
                    "generated_at": insight.generated_at.isoformat()
                }
                for insight in recent_insights
            ],
            "recent_recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "recommendation_type": rec.recommendation_type.value,
                    "title": rec.title,
                    "confidence_score": rec.confidence_score,
                    "expected_return": rec.expected_return,
                    "risk_level": rec.risk_level,
                    "generated_at": rec.generated_at.isoformat()
                }
                for rec in recent_recommendations
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting AI insights dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_ai_insights():
    """Start AI insights engine"""
    try:
        await ai_insights_engine.start_ai_insights_engine()
        return JSONResponse(content={"message": "AI insights engine started"})
        
    except Exception as e:
        logger.error(f"Error starting AI insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_ai_insights():
    """Stop AI insights engine"""
    try:
        await ai_insights_engine.stop_ai_insights_engine()
        return JSONResponse(content={"message": "AI insights engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping AI insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))
