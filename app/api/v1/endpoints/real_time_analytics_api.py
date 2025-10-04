"""
Real-time Analytics API
API endpoints for real-time analytics and streaming data processing
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.real_time_analytics_engine import real_time_analytics_engine, StreamType, AnalyticsType, ProcessingMode

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class DataIngestionRequest(BaseModel):
    """Data ingestion request model"""
    stream_type: str
    data: Dict[str, Any]
    source: str = "api"


class StreamProcessorRequest(BaseModel):
    """Stream processor request model"""
    stream_type: str
    processing_mode: str
    window_size: int
    batch_size: int
    filters: Optional[List[Dict[str, Any]]] = None
    transformations: Optional[List[Dict[str, Any]]] = None
    aggregations: Optional[List[Dict[str, Any]]] = None


# API Endpoints
@router.get("/status")
async def get_real_time_analytics_status():
    """Get real-time analytics system status"""
    try:
        summary = real_time_analytics_engine.get_analytics_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting real-time analytics status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_data(data_request: DataIngestionRequest):
    """Ingest data into real-time analytics stream"""
    try:
        stream_type = StreamType(data_request.stream_type)
        
        success = await real_time_analytics_engine.ingest_data(
            stream_type=stream_type,
            data=data_request.data,
            source=data_request.source
        )
        
        return JSONResponse(content={
            "message": "Data ingested successfully" if success else "Failed to ingest data",
            "stream_type": data_request.stream_type,
            "source": data_request.source,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams")
async def get_data_streams():
    """Get data streams information"""
    try:
        streams_info = {}
        for stream_type, stream in real_time_analytics_engine.data_streams.items():
            streams_info[stream_type.value] = {
                "stream_type": stream_type.value,
                "data_points": len(stream),
                "latest_timestamp": stream[-1].timestamp.isoformat() if stream else None,
                "data_quality": sum(dp.quality_score for dp in stream) / len(stream) if stream else 0,
                "sample_data": [
                    {
                        "timestamp": dp.timestamp.isoformat(),
                        "data": dp.data,
                        "source": dp.source,
                        "quality_score": dp.quality_score
                    }
                    for dp in list(stream)[-5:]  # Last 5 data points
                ]
            }
            
        return JSONResponse(content={"streams": streams_info})
        
    except Exception as e:
        logger.error(f"Error getting data streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_type}")
async def get_stream_data(stream_type: str, limit: int = 100):
    """Get data from specific stream"""
    try:
        stream_enum = StreamType(stream_type)
        stream = real_time_analytics_engine.data_streams[stream_enum]
        
        # Get recent data points
        recent_data = list(stream)[-limit:] if stream else []
        
        stream_data = [
            {
                "timestamp": dp.timestamp.isoformat(),
                "data": dp.data,
                "source": dp.source,
                "quality_score": dp.quality_score,
                "metadata": dp.metadata
            }
            for dp in recent_data
        ]
        
        return JSONResponse(content={
            "stream_type": stream_type,
            "data_points": stream_data,
            "total_points": len(stream),
            "limit": limit
        })
        
    except Exception as e:
        logger.error(f"Error getting stream data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics_results(analytics_type: Optional[str] = None, limit: int = 50):
    """Get analytics results"""
    try:
        results = []
        for result_id, result in real_time_analytics_engine.analytics_results.items():
            # Filter by type if specified
            if analytics_type and result.analytics_type.value != analytics_type:
                continue
                
            results.append({
                "result_id": result.result_id,
                "analytics_type": result.analytics_type.value,
                "timestamp": result.timestamp.isoformat(),
                "data": result.data,
                "metrics": result.metrics,
                "insights": result.insights,
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms,
                "metadata": result.metadata
            })
            
        # Sort by timestamp and limit
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        results = results[:limit]
        
        return JSONResponse(content={"analytics_results": results})
        
    except Exception as e:
        logger.error(f"Error getting analytics results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{result_id}")
async def get_analytics_result(result_id: str):
    """Get specific analytics result"""
    try:
        result = real_time_analytics_engine.analytics_results.get(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Analytics result not found")
            
        return JSONResponse(content={
            "result_id": result.result_id,
            "analytics_type": result.analytics_type.value,
            "timestamp": result.timestamp.isoformat(),
            "data": result.data,
            "metrics": result.metrics,
            "insights": result.insights,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms,
            "metadata": result.metadata
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processors")
async def get_stream_processors():
    """Get stream processors information"""
    try:
        processors = []
        for processor_id, processor in real_time_analytics_engine.stream_processors.items():
            processors.append({
                "processor_id": processor.processor_id,
                "stream_type": processor.stream_type.value,
                "processing_mode": processor.processing_mode.value,
                "window_size": processor.window_size,
                "batch_size": processor.batch_size,
                "filters": processor.filters,
                "transformations": processor.transformations,
                "aggregations": processor.aggregations,
                "output_schema": processor.output_schema,
                "status": processor.status
            })
            
        return JSONResponse(content={"processors": processors})
        
    except Exception as e:
        logger.error(f"Error getting stream processors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processors")
async def create_stream_processor(processor_request: StreamProcessorRequest):
    """Create new stream processor"""
    try:
        processor_id = f"processor_{int(time.time())}_{secrets.token_hex(4)}"
        
        processor = StreamProcessor(
            processor_id=processor_id,
            stream_type=StreamType(processor_request.stream_type),
            processing_mode=ProcessingMode(processor_request.processing_mode),
            window_size=processor_request.window_size,
            batch_size=processor_request.batch_size,
            filters=processor_request.filters or [],
            transformations=processor_request.transformations or [],
            aggregations=processor_request.aggregations or [],
            output_schema={}
        )
        
        real_time_analytics_engine.stream_processors[processor_id] = processor
        
        return JSONResponse(content={
            "message": "Stream processor created successfully",
            "processor_id": processor_id,
            "stream_type": processor_request.stream_type,
            "processing_mode": processor_request.processing_mode
        })
        
    except Exception as e:
        logger.error(f"Error creating stream processor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_real_time_dashboard():
    """Get real-time analytics dashboard data"""
    try:
        summary = real_time_analytics_engine.get_analytics_summary()
        
        # Get latest analytics results
        latest_results = list(real_time_analytics_engine.analytics_results.values())[-10:]
        
        # Get stream health
        stream_health = {}
        for stream_type, stream in real_time_analytics_engine.data_streams.items():
            stream_health[stream_type.value] = {
                "data_points": len(stream),
                "health_status": "healthy" if len(stream) > 0 else "empty",
                "latest_activity": stream[-1].timestamp.isoformat() if stream else None
            }
            
        dashboard_data = {
            "summary": summary,
            "stream_health": stream_health,
            "latest_analytics": [
                {
                    "result_id": result.result_id,
                    "analytics_type": result.analytics_type.value,
                    "timestamp": result.timestamp.isoformat(),
                    "insights": result.insights,
                    "confidence": result.confidence,
                    "processing_time_ms": result.processing_time_ms
                }
                for result in latest_results
            ],
            "performance_metrics": {
                "data_points_processed": summary.get("total_data_points", 0),
                "analytics_generated": summary.get("total_analytics_results", 0),
                "anomalies_detected": summary.get("stats", {}).get("anomalies_detected", 0),
                "trends_identified": summary.get("stats", {}).get("trends_identified", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting real-time dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_analytics_metrics():
    """Get comprehensive analytics metrics"""
    try:
        summary = real_time_analytics_engine.get_analytics_summary()
        
        return JSONResponse(content={
            "overall_metrics": {
                "total_data_points": summary.get("total_data_points", 0),
                "total_analytics_results": summary.get("total_analytics_results", 0),
                "total_processors": summary.get("total_processors", 0),
                "total_subscribers": summary.get("total_subscribers", 0)
            },
            "stream_metrics": summary.get("stream_stats", {}),
            "analytics_metrics": summary.get("analytics_by_type", {}),
            "processor_metrics": summary.get("processor_stats", {}),
            "performance_metrics": summary.get("stats", {}),
            "timestamp": summary.get("timestamp")
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_real_time_analytics():
    """Start real-time analytics engine"""
    try:
        await real_time_analytics_engine.start_analytics_engine()
        return JSONResponse(content={"message": "Real-time analytics engine started"})
        
    except Exception as e:
        logger.error(f"Error starting real-time analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_real_time_analytics():
    """Stop real-time analytics engine"""
    try:
        await real_time_analytics_engine.stop_analytics_engine()
        return JSONResponse(content={"message": "Real-time analytics engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping real-time analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
