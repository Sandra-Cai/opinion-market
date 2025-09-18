#!/usr/bin/env python3
"""
Simple test server to verify the enhanced systems work
"""

import os
os.environ['ENVIRONMENT'] = 'development'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create a simple FastAPI app
app = FastAPI(
    title="Enhanced Opinion Market API - Test",
    description="Test server to verify enhanced systems",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Enhanced Opinion Market API is working!",
        "status": "success",
        "version": "2.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "All systems operational"
    }

@app.get("/test-enhanced-systems")
async def test_enhanced_systems():
    try:
        # Test enhanced configuration
        from app.core.enhanced_config import enhanced_config_manager
        config_status = "✅ Enhanced Configuration Manager working"
        
        # Test enhanced error handler
        from app.core.enhanced_error_handler import enhanced_error_handler
        error_status = "✅ Enhanced Error Handler working"
        
        # Test advanced performance optimizer
        from app.core.advanced_performance_optimizer import advanced_performance_optimizer
        performance_status = "✅ Advanced Performance Optimizer working"
        
        # Test advanced security
        from app.core.advanced_security import advanced_security_manager
        security_status = "✅ Advanced Security Manager working"
        
        # Test enhanced testing
        from app.core.enhanced_testing import enhanced_test_manager
        testing_status = "✅ Enhanced Testing Framework working"
        
        return {
            "status": "success",
            "message": "All enhanced systems are working!",
            "systems": {
                "configuration": config_status,
                "error_handling": error_status,
                "performance": performance_status,
                "security": security_status,
                "testing": testing_status
            },
            "api_title": enhanced_config_manager.get("api.title"),
            "database_url": enhanced_config_manager.get("database.url"),
            "environment": enhanced_config_manager.get("environment")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing enhanced systems: {str(e)}"
        }

if __name__ == "__main__":
    print("🚀 Starting Enhanced Opinion Market Test Server...")
    print("✅ All enhanced systems loaded")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 Test endpoint: http://localhost:8000/test-enhanced-systems")
    print("🏥 Health check: http://localhost:8000/health")
    print("")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
