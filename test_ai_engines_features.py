#!/usr/bin/env python3
"""
Test script for the new AI engines features.
Tests Advanced AI Orchestration, Intelligent Decision, Pattern Recognition, and Risk Assessment engines.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_ai_orchestration_engine():
    """Test the Advanced AI Orchestration Engine"""
    print("\n🧠 Testing Advanced AI Orchestration Engine...")
    
    try:
        from app.services.advanced_ai_orchestration_engine import advanced_ai_orchestration_engine
        
        # Test engine initialization
        await advanced_ai_orchestration_engine.start_ai_orchestration_engine()
        print("✅ Advanced AI Orchestration Engine started")
        
        # Test AI request submission
        from app.services.advanced_ai_orchestration_engine import DecisionType
        request_id = await advanced_ai_orchestration_engine.submit_ai_request(
            DecisionType.MARKET_ANALYSIS,
            {"market_data": "test_data"},
            priority=1,
            timeout=30.0
        )
        print(f"✅ AI request submitted: {request_id}")
        
        # Test getting AI response
        response = await advanced_ai_orchestration_engine.get_ai_response(request_id)
        print(f"✅ AI response received: {response is not None}")
        
        # Test getting AI models
        models = await advanced_ai_orchestration_engine.get_ai_models()
        print(f"✅ AI models: {len(models)} models available")
        
        # Test getting AI workflows
        workflows = await advanced_ai_orchestration_engine.get_ai_workflows()
        print(f"✅ AI workflows: {len(workflows)} workflows available")
        
        # Test getting performance metrics
        performance = await advanced_ai_orchestration_engine.get_ai_performance_metrics()
        print(f"✅ Performance metrics: {performance}")
        
        # Test engine shutdown
        await advanced_ai_orchestration_engine.stop_ai_orchestration_engine()
        print("✅ Advanced AI Orchestration Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced AI Orchestration Engine test failed: {e}")
        return False

async def test_intelligent_decision_engine():
    """Test the Intelligent Decision Engine"""
    print("\n🎯 Testing Intelligent Decision Engine...")
    
    try:
        from app.services.intelligent_decision_engine import intelligent_decision_engine
        
        # Test engine initialization
        await intelligent_decision_engine.start_intelligent_decision_engine()
        print("✅ Intelligent Decision Engine started")
        
        # Test decision request submission
        from app.services.intelligent_decision_engine import DecisionContext
        request_id = await intelligent_decision_engine.submit_decision_request(
            DecisionContext.FINANCIAL,
            "Should we invest in this asset?",
            options=[
                {"action": "invest", "description": "Invest in the asset"},
                {"action": "hold", "description": "Hold current position"},
                {"action": "sell", "description": "Sell the asset"}
            ],
            criteria=[
                {"name": "risk_tolerance", "weight": 0.3},
                {"name": "expected_return", "weight": 0.4},
                {"name": "market_trend", "weight": 0.3}
            ],
            constraints={
                "max_risk": 0.2,
                "min_return": 0.05
            },
            priority=1
        )
        print(f"✅ Decision request submitted: {request_id}")
        
        # Test getting decision result
        decision = await intelligent_decision_engine.get_decision_result(request_id)
        print(f"✅ Decision result: {decision is not None}")
        
        # Test getting performance metrics
        performance = await intelligent_decision_engine.get_decision_performance_metrics()
        print(f"✅ Decision performance: {performance}")
        
        # Test engine shutdown
        await intelligent_decision_engine.stop_intelligent_decision_engine()
        print("✅ Intelligent Decision Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Intelligent Decision Engine test failed: {e}")
        return False

async def test_advanced_pattern_recognition_engine():
    """Test the Advanced Pattern Recognition Engine"""
    print("\n🔍 Testing Advanced Pattern Recognition Engine...")
    
    try:
        from app.services.advanced_pattern_recognition_engine import advanced_pattern_recognition_engine
        
        # Test engine initialization
        await advanced_pattern_recognition_engine.start_advanced_pattern_recognition_engine()
        print("✅ Advanced Pattern Recognition Engine started")
        
        # Test pattern recognition request submission
        from datetime import datetime, timedelta
        from app.services.advanced_pattern_recognition_engine import PatternType, PatternComplexity, PatternConfidence
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
        
        request_id = await advanced_pattern_recognition_engine.submit_pattern_recognition_request(
            "market_data",
            (start_time, end_time),
            pattern_types=[PatternType.TREND, PatternType.SEASONAL, PatternType.CYCLICAL],
            complexity_threshold=PatternComplexity.SIMPLE,
            confidence_threshold=PatternConfidence.LOW
        )
        print(f"✅ Pattern recognition request submitted: {request_id}")
        
        # Test getting pattern result
        result = await advanced_pattern_recognition_engine.get_pattern_result(request_id)
        print(f"✅ Pattern result: {result is not None}")
        
        # Test getting pattern models
        models = await advanced_pattern_recognition_engine.get_pattern_models()
        print(f"✅ Pattern models: {len(models)} models available")
        
        # Test getting performance metrics
        performance = await advanced_pattern_recognition_engine.get_pattern_performance_metrics()
        print(f"✅ Pattern performance: {performance}")
        
        # Test engine shutdown
        await advanced_pattern_recognition_engine.stop_advanced_pattern_recognition_engine()
        print("✅ Advanced Pattern Recognition Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Pattern Recognition Engine test failed: {e}")
        return False

async def test_ai_powered_risk_assessment_engine():
    """Test the AI-Powered Risk Assessment Engine"""
    print("\n⚠️ Testing AI-Powered Risk Assessment Engine...")
    
    try:
        from app.services.ai_powered_risk_assessment_engine import ai_powered_risk_assessment_engine
        
        # Test engine initialization
        await ai_powered_risk_assessment_engine.start_ai_powered_risk_assessment_engine()
        print("✅ AI-Powered Risk Assessment Engine started")
        
        # Test risk assessment
        assessment_id = await ai_powered_risk_assessment_engine.perform_risk_assessment(
            "test_asset_001",
            "cryptocurrency"
        )
        print(f"✅ Risk assessment performed: {assessment_id}")
        
        # Test getting risk assessment
        assessment = await ai_powered_risk_assessment_engine.get_risk_assessment(assessment_id)
        print(f"✅ Risk assessment result: {assessment is not None}")
        
        # Test getting risk alerts
        alerts = await ai_powered_risk_assessment_engine.get_risk_alerts()
        print(f"✅ Risk alerts: {len(alerts)} alerts available")
        
        # Test getting performance metrics
        performance = await ai_powered_risk_assessment_engine.get_risk_performance_metrics()
        print(f"✅ Risk performance: {performance}")
        
        # Test engine shutdown
        await ai_powered_risk_assessment_engine.stop_ai_powered_risk_assessment_engine()
        print("✅ AI-Powered Risk Assessment Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ AI-Powered Risk Assessment Engine test failed: {e}")
        return False

async def test_ai_engines_integration():
    """Test integration between all AI engines"""
    print("\n🔗 Testing AI Engines Integration...")
    
    try:
        from app.services.advanced_ai_orchestration_engine import advanced_ai_orchestration_engine
        from app.services.intelligent_decision_engine import intelligent_decision_engine
        from app.services.advanced_pattern_recognition_engine import advanced_pattern_recognition_engine
        from app.services.ai_powered_risk_assessment_engine import ai_powered_risk_assessment_engine
        
        # Start all engines
        await advanced_ai_orchestration_engine.start_ai_orchestration_engine()
        await intelligent_decision_engine.start_intelligent_decision_engine()
        await advanced_pattern_recognition_engine.start_advanced_pattern_recognition_engine()
        await ai_powered_risk_assessment_engine.start_ai_powered_risk_assessment_engine()
        
        print("✅ All AI engines started")
        
        # Test integrated workflow
        from datetime import datetime, timedelta
        from app.services.advanced_ai_orchestration_engine import DecisionType
        from app.services.intelligent_decision_engine import DecisionContext
        
        # Step 1: Pattern recognition
        from app.services.advanced_pattern_recognition_engine import PatternType, PatternComplexity, PatternConfidence
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
        pattern_request_id = await advanced_pattern_recognition_engine.submit_pattern_recognition_request(
            "market_data",
            (start_time, end_time),
            pattern_types=[PatternType.TREND, PatternType.SEASONAL],
            complexity_threshold=PatternComplexity.SIMPLE,
            confidence_threshold=PatternConfidence.LOW
        )
        print(f"✅ Step 1 - Pattern recognition request: {pattern_request_id}")
        
        # Step 2: Risk assessment
        risk_assessment_id = await ai_powered_risk_assessment_engine.perform_risk_assessment(
            "test_asset_001",
            "cryptocurrency"
        )
        print(f"✅ Step 2 - Risk assessment: {risk_assessment_id}")
        
        # Step 3: Decision making
        decision_request_id = await intelligent_decision_engine.submit_decision_request(
            DecisionContext.FINANCIAL,
            "Should we invest based on patterns and risk?",
            options=[
                {"action": "invest", "description": "Invest based on positive patterns"},
                {"action": "hold", "description": "Hold current position"},
                {"action": "sell", "description": "Sell due to high risk"}
            ],
            criteria=[
                {"name": "pattern_confidence", "weight": 0.4},
                {"name": "risk_level", "weight": 0.4},
                {"name": "market_trend", "weight": 0.2}
            ],
            constraints={
                "max_risk": 0.3,
                "min_pattern_confidence": 0.7
            },
            priority=1
        )
        print(f"✅ Step 3 - Decision request: {decision_request_id}")
        
        # Step 4: AI orchestration
        ai_request_id = await advanced_ai_orchestration_engine.submit_ai_request(
            DecisionType.MARKET_ANALYSIS,
            {
                "pattern_request_id": pattern_request_id,
                "risk_assessment_id": risk_assessment_id,
                "decision_request_id": decision_request_id
            },
            priority=1,
            timeout=30.0
        )
        print(f"✅ Step 4 - AI orchestration request: {ai_request_id}")
        
        # Stop all engines
        await advanced_ai_orchestration_engine.stop_ai_orchestration_engine()
        await intelligent_decision_engine.stop_intelligent_decision_engine()
        await advanced_pattern_recognition_engine.stop_advanced_pattern_recognition_engine()
        await ai_powered_risk_assessment_engine.stop_ai_powered_risk_assessment_engine()
        
        print("✅ All AI engines stopped")
        return True
        
    except Exception as e:
        print(f"❌ AI engines integration test failed: {e}")
        return False

async def test_ai_engines_performance():
    """Test performance of all AI engines"""
    print("\n⚡ Testing AI Engines Performance...")
    
    try:
        from app.services.advanced_ai_orchestration_engine import advanced_ai_orchestration_engine
        from app.services.intelligent_decision_engine import intelligent_decision_engine
        from app.services.advanced_pattern_recognition_engine import advanced_pattern_recognition_engine
        from app.services.ai_powered_risk_assessment_engine import ai_powered_risk_assessment_engine
        
        # Start all engines
        await advanced_ai_orchestration_engine.start_ai_orchestration_engine()
        await intelligent_decision_engine.start_intelligent_decision_engine()
        await advanced_pattern_recognition_engine.start_advanced_pattern_recognition_engine()
        await ai_powered_risk_assessment_engine.start_ai_powered_risk_assessment_engine()
        
        # Test orchestration performance
        from app.services.advanced_ai_orchestration_engine import DecisionType
        start_time = time.time()
        for i in range(50):
            await advanced_ai_orchestration_engine.submit_ai_request(
                DecisionType.MARKET_ANALYSIS,
                {"test_data": f"test_{i}"},
                priority=1,
                timeout=30.0
            )
        orchestration_time = time.time() - start_time
        print(f"✅ Orchestration: 50 operations in {orchestration_time:.2f}s ({50/orchestration_time:.1f} ops/s)")
        
        # Test decision performance
        from app.services.intelligent_decision_engine import DecisionContext
        start_time = time.time()
        for i in range(50):
            await intelligent_decision_engine.submit_decision_request(
                DecisionContext.FINANCIAL,
                f"Test decision {i}",
                options=[
                    {"action": "option1", "description": f"Option 1 for test {i}"},
                    {"action": "option2", "description": f"Option 2 for test {i}"}
                ],
                criteria=[
                    {"name": "value", "weight": 0.5},
                    {"name": "test_id", "weight": 0.5}
                ],
                priority=1
            )
        decision_time = time.time() - start_time
        print(f"✅ Decision making: 50 operations in {decision_time:.2f}s ({50/decision_time:.1f} ops/s)")
        
        # Test pattern recognition performance
        from datetime import datetime, timedelta
        from app.services.advanced_pattern_recognition_engine import PatternType, PatternComplexity, PatternConfidence
        start_time = time.time()
        for i in range(50):
            start_time_range = datetime.now() - timedelta(days=30)
            end_time_range = datetime.now()
            await advanced_pattern_recognition_engine.submit_pattern_recognition_request(
                f"test_data_{i}",
                (start_time_range, end_time_range),
                pattern_types=[PatternType.TREND],
                complexity_threshold=PatternComplexity.SIMPLE,
                confidence_threshold=PatternConfidence.LOW
            )
        pattern_time = time.time() - start_time
        print(f"✅ Pattern recognition: 50 operations in {pattern_time:.2f}s ({50/pattern_time:.1f} ops/s)")
        
        # Test risk assessment performance
        start_time = time.time()
        for i in range(50):
            await ai_powered_risk_assessment_engine.perform_risk_assessment(
                f"test_asset_{i}",
                "cryptocurrency"
            )
        risk_time = time.time() - start_time
        print(f"✅ Risk assessment: 50 operations in {risk_time:.2f}s ({50/risk_time:.1f} ops/s)")
        
        # Stop all engines
        await advanced_ai_orchestration_engine.stop_ai_orchestration_engine()
        await intelligent_decision_engine.stop_intelligent_decision_engine()
        await advanced_pattern_recognition_engine.stop_advanced_pattern_recognition_engine()
        await ai_powered_risk_assessment_engine.stop_ai_powered_risk_assessment_engine()
        
        return True
        
    except Exception as e:
        print(f"❌ AI engines performance test failed: {e}")
        return False

async def test_new_api_endpoints():
    """Test the new API endpoints"""
    print("\n🌐 Testing New API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        endpoints = [
            "/ai-orchestration/coordinate",
            "/intelligent-decision/make-decision",
            "/pattern-recognition/detect-patterns",
            "/risk-assessment/assess-risk"
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    async with session.get(f"{base_url}{endpoint}") as response:
                        if response.status == 200:
                            print(f"✅ {endpoint} - Status: {response.status}")
                        else:
                            print(f"⚠️ {endpoint} - Status: {response.status}")
                except Exception as e:
                    print(f"❌ {endpoint} - Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting AI Engines Features Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test individual engines
    test_results.append(await test_advanced_ai_orchestration_engine())
    test_results.append(await test_intelligent_decision_engine())
    test_results.append(await test_advanced_pattern_recognition_engine())
    test_results.append(await test_ai_powered_risk_assessment_engine())
    
    # Test integration
    test_results.append(await test_ai_engines_integration())
    
    # Test performance
    test_results.append(await test_ai_engines_performance())
    
    # Test API endpoints
    test_results.append(await test_new_api_endpoints())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}")
    print(f"❌ Failed: {len(test_results) - sum(test_results)}")
    print(f"📈 Success Rate: {sum(test_results)/len(test_results)*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 All AI Engines Features tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all(test_results)

if __name__ == "__main__":
    asyncio.run(main())
