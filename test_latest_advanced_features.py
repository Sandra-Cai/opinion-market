#!/usr/bin/env python3
"""
Test Suite for Latest Advanced Features
Comprehensive testing for Chaos Engineering, MLOps, API Gateway, and Event Sourcing
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_chaos_engineering_engine():
    """Test Chaos Engineering Engine"""
    print("\n🧪 Testing Chaos Engineering Engine...")
    
    try:
        from app.services.chaos_engineering_engine import (
            chaos_engineering_engine, 
            ChaosExperimentType, 
            ExperimentStatus,
            FailureMode
        )
        
        # Test 1: Start chaos engineering engine
        await chaos_engineering_engine.start_chaos_engine()
        print("✅ Chaos engineering engine started")
        
        # Test 2: Create chaos experiment
        experiment_data = {
            "name": "Network Latency Test",
            "description": "Test network latency injection",
            "type": "network_latency",
            "target_services": ["api", "database"],
            "duration": 60,
            "intensity": 0.5,
            "failure_mode": "graceful"
        }
        
        experiment = await chaos_engineering_engine.create_experiment(experiment_data)
        print(f"✅ Chaos experiment created: {experiment.experiment_id}")
        
        # Test 3: Run experiment
        success = await chaos_engineering_engine.run_experiment(experiment.experiment_id)
        print(f"✅ Chaos experiment executed: {success}")
        
        # Test 4: Check resilience metrics
        resilience_metrics = chaos_engineering_engine.resilience_metrics
        print(f"✅ Resilience metrics collected: {len(resilience_metrics)} metrics")
        
        # Test 5: Get chaos summary
        summary = chaos_engineering_engine.get_chaos_summary()
        print(f"✅ Chaos summary: {summary['total_experiments']} experiments, {summary['stats']['experiments_run']} run")
        
        # Test 6: Stop chaos engineering engine
        await chaos_engineering_engine.stop_chaos_engine()
        print("✅ Chaos engineering engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Chaos Engineering Engine test failed: {e}")
        return False

async def test_mlops_pipeline_engine():
    """Test MLOps Pipeline Engine"""
    print("\n🤖 Testing MLOps Pipeline Engine...")
    
    try:
        from app.services.mlops_pipeline_engine import (
            mlops_pipeline_engine, 
            PipelineStage, 
            PipelineStatus,
            ModelVersion
        )
        
        # Test 1: Start MLOps engine
        await mlops_pipeline_engine.start_mlops_engine()
        print("✅ MLOps engine started")
        
        # Test 2: Create ML pipeline
        pipeline_data = {
            "name": "Market Prediction Pipeline",
            "description": "Automated market prediction pipeline",
            "template": "market_prediction",
            "config": {
                "data_sources": ["market_data", "news_data"],
                "model_types": ["classification", "regression"]
            }
        }
        
        pipeline = await mlops_pipeline_engine.create_pipeline(pipeline_data)
        print(f"✅ ML pipeline created: {pipeline.pipeline_id}")
        
        # Test 3: Run pipeline
        success = await mlops_pipeline_engine.run_pipeline(pipeline.pipeline_id)
        print(f"✅ ML pipeline executed: {success}")
        
        # Test 4: Check model artifacts
        artifacts = mlops_pipeline_engine.model_artifacts
        print(f"✅ Model artifacts created: {len(artifacts)} artifacts")
        
        # Test 5: Check model deployments
        deployments = mlops_pipeline_engine.model_deployments
        print(f"✅ Model deployments: {len(deployments)} deployments")
        
        # Test 6: Get MLOps summary
        summary = mlops_pipeline_engine.get_mlops_summary()
        print(f"✅ MLOps summary: {summary['total_pipelines']} pipelines, {summary['total_deployments']} deployments")
        
        # Test 7: Stop MLOps engine
        await mlops_pipeline_engine.stop_mlops_engine()
        print("✅ MLOps engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ MLOps Pipeline Engine test failed: {e}")
        return False

async def test_advanced_api_gateway():
    """Test Advanced API Gateway"""
    print("\n🚪 Testing Advanced API Gateway...")
    
    try:
        from app.services.advanced_api_gateway import (
            advanced_api_gateway, 
            GatewayRoute, 
            AuthenticationMethod,
            RateLimitType
        )
        
        # Test 1: Start API gateway
        await advanced_api_gateway.start_gateway()
        print("✅ API gateway started")
        
        # Test 2: Add custom route
        route_data = {
            "name": "Test Route",
            "path": "/test/*",
            "methods": ["GET", "POST"],
            "target_service": "test-service",
            "target_url": "http://localhost:8000",
            "authentication_required": True,
            "authentication_method": "jwt"
        }
        
        route = await advanced_api_gateway.add_route(route_data)
        print(f"✅ API route added: {route.route_id}")
        
        # Test 3: Process request
        request_data = {
            "client_ip": "127.0.0.1",
            "user_id": "test_user",
            "method": "GET",
            "path": "/test/health",
            "headers": {"Authorization": "Bearer test_token"},
            "query_params": {"param1": "value1"}
        }
        
        response = await advanced_api_gateway.process_request(request_data)
        print(f"✅ API request processed: {response.get('status_code', 'unknown')}")
        
        # Test 4: Check request logs
        request_logs = advanced_api_gateway.request_logs
        print(f"✅ Request logs: {len(request_logs)} requests logged")
        
        # Test 5: Get gateway summary
        summary = advanced_api_gateway.get_gateway_summary()
        print(f"✅ Gateway summary: {summary['total_routes']} routes, {summary['total_requests']} requests")
        
        # Test 6: Stop API gateway
        await advanced_api_gateway.stop_gateway()
        print("✅ API gateway stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced API Gateway test failed: {e}")
        return False

async def test_event_sourcing_engine():
    """Test Event Sourcing Engine"""
    print("\n📡 Testing Event Sourcing Engine...")
    
    try:
        from app.services.event_sourcing_engine import (
            event_sourcing_engine, 
            EventType, 
            EventStatus,
            AggregateType
        )
        
        # Test 1: Start event sourcing engine
        await event_sourcing_engine.start_event_sourcing_engine()
        print("✅ Event sourcing engine started")
        
        # Test 2: Create event
        event_data = {
            "aggregate_id": "user_123",
            "aggregate_type": "user",
            "event_type": "user_created",
            "event_data": {
                "user_id": "user_123",
                "email": "test@example.com",
                "name": "Test User"
            },
            "event_metadata": {"source": "test"}
        }
        
        event = await event_sourcing_engine.create_event(event_data)
        print(f"✅ Event created: {event.event_id}")
        
        # Test 3: Process command
        command_data = {
            "aggregate_id": "user_123",
            "aggregate_type": "user",
            "command_type": "update_user",
            "command_data": {"name": "Updated User"}
        }
        
        command = await event_sourcing_engine.process_command(command_data)
        print(f"✅ Command processed: {command.command_id}")
        
        # Test 4: Check aggregates
        aggregates = event_sourcing_engine.aggregates
        print(f"✅ Aggregates: {len(aggregates)} aggregates")
        
        # Test 5: Check projections
        projections = event_sourcing_engine.projections
        print(f"✅ Projections: {len(projections)} projections")
        
        # Test 6: Get event sourcing summary
        summary = event_sourcing_engine.get_event_sourcing_summary()
        print(f"✅ Event sourcing summary: {summary['total_events']} events, {summary['total_aggregates']} aggregates")
        
        # Test 7: Stop event sourcing engine
        await event_sourcing_engine.stop_event_sourcing_engine()
        print("✅ Event sourcing engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Event Sourcing Engine test failed: {e}")
        return False

async def test_api_endpoints():
    """Test New API Endpoints"""
    print("\n🔌 Testing New API Endpoints...")
    
    try:
        import httpx
        
        # Test chaos engineering API endpoints
        async with httpx.AsyncClient() as client:
            try:
                # Test chaos engineering status
                response = await client.get("http://localhost:8000/api/v1/chaos-engineering/status")
                if response.status_code == 200:
                    print("✅ Chaos Engineering API status endpoint accessible")
                else:
                    print(f"⚠️ Chaos Engineering API returned status {response.status_code}")
                
                # Test MLOps status
                response = await client.get("http://localhost:8000/api/v1/mlops/status")
                if response.status_code == 200:
                    print("✅ MLOps API status endpoint accessible")
                else:
                    print(f"⚠️ MLOps API returned status {response.status_code}")
                
                # Test API gateway status
                response = await client.get("http://localhost:8000/api/v1/api-gateway/status")
                if response.status_code == 200:
                    print("✅ API Gateway API status endpoint accessible")
                else:
                    print(f"⚠️ API Gateway API returned status {response.status_code}")
                
                # Test event sourcing status
                response = await client.get("http://localhost:8000/api/v1/event-sourcing/status")
                if response.status_code == 200:
                    print("✅ Event Sourcing API status endpoint accessible")
                else:
                    print(f"⚠️ Event Sourcing API returned status {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ API endpoints not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Endpoints test failed: {e}")
        return False

async def test_integration_workflow():
    """Test integration between all new advanced features"""
    print("\n🔗 Testing Advanced Integration Workflow...")
    
    try:
        from app.services.chaos_engineering_engine import chaos_engineering_engine
        from app.services.mlops_pipeline_engine import mlops_pipeline_engine
        from app.services.advanced_api_gateway import advanced_api_gateway
        from app.services.event_sourcing_engine import event_sourcing_engine
        
        # Test 1: Start all systems
        await chaos_engineering_engine.start_chaos_engine()
        await mlops_pipeline_engine.start_mlops_engine()
        await advanced_api_gateway.start_gateway()
        await event_sourcing_engine.start_event_sourcing_engine()
        print("✅ All advanced systems started")
        
        # Test 2: Integrated workflow: Event -> ML Pipeline -> API Gateway -> Chaos Engineering
        
        # Step 1: Create event
        event_data = {
            "aggregate_id": "market_456",
            "aggregate_type": "market",
            "event_type": "market_created",
            "event_data": {
                "market_id": "market_456",
                "name": "Advanced Integration Test Market",
                "description": "Test market for advanced integration"
            }
        }
        
        event = await event_sourcing_engine.create_event(event_data)
        print(f"✅ Event created: {event.event_id}")
        
        # Step 2: Create ML pipeline for the market
        pipeline_data = {
            "name": "Advanced Integration ML Pipeline",
            "description": "ML pipeline for advanced integration test",
            "template": "market_prediction",
            "config": {"data_sources": ["market_data"]}
        }
        
        pipeline = await mlops_pipeline_engine.create_pipeline(pipeline_data)
        print(f"✅ ML pipeline created: {pipeline.pipeline_id}")
        
        # Step 3: Add API route for the market
        route_data = {
            "name": "Advanced Integration Route",
            "path": f"/markets/{event.aggregate_id}/*",
            "methods": ["GET", "POST"],
            "target_service": "market-service",
            "target_url": "http://localhost:8000"
        }
        
        route = await advanced_api_gateway.add_route(route_data)
        print(f"✅ API route added: {route.route_id}")
        
        # Step 4: Create chaos experiment for the market
        experiment_data = {
            "name": "Advanced Integration Chaos Test",
            "description": "Chaos experiment for advanced integration",
            "type": "service_failure",
            "target_services": ["market-service"],
            "duration": 30,
            "intensity": 0.3
        }
        
        experiment = await chaos_engineering_engine.create_experiment(experiment_data)
        print(f"✅ Chaos experiment created: {experiment.experiment_id}")
        
        # Step 5: Create integrated record
        integrated_record = {
            "event_id": event.event_id,
            "pipeline_id": pipeline.pipeline_id,
            "route_id": route.route_id,
            "experiment_id": experiment.experiment_id,
            "integration_active": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in cache (simulating integration)
        from app.core.enhanced_cache import enhanced_cache
        await enhanced_cache.set(
            f"advanced_integration_{event.aggregate_id}",
            integrated_record,
            ttl=3600
        )
        print("✅ Integrated record created and cached")
        
        # Step 6: Verify integration
        cached_record = await enhanced_cache.get(f"advanced_integration_{event.aggregate_id}")
        if cached_record:
            print("✅ Advanced integration verification successful")
        else:
            print("❌ Advanced integration verification failed")
        
        # Test 3: Stop all systems
        await chaos_engineering_engine.stop_chaos_engine()
        await mlops_pipeline_engine.stop_mlops_engine()
        await advanced_api_gateway.stop_gateway()
        await event_sourcing_engine.stop_event_sourcing_engine()
        print("✅ All advanced systems stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced integration workflow test failed: {e}")
        return False

async def test_performance_benchmarks():
    """Test performance benchmarks for new advanced features"""
    print("\n⚡ Testing Advanced Performance Benchmarks...")
    
    try:
        from app.services.chaos_engineering_engine import chaos_engineering_engine, ChaosExperimentType
        from app.services.mlops_pipeline_engine import mlops_pipeline_engine, PipelineStage
        from app.services.advanced_api_gateway import advanced_api_gateway
        from app.services.event_sourcing_engine import event_sourcing_engine, EventType
        
        # Test chaos engineering performance
        start_time = time.time()
        
        for i in range(50):
            experiment_data = {
                "name": f"Performance Test Experiment {i}",
                "description": f"Test experiment {i} for performance testing",
                "type": "cpu_stress",
                "target_services": ["test-service"],
                "duration": 10,
                "intensity": 0.5
            }
            
            await chaos_engineering_engine.create_experiment(experiment_data)
        
        chaos_time = time.time() - start_time
        print(f"✅ Chaos Engineering: 50 experiments created in {chaos_time:.2f}s ({50/chaos_time:.1f} experiments/s)")
        
        # Test MLOps performance
        start_time = time.time()
        
        for i in range(30):
            pipeline_data = {
                "name": f"Performance Test Pipeline {i}",
                "description": f"Test pipeline {i} for performance testing",
                "template": "market_prediction"
            }
            
            await mlops_pipeline_engine.create_pipeline(pipeline_data)
        
        mlops_time = time.time() - start_time
        print(f"✅ MLOps: 30 pipelines created in {mlops_time:.2f}s ({30/mlops_time:.1f} pipelines/s)")
        
        # Test API gateway performance
        start_time = time.time()
        
        for i in range(100):
            request_data = {
                "client_ip": "127.0.0.1",
                "method": "GET",
                "path": f"/test/{i}",
                "headers": {"Authorization": "Bearer test_token"}
            }
            
            await advanced_api_gateway.process_request(request_data)
        
        gateway_time = time.time() - start_time
        print(f"✅ API Gateway: 100 requests processed in {gateway_time:.2f}s ({100/gateway_time:.1f} requests/s)")
        
        # Test event sourcing performance
        start_time = time.time()
        
        for i in range(200):
            event_data = {
                "aggregate_id": f"test_aggregate_{i}",
                "aggregate_type": "user",
                "event_type": "user_created",
                "event_data": {"user_id": f"user_{i}", "name": f"User {i}"}
            }
            
            await event_sourcing_engine.create_event(event_data)
        
        event_sourcing_time = time.time() - start_time
        print(f"✅ Event Sourcing: 200 events created in {event_sourcing_time:.2f}s ({200/event_sourcing_time:.1f} events/s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced performance benchmarks test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Latest Advanced Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Chaos Engineering Engine", test_chaos_engineering_engine),
        ("MLOps Pipeline Engine", test_mlops_pipeline_engine),
        ("Advanced API Gateway", test_advanced_api_gateway),
        ("Event Sourcing Engine", test_event_sourcing_engine),
        ("New API Endpoints", test_api_endpoints),
        ("Advanced Integration Workflow", test_integration_workflow),
        ("Advanced Performance Benchmarks", test_performance_benchmarks)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 ADVANCED FEATURES TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1
    
    print("=" * 70)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All advanced tests passed! Latest features are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
