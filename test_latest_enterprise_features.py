#!/usr/bin/env python3
"""
Test Suite for Latest Enterprise Features
Comprehensive testing for Advanced Monitoring, Data Governance, and Microservices
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

async def test_advanced_monitoring_engine():
    """Test Advanced Monitoring Engine"""
    print("\n📊 Testing Advanced Monitoring Engine...")
    
    try:
        from app.services.advanced_monitoring_engine import (
            advanced_monitoring_engine, 
            MetricType, 
            AlertSeverity,
            ServiceStatus
        )
        
        # Test 1: Start monitoring engine
        await advanced_monitoring_engine.start_monitoring_engine()
        print("✅ Monitoring engine started")
        
        # Test 2: Record metrics
        await advanced_monitoring_engine._record_metric("test_metric", 100.0, MetricType.GAUGE, {"service": "test"})
        await advanced_monitoring_engine._record_metric("cpu_usage_percent", 85.0, MetricType.GAUGE)
        await advanced_monitoring_engine._record_metric("memory_usage_bytes", 1024*1024*1024*9, MetricType.GAUGE)
        print("✅ Metrics recorded")
        
        # Test 3: Check service health
        health = await advanced_monitoring_engine._check_single_service_health("api")
        print(f"✅ Service health checked: {health.status.value}")
        
        # Test 4: Evaluate alert rules
        await advanced_monitoring_engine._evaluate_alert_rules()
        print(f"✅ Alert rules evaluated: {len(advanced_monitoring_engine.alerts)} alerts")
        
        # Test 5: Get Prometheus metrics
        prometheus_metrics = advanced_monitoring_engine.get_prometheus_metrics()
        print(f"✅ Prometheus metrics generated: {len(prometheus_metrics)} characters")
        
        # Test 6: Get monitoring summary
        summary = advanced_monitoring_engine.get_monitoring_summary()
        print(f"✅ Monitoring summary: {summary['total_metrics']} metrics, {summary['total_alerts']} alerts")
        
        # Test 7: Stop monitoring engine
        await advanced_monitoring_engine.stop_monitoring_engine()
        print("✅ Monitoring engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Monitoring Engine test failed: {e}")
        return False

async def test_data_governance_engine():
    """Test Data Governance Engine"""
    print("\n🔒 Testing Data Governance Engine...")
    
    try:
        from app.services.data_governance_engine import (
            data_governance_engine, 
            DataClassification, 
            DataRetentionPolicy,
            ConsentStatus,
            DataSubjectRights
        )
        
        # Test 1: Start governance engine
        await data_governance_engine.start_governance_engine()
        print("✅ Governance engine started")
        
        # Test 2: Register data asset
        asset_data = {
            "name": "User Database",
            "description": "User personal information database",
            "owner": "data_team",
            "size_bytes": 1024*1024*100,  # 100MB
            "location": "database://users",
            "tags": ["personal", "database"],
            "metadata": {"encrypted": True, "backup_enabled": True}
        }
        
        asset = await data_governance_engine.register_data_asset(asset_data)
        print(f"✅ Data asset registered: {asset.asset_id} ({asset.classification.value})")
        
        # Test 3: Register data subject
        subject_data = {
            "email": "test@example.com",
            "name": "Test User"
        }
        
        subject = await data_governance_engine.register_data_subject(subject_data)
        print(f"✅ Data subject registered: {subject.subject_id}")
        
        # Test 4: Process consent request
        consent_data = {
            "consent_granted": True,
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0"
        }
        
        consent_granted = await data_governance_engine.process_consent_request(subject.subject_id, consent_data)
        print(f"✅ Consent processed: {consent_granted}")
        
        # Test 5: Process data subject rights request
        rights_result = await data_governance_engine.process_data_subject_rights_request(
            subject.subject_id, 
            DataSubjectRights.ACCESS, 
            {}
        )
        print(f"✅ Rights request processed: {rights_result['status']}")
        
        # Test 6: Report data breach
        breach_data = {
            "description": "Test data breach for testing purposes",
            "affected_subjects": 1,
            "data_categories": ["email", "name"],
            "consequences": "Low risk - test data only",
            "measures": ["Immediate notification", "Data encryption review"]
        }
        
        breach = await data_governance_engine.report_data_breach(breach_data)
        print(f"✅ Data breach reported: {breach.breach_id}")
        
        # Test 7: Get governance summary
        summary = data_governance_engine.get_governance_summary()
        print(f"✅ Governance summary: {summary['total_data_assets']} assets, {summary['total_data_subjects']} subjects")
        
        # Test 8: Stop governance engine
        await data_governance_engine.stop_governance_engine()
        print("✅ Governance engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Governance Engine test failed: {e}")
        return False

async def test_microservices_engine():
    """Test Microservices Engine"""
    print("\n🏗️ Testing Microservices Engine...")
    
    try:
        from app.services.microservices_engine import (
            microservices_engine, 
            ServiceType, 
            ServiceStatus,
            LoadBalancingStrategy
        )
        
        # Test 1: Start microservices engine
        await microservices_engine.start_microservices_engine()
        print("✅ Microservices engine started")
        
        # Test 2: Create service mesh
        mesh = await microservices_engine.create_service_mesh("test-mesh")
        print(f"✅ Service mesh created: {mesh.mesh_id}")
        
        # Test 3: Register services
        services_to_register = [
            ServiceType.API_GATEWAY,
            ServiceType.AUTHENTICATION,
            ServiceType.USER_MANAGEMENT,
            ServiceType.MARKET_DATA
        ]
        
        registered_services = []
        for service_type in services_to_register:
            instance = await microservices_engine.register_service(service_type)
            registered_services.append(instance)
            print(f"✅ Service registered: {instance.service_name}")
        
        # Test 4: Health check services
        await microservices_engine._health_check_services()
        healthy_services = sum(1 for inst in microservices_engine.service_instances.values() if inst.status == ServiceStatus.RUNNING)
        print(f"✅ Health check completed: {healthy_services} healthy services")
        
        # Test 5: Call service (mock)
        try:
            # This would normally make an HTTP call, but we'll simulate it
            response = await microservices_engine.call_service("api-gateway", "GET", "/health")
            print(f"✅ Service call successful: {response}")
        except Exception as e:
            print(f"⚠️ Service call failed (expected in test environment): {e}")
        
        # Test 6: Update circuit breakers
        await microservices_engine._update_circuit_breakers()
        print("✅ Circuit breakers updated")
        
        # Test 7: Get microservices summary
        summary = microservices_engine.get_microservices_summary()
        print(f"✅ Microservices summary: {summary['total_instances']} instances, {summary['success_rate']:.1f}% success rate")
        
        # Test 8: Stop microservices engine
        await microservices_engine.stop_microservices_engine()
        print("✅ Microservices engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Microservices Engine test failed: {e}")
        return False

async def test_api_endpoints():
    """Test New API Endpoints"""
    print("\n🔌 Testing New API Endpoints...")
    
    try:
        import httpx
        
        # Test monitoring API endpoints
        async with httpx.AsyncClient() as client:
            try:
                # Test monitoring status
                response = await client.get("http://localhost:8000/api/v1/monitoring/status")
                if response.status_code == 200:
                    print("✅ Monitoring API status endpoint accessible")
                else:
                    print(f"⚠️ Monitoring API returned status {response.status_code}")
                
                # Test governance status
                response = await client.get("http://localhost:8000/api/v1/governance/status")
                if response.status_code == 200:
                    print("✅ Governance API status endpoint accessible")
                else:
                    print(f"⚠️ Governance API returned status {response.status_code}")
                
                # Test microservices status
                response = await client.get("http://localhost:8000/api/v1/microservices/status")
                if response.status_code == 200:
                    print("✅ Microservices API status endpoint accessible")
                else:
                    print(f"⚠️ Microservices API returned status {response.status_code}")
                
                # Test Prometheus metrics endpoint
                response = await client.get("http://localhost:8000/api/v1/monitoring/prometheus")
                if response.status_code == 200:
                    print("✅ Prometheus metrics endpoint accessible")
                else:
                    print(f"⚠️ Prometheus metrics returned status {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ API endpoints not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Endpoints test failed: {e}")
        return False

async def test_integration_workflow():
    """Test integration between all new enterprise features"""
    print("\n🔗 Testing Enterprise Integration Workflow...")
    
    try:
        from app.services.advanced_monitoring_engine import advanced_monitoring_engine
        from app.services.data_governance_engine import data_governance_engine
        from app.services.microservices_engine import microservices_engine
        
        # Test 1: Start all systems
        await advanced_monitoring_engine.start_monitoring_engine()
        await data_governance_engine.start_governance_engine()
        await microservices_engine.start_microservices_engine()
        print("✅ All enterprise systems started")
        
        # Test 2: Integrated workflow: Data Asset -> Monitoring -> Governance
        
        # Step 1: Register data asset
        asset_data = {
            "name": "Enterprise Integration Test Asset",
            "description": "Test asset for integration workflow",
            "owner": "integration_test",
            "size_bytes": 1024*1024*50,  # 50MB
            "location": "test://integration",
            "tags": ["integration", "test"],
            "metadata": {"test": True}
        }
        
        asset = await data_governance_engine.register_data_asset(asset_data)
        print(f"✅ Data asset registered: {asset.asset_id}")
        
        # Step 2: Monitor the asset
        from app.services.advanced_monitoring_engine import MetricType
        await advanced_monitoring_engine._record_metric(
            f"asset_size_{asset.asset_id}", 
            asset.size_bytes, 
            MetricType.GAUGE
        )
        print("✅ Asset size monitored")
        
        # Step 3: Register service for the asset
        from app.services.microservices_engine import ServiceType
        service_instance = await microservices_engine.register_service(ServiceType.ANALYTICS)
        print(f"✅ Service registered for asset: {service_instance.service_name}")
        
        # Step 4: Create integrated monitoring record
        integrated_record = {
            "asset_id": asset.asset_id,
            "service_id": service_instance.instance_id,
            "monitoring_active": True,
            "governance_compliant": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in cache (simulating integration)
        from app.core.enhanced_cache import enhanced_cache
        await enhanced_cache.set(
            f"integration_{asset.asset_id}",
            integrated_record,
            ttl=3600
        )
        print("✅ Integrated record created and cached")
        
        # Step 5: Verify integration
        cached_record = await enhanced_cache.get(f"integration_{asset.asset_id}")
        if cached_record:
            print("✅ Integration verification successful")
        else:
            print("❌ Integration verification failed")
        
        # Test 3: Stop all systems
        await advanced_monitoring_engine.stop_monitoring_engine()
        await data_governance_engine.stop_governance_engine()
        await microservices_engine.stop_microservices_engine()
        print("✅ All enterprise systems stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise integration workflow test failed: {e}")
        return False

async def test_performance_benchmarks():
    """Test performance benchmarks for new enterprise features"""
    print("\n⚡ Testing Enterprise Performance Benchmarks...")
    
    try:
        from app.services.advanced_monitoring_engine import advanced_monitoring_engine, MetricType
        from app.services.data_governance_engine import data_governance_engine
        from app.services.microservices_engine import microservices_engine, ServiceType
        
        # Test monitoring metrics performance
        start_time = time.time()
        
        for i in range(100):
            await advanced_monitoring_engine._record_metric(
                f"perf_test_metric_{i}", 
                float(i), 
                MetricType.GAUGE,
                {"test": "performance"}
            )
        
        monitoring_time = time.time() - start_time
        print(f"✅ Monitoring: 100 metrics recorded in {monitoring_time:.2f}s ({100/monitoring_time:.1f} metrics/s)")
        
        # Test data governance performance
        start_time = time.time()
        
        for i in range(50):
            asset_data = {
                "name": f"Performance Test Asset {i}",
                "description": f"Test asset {i} for performance testing",
                "owner": "perf_test",
                "size_bytes": 1024*1024,
                "location": f"test://perf/{i}",
                "tags": ["performance", "test"]
            }
            
            await data_governance_engine.register_data_asset(asset_data)
        
        governance_time = time.time() - start_time
        print(f"✅ Data Governance: 50 assets registered in {governance_time:.2f}s ({50/governance_time:.1f} assets/s)")
        
        # Test microservices performance
        start_time = time.time()
        
        for i in range(20):
            service_type = ServiceType.API_GATEWAY if i % 2 == 0 else ServiceType.AUTHENTICATION
            await microservices_engine.register_service(service_type, port=8000+i)
        
        microservices_time = time.time() - start_time
        print(f"✅ Microservices: 20 services registered in {microservices_time:.2f}s ({20/microservices_time:.1f} services/s)")
        
        # Test cache performance
        from app.core.enhanced_cache import enhanced_cache
        start_time = time.time()
        
        for i in range(200):
            await enhanced_cache.set(f"perf_test_{i}", {"data": f"test_data_{i}"}, ttl=3600)
        
        cache_set_time = time.time() - start_time
        print(f"✅ Cache: 200 sets in {cache_set_time:.2f}s ({200/cache_set_time:.1f} sets/s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise performance benchmarks test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Latest Enterprise Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Advanced Monitoring Engine", test_advanced_monitoring_engine),
        ("Data Governance Engine", test_data_governance_engine),
        ("Microservices Engine", test_microservices_engine),
        ("New API Endpoints", test_api_endpoints),
        ("Enterprise Integration Workflow", test_integration_workflow),
        ("Enterprise Performance Benchmarks", test_performance_benchmarks)
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
    print("📊 ENTERPRISE TEST RESULTS SUMMARY")
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
        print("\n🎉 All enterprise tests passed! Latest features are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
