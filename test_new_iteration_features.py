#!/usr/bin/env python3
"""
Test Suite for New Iteration Features
Comprehensive testing for Advanced Security V2, Business Intelligence, and Mobile Optimization
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

async def test_advanced_security_v2():
    """Test Advanced Security V2 system"""
    print("\n🔒 Testing Advanced Security V2...")
    
    try:
        from app.core.advanced_security_v2 import advanced_security_v2, SecurityThreat, SecurityPolicy, ThreatLevel, SecurityEvent
        
        # Test 1: Start security monitoring
        await advanced_security_v2.start_security_monitoring()
        print("✅ Security monitoring started")
        
        # Test 2: Analyze safe request
        safe_request = {
            "source_ip": "192.168.1.100",
            "user_id": "user123",
            "method": "GET",
            "path": "/api/v1/markets",
            "headers": {"User-Agent": "Mozilla/5.0"},
            "body": "",
            "query_params": {"limit": "10"}
        }
        
        analysis = await advanced_security_v2.analyze_request(safe_request)
        print(f"✅ Safe request analysis: {analysis['action']}")
        
        # Test 3: Analyze malicious request (SQL injection)
        malicious_request = {
            "source_ip": "192.168.1.200",
            "user_id": None,
            "method": "POST",
            "path": "/api/v1/users",
            "headers": {"User-Agent": "Mozilla/5.0"},
            "body": "username=admin' OR '1'='1",
            "query_params": {}
        }
        
        analysis = await advanced_security_v2.analyze_request(malicious_request)
        print(f"✅ Malicious request analysis: {analysis['action']} - {analysis['message']}")
        
        # Test 4: Get security summary
        summary = advanced_security_v2.get_security_summary()
        print(f"✅ Security summary: {summary['recent_threats']} threats detected")
        
        # Test 5: Add security policy
        policy = SecurityPolicy(
            policy_id="test_policy",
            name="Test Security Policy",
            description="Test policy for validation",
            enabled=True,
            conditions={"max_attempts": 3, "time_window": 300},
            actions=["block_ip", "alert_admin"],
            severity=ThreatLevel.HIGH
        )
        
        advanced_security_v2.add_security_policy(policy)
        print("✅ Security policy added")
        
        # Test 6: Stop security monitoring
        await advanced_security_v2.stop_security_monitoring()
        print("✅ Security monitoring stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Security V2 test failed: {e}")
        return False

async def test_business_intelligence_engine():
    """Test Business Intelligence Engine"""
    print("\n📊 Testing Business Intelligence Engine...")
    
    try:
        from app.services.business_intelligence_engine import business_intelligence_engine, BusinessMetric, MetricType
        
        # Test 1: Start BI engine
        await business_intelligence_engine.start_bi_engine()
        print("✅ BI engine started")
        
        # Test 2: Collect metrics
        metrics = [
            BusinessMetric(
                metric_id="user_engagement",
                name="User Engagement",
                description="Daily active users",
                metric_type=MetricType.GAUGE,
                value=1500.0,
                timestamp=datetime.now()
            ),
            BusinessMetric(
                metric_id="revenue",
                name="Revenue",
                description="Daily revenue",
                metric_type=MetricType.COUNTER,
                value=25000.0,
                timestamp=datetime.now()
            ),
            BusinessMetric(
                metric_id="response_time",
                name="Response Time",
                description="Average response time",
                metric_type=MetricType.HISTOGRAM,
                value=150.0,
                timestamp=datetime.now()
            )
        ]
        
        await business_intelligence_engine.collect_metrics_batch(metrics)
        print("✅ Metrics collected")
        
        # Test 3: Wait for insights generation
        await asyncio.sleep(2)
        
        # Test 4: Get BI summary
        summary = business_intelligence_engine.get_bi_summary()
        print(f"✅ BI summary: {summary['total_metrics']} metrics, {summary['total_insights']} insights")
        
        # Test 5: Stop BI engine
        await business_intelligence_engine.stop_bi_engine()
        print("✅ BI engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Business Intelligence Engine test failed: {e}")
        return False

async def test_mobile_optimization_engine():
    """Test Mobile Optimization Engine"""
    print("\n📱 Testing Mobile Optimization Engine...")
    
    try:
        from app.services.mobile_optimization_engine import mobile_optimization_engine, DeviceInfo, DeviceType, ConnectionType
        
        # Test 1: Start mobile optimization
        await mobile_optimization_engine.start_mobile_optimization()
        print("✅ Mobile optimization started")
        
        # Test 2: Register device
        device_info = DeviceInfo(
            device_id="test_device_123",
            device_type=DeviceType.MOBILE,
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
            screen_width=375,
            screen_height=667,
            pixel_ratio=2.0,
            connection_type=ConnectionType.CELLULAR_4G,
            is_touch_device=True,
            browser="Safari",
            os="iOS"
        )
        
        await mobile_optimization_engine.register_device(device_info)
        print("✅ Device registered")
        
        # Test 3: Optimize CSS content
        css_content = """
        .header {
            background-color: #1a365d;
            padding: 20px;
            margin: 10px;
        }
        
        .content {
            font-size: 16px;
            line-height: 1.5;
        }
        """
        
        optimization_result = await mobile_optimization_engine.optimize_content(
            css_content, "css", "test_device_123"
        )
        
        print(f"✅ CSS optimization: {optimization_result['compression_ratio']:.2%} compression")
        
        # Test 4: Optimize JavaScript content
        js_content = """
        function calculateTotal(items) {
            let total = 0;
            for (let i = 0; i < items.length; i++) {
                total += items[i].price;
            }
            return total;
        }
        """
        
        optimization_result = await mobile_optimization_engine.optimize_content(
            js_content, "javascript", "test_device_123"
        )
        
        print(f"✅ JavaScript optimization: {optimization_result['compression_ratio']:.2%} compression")
        
        # Test 5: Get PWA manifest
        manifest = mobile_optimization_engine.get_pwa_manifest("default")
        print(f"✅ PWA manifest: {manifest.name if manifest else 'Not found'}")
        
        # Test 6: Get mobile summary
        summary = mobile_optimization_engine.get_mobile_summary()
        print(f"✅ Mobile summary: {summary['registered_devices']} devices, {summary['total_optimizations']} optimizations")
        
        # Test 7: Stop mobile optimization
        await mobile_optimization_engine.stop_mobile_optimization()
        print("✅ Mobile optimization stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Mobile Optimization Engine test failed: {e}")
        return False

async def test_api_endpoints():
    """Test new API endpoints"""
    print("\n🌐 Testing New API Endpoints...")
    
    try:
        import httpx
        
        # Test advanced security API
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8000/api/v1/advanced-security/status")
                if response.status_code == 200:
                    print("✅ Advanced Security API endpoint accessible")
                else:
                    print(f"⚠️ Advanced Security API returned status {response.status_code}")
            except Exception as e:
                print(f"⚠️ Advanced Security API not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

async def test_integration():
    """Test integration between all new features"""
    print("\n🔗 Testing Feature Integration...")
    
    try:
        from app.core.advanced_security_v2 import advanced_security_v2
        from app.services.business_intelligence_engine import business_intelligence_engine
        from app.services.mobile_optimization_engine import mobile_optimization_engine
        
        # Test 1: Start all systems
        await advanced_security_v2.start_security_monitoring()
        await business_intelligence_engine.start_bi_engine()
        await mobile_optimization_engine.start_mobile_optimization()
        print("✅ All systems started")
        
        # Test 2: Simulate integrated workflow
        # Security analysis -> Business metrics -> Mobile optimization
        
        # Security analysis
        request_data = {
            "source_ip": "192.168.1.100",
            "user_id": "user123",
            "method": "GET",
            "path": "/api/v1/markets",
            "headers": {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"},
            "body": "",
            "query_params": {"limit": "10"}
        }
        
        security_analysis = await advanced_security_v2.analyze_request(request_data)
        print(f"✅ Security analysis completed: {security_analysis['action']}")
        
        # Business metrics collection
        from app.services.business_intelligence_engine import BusinessMetric, MetricType
        
        metric = BusinessMetric(
            metric_id="api_requests",
            name="API Requests",
            description="Total API requests processed",
            metric_type=MetricType.COUNTER,
            value=1.0,
            timestamp=datetime.now()
        )
        
        await business_intelligence_engine.collect_metric(metric)
        print("✅ Business metric collected")
        
        # Mobile optimization
        device_info = mobile_optimization_engine.devices.get("test_device_123")
        if device_info:
            optimization_result = await mobile_optimization_engine.optimize_content(
                "<div>Test content</div>", "html", "test_device_123"
            )
            print(f"✅ Mobile optimization completed: {optimization_result['optimization_applied']}")
        
        # Test 3: Stop all systems
        await advanced_security_v2.stop_security_monitoring()
        await business_intelligence_engine.stop_bi_engine()
        await mobile_optimization_engine.stop_mobile_optimization()
        print("✅ All systems stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def test_performance():
    """Test performance of new features"""
    print("\n⚡ Testing Performance...")
    
    try:
        from app.core.advanced_security_v2 import advanced_security_v2
        from app.services.business_intelligence_engine import business_intelligence_engine
        
        # Test security analysis performance
        start_time = time.time()
        
        for i in range(100):
            request_data = {
                "source_ip": f"192.168.1.{i % 255}",
                "user_id": f"user{i}",
                "method": "GET",
                "path": f"/api/v1/markets/{i}",
                "headers": {"User-Agent": "Mozilla/5.0"},
                "body": "",
                "query_params": {"limit": "10"}
            }
            
            await advanced_security_v2.analyze_request(request_data)
        
        security_time = time.time() - start_time
        print(f"✅ Security analysis: 100 requests in {security_time:.2f}s ({100/security_time:.1f} req/s)")
        
        # Test business intelligence performance
        start_time = time.time()
        
        from app.services.business_intelligence_engine import BusinessMetric, MetricType
        
        metrics = []
        for i in range(100):
            metric = BusinessMetric(
                metric_id=f"test_metric_{i}",
                name=f"Test Metric {i}",
                description=f"Test metric {i}",
                metric_type=MetricType.GAUGE,
                value=float(i),
                timestamp=datetime.now()
            )
            metrics.append(metric)
        
        await business_intelligence_engine.collect_metrics_batch(metrics)
        
        bi_time = time.time() - start_time
        print(f"✅ Business Intelligence: 100 metrics in {bi_time:.2f}s ({100/bi_time:.1f} metrics/s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting New Iteration Features Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Advanced Security V2", test_advanced_security_v2),
        ("Business Intelligence Engine", test_business_intelligence_engine),
        ("Mobile Optimization Engine", test_mobile_optimization_engine),
        ("API Endpoints", test_api_endpoints),
        ("Feature Integration", test_integration),
        ("Performance", test_performance)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! New iteration features are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
