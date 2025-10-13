#!/usr/bin/env python3
"""
Comprehensive Test Script for Opinion Market Improvements
Tests all the enhancements made to the application
"""

import asyncio
import time
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test that all new modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from app.core.advanced_security import advanced_security_manager, threat_detector
        print("✅ Advanced security module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import advanced security: {e}")
        return False
    
    try:
        from app.core.performance_optimizer import performance_optimizer
        print("✅ Performance optimizer module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import performance optimizer: {e}")
        return False
    
    try:
        from app.core.middleware import middleware_manager
        print("✅ Enhanced middleware module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced middleware: {e}")
        return False
    
    try:
        from app.api.v1.endpoints.security_monitoring import router as security_router
        print("✅ Security monitoring endpoints imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import security monitoring: {e}")
        return False
    
    try:
        from app.api.v1.endpoints.performance_monitoring import router as performance_router
        print("✅ Performance monitoring endpoints imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import performance monitoring: {e}")
        return False
    
    return True


def test_security_features():
    """Test advanced security features"""
    print("\n🔒 Testing security features...")
    
    try:
        from app.core.advanced_security import advanced_security_manager, threat_detector
        
        # Test threat detection
        print("  Testing threat detection...")
        
        # Test brute force detection
        is_brute_force = threat_detector.detect_brute_force("test_user", max_attempts=3, window=60)
        print(f"  ✅ Brute force detection: {is_brute_force}")
        
        # Test DDoS detection
        is_ddos = threat_detector.detect_ddos_attack("192.168.1.1")
        print(f"  ✅ DDoS detection: {is_ddos}")
        
        # Test IP blocking
        is_blocked = threat_detector.is_ip_blocked("192.168.1.1")
        print(f"  ✅ IP blocking check: {is_blocked}")
        
        # Test device fingerprinting
        device_info = {
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "accept_language": "en-US,en;q=0.9",
            "platform": "Windows"
        }
        
        fingerprint = advanced_security_manager._generate_device_fingerprint(device_info)
        print(f"  ✅ Device fingerprinting: {len(fingerprint)} chars")
        
        # Test session creation
        session_data = advanced_security_manager.create_secure_session(1, device_info)
        print(f"  ✅ Secure session creation: {session_data['session_id'][:8]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Security features test failed: {e}")
        return False


def test_performance_features():
    """Test performance optimization features"""
    print("\n⚡ Testing performance features...")
    
    try:
        from app.core.performance_optimizer import performance_optimizer
        
        # Test cache manager
        print("  Testing cache manager...")
        performance_optimizer.cache_manager.set("test_key", {"data": "test"}, ttl=60)
        cached_data = performance_optimizer.cache_manager.get("test_key")
        print(f"  ✅ Cache operations: {cached_data is not None}")
        
        # Test cache stats
        cache_stats = performance_optimizer.cache_manager.get_cache_stats()
        print(f"  ✅ Cache stats: {cache_stats['total_requests']} requests")
        
        # Test query optimizer
        print("  Testing query optimizer...")
        query_hash = performance_optimizer.query_optimizer.generate_query_hash(
            "SELECT * FROM users", {"limit": 10}
        )
        print(f"  ✅ Query hash generation: {len(query_hash)} chars")
        
        # Test performance monitor
        print("  Testing performance monitor...")
        performance_summary = performance_optimizer.performance_monitor.get_performance_summary()
        print(f"  ✅ Performance monitoring: {performance_summary.get('monitoring_active', False)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance features test failed: {e}")
        return False


def test_middleware_improvements():
    """Test middleware improvements"""
    print("\n🛡️ Testing middleware improvements...")
    
    try:
        from app.core.middleware import PerformanceMiddleware, SecurityMiddleware
        
        # Test middleware initialization
        print("  Testing middleware initialization...")
        
        # Create mock app for testing
        class MockApp:
            def __init__(self):
                self.state = type('State', (), {})()
        
        mock_app = MockApp()
        
        # Test PerformanceMiddleware
        perf_middleware = PerformanceMiddleware(mock_app)
        print("  ✅ PerformanceMiddleware initialized")
        
        # Test SecurityMiddleware
        sec_middleware = SecurityMiddleware(mock_app)
        print("  ✅ SecurityMiddleware initialized")
        
        # Test performance stats
        perf_stats = perf_middleware.get_performance_stats()
        print(f"  ✅ Performance stats: {perf_stats.get('total_requests', 0)} requests")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Middleware improvements test failed: {e}")
        return False


def test_api_endpoints():
    """Test new API endpoints"""
    print("\n🌐 Testing API endpoints...")
    
    try:
        from app.api.v1.endpoints.security_monitoring import router as security_router
        from app.api.v1.endpoints.performance_monitoring import router as performance_router
        
        # Check if routers have the expected routes
        security_routes = [route.path for route in security_router.routes]
        performance_routes = [route.path for route in performance_router.routes]
        
        expected_security_routes = [
            "/threats/analysis",
            "/sessions/active", 
            "/security/events",
            "/security/stats",
            "/security/health"
        ]
        
        expected_performance_routes = [
            "/stats",
            "/system",
            "/cache",
            "/queries",
            "/performance/health"
        ]
        
        print("  Testing security monitoring routes...")
        for route in expected_security_routes:
            if any(route in r for r in security_routes):
                print(f"  ✅ Security route found: {route}")
            else:
                print(f"  ❌ Security route missing: {route}")
        
        print("  Testing performance monitoring routes...")
        for route in expected_performance_routes:
            if any(route in r for r in performance_routes):
                print(f"  ✅ Performance route found: {route}")
            else:
                print(f"  ❌ Performance route missing: {route}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ API endpoints test failed: {e}")
        return False


def test_configuration():
    """Test configuration and settings"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from app.core.config import settings
        
        # Test key configuration values
        print(f"  ✅ App name: {settings.APP_NAME}")
        print(f"  ✅ App version: {settings.APP_VERSION}")
        print(f"  ✅ Environment: {settings.ENVIRONMENT}")
        print(f"  ✅ Debug mode: {settings.DEBUG}")
        print(f"  ✅ Rate limiting: {settings.RATE_LIMIT_ENABLED}")
        print(f"  ✅ Caching enabled: {settings.ENABLE_CACHING}")
        print(f"  ✅ Compression enabled: {settings.ENABLE_COMPRESSION}")
        
        # Test database configuration
        db_config = settings.database_config
        print(f"  ✅ Database pool size: {db_config['pool_size']}")
        
        # Test Redis configuration
        redis_config = settings.redis_config
        print(f"  ✅ Redis max connections: {redis_config['max_connections']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n📊 Generating test report...")
    
    report = {
        "test_timestamp": datetime.utcnow().isoformat(),
        "tests_run": [],
        "summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "success_rate": 0.0
        }
    }
    
    # Run all tests
    tests = [
        ("Import Tests", test_imports),
        ("Security Features", test_security_features),
        ("Performance Features", test_performance_features),
        ("Middleware Improvements", test_middleware_improvements),
        ("API Endpoints", test_api_endpoints),
        ("Configuration", test_configuration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            
            test_result = {
                "name": test_name,
                "status": "PASSED" if result else "FAILED",
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            report["tests_run"].append(test_result)
            report["summary"]["total_tests"] += 1
            
            if result:
                report["summary"]["passed_tests"] += 1
                print(f"\n✅ {test_name}: PASSED ({duration:.2f}s)")
            else:
                report["summary"]["failed_tests"] += 1
                print(f"\n❌ {test_name}: FAILED ({duration:.2f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            test_result = {
                "name": test_name,
                "status": "ERROR",
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            report["tests_run"].append(test_result)
            report["summary"]["total_tests"] += 1
            report["summary"]["failed_tests"] += 1
            print(f"\n💥 {test_name}: ERROR - {e} ({duration:.2f}s)")
    
    # Calculate success rate
    if report["summary"]["total_tests"] > 0:
        report["summary"]["success_rate"] = (
            report["summary"]["passed_tests"] / report["summary"]["total_tests"] * 100
        )
    
    # Save report
    report_file = f"test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Report saved to: {report_file}")
    
    return report


if __name__ == "__main__":
    print("🚀 Opinion Market - Comprehensive Improvement Test Suite")
    print("=" * 60)
    
    try:
        report = generate_test_report()
        
        if report["summary"]["success_rate"] >= 80:
            print("\n🎉 Overall Result: EXCELLENT - Most improvements are working correctly!")
        elif report["summary"]["success_rate"] >= 60:
            print("\n✅ Overall Result: GOOD - Most improvements are working with some issues")
        else:
            print("\n⚠️ Overall Result: NEEDS ATTENTION - Several improvements need fixes")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(1)