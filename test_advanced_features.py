#!/usr/bin/env python3
"""
Test Advanced Features
Simple test script to verify advanced features work correctly
"""

import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_performance_optimizer_v2():
    """Test Performance Optimizer V2"""
    print("🔍 Testing Performance Optimizer V2...")
    try:
        from app.core.performance_optimizer_v2 import performance_optimizer_v2
        
        # Test basic functionality
        summary = performance_optimizer_v2.get_performance_summary()
        print(f"✅ Performance summary generated: Score {summary.get('performance_score', 0)}")
        
        # Test strategy setting
        performance_optimizer_v2.set_optimization_strategy("balanced")
        print("✅ Strategy set to balanced")
        
        return True
    except Exception as e:
        print(f"❌ Performance Optimizer V2 test failed: {e}")
        return False

def test_intelligent_alerting():
    """Test Intelligent Alerting System"""
    print("\n🔍 Testing Intelligent Alerting System...")
    try:
        from app.core.intelligent_alerting import intelligent_alerting_system
        
        # Test basic functionality
        summary = intelligent_alerting_system.get_alerting_summary()
        active_alerts = summary.get('active_alerts', {})
        total_alerts = active_alerts.get('total', 0) if isinstance(active_alerts, dict) else 0
        print(f"✅ Alerting summary generated: {total_alerts} active alerts")
        
        # Test rule management
        from app.core.intelligent_alerting import AlertRule, AlertSeverity
        test_rule = AlertRule(
            id="test_rule",
            name="Test Rule",
            description="Test alert rule",
            metric_name="test_metric",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.WARNING
        )
        intelligent_alerting_system.add_alert_rule(test_rule)
        print("✅ Test alert rule added")
        
        return True
    except Exception as e:
        print(f"❌ Intelligent Alerting System test failed: {e}")
        return False

def test_advanced_dashboard():
    """Test Advanced Dashboard"""
    print("\n🔍 Testing Advanced Dashboard...")
    try:
        from app.api.v1.endpoints.advanced_dashboard import dashboard_manager
        
        # Test basic functionality
        dashboard_manager.add_metric('test_metric', 50.0)
        print("✅ Dashboard manager metric added")
        
        # Test trend calculation
        for i in range(10):
            dashboard_manager.add_metric('trend_test', 50.0 + i * 2)
        trend = dashboard_manager.get_metric_trend('trend_test')
        print(f"✅ Trend calculation: {trend}")
        
        return True
    except Exception as e:
        print(f"❌ Advanced Dashboard test failed: {e}")
        return False

async def test_analytics_engine():
    """Test Advanced Analytics Engine"""
    print("\n🔍 Testing Advanced Analytics Engine...")
    try:
        from app.services.advanced_analytics_engine import advanced_analytics_engine
        
        # Test basic functionality
        advanced_analytics_engine.add_data_point('test_metric', 75.5)
        summary = advanced_analytics_engine.get_analytics_summary()
        print(f"✅ Analytics summary generated: {summary.get('data_points_collected', 0)} data points")
        
        # Test starting analytics
        await advanced_analytics_engine.start_analytics()
        await asyncio.sleep(1)
        await advanced_analytics_engine.stop_analytics()
        print("✅ Analytics engine start/stop test passed")
        
        return True
    except Exception as e:
        print(f"❌ Advanced Analytics Engine test failed: {e}")
        return False

async def test_auto_scaling():
    """Test Auto-Scaling Manager"""
    print("\n🔍 Testing Auto-Scaling Manager...")
    try:
        # Mock missing dependencies
        import types
        service_discovery = types.ModuleType('service_discovery')
        service_registry = types.ModuleType('service_registry')
        service_registry.ServiceRegistry = type('ServiceRegistry', (), {})
        service_discovery.service_registry = service_registry
        sys.modules['app.services.service_discovery'] = service_discovery
        sys.modules['app.services.service_registry'] = service_registry
        
        from app.services.auto_scaling_manager import auto_scaling_manager
        
        # Test basic functionality
        summary = auto_scaling_manager.get_scaling_summary()
        print(f"✅ Scaling summary generated: {len(summary.get('policies', {}))} policies")
        
        # Test policy management
        from app.services.auto_scaling_manager import ScalingPolicy
        test_policy = ScalingPolicy(
            name="test_policy",
            metric_name="test_metric",
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            min_instances=1,
            max_instances=10
        )
        auto_scaling_manager.add_scaling_policy(test_policy)
        print("✅ Test scaling policy added")
        
        # Test starting scaling
        await auto_scaling_manager.start_scaling()
        await asyncio.sleep(1)
        await auto_scaling_manager.stop_scaling()
        print("✅ Auto-scaling start/stop test passed")
        
        return True
    except Exception as e:
        print(f"❌ Auto-Scaling Manager test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoint imports"""
    print("\n🔍 Testing API Endpoints...")
    try:
        from app.api.v1.endpoints import advanced_analytics_api
        from app.api.v1.endpoints import auto_scaling_api
        from app.api.v1.endpoints import intelligent_alerting_api
        from app.api.v1.endpoints import advanced_dashboard
        
        print("✅ All advanced API endpoints imported successfully")
        return True
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 Testing Advanced Features")
    print("=" * 50)
    
    test_results = []
    
    # Test individual components
    test_results.append(test_performance_optimizer_v2())
    test_results.append(test_intelligent_alerting())
    test_results.append(test_advanced_dashboard())
    test_results.append(await test_analytics_engine())
    test_results.append(await test_auto_scaling())
    test_results.append(test_api_endpoints())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All advanced features are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)