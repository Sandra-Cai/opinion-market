#!/usr/bin/env python3
"""
Test script to validate the new performance improvements
"""

import asyncio
import time
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.advanced_performance_optimizer import advanced_performance_optimizer
from app.core.enhanced_cache import enhanced_cache


async def test_advanced_performance_optimizer():
    """Test the advanced performance optimizer"""
    print("🚀 Testing Advanced Performance Optimizer...")
    
    try:
        # Test initialization
        print("  ✓ Testing initialization...")
        await advanced_performance_optimizer.start_monitoring()
        await advanced_performance_optimizer.start_optimization()
        
        # Wait for metrics collection
        print("  ✓ Collecting metrics...")
        await asyncio.sleep(5)
        
        # Test performance summary
        print("  ✓ Testing performance summary...")
        summary = advanced_performance_optimizer.get_performance_summary()
        
        print(f"    - Monitoring active: {summary['monitoring_active']}")
        print(f"    - Optimization active: {summary['optimization_active']}")
        print(f"    - Performance score: {summary['performance_score']:.1f}")
        print(f"    - Metrics collected: {len(summary['metrics'])}")
        
        # Test metrics collection
        print("  ✓ Testing metrics collection...")
        metrics = summary['metrics']
        for metric_name, metric_data in metrics.items():
            print(f"    - {metric_name}: {metric_data['current']:.2f}")
        
        # Test optimization actions
        print("  ✓ Testing optimization actions...")
        await advanced_performance_optimizer._create_optimization_action(
            action_type="cache_optimization",
            target="cache",
            parameters={"test": "manual_optimization"},
            priority=1
        )
        
        # Execute optimizations
        await advanced_performance_optimizer._execute_optimizations()
        
        # Stop services
        await advanced_performance_optimizer.stop_optimization()
        await advanced_performance_optimizer.stop_monitoring()
        
        print("  ✅ Advanced Performance Optimizer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Advanced Performance Optimizer test failed: {e}")
        return False


async def test_enhanced_cache():
    """Test the enhanced cache system"""
    print("🚀 Testing Enhanced Cache System...")
    
    try:
        # Test cache operations
        print("  ✓ Testing cache operations...")
        
        # Set some test data
        for i in range(10):
            await enhanced_cache.set(f"test_key_{i}", f"test_value_{i}")
        
        # Get test data
        for i in range(10):
            value = await enhanced_cache.get(f"test_key_{i}")
            assert value == f"test_value_{i}"
        
        # Test cache stats
        print("  ✓ Testing cache statistics...")
        stats = enhanced_cache.get_stats()
        print(f"    - Hit rate: {stats.get('hit_rate', 0):.2%}")
        print(f"    - Entry count: {stats.get('entry_count', 0)}")
        print(f"    - Memory usage: {stats.get('memory_usage_mb', 0):.2f} MB")
        
        # Clear cache
        await enhanced_cache.clear()
        
        print("  ✅ Enhanced Cache System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced Cache System test failed: {e}")
        return False


async def test_integration():
    """Test integration between systems"""
    print("🚀 Testing System Integration...")
    
    try:
        # Start performance optimizer
        await advanced_performance_optimizer.start_monitoring()
        
        # Perform cache operations
        print("  ✓ Testing cache operations with monitoring...")
        for i in range(20):
            await enhanced_cache.set(f"integration_test_{i}", f"value_{i}")
            await enhanced_cache.get(f"integration_test_{i}")
        
        # Wait for metrics collection
        await asyncio.sleep(3)
        
        # Check that cache metrics are being monitored
        summary = advanced_performance_optimizer.get_performance_summary()
        cache_metrics = summary['metrics'].get('cache.hit_rate', {})
        
        print(f"    - Cache hit rate monitored: {cache_metrics.get('current', 0):.2f}%")
        
        # Stop monitoring
        await advanced_performance_optimizer.stop_monitoring()
        
        print("  ✅ System Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ System Integration test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🎯 Opinion Market Performance Improvements Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        test_enhanced_cache(),
        test_advanced_performance_optimizer(),
        test_integration(),
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Report results
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Test {i+1} failed with exception: {result}")
            failed += 1
        elif result:
            print(f"✅ Test {i+1} passed")
            passed += 1
        else:
            print(f"❌ Test {i+1} failed")
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    
    if failed == 0:
        print("\n🎉 All tests passed! Performance improvements are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
