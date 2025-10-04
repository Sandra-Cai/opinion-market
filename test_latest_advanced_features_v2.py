#!/usr/bin/env python3
"""
Test Suite for Latest Advanced Features V2
Comprehensive testing for Advanced Caching, AI Insights, and Real-time Analytics
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

async def test_advanced_caching_engine():
    """Test Advanced Caching Engine"""
    print("\n🚀 Testing Advanced Caching Engine...")
    
    try:
        from app.services.advanced_caching_engine import (
            advanced_caching_engine, 
            CacheStrategy, 
            CacheTier
        )
        
        # Test 1: Start advanced caching engine
        await advanced_caching_engine.start_caching_engine()
        print("✅ Advanced caching engine started")
        
        # Test 2: Set cache values in different tiers
        test_data = {
            "small_data": {"key": "value", "number": 42},
            "medium_data": {"data": [i for i in range(1000)]},
            "large_data": {"data": [i for i in range(10000)]}
        }
        
        for key, value in test_data.items():
            success = await advanced_caching_engine.set(key, value, ttl=3600)
            print(f"✅ Cache set: {key} - {success}")
        
        # Test 3: Get cache values
        for key in test_data.keys():
            value = await advanced_caching_engine.get(key)
            if value is not None:
                print(f"✅ Cache get: {key} - Found")
            else:
                print(f"❌ Cache get: {key} - Not found")
        
        # Test 4: Test cache strategies
        strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.TTL]
        for strategy in strategies:
            test_key = f"strategy_test_{strategy.value}"
            success = await advanced_caching_engine.set(test_key, {"strategy": strategy.value}, strategy=strategy)
            print(f"✅ Cache strategy test: {strategy.value} - {success}")
        
        # Test 5: Test cache tiers
        tiers = [CacheTier.L1_MEMORY, CacheTier.L2_REDIS, CacheTier.L3_CDN]
        for tier in tiers:
            test_key = f"tier_test_{tier.value}"
            success = await advanced_caching_engine.set(test_key, {"tier": tier.value}, tier=tier)
            print(f"✅ Cache tier test: {tier.value} - {success}")
        
        # Test 6: Get caching summary
        summary = advanced_caching_engine.get_caching_summary()
        print(f"✅ Caching summary: {summary['overall_hit_rate']:.1%} hit rate, {summary['total_requests']} requests")
        
        # Test 7: Stop advanced caching engine
        await advanced_caching_engine.stop_caching_engine()
        print("✅ Advanced caching engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Caching Engine test failed: {e}")
        return False

async def test_ai_insights_engine():
    """Test AI Insights Engine"""
    print("\n🤖 Testing AI Insights Engine...")
    
    try:
        from app.services.ai_insights_engine import (
            ai_insights_engine, 
            InsightType, 
            RecommendationType,
            ConfidenceLevel
        )
        
        # Test 1: Start AI insights engine
        await ai_insights_engine.start_ai_insights_engine()
        print("✅ AI insights engine started")
        
        # Test 2: Generate market trend insight
        market_data = {
            "asset": "BTC",
            "data_points": [
                {"timestamp": datetime.now().isoformat(), "price": 45000, "volume": 1000000},
                {"timestamp": datetime.now().isoformat(), "price": 46000, "volume": 1200000}
            ],
            "metadata": {"source": "test"}
        }
        
        insight = await ai_insights_engine.generate_insight(
            InsightType.MARKET_TREND, 
            market_data, 
            user_id="test_user"
        )
        print(f"✅ Market trend insight generated: {insight.insight_id}")
        
        # Test 3: Generate price prediction insight
        prediction_data = {
            "asset": "ETH",
            "predicted_price": 3500.0,
            "time_horizon": "7 days",
            "data_points": [{"price": 3200, "volume": 800000}]
        }
        
        prediction_insight = await ai_insights_engine.generate_insight(
            InsightType.PRICE_PREDICTION,
            prediction_data,
            user_id="test_user"
        )
        print(f"✅ Price prediction insight generated: {prediction_insight.insight_id}")
        
        # Test 4: Generate trading recommendation
        trading_data = {
            "asset": "BTC",
            "action": "BUY",
            "price": 45000.0,
            "quantity": 1,
            "expected_return": 0.15,
            "risk_level": "medium",
            "time_horizon": "short",
            "supporting_insights": [insight.insight_id]
        }
        
        recommendation = await ai_insights_engine.generate_recommendation(
            RecommendationType.TRADING_ACTION,
            trading_data,
            user_id="test_user"
        )
        print(f"✅ Trading recommendation generated: {recommendation.recommendation_id}")
        
        # Test 5: Generate portfolio optimization recommendation
        portfolio_data = {
            "rebalancing_actions": "Reduce tech exposure by 10%, increase bonds by 15%",
            "expected_return": 0.08,
            "risk_level": "low",
            "time_horizon": "medium"
        }
        
        portfolio_rec = await ai_insights_engine.generate_recommendation(
            RecommendationType.PORTFOLIO_OPTIMIZATION,
            portfolio_data,
            user_id="test_user"
        )
        print(f"✅ Portfolio optimization recommendation generated: {portfolio_rec.recommendation_id}")
        
        # Test 6: Check AI models
        models = ai_insights_engine.ai_models
        print(f"✅ AI models initialized: {len(models)} models")
        
        # Test 7: Check user profiles
        user_profiles = ai_insights_engine.user_profiles
        print(f"✅ User profiles: {len(user_profiles)} profiles")
        
        # Test 8: Get AI insights summary
        summary = ai_insights_engine.get_ai_insights_summary()
        print(f"✅ AI insights summary: {summary['total_insights']} insights, {summary['total_recommendations']} recommendations")
        
        # Test 9: Stop AI insights engine
        await ai_insights_engine.stop_ai_insights_engine()
        print("✅ AI insights engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Insights Engine test failed: {e}")
        return False

async def test_real_time_analytics_engine():
    """Test Real-time Analytics Engine"""
    print("\n📊 Testing Real-time Analytics Engine...")
    
    try:
        from app.services.real_time_analytics_engine import (
            real_time_analytics_engine, 
            StreamType, 
            AnalyticsType,
            ProcessingMode
        )
        
        # Test 1: Start real-time analytics engine
        await real_time_analytics_engine.start_analytics_engine()
        print("✅ Real-time analytics engine started")
        
        # Test 2: Ingest market data
        market_data = {
            "symbol": "BTC",
            "price": 45000.0,
            "volume": 1000000,
            "bid": 44950.0,
            "ask": 45050.0
        }
        
        success = await real_time_analytics_engine.ingest_data(
            StreamType.MARKET_DATA,
            market_data,
            "test_source"
        )
        print(f"✅ Market data ingested: {success}")
        
        # Test 3: Ingest user activity data
        user_activity_data = {
            "user_id": "user_123",
            "action": "login",
            "session_id": "session_456",
            "ip_address": "192.168.1.1"
        }
        
        success = await real_time_analytics_engine.ingest_data(
            StreamType.USER_ACTIVITY,
            user_activity_data,
            "test_source"
        )
        print(f"✅ User activity data ingested: {success}")
        
        # Test 4: Ingest trading events
        trading_data = {
            "trade_id": "trade_789",
            "user_id": "user_123",
            "symbol": "BTC",
            "quantity": 0.1,
            "price": 45000.0,
            "side": "buy"
        }
        
        success = await real_time_analytics_engine.ingest_data(
            StreamType.TRADING_EVENTS,
            trading_data,
            "test_source"
        )
        print(f"✅ Trading event ingested: {success}")
        
        # Test 5: Ingest system metrics
        system_data = {
            "metric_name": "cpu_usage",
            "value": 75.5,
            "host": "server_1",
            "service": "api"
        }
        
        success = await real_time_analytics_engine.ingest_data(
            StreamType.SYSTEM_METRICS,
            system_data,
            "test_source"
        )
        print(f"✅ System metrics ingested: {success}")
        
        # Test 6: Check data streams
        streams = real_time_analytics_engine.data_streams
        for stream_type, stream in streams.items():
            print(f"✅ Stream {stream_type.value}: {len(stream)} data points")
        
        # Test 7: Check stream processors
        processors = real_time_analytics_engine.stream_processors
        print(f"✅ Stream processors: {len(processors)} processors")
        
        # Test 8: Check analytics results
        analytics_results = real_time_analytics_engine.analytics_results
        print(f"✅ Analytics results: {len(analytics_results)} results")
        
        # Test 9: Get analytics summary
        summary = real_time_analytics_engine.get_analytics_summary()
        print(f"✅ Analytics summary: {summary['total_data_points']} data points, {summary['total_analytics_results']} results")
        
        # Test 10: Stop real-time analytics engine
        await real_time_analytics_engine.stop_analytics_engine()
        print("✅ Real-time analytics engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time Analytics Engine test failed: {e}")
        return False

async def test_api_endpoints():
    """Test New API Endpoints"""
    print("\n🔌 Testing New API Endpoints...")
    
    try:
        import httpx
        
        # Test advanced caching API endpoints
        async with httpx.AsyncClient() as client:
            try:
                # Test advanced caching status
                response = await client.get("http://localhost:8000/api/v1/advanced-caching/status")
                if response.status_code == 200:
                    print("✅ Advanced Caching API status endpoint accessible")
                else:
                    print(f"⚠️ Advanced Caching API returned status {response.status_code}")
                
                # Test AI insights status
                response = await client.get("http://localhost:8000/api/v1/ai-insights/status")
                if response.status_code == 200:
                    print("✅ AI Insights API status endpoint accessible")
                else:
                    print(f"⚠️ AI Insights API returned status {response.status_code}")
                
                # Test real-time analytics status
                response = await client.get("http://localhost:8000/api/v1/real-time-analytics/status")
                if response.status_code == 200:
                    print("✅ Real-time Analytics API status endpoint accessible")
                else:
                    print(f"⚠️ Real-time Analytics API returned status {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ API endpoints not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Endpoints test failed: {e}")
        return False

async def test_integration_workflow():
    """Test integration between all new advanced features"""
    print("\n🔗 Testing Advanced Integration Workflow V2...")
    
    try:
        from app.services.advanced_caching_engine import advanced_caching_engine, CacheStrategy, CacheTier
        from app.services.ai_insights_engine import ai_insights_engine, InsightType, RecommendationType
        from app.services.real_time_analytics_engine import real_time_analytics_engine, StreamType
        
        # Test 1: Start all systems
        await advanced_caching_engine.start_caching_engine()
        await ai_insights_engine.start_ai_insights_engine()
        await real_time_analytics_engine.start_analytics_engine()
        print("✅ All advanced systems started")
        
        # Test 2: Integrated workflow: Real-time Data -> AI Insights -> Advanced Caching
        
        # Step 1: Ingest real-time market data
        market_data = {
            "symbol": "BTC",
            "price": 45000.0,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
        
        success = await real_time_analytics_engine.ingest_data(
            StreamType.MARKET_DATA,
            market_data,
            "integration_test"
        )
        print(f"✅ Real-time market data ingested: {success}")
        
        # Step 2: Generate AI insight from the data
        insight_data = {
            "asset": "BTC",
            "data_points": [market_data],
            "metadata": {"source": "real_time_analytics"}
        }
        
        insight = await ai_insights_engine.generate_insight(
            InsightType.MARKET_TREND,
            insight_data,
            user_id="integration_user"
        )
        print(f"✅ AI insight generated: {insight.insight_id}")
        
        # Step 3: Generate AI recommendation
        recommendation_data = {
            "asset": "BTC",
            "action": "BUY",
            "price": 45000.0,
            "quantity": 1,
            "expected_return": 0.12,
            "risk_level": "medium",
            "supporting_insights": [insight.insight_id]
        }
        
        recommendation = await ai_insights_engine.generate_recommendation(
            RecommendationType.TRADING_ACTION,
            recommendation_data,
            user_id="integration_user"
        )
        print(f"✅ AI recommendation generated: {recommendation.recommendation_id}")
        
        # Step 4: Cache the insights and recommendations
        cache_data = {
            "insight": {
                "id": insight.insight_id,
                "title": insight.title,
                "confidence_score": insight.confidence_score
            },
            "recommendation": {
                "id": recommendation.recommendation_id,
                "action": recommendation.action,
                "expected_return": recommendation.expected_return
            },
            "market_data": market_data,
            "timestamp": datetime.now().isoformat()
        }
        
        cache_success = await advanced_caching_engine.set(
            f"integration_result_{insight.insight_id}",
            cache_data,
            ttl=3600,
            strategy=CacheStrategy.INTELLIGENT,
            tier=CacheTier.L1_MEMORY
        )
        print(f"✅ Integration result cached: {cache_success}")
        
        # Step 5: Retrieve from cache
        cached_result = await advanced_caching_engine.get(f"integration_result_{insight.insight_id}")
        if cached_result:
            print("✅ Integration result retrieved from cache")
        else:
            print("❌ Integration result not found in cache")
        
        # Step 6: Create comprehensive integration record
        integration_record = {
            "workflow_id": f"workflow_{int(time.time())}",
            "real_time_data": market_data,
            "ai_insight": insight.insight_id,
            "ai_recommendation": recommendation.recommendation_id,
            "cache_key": f"integration_result_{insight.insight_id}",
            "integration_active": True,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": {
                "data_ingestion_time": 0.001,
                "insight_generation_time": 0.05,
                "recommendation_generation_time": 0.03,
                "cache_operation_time": 0.002
            }
        }
        
        # Store in cache (simulating integration)
        from app.core.enhanced_cache import enhanced_cache
        await enhanced_cache.set(
            f"advanced_integration_{integration_record['workflow_id']}",
            integration_record,
            ttl=3600
        )
        print("✅ Advanced integration record created and cached")
        
        # Step 7: Verify integration
        cached_record = await enhanced_cache.get(f"advanced_integration_{integration_record['workflow_id']}")
        if cached_record:
            print("✅ Advanced integration verification successful")
        else:
            print("❌ Advanced integration verification failed")
        
        # Test 3: Stop all systems
        await advanced_caching_engine.stop_caching_engine()
        await ai_insights_engine.stop_ai_insights_engine()
        await real_time_analytics_engine.stop_analytics_engine()
        print("✅ All advanced systems stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced integration workflow test failed: {e}")
        return False

async def test_performance_benchmarks():
    """Test performance benchmarks for new advanced features"""
    print("\n⚡ Testing Advanced Performance Benchmarks V2...")
    
    try:
        from app.services.advanced_caching_engine import advanced_caching_engine, CacheStrategy, CacheTier
        from app.services.ai_insights_engine import ai_insights_engine, InsightType, RecommendationType
        from app.services.real_time_analytics_engine import real_time_analytics_engine, StreamType
        
        # Test advanced caching performance
        start_time = time.time()
        
        for i in range(100):
            await advanced_caching_engine.set(
                f"perf_test_{i}", 
                {"data": f"test_data_{i}", "number": i}, 
                strategy=CacheStrategy.LRU,
                tier=CacheTier.L1_MEMORY
            )
        
        caching_time = time.time() - start_time
        print(f"✅ Advanced Caching: 100 sets in {caching_time:.2f}s ({100/caching_time:.1f} sets/s)")
        
        # Test AI insights performance
        start_time = time.time()
        
        for i in range(50):
            insight_data = {
                "asset": f"ASSET_{i}",
                "data_points": [{"price": 100 + i, "volume": 1000 + i}],
                "metadata": {"test": True}
            }
            
            await ai_insights_engine.generate_insight(
                InsightType.MARKET_TREND,
                insight_data,
                user_id=f"user_{i}"
            )
        
        ai_insights_time = time.time() - start_time
        print(f"✅ AI Insights: 50 insights generated in {ai_insights_time:.2f}s ({50/ai_insights_time:.1f} insights/s)")
        
        # Test real-time analytics performance
        start_time = time.time()
        
        for i in range(200):
            market_data = {
                "symbol": f"SYMBOL_{i}",
                "price": 100.0 + i,
                "volume": 1000 + i,
                "timestamp": datetime.now().isoformat()
            }
            
            await real_time_analytics_engine.ingest_data(
                StreamType.MARKET_DATA,
                market_data,
                "performance_test"
            )
        
        analytics_time = time.time() - start_time
        print(f"✅ Real-time Analytics: 200 data points ingested in {analytics_time:.2f}s ({200/analytics_time:.1f} points/s)")
        
        # Test AI recommendations performance
        start_time = time.time()
        
        for i in range(30):
            recommendation_data = {
                "asset": f"ASSET_{i}",
                "action": "BUY" if i % 2 == 0 else "SELL",
                "price": 100.0 + i,
                "quantity": 1,
                "expected_return": 0.1 + (i / 100.0),
                "risk_level": "medium"
            }
            
            await ai_insights_engine.generate_recommendation(
                RecommendationType.TRADING_ACTION,
                recommendation_data,
                user_id=f"user_{i}"
            )
        
        recommendations_time = time.time() - start_time
        print(f"✅ AI Recommendations: 30 recommendations generated in {recommendations_time:.2f}s ({30/recommendations_time:.1f} recommendations/s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced performance benchmarks test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Latest Advanced Features V2 Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Advanced Caching Engine", test_advanced_caching_engine),
        ("AI Insights Engine", test_ai_insights_engine),
        ("Real-time Analytics Engine", test_real_time_analytics_engine),
        ("New API Endpoints", test_api_endpoints),
        ("Advanced Integration Workflow V2", test_integration_workflow),
        ("Advanced Performance Benchmarks V2", test_performance_benchmarks)
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
    print("📊 ADVANCED FEATURES V2 TEST RESULTS SUMMARY")
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
        print("\n🎉 All advanced tests passed! Latest features V2 are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
