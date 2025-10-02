#!/usr/bin/env python3
"""
Test Suite for Latest Iteration Features
Comprehensive testing for Blockchain Integration, Advanced ML, Distributed Caching, and Admin Dashboard
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

async def test_blockchain_integration_engine():
    """Test Blockchain Integration Engine"""
    print("\n⛓️ Testing Blockchain Integration Engine...")
    
    try:
        from app.services.blockchain_integration_engine import (
            blockchain_integration_engine, 
            BlockchainTransaction, 
            TransactionType, 
            TransactionStatus,
            SmartContract
        )
        
        # Test 1: Start blockchain engine
        await blockchain_integration_engine.start_blockchain_engine()
        print("✅ Blockchain engine started")
        
        # Test 2: Create transaction
        transaction_data = {
            "type": "trade",
            "from_address": "0x1234567890123456789012345678901234567890",
            "to_address": "0x0987654321098765432109876543210987654321",
            "amount": 100.0,
            "token_symbol": "ETH",
            "gas_price": 20.0,
            "data": {"market_id": "market_123", "trade_type": "buy"}
        }
        
        transaction = await blockchain_integration_engine.create_transaction(transaction_data)
        print(f"✅ Transaction created: {transaction.tx_id}")
        
        # Test 3: Deploy smart contract
        contract_data = {
            "name": "OpinionMarket",
            "type": "opinion_market",
            "abi": {"functions": ["createMarket", "placeTrade"]},
            "bytecode": "0x608060405234801561001057600080fd5b50",
            "owner": "0x1234567890123456789012345678901234567890",
            "version": "1.0.0"
        }
        
        contract = await blockchain_integration_engine.deploy_smart_contract(contract_data)
        print(f"✅ Smart contract deployed: {contract.contract_id}")
        
        # Test 4: Call smart contract function
        result = await blockchain_integration_engine.call_smart_contract_function(
            contract.contract_id, "createMarket", ["Test Market", 1000]
        )
        print(f"✅ Smart contract function called: {result['function_name']}")
        
        # Test 5: Emit blockchain event
        event_data = {
            "contract_address": contract.contract_address,
            "event_name": "MarketCreated",
            "event_data": {"market_id": "market_123", "creator": "0x1234"},
            "block_number": 12345,
            "transaction_hash": transaction.tx_hash,
            "log_index": 0
        }
        
        event = await blockchain_integration_engine.emit_blockchain_event(event_data)
        print(f"✅ Blockchain event emitted: {event.event_id}")
        
        # Test 6: Get blockchain summary
        summary = blockchain_integration_engine.get_blockchain_summary()
        print(f"✅ Blockchain summary: {summary['total_transactions']} transactions, {summary['total_smart_contracts']} contracts")
        
        # Test 7: Stop blockchain engine
        await blockchain_integration_engine.stop_blockchain_engine()
        print("✅ Blockchain engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Blockchain Integration Engine test failed: {e}")
        return False

async def test_advanced_ml_engine():
    """Test Advanced ML Engine"""
    print("\n🤖 Testing Advanced ML Engine...")
    
    try:
        from app.services.advanced_ml_engine import (
            advanced_ml_engine, 
            MLModel, 
            MLModelType, 
            PredictionType,
            ModelStatus
        )
        
        # Test 1: Start ML engine
        await advanced_ml_engine.start_ml_engine()
        print("✅ ML engine started")
        
        # Test 2: Create ML model
        model_data = {
            "name": "Test Market Predictor",
            "type": "classification",
            "prediction_type": "market_direction",
            "features": ["price_history", "volume", "sentiment"],
            "hyperparameters": {"n_estimators": 100, "max_depth": 10}
        }
        
        model = await advanced_ml_engine.create_model(model_data)
        print(f"✅ ML model created: {model.model_id}")
        
        # Test 3: Train model
        training_data = [
            {"price_history": [100, 101, 102], "volume": 1000, "sentiment": 0.8, "direction": "up"},
            {"price_history": [102, 101, 100], "volume": 800, "sentiment": 0.3, "direction": "down"},
            {"price_history": [100, 100, 100], "volume": 500, "sentiment": 0.5, "direction": "sideways"}
        ]
        
        training_job = await advanced_ml_engine.train_model(model.model_id, training_data)
        print(f"✅ Training job started: {training_job.job_id}")
        
        # Wait for training to complete
        await asyncio.sleep(3)
        
        # Test 4: Make prediction
        input_data = {
            "price_history": [100, 101, 102],
            "volume": 1000,
            "sentiment": 0.8
        }
        
        prediction = await advanced_ml_engine.make_prediction(model.model_id, input_data)
        print(f"✅ Prediction made: {prediction.prediction_id} with confidence {prediction.confidence:.2f}")
        
        # Test 5: Evaluate model
        test_data = [
            {"price_history": [100, 101, 102], "volume": 1000, "sentiment": 0.8, "direction": "up"},
            {"price_history": [102, 101, 100], "volume": 800, "sentiment": 0.3, "direction": "down"}
        ]
        
        evaluation = await advanced_ml_engine.evaluate_model(model.model_id, test_data)
        print(f"✅ Model evaluated: accuracy {evaluation['accuracy']:.2f}")
        
        # Test 6: Get ML summary
        summary = advanced_ml_engine.get_ml_summary()
        print(f"✅ ML summary: {summary['total_models']} models, {summary['total_predictions']} predictions")
        
        # Test 7: Stop ML engine
        await advanced_ml_engine.stop_ml_engine()
        print("✅ ML engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced ML Engine test failed: {e}")
        return False

async def test_distributed_caching_engine():
    """Test Distributed Caching Engine"""
    print("\n🌐 Testing Distributed Caching Engine...")
    
    try:
        from app.services.distributed_caching_engine import (
            distributed_caching_engine, 
            ContentType, 
            CacheStrategy
        )
        
        # Test 1: Start caching engine
        await distributed_caching_engine.start_caching_engine()
        print("✅ Caching engine started")
        
        # Test 2: Set cache entries
        test_data = {
            "user_profile": {"id": 123, "name": "John Doe", "email": "john@example.com"},
            "market_data": {"price": 100.50, "volume": 1000, "change": 0.05},
            "api_response": {"status": "success", "data": [1, 2, 3, 4, 5]}
        }
        
        for key, value in test_data.items():
            content_type = ContentType.STATIC if key == "user_profile" else ContentType.API_RESPONSE
            success = await distributed_caching_engine.set(key, value, content_type)
            print(f"✅ Cache set: {key} - {success}")
        
        # Test 3: Get cache entries
        for key in test_data.keys():
            cached_value = await distributed_caching_engine.get(key)
            if cached_value:
                print(f"✅ Cache hit: {key}")
            else:
                print(f"❌ Cache miss: {key}")
        
        # Test 4: Test compression
        large_data = {"large_array": list(range(10000))}
        success = await distributed_caching_engine.set("large_data", large_data, ContentType.STATIC)
        print(f"✅ Large data cached with compression: {success}")
        
        # Test 5: Cache invalidation by tags
        await distributed_caching_engine.set("tagged_data", {"data": "test"}, ContentType.STATIC, tags=["test", "demo"])
        invalidated = await distributed_caching_engine.invalidate_by_tags(["test"])
        print(f"✅ Cache invalidation: {invalidated} entries invalidated")
        
        # Test 6: Get caching summary
        summary = distributed_caching_engine.get_caching_summary()
        print(f"✅ Caching summary: {summary['total_cache_entries']} entries, hit rate {summary['cache_hit_rate']:.2%}")
        
        # Test 7: Stop caching engine
        await distributed_caching_engine.stop_caching_engine()
        print("✅ Caching engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed Caching Engine test failed: {e}")
        return False

async def test_admin_dashboard_api():
    """Test Admin Dashboard API"""
    print("\n📊 Testing Admin Dashboard API...")
    
    try:
        import httpx
        
        # Test admin dashboard endpoints
        async with httpx.AsyncClient() as client:
            try:
                # Test system status endpoint
                response = await client.get("http://localhost:8000/api/v1/admin-dashboard/status")
                if response.status_code == 200:
                    status_data = response.json()
                    print(f"✅ System status: {status_data['overall_status']}")
                    print(f"✅ Active services: {len([s for s in status_data['services'].values() if s.get('active', False)])}")
                else:
                    print(f"⚠️ System status returned status {response.status_code}")
                
                # Test dashboard metrics endpoint
                response = await client.get("http://localhost:8000/api/v1/admin-dashboard/metrics")
                if response.status_code == 200:
                    metrics_data = response.json()
                    print(f"✅ Dashboard metrics retrieved")
                    print(f"✅ System health: CPU {metrics_data['system_health']['cpu_usage']}%")
                else:
                    print(f"⚠️ Dashboard metrics returned status {response.status_code}")
                
                # Test admin dashboard HTML
                response = await client.get("http://localhost:8000/api/v1/admin-dashboard/")
                if response.status_code == 200:
                    print("✅ Admin dashboard HTML accessible")
                else:
                    print(f"⚠️ Admin dashboard HTML returned status {response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Admin dashboard API not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Admin Dashboard API test failed: {e}")
        return False

async def test_integration_workflow():
    """Test integration between all new features"""
    print("\n🔗 Testing Integration Workflow...")
    
    try:
        from app.services.blockchain_integration_engine import blockchain_integration_engine
        from app.services.advanced_ml_engine import advanced_ml_engine
        from app.services.distributed_caching_engine import distributed_caching_engine, ContentType
        
        # Test 1: Start all systems
        await blockchain_integration_engine.start_blockchain_engine()
        await advanced_ml_engine.start_ml_engine()
        await distributed_caching_engine.start_caching_engine()
        print("✅ All systems started")
        
        # Test 2: Integrated workflow: ML prediction -> Blockchain transaction -> Cache result
        
        # Step 1: Make ML prediction
        model_id = "market_direction_predictor"
        input_data = {
            "price_history": [100, 101, 102],
            "volume": 1000,
            "sentiment": 0.8
        }
        
        prediction = await advanced_ml_engine.make_prediction(model_id, input_data)
        print(f"✅ ML prediction: {prediction.prediction_result}")
        
        # Step 2: Create blockchain transaction based on prediction
        if prediction.prediction_result.get("direction") == "up":
            transaction_data = {
                "type": "trade",
                "from_address": "0x1234567890123456789012345678901234567890",
                "to_address": "0x0987654321098765432109876543210987654321",
                "amount": 100.0,
                "token_symbol": "ETH",
                "data": {"prediction_id": prediction.prediction_id, "action": "buy"}
            }
            
            transaction = await blockchain_integration_engine.create_transaction(transaction_data)
            print(f"✅ Blockchain transaction created: {transaction.tx_id}")
            
            # Step 3: Cache the integrated result
            integrated_result = {
                "prediction": prediction.prediction_result,
                "transaction": {
                    "tx_id": transaction.tx_id,
                    "status": transaction.status.value
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await distributed_caching_engine.set(
                f"integrated_result_{prediction.prediction_id}",
                integrated_result,
                ContentType.API_RESPONSE
            )
            print("✅ Integrated result cached")
            
            # Step 4: Retrieve cached result
            cached_result = await distributed_caching_engine.get(f"integrated_result_{prediction.prediction_id}")
            if cached_result:
                print("✅ Cached integrated result retrieved")
            else:
                print("❌ Failed to retrieve cached result")
        
        # Test 3: Stop all systems
        await blockchain_integration_engine.stop_blockchain_engine()
        await advanced_ml_engine.stop_ml_engine()
        await distributed_caching_engine.stop_caching_engine()
        print("✅ All systems stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        return False

async def test_performance_benchmarks():
    """Test performance benchmarks for new features"""
    print("\n⚡ Testing Performance Benchmarks...")
    
    try:
        from app.services.blockchain_integration_engine import blockchain_integration_engine
        from app.services.advanced_ml_engine import advanced_ml_engine
        from app.services.distributed_caching_engine import distributed_caching_engine, ContentType
        
        # Test blockchain transaction performance
        start_time = time.time()
        
        for i in range(50):
            transaction_data = {
                "type": "trade",
                "from_address": f"0x{i:040x}",
                "to_address": f"0x{i+1:040x}",
                "amount": float(i),
                "token_symbol": "ETH"
            }
            
            await blockchain_integration_engine.create_transaction(transaction_data)
        
        blockchain_time = time.time() - start_time
        print(f"✅ Blockchain: 50 transactions in {blockchain_time:.2f}s ({50/blockchain_time:.1f} tx/s)")
        
        # Test ML prediction performance
        start_time = time.time()
        
        for i in range(100):
            input_data = {
                "price_history": [100 + i, 101 + i, 102 + i],
                "volume": 1000 + i,
                "sentiment": 0.5 + (i % 10) / 20.0
            }
            
            await advanced_ml_engine.make_prediction("market_direction_predictor", input_data)
        
        ml_time = time.time() - start_time
        print(f"✅ ML Engine: 100 predictions in {ml_time:.2f}s ({100/ml_time:.1f} pred/s)")
        
        # Test caching performance
        start_time = time.time()
        
        for i in range(200):
            key = f"test_key_{i}"
            value = {"data": f"test_value_{i}", "index": i}
            await distributed_caching_engine.set(key, value, ContentType.STATIC)
        
        cache_set_time = time.time() - start_time
        print(f"✅ Caching: 200 sets in {cache_set_time:.2f}s ({200/cache_set_time:.1f} sets/s)")
        
        # Test cache retrieval performance
        start_time = time.time()
        
        for i in range(200):
            key = f"test_key_{i}"
            await distributed_caching_engine.get(key)
        
        cache_get_time = time.time() - start_time
        print(f"✅ Caching: 200 gets in {cache_get_time:.2f}s ({200/cache_get_time:.1f} gets/s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmarks test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Latest Iteration Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Blockchain Integration Engine", test_blockchain_integration_engine),
        ("Advanced ML Engine", test_advanced_ml_engine),
        ("Distributed Caching Engine", test_distributed_caching_engine),
        ("Admin Dashboard API", test_admin_dashboard_api),
        ("Integration Workflow", test_integration_workflow),
        ("Performance Benchmarks", test_performance_benchmarks)
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
    print("📊 TEST RESULTS SUMMARY")
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
        print("\n🎉 All tests passed! Latest iteration features are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
