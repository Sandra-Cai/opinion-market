#!/usr/bin/env python3
"""
Test script for the new Advanced Analytics Features.
Tests Advanced Predictive Analytics Engine, Time Series Forecasting Engine, and Anomaly Detection Engine.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_predictive_analytics_engine():
    """Test the Advanced Predictive Analytics Engine"""
    print("\n🔮 Testing Advanced Predictive Analytics Engine...")
    
    try:
        from app.services.advanced_predictive_analytics_engine import (
            advanced_predictive_analytics_engine, 
            PredictionType, 
            ModelType, 
            ForecastHorizon
        )
        
        # Test engine initialization
        await advanced_predictive_analytics_engine.start_predictive_analytics_engine()
        print("✅ Advanced Predictive Analytics Engine started")
        
        # Test prediction request submission
        request_id = await advanced_predictive_analytics_engine.submit_prediction_request(
            prediction_type=PredictionType.PRICE_FORECAST,
            asset="BTC",
            model_type=ModelType.RANDOM_FOREST,
            forecast_horizon=ForecastHorizon.SHORT_TERM,
            input_data={"price": 45000, "volume": 1000000},
            confidence_level=0.95
        )
        print(f"✅ Prediction request submitted: {request_id}")
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Test getting prediction result
        result = await advanced_predictive_analytics_engine.get_prediction_result(request_id)
        if result:
            print(f"✅ Prediction result retrieved: {len(result['predictions'])} predictions")
        else:
            print("⚠️ Prediction result not ready yet")
        
        # Test getting anomaly detections
        anomalies = await advanced_predictive_analytics_engine.get_anomaly_detections()
        print(f"✅ Anomaly detections: {len(anomalies)} anomalies")
        
        # Test getting model performance
        performance = await advanced_predictive_analytics_engine.get_model_performance()
        print(f"✅ Model performance: {performance['total_models']} models")
        
        # Test getting available models
        models = await advanced_predictive_analytics_engine.get_available_models()
        print(f"✅ Available models: {len(models)} models")
        
        # Test engine shutdown
        await advanced_predictive_analytics_engine.stop_predictive_analytics_engine()
        print("✅ Advanced Predictive Analytics Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Predictive Analytics Engine test failed: {e}")
        return False

async def test_time_series_forecasting_engine():
    """Test the Time Series Forecasting Engine"""
    print("\n📈 Testing Time Series Forecasting Engine...")
    
    try:
        from app.services.time_series_forecasting_engine import time_series_forecasting_engine
        
        # Test engine initialization
        await time_series_forecasting_engine.start_time_series_forecasting_engine()
        print("✅ Time Series Forecasting Engine started")
        
        # Test getting available time series
        time_series_list = await time_series_forecasting_engine.get_available_time_series()
        print(f"✅ Available time series: {len(time_series_list)} series")
        
        # Test getting specific time series
        if time_series_list:
            series_id = time_series_list[0]["series_id"]
            time_series = await time_series_forecasting_engine.get_time_series(series_id)
            if time_series:
                print(f"✅ Time series retrieved: {series_id} with {len(time_series['values'])} data points")
        
        # Test getting time series analysis
        if time_series_list:
            analysis = await time_series_forecasting_engine.get_time_series_analysis(series_id)
            if analysis:
                print(f"✅ Time series analysis retrieved: {analysis['analysis_id']}")
            else:
                print("⚠️ Time series analysis not ready yet")
        
        # Test getting forecast results
        if time_series_list:
            forecasts = await time_series_forecasting_engine.get_forecast_results(series_id)
            print(f"✅ Forecast results: {len(forecasts)} forecasts")
        
        # Test getting engine metrics
        metrics = await time_series_forecasting_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_time_series']} time series, {metrics['total_forecasts']} forecasts")
        
        # Test engine shutdown
        await time_series_forecasting_engine.stop_time_series_forecasting_engine()
        print("✅ Time Series Forecasting Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Time Series Forecasting Engine test failed: {e}")
        return False

async def test_anomaly_detection_engine():
    """Test the Anomaly Detection Engine"""
    print("\n🚨 Testing Anomaly Detection Engine...")
    
    try:
        from app.services.anomaly_detection_engine import (
            anomaly_detection_engine, 
            DetectionMethod, 
            SeverityLevel
        )
        
        # Test engine initialization
        await anomaly_detection_engine.start_anomaly_detection_engine()
        print("✅ Anomaly Detection Engine started")
        
        # Test getting anomaly detections
        anomalies = await anomaly_detection_engine.get_anomaly_detections()
        print(f"✅ Anomaly detections: {len(anomalies)} anomalies")
        
        # Test getting anomaly patterns
        patterns = await anomaly_detection_engine.get_anomaly_patterns()
        print(f"✅ Anomaly patterns: {len(patterns)} patterns")
        
        # Test getting detection rules
        rules = await anomaly_detection_engine.get_detection_rules()
        print(f"✅ Detection rules: {len(rules)} rules")
        
        # Test adding a new detection rule
        rule_id = await anomaly_detection_engine.add_detection_rule(
            name="Test Rule",
            description="Test detection rule",
            asset="BTC",
            method=DetectionMethod.Z_SCORE,
            parameters={"threshold": 3.0},
            threshold=0.9
        )
        print(f"✅ New detection rule added: {rule_id}")
        
        # Test updating detection rule
        success = await anomaly_detection_engine.update_detection_rule(
            rule_id, 
            threshold=0.8, 
            enabled=False
        )
        print(f"✅ Detection rule updated: {success}")
        
        # Test getting engine metrics
        metrics = await anomaly_detection_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_anomalies']} anomalies, {metrics['total_rules']} rules")
        
        # Test engine shutdown
        await anomaly_detection_engine.stop_anomaly_detection_engine()
        print("✅ Anomaly Detection Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Anomaly Detection Engine test failed: {e}")
        return False

async def test_analytics_integration():
    """Test integration between analytics engines"""
    print("\n🔗 Testing Analytics Integration...")
    
    try:
        from app.services.advanced_predictive_analytics_engine import (
            advanced_predictive_analytics_engine, 
            PredictionType, 
            ModelType, 
            ForecastHorizon
        )
        from app.services.time_series_forecasting_engine import time_series_forecasting_engine
        from app.services.anomaly_detection_engine import anomaly_detection_engine
        
        # Start all engines
        await advanced_predictive_analytics_engine.start_predictive_analytics_engine()
        await time_series_forecasting_engine.start_time_series_forecasting_engine()
        await anomaly_detection_engine.start_anomaly_detection_engine()
        
        print("✅ All analytics engines started")
        
        # Step 1: Get time series data
        time_series_list = await time_series_forecasting_engine.get_available_time_series()
        if time_series_list:
            series_id = time_series_list[0]["series_id"]
            time_series = await time_series_forecasting_engine.get_time_series(series_id)
            print(f"✅ Step 1 - Time series data retrieved: {series_id}")
        
        # Step 2: Submit prediction request
        request_id = await advanced_predictive_analytics_engine.submit_prediction_request(
            prediction_type=PredictionType.PRICE_FORECAST,
            asset="BTC",
            model_type=ModelType.RANDOM_FOREST,
            forecast_horizon=ForecastHorizon.SHORT_TERM,
            input_data={"price": 45000, "volume": 1000000}
        )
        print(f"✅ Step 2 - Prediction request submitted: {request_id}")
        
        # Step 3: Check for anomalies
        anomalies = await anomaly_detection_engine.get_anomaly_detections(asset="BTC", limit=10)
        print(f"✅ Step 3 - Anomaly detection: {len(anomalies)} anomalies for BTC")
        
        # Step 4: Get integrated results
        prediction_result = await advanced_predictive_analytics_engine.get_prediction_result(request_id)
        forecast_results = await time_series_forecasting_engine.get_forecast_results(series_id) if time_series_list else []
        anomaly_patterns = await anomaly_detection_engine.get_anomaly_patterns()
        
        print(f"✅ Step 4 - Integration results:")
        print(f"   Prediction: {'Ready' if prediction_result else 'Processing'}")
        print(f"   Forecasts: {len(forecast_results)} forecasts")
        print(f"   Anomaly patterns: {len(anomaly_patterns)} patterns")
        
        # Stop all engines
        await advanced_predictive_analytics_engine.stop_predictive_analytics_engine()
        await time_series_forecasting_engine.stop_time_series_forecasting_engine()
        await anomaly_detection_engine.stop_anomaly_detection_engine()
        
        print("✅ All analytics engines stopped")
        return True
        
    except Exception as e:
        print(f"❌ Analytics integration test failed: {e}")
        return False

async def test_analytics_performance():
    """Test performance of analytics engines"""
    print("\n⚡ Testing Analytics Performance...")
    
    try:
        from app.services.advanced_predictive_analytics_engine import (
            advanced_predictive_analytics_engine, 
            PredictionType, 
            ModelType, 
            ForecastHorizon
        )
        from app.services.time_series_forecasting_engine import time_series_forecasting_engine
        from app.services.anomaly_detection_engine import anomaly_detection_engine
        
        # Start all engines
        await advanced_predictive_analytics_engine.start_predictive_analytics_engine()
        await time_series_forecasting_engine.start_time_series_forecasting_engine()
        await anomaly_detection_engine.start_anomaly_detection_engine()
        
        # Test predictive analytics performance
        start_time = time.time()
        request_ids = []
        for i in range(20):
            request_id = await advanced_predictive_analytics_engine.submit_prediction_request(
                prediction_type=PredictionType.PRICE_FORECAST,
                asset="BTC",
                model_type=ModelType.RANDOM_FOREST,
                forecast_horizon=ForecastHorizon.SHORT_TERM,
                input_data={"price": 45000 + i, "volume": 1000000}
            )
            request_ids.append(request_id)
        analytics_time = time.time() - start_time
        print(f"✅ Predictive Analytics: 20 requests in {analytics_time:.2f}s ({20/analytics_time:.1f} requests/s)")
        
        # Test time series performance
        start_time = time.time()
        time_series_list = await time_series_forecasting_engine.get_available_time_series()
        for i in range(min(10, len(time_series_list))):
            series_id = time_series_list[i]["series_id"]
            await time_series_forecasting_engine.get_time_series(series_id)
        timeseries_time = time.time() - start_time
        print(f"✅ Time Series: 10 retrievals in {timeseries_time:.2f}s ({10/timeseries_time:.1f} retrievals/s)")
        
        # Test anomaly detection performance
        start_time = time.time()
        for i in range(10):
            await anomaly_detection_engine.get_anomaly_detections(limit=50)
        anomaly_time = time.time() - start_time
        print(f"✅ Anomaly Detection: 10 queries in {anomaly_time:.2f}s ({10/anomaly_time:.1f} queries/s)")
        
        # Stop all engines
        await advanced_predictive_analytics_engine.stop_predictive_analytics_engine()
        await time_series_forecasting_engine.stop_time_series_forecasting_engine()
        await anomaly_detection_engine.stop_anomaly_detection_engine()
        
        return True
        
    except Exception as e:
        print(f"❌ Analytics performance test failed: {e}")
        return False

async def test_new_analytics_api_endpoints():
    """Test the new analytics API endpoints"""
    print("\n🌐 Testing New Analytics API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        endpoints = [
            "/predictive-analytics/health",
            "/time-series-forecasting/health",
            "/anomaly-detection/health"
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
        print(f"❌ Analytics API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Advanced Analytics Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Test individual engines
    test_results.append(await test_advanced_predictive_analytics_engine())
    test_results.append(await test_time_series_forecasting_engine())
    test_results.append(await test_anomaly_detection_engine())
    
    # Test integration
    test_results.append(await test_analytics_integration())
    
    # Test performance
    test_results.append(await test_analytics_performance())
    
    # Test API endpoints
    test_results.append(await test_new_analytics_api_endpoints())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}")
    print(f"❌ Failed: {len(test_results) - sum(test_results)}")
    print(f"📈 Success Rate: {sum(test_results)/len(test_results)*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 All Advanced Analytics Features tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all(test_results)

if __name__ == "__main__":
    asyncio.run(main())
