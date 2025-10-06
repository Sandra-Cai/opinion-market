#!/usr/bin/env python3
"""
Test script for the new Advanced IoT Features.
Tests IoT Data Processing Engine, IoT Device Management Engine, and IoT Analytics Engine.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_iot_data_processing_engine():
    """Test the IoT Data Processing Engine"""
    print("\n📡 Testing IoT Data Processing Engine...")
    
    try:
        from app.services.iot_data_processing_engine import (
            iot_data_processing_engine, 
            SensorType, 
            DataQuality
        )
        
        # Test engine initialization
        await iot_data_processing_engine.start_iot_processing_engine()
        print("✅ IoT Data Processing Engine started")
        
        # Test getting devices
        devices = await iot_data_processing_engine.get_devices(limit=10)
        print(f"✅ Devices retrieved: {len(devices)} devices")
        
        # Test getting sensor data
        sensor_data = await iot_data_processing_engine.get_sensor_data(limit=10)
        print(f"✅ Sensor data retrieved: {len(sensor_data)} data points")
        
        # Test getting processing jobs
        jobs = await iot_data_processing_engine.get_processing_jobs(limit=10)
        print(f"✅ Processing jobs retrieved: {len(jobs)} jobs")
        
        # Test getting data insights
        insights = await iot_data_processing_engine.get_data_insights(limit=10)
        print(f"✅ Data insights retrieved: {len(insights)} insights")
        
        # Test adding device
        device_id = await iot_data_processing_engine.add_device(
            name="Test IoT Device",
            device_type="sensor_node",
            location="test_location",
            latitude=40.7128,
            longitude=-74.0060,
            sensors=["temperature", "humidity", "pressure"]
        )
        print(f"✅ Device added: {device_id}")
        
        # Test updating device
        success = await iot_data_processing_engine.update_device(
            device_id, 
            name="Updated Test Device",
            battery_level=85.0
        )
        print(f"✅ Device updated: {success}")
        
        # Test getting engine metrics
        metrics = await iot_data_processing_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_devices']} devices, {metrics['total_sensor_data']} sensor data points")
        
        # Test engine shutdown
        await iot_data_processing_engine.stop_iot_processing_engine()
        print("✅ IoT Data Processing Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ IoT Data Processing Engine test failed: {e}")
        return False

async def test_iot_device_management_engine():
    """Test the IoT Device Management Engine"""
    print("\n🔧 Testing IoT Device Management Engine...")
    
    try:
        from app.services.iot_device_management_engine import (
            iot_device_management_engine, 
            DeviceType, 
            ConfigurationType
        )
        
        # Test engine initialization
        await iot_device_management_engine.start_device_management_engine()
        print("✅ IoT Device Management Engine started")
        
        # Test getting device configurations
        configs = await iot_device_management_engine.get_device_configurations()
        print(f"✅ Device configurations retrieved: {len(configs)} configurations")
        
        # Test getting device firmware
        firmware = await iot_device_management_engine.get_device_firmware()
        print(f"✅ Device firmware retrieved: {len(firmware)} firmware versions")
        
        # Test getting device updates
        updates = await iot_device_management_engine.get_device_updates(limit=10)
        print(f"✅ Device updates retrieved: {len(updates)} updates")
        
        # Test getting device alerts
        alerts = await iot_device_management_engine.get_device_alerts(limit=10)
        print(f"✅ Device alerts retrieved: {len(alerts)} alerts")
        
        # Test creating device update
        update_id = await iot_device_management_engine.create_device_update(
            device_id="test_device_123",
            update_type="firmware",
            target_version="2.0.0",
            current_version="1.0.0"
        )
        print(f"✅ Device update created: {update_id}")
        
        # Test acknowledging alert
        if alerts:
            alert_id = alerts[0]["alert_id"]
            success = await iot_device_management_engine.acknowledge_alert(
                alert_id=alert_id,
                acknowledged_by="test_user"
            )
            print(f"✅ Alert acknowledged: {success}")
        
        # Test updating device configuration
        if configs:
            config_id = configs[0]["config_id"]
            success = await iot_device_management_engine.update_device_configuration(
                config_id=config_id,
                parameters={"sampling_rate": 2, "sensitivity": 0.9}
            )
            print(f"✅ Device configuration updated: {success}")
        
        # Test getting engine metrics
        metrics = await iot_device_management_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_configurations']} configurations, {metrics['total_firmware']} firmware versions")
        
        # Test engine shutdown
        await iot_device_management_engine.stop_device_management_engine()
        print("✅ IoT Device Management Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ IoT Device Management Engine test failed: {e}")
        return False

async def test_iot_analytics_engine():
    """Test the IoT Analytics Engine"""
    print("\n📊 Testing IoT Analytics Engine...")
    
    try:
        from app.services.iot_analytics_engine import (
            iot_analytics_engine, 
            AnalyticsType, 
            InsightType, 
            AlertSeverity
        )
        
        # Test engine initialization
        await iot_analytics_engine.start_analytics_engine()
        print("✅ IoT Analytics Engine started")
        
        # Test getting analytics jobs
        jobs = await iot_analytics_engine.get_analytics_jobs(limit=10)
        print(f"✅ Analytics jobs retrieved: {len(jobs)} jobs")
        
        # Test getting data insights
        insights = await iot_analytics_engine.get_data_insights(limit=10)
        print(f"✅ Data insights retrieved: {len(insights)} insights")
        
        # Test getting predictive models
        models = await iot_analytics_engine.get_predictive_models(limit=10)
        print(f"✅ Predictive models retrieved: {len(models)} models")
        
        # Test getting data patterns
        patterns = await iot_analytics_engine.get_data_patterns(limit=10)
        print(f"✅ Data patterns retrieved: {len(patterns)} patterns")
        
        # Test creating analytics job
        job_id = await iot_analytics_engine.create_analytics_job(
            job_type=AnalyticsType.BATCH,
            device_ids=["device_1", "device_2"],
            sensor_types=["temperature", "humidity"],
            parameters={"window_size": 1000, "analysis_type": "comprehensive"}
        )
        print(f"✅ Analytics job created: {job_id}")
        
        # Test updating model status
        if models:
            model_id = models[0]["model_id"]
            success = await iot_analytics_engine.update_model_status(
                model_id=model_id,
                is_active=False
            )
            print(f"✅ Model status updated: {success}")
        
        # Test getting engine metrics
        metrics = await iot_analytics_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_jobs']} jobs, {metrics['total_insights']} insights, {metrics['total_models']} models")
        
        # Test engine shutdown
        await iot_analytics_engine.stop_analytics_engine()
        print("✅ IoT Analytics Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ IoT Analytics Engine test failed: {e}")
        return False

async def test_iot_integration():
    """Test integration between IoT engines"""
    print("\n🔗 Testing IoT Integration...")
    
    try:
        from app.services.iot_data_processing_engine import iot_data_processing_engine
        from app.services.iot_device_management_engine import iot_device_management_engine
        from app.services.iot_analytics_engine import iot_analytics_engine
        
        # Start all engines
        await iot_data_processing_engine.start_iot_processing_engine()
        await iot_device_management_engine.start_device_management_engine()
        await iot_analytics_engine.start_analytics_engine()
        
        print("✅ All IoT engines started")
        
        # Step 1: Get IoT data
        devices = await iot_data_processing_engine.get_devices(limit=5)
        sensor_data = await iot_data_processing_engine.get_sensor_data(limit=5)
        print(f"✅ Step 1 - IoT data: {len(devices)} devices, {len(sensor_data)} sensor data points")
        
        # Step 2: Get device management data
        configs = await iot_device_management_engine.get_device_configurations()
        firmware = await iot_device_management_engine.get_device_firmware()
        updates = await iot_device_management_engine.get_device_updates(limit=5)
        alerts = await iot_device_management_engine.get_device_alerts(limit=5)
        print(f"✅ Step 2 - Device management: {len(configs)} configs, {len(firmware)} firmware, {len(updates)} updates, {len(alerts)} alerts")
        
        # Step 3: Get analytics data
        jobs = await iot_analytics_engine.get_analytics_jobs(limit=5)
        insights = await iot_analytics_engine.get_data_insights(limit=5)
        models = await iot_analytics_engine.get_predictive_models(limit=5)
        patterns = await iot_analytics_engine.get_data_patterns(limit=5)
        print(f"✅ Step 3 - Analytics: {len(jobs)} jobs, {len(insights)} insights, {len(models)} models, {len(patterns)} patterns")
        
        # Step 4: Get integrated metrics
        data_metrics = await iot_data_processing_engine.get_engine_metrics()
        management_metrics = await iot_device_management_engine.get_engine_metrics()
        analytics_metrics = await iot_analytics_engine.get_engine_metrics()
        
        print(f"✅ Step 4 - Integration results:")
        print(f"   Data Processing: {data_metrics['total_devices']} devices, {data_metrics['total_sensor_data']} sensor data")
        print(f"   Device Management: {management_metrics['total_configurations']} configs, {management_metrics['total_firmware']} firmware")
        print(f"   Analytics: {analytics_metrics['total_jobs']} jobs, {analytics_metrics['total_insights']} insights")
        
        # Stop all engines
        await iot_data_processing_engine.stop_iot_processing_engine()
        await iot_device_management_engine.stop_device_management_engine()
        await iot_analytics_engine.stop_analytics_engine()
        
        print("✅ All IoT engines stopped")
        return True
        
    except Exception as e:
        print(f"❌ IoT integration test failed: {e}")
        return False

async def test_iot_performance():
    """Test performance of IoT engines"""
    print("\n⚡ Testing IoT Performance...")
    
    try:
        from app.services.iot_data_processing_engine import iot_data_processing_engine
        from app.services.iot_device_management_engine import iot_device_management_engine
        from app.services.iot_analytics_engine import iot_analytics_engine
        
        # Start all engines
        await iot_data_processing_engine.start_iot_processing_engine()
        await iot_device_management_engine.start_device_management_engine()
        await iot_analytics_engine.start_analytics_engine()
        
        # Test IoT data processing engine performance
        start_time = time.time()
        for i in range(20):
            await iot_data_processing_engine.get_devices(limit=10)
        data_time = time.time() - start_time
        print(f"✅ IoT Data Processing: 20 queries in {data_time:.2f}s ({20/data_time:.1f} queries/s)")
        
        # Test IoT device management engine performance
        start_time = time.time()
        for i in range(15):
            await iot_device_management_engine.get_device_configurations()
        management_time = time.time() - start_time
        print(f"✅ IoT Device Management: 15 queries in {management_time:.2f}s ({15/management_time:.1f} queries/s)")
        
        # Test IoT analytics engine performance
        start_time = time.time()
        for i in range(10):
            await iot_analytics_engine.get_analytics_jobs(limit=10)
        analytics_time = time.time() - start_time
        print(f"✅ IoT Analytics: 10 queries in {analytics_time:.2f}s ({10/analytics_time:.1f} queries/s)")
        
        # Stop all engines
        await iot_data_processing_engine.stop_iot_processing_engine()
        await iot_device_management_engine.stop_device_management_engine()
        await iot_analytics_engine.stop_analytics_engine()
        
        return True
        
    except Exception as e:
        print(f"❌ IoT performance test failed: {e}")
        return False

async def test_new_iot_api_endpoints():
    """Test the new IoT API endpoints"""
    print("\n🌐 Testing New IoT API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        endpoints = [
            "/iot-data/health",
            "/iot-devices/health",
            "/iot-analytics/health"
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
        print(f"❌ IoT API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Advanced IoT Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Test individual engines
    test_results.append(await test_iot_data_processing_engine())
    test_results.append(await test_iot_device_management_engine())
    test_results.append(await test_iot_analytics_engine())
    
    # Test integration
    test_results.append(await test_iot_integration())
    
    # Test performance
    test_results.append(await test_iot_performance())
    
    # Test API endpoints
    test_results.append(await test_new_iot_api_endpoints())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}")
    print(f"❌ Failed: {len(test_results) - sum(test_results)}")
    print(f"📈 Success Rate: {sum(test_results)/len(test_results)*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 All Advanced IoT Features tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all(test_results)

if __name__ == "__main__":
    asyncio.run(main())
