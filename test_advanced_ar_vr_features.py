#!/usr/bin/env python3
"""
Test script for the new Advanced AR/VR Features.
Tests AR/VR Experience Engine and Immersive Content Engine.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ar_vr_experience_engine():
    """Test the AR/VR Experience Engine"""
    print("\n🥽 Testing AR/VR Experience Engine...")
    
    try:
        from app.services.ar_vr_experience_engine import (
            ar_vr_experience_engine, 
            ExperienceType, 
            DeviceType, 
            InteractionType, 
            EnvironmentType
        )
        
        # Test engine initialization
        await ar_vr_experience_engine.start_ar_vr_engine()
        print("✅ AR/VR Experience Engine started")
        
        # Test getting environments
        environments = await ar_vr_experience_engine.get_environments(limit=10)
        print(f"✅ Environments retrieved: {len(environments)} environments")
        
        # Test getting users
        users = await ar_vr_experience_engine.get_users(limit=10)
        print(f"✅ Users retrieved: {len(users)} users")
        
        # Test getting objects
        objects = await ar_vr_experience_engine.get_objects(limit=10)
        print(f"✅ Objects retrieved: {len(objects)} objects")
        
        # Test getting events
        events = await ar_vr_experience_engine.get_events(limit=10)
        print(f"✅ Events retrieved: {len(events)} events")
        
        # Test getting sessions
        sessions = await ar_vr_experience_engine.get_sessions(limit=10)
        print(f"✅ Sessions retrieved: {len(sessions)} sessions")
        
        # Test creating environment
        environment_id = await ar_vr_experience_engine.create_environment(
            name="Test VR Environment",
            experience_type=ExperienceType.VIRTUAL_REALITY,
            environment_type=EnvironmentType.VIRTUAL_WORLD,
            description="A test virtual reality environment",
            world_size=(100.0, 20.0, 100.0),
            max_users=25
        )
        print(f"✅ Environment created: {environment_id}")
        
        # Test adding user
        user_id = await ar_vr_experience_engine.add_user(
            username="TestVRUser",
            device_type=DeviceType.VR_HEADSET,
            interaction_type=InteractionType.HAND_TRACKING
        )
        print(f"✅ User added: {user_id}")
        
        # Test creating object
        object_id = await ar_vr_experience_engine.create_object(
            name="Test Cube",
            object_type="cube",
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0)
        )
        print(f"✅ Object created: {object_id}")
        
        # Test getting engine metrics
        metrics = await ar_vr_experience_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_environments']} environments, {metrics['total_users']} users")
        
        # Test engine shutdown
        await ar_vr_experience_engine.stop_ar_vr_engine()
        print("✅ AR/VR Experience Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ AR/VR Experience Engine test failed: {e}")
        return False

async def test_immersive_content_engine():
    """Test the Immersive Content Engine"""
    print("\n🎨 Testing Immersive Content Engine...")
    
    try:
        from app.services.immersive_content_engine import (
            immersive_content_engine, 
            ContentType, 
            MediaFormat, 
            AudioType, 
            QualityLevel
        )
        
        # Test engine initialization
        await immersive_content_engine.start_content_engine()
        print("✅ Immersive Content Engine started")
        
        # Test getting immersive content
        content = await immersive_content_engine.get_immersive_content(limit=10)
        print(f"✅ Immersive content retrieved: {len(content)} content items")
        
        # Test getting spatial audio
        audio = await immersive_content_engine.get_spatial_audio(limit=10)
        print(f"✅ Spatial audio retrieved: {len(audio)} audio items")
        
        # Test getting 3D models
        models = await immersive_content_engine.get_three_d_models(limit=10)
        print(f"✅ 3D models retrieved: {len(models)} models")
        
        # Test getting immersive scenes
        scenes = await immersive_content_engine.get_immersive_scenes(limit=10)
        print(f"✅ Immersive scenes retrieved: {len(scenes)} scenes")
        
        # Test getting content optimizations
        optimizations = await immersive_content_engine.get_content_optimizations(limit=10)
        print(f"✅ Content optimizations retrieved: {len(optimizations)} optimizations")
        
        # Test creating immersive content
        content_id = await immersive_content_engine.create_immersive_content(
            name="Test 3D Model",
            content_type=ContentType.THREE_D_MODEL,
            media_format=MediaFormat.GLTF,
            file_path="/content/test_model.gltf",
            file_size=1024000
        )
        print(f"✅ Immersive content created: {content_id}")
        
        # Test creating spatial audio
        audio_id = await immersive_content_engine.create_spatial_audio(
            name="Test Spatial Audio",
            audio_type=AudioType.SPATIAL_AUDIO,
            file_path="/audio/test_spatial.wav",
            position=(10.0, 5.0, 10.0)
        )
        print(f"✅ Spatial audio created: {audio_id}")
        
        # Test creating immersive scene
        scene_id = await immersive_content_engine.create_immersive_scene(
            name="Test Scene",
            description="A test immersive scene",
            content_objects=[content_id],
            spatial_audio=[audio_id]
        )
        print(f"✅ Immersive scene created: {scene_id}")
        
        # Test getting engine metrics
        metrics = await immersive_content_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_content']} content, {metrics['total_scenes']} scenes")
        
        # Test engine shutdown
        await immersive_content_engine.stop_content_engine()
        print("✅ Immersive Content Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Immersive Content Engine test failed: {e}")
        return False

async def test_ar_vr_integration():
    """Test integration between AR/VR engines"""
    print("\n🔗 Testing AR/VR Integration...")
    
    try:
        from app.services.ar_vr_experience_engine import ar_vr_experience_engine
        from app.services.immersive_content_engine import immersive_content_engine
        
        # Start all engines
        await ar_vr_experience_engine.start_ar_vr_engine()
        await immersive_content_engine.start_content_engine()
        
        print("✅ All AR/VR engines started")
        
        # Step 1: Get AR/VR data
        environments = await ar_vr_experience_engine.get_environments(limit=5)
        users = await ar_vr_experience_engine.get_users(limit=5)
        objects = await ar_vr_experience_engine.get_objects(limit=5)
        events = await ar_vr_experience_engine.get_events(limit=5)
        sessions = await ar_vr_experience_engine.get_sessions(limit=5)
        print(f"✅ Step 1 - AR/VR data: {len(environments)} environments, {len(users)} users, {len(objects)} objects, {len(events)} events, {len(sessions)} sessions")
        
        # Step 2: Get immersive content data
        content = await immersive_content_engine.get_immersive_content(limit=5)
        audio = await immersive_content_engine.get_spatial_audio(limit=5)
        models = await immersive_content_engine.get_three_d_models(limit=5)
        scenes = await immersive_content_engine.get_immersive_scenes(limit=5)
        optimizations = await immersive_content_engine.get_content_optimizations(limit=5)
        print(f"✅ Step 2 - Immersive content: {len(content)} content, {len(audio)} audio, {len(models)} models, {len(scenes)} scenes, {len(optimizations)} optimizations")
        
        # Step 3: Get integrated metrics
        ar_vr_metrics = await ar_vr_experience_engine.get_engine_metrics()
        content_metrics = await immersive_content_engine.get_engine_metrics()
        
        print(f"✅ Step 3 - Integration results:")
        print(f"   AR/VR Experience: {ar_vr_metrics['total_environments']} environments, {ar_vr_metrics['total_users']} users")
        print(f"   Immersive Content: {content_metrics['total_content']} content, {content_metrics['total_scenes']} scenes")
        
        # Stop all engines
        await ar_vr_experience_engine.stop_ar_vr_engine()
        await immersive_content_engine.stop_content_engine()
        
        print("✅ All AR/VR engines stopped")
        return True
        
    except Exception as e:
        print(f"❌ AR/VR integration test failed: {e}")
        return False

async def test_ar_vr_performance():
    """Test performance of AR/VR engines"""
    print("\n⚡ Testing AR/VR Performance...")
    
    try:
        from app.services.ar_vr_experience_engine import ar_vr_experience_engine
        from app.services.immersive_content_engine import immersive_content_engine
        
        # Start all engines
        await ar_vr_experience_engine.start_ar_vr_engine()
        await immersive_content_engine.start_content_engine()
        
        # Test AR/VR experience engine performance
        start_time = time.time()
        for i in range(20):
            await ar_vr_experience_engine.get_environments(limit=10)
        ar_vr_time = time.time() - start_time
        print(f"✅ AR/VR Experience: 20 queries in {ar_vr_time:.2f}s ({20/ar_vr_time:.1f} queries/s)")
        
        # Test immersive content engine performance
        start_time = time.time()
        for i in range(15):
            await immersive_content_engine.get_immersive_content(limit=10)
        content_time = time.time() - start_time
        print(f"✅ Immersive Content: 15 queries in {content_time:.2f}s ({15/content_time:.1f} queries/s)")
        
        # Test combined performance
        start_time = time.time()
        for i in range(10):
            await ar_vr_experience_engine.get_users(limit=5)
            await immersive_content_engine.get_spatial_audio(limit=5)
        combined_time = time.time() - start_time
        print(f"✅ Combined AR/VR: 20 queries in {combined_time:.2f}s ({20/combined_time:.1f} queries/s)")
        
        # Stop all engines
        await ar_vr_experience_engine.stop_ar_vr_engine()
        await immersive_content_engine.stop_content_engine()
        
        return True
        
    except Exception as e:
        print(f"❌ AR/VR performance test failed: {e}")
        return False

async def test_new_ar_vr_api_endpoints():
    """Test the new AR/VR API endpoints"""
    print("\n🌐 Testing New AR/VR API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        endpoints = [
            "/ar-vr/health",
            "/immersive-content/health"
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
        print(f"❌ AR/VR API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Advanced AR/VR Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Test individual engines
    test_results.append(await test_ar_vr_experience_engine())
    test_results.append(await test_immersive_content_engine())
    
    # Test integration
    test_results.append(await test_ar_vr_integration())
    
    # Test performance
    test_results.append(await test_ar_vr_performance())
    
    # Test API endpoints
    test_results.append(await test_new_ar_vr_api_endpoints())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}")
    print(f"❌ Failed: {len(test_results) - sum(test_results)}")
    print(f"📈 Success Rate: {sum(test_results)/len(test_results)*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 All Advanced AR/VR Features tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all(test_results)

if __name__ == "__main__":
    asyncio.run(main())
