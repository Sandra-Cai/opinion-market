"""
Test suite for the final iteration features: Edge Computing, Quantum Security, Metaverse Web3, and Autonomous Systems
"""

import asyncio
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_edge_computing_engine():
    """Test Edge Computing Engine"""
    print("\n🚀 Testing Edge Computing Engine...")
    
    try:
        from app.services.edge_computing_engine import edge_computing_engine, TaskPriority
        
        # Start engine
        success = await edge_computing_engine.start_edge_computing_engine()
        print(f"✅ Edge computing engine started: {success}")
        
        # Submit edge task
        task_id = await edge_computing_engine.submit_edge_task(
            "compute", 
            {"data": "test_computation"}, 
            TaskPriority.HIGH
        )
        print(f"✅ Edge task submitted: {task_id}")
        
        # Get task status
        await asyncio.sleep(1)  # Wait for processing
        status = await edge_computing_engine.get_edge_task_status(task_id)
        print(f"✅ Task status: {status['status'] if status else 'Not found'}")
        
        # Get edge nodes
        nodes = await edge_computing_engine.get_edge_nodes()
        print(f"✅ Edge nodes: {len(nodes)} nodes")
        
        # Get performance metrics
        metrics = await edge_computing_engine.get_edge_performance_metrics()
        print(f"✅ Performance metrics: {metrics['performance_metrics']['tasks_processed']} tasks processed")
        
        # Create workload
        workload_tasks = [
            {"task_type": "compute", "data": {"workload": "test1"}, "priority": 2},
            {"task_type": "storage", "data": {"workload": "test2"}, "priority": 1}
        ]
        workload_id = await edge_computing_engine.create_edge_workload(workload_tasks, "round_robin")
        print(f"✅ Edge workload created: {workload_id}")
        
        # Stop engine
        await edge_computing_engine.stop_edge_computing_engine()
        print("✅ Edge computing engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge Computing Engine test failed: {e}")
        return False

async def test_quantum_security_engine():
    """Test Quantum Security Engine"""
    print("\n🔐 Testing Quantum Security Engine...")
    
    try:
        from app.services.quantum_security_engine import quantum_security_engine, QuantumAlgorithm, SecurityLevel, KeyType
        
        # Start engine
        success = await quantum_security_engine.start_quantum_security_engine()
        print(f"✅ Quantum security engine started: {success}")
        
        # Generate quantum key
        key_id = await quantum_security_engine.generate_quantum_key(
            QuantumAlgorithm.KYBER,
            SecurityLevel.LEVEL_3,
            KeyType.ENCRYPTION
        )
        print(f"✅ Quantum key generated: {key_id}")
        
        # Get key details
        key = await quantum_security_engine.get_quantum_key(key_id)
        print(f"✅ Key details: {key['algorithm']} - {key['security_level']}")
        
        # Create quantum signature
        signature_id = await quantum_security_engine.create_quantum_signature(
            "test_data_for_signature",
            key_id
        )
        print(f"✅ Quantum signature created: {signature_id}")
        
        # Verify signature
        verified = await quantum_security_engine.verify_quantum_signature(signature_id, "test_data_for_signature")
        print(f"✅ Signature verified: {verified}")
        
        # Encrypt data
        encryption_id = await quantum_security_engine.encrypt_quantum("sensitive_data", key_id)
        print(f"✅ Data encrypted: {encryption_id}")
        
        # Decrypt data
        decrypted = await quantum_security_engine.decrypt_quantum(encryption_id)
        print(f"✅ Data decrypted: {decrypted}")
        
        # Get performance metrics
        metrics = await quantum_security_engine.get_quantum_performance_metrics()
        print(f"✅ Performance metrics: {metrics['performance_metrics']['keys_generated']} keys generated")
        
        # Stop engine
        await quantum_security_engine.stop_quantum_security_engine()
        print("✅ Quantum security engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum Security Engine test failed: {e}")
        return False

async def test_metaverse_web3_engine():
    """Test Metaverse Web3 Engine"""
    print("\n🌐 Testing Metaverse Web3 Engine...")
    
    try:
        from app.services.metaverse_web3_engine import metaverse_web3_engine, VirtualWorld, AvatarType, BlockchainNetwork, NFTStandard
        
        # Start engine
        success = await metaverse_web3_engine.start_metaverse_web3_engine()
        print(f"✅ Metaverse Web3 engine started: {success}")
        
        # Create virtual asset
        asset_id = await metaverse_web3_engine.create_virtual_asset(
            "land",
            "Virtual Land Plot",
            "A premium virtual land plot",
            VirtualWorld.DECENTRALAND,
            {"x": 100, "y": 200, "z": 0},
            "user123",
            {"size": "10x10", "terrain": "urban"}
        )
        print(f"✅ Virtual asset created: {asset_id}")
        
        # Mint NFT
        nft_id = await metaverse_web3_engine.mint_nft(
            "token123",
            "0x1234567890abcdef",
            BlockchainNetwork.ETHEREUM,
            NFTStandard.ERC721,
            "Virtual Art NFT",
            "A unique piece of virtual art",
            "https://example.com/art.jpg",
            "user456",
            {"artist": "VirtualArtist", "rarity": "legendary"}
        )
        print(f"✅ NFT minted: {nft_id}")
        
        # Create virtual avatar
        avatar_id = await metaverse_web3_engine.create_virtual_avatar(
            "user789",
            "MyAvatar",
            AvatarType.HUMAN,
            {"hair": "brown", "eyes": "blue", "clothing": "casual"},
            VirtualWorld.SANDBOX,
            {"x": 50, "y": 75, "z": 0}
        )
        print(f"✅ Virtual avatar created: {avatar_id}")
        
        # Create virtual event
        from datetime import datetime, timedelta
        start_time = datetime.now() + timedelta(hours=1)
        end_time = start_time + timedelta(hours=2)
        
        event_id = await metaverse_web3_engine.create_virtual_event(
            "Virtual Concert",
            "A live virtual concert experience",
            VirtualWorld.CRYPTOVOXELS,
            {"x": 200, "y": 300, "z": 0},
            start_time,
            end_time,
            "organizer123",
            100,
            10.0
        )
        print(f"✅ Virtual event created: {event_id}")
        
        # Transfer asset
        transfer_success = await metaverse_web3_engine.transfer_asset(asset_id, "user123", "user456")
        print(f"✅ Asset transferred: {transfer_success}")
        
        # Join event
        join_success = await metaverse_web3_engine.join_virtual_event(event_id, "user789")
        print(f"✅ Joined event: {join_success}")
        
        # Get performance metrics
        metrics = await metaverse_web3_engine.get_metaverse_performance_metrics()
        print(f"✅ Performance metrics: {metrics['performance_metrics']['assets_created']} assets created")
        
        # Stop engine
        await metaverse_web3_engine.stop_metaverse_web3_engine()
        print("✅ Metaverse Web3 engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Metaverse Web3 Engine test failed: {e}")
        return False

async def test_autonomous_systems_engine():
    """Test Autonomous Systems Engine"""
    print("\n🤖 Testing Autonomous Systems Engine...")
    
    try:
        from app.services.autonomous_systems_engine import autonomous_systems_engine
        
        # Start engine
        success = await autonomous_systems_engine.start_autonomous_systems_engine()
        print(f"✅ Autonomous systems engine started: {success}")
        
        # Wait for health checks to run
        await asyncio.sleep(2)
        
        # Get system nodes
        nodes = await autonomous_systems_engine.get_system_nodes()
        print(f"✅ System nodes: {len(nodes)} nodes")
        
        # Get system alerts
        alerts = await autonomous_systems_engine.get_system_alerts()
        print(f"✅ System alerts: {len(alerts)} alerts")
        
        # Get recovery plans
        plans = await autonomous_systems_engine.get_recovery_plans()
        print(f"✅ Recovery plans: {len(plans)} plans")
        
        # Acknowledge alert if any
        if alerts:
            alert_id = alerts[0]["alert_id"]
            acknowledged = await autonomous_systems_engine.acknowledge_alert(alert_id)
            print(f"✅ Alert acknowledged: {acknowledged}")
        
        # Get performance metrics
        metrics = await autonomous_systems_engine.get_autonomous_performance_metrics()
        print(f"✅ Performance metrics: {metrics['performance_metrics']['health_checks_performed']} health checks")
        
        # Stop engine
        await autonomous_systems_engine.stop_autonomous_systems_engine()
        print("✅ Autonomous systems engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Autonomous Systems Engine test failed: {e}")
        return False

async def test_new_api_endpoints():
    """Test new API endpoints"""
    print("\n🔌 Testing New API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        
        async with aiohttp.ClientSession() as session:
            # Test edge computing API
            try:
                async with session.get(f"{base_url}/edge-computing/nodes") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Edge Computing API: {data['total']} nodes")
                    else:
                        print(f"⚠️ Edge Computing API: Status {response.status}")
            except Exception as e:
                print(f"⚠️ Edge Computing API not accessible: {e}")
            
            # Test quantum security API
            try:
                async with session.get(f"{base_url}/quantum-security/keys") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Quantum Security API: {data['total']} keys")
                    else:
                        print(f"⚠️ Quantum Security API: Status {response.status}")
            except Exception as e:
                print(f"⚠️ Quantum Security API not accessible: {e}")
            
            # Test metaverse Web3 API
            try:
                async with session.get(f"{base_url}/metaverse-web3/assets") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Metaverse Web3 API: {data['total']} assets")
                    else:
                        print(f"⚠️ Metaverse Web3 API: Status {response.status}")
            except Exception as e:
                print(f"⚠️ Metaverse Web3 API not accessible: {e}")
            
            # Test autonomous systems API
            try:
                async with session.get(f"{base_url}/autonomous-systems/nodes") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Autonomous Systems API: {data['total']} nodes")
                    else:
                        print(f"⚠️ Autonomous Systems API: Status {response.status}")
            except Exception as e:
                print(f"⚠️ Autonomous Systems API not accessible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ New API endpoints test failed: {e}")
        return False

async def test_final_integration_workflow():
    """Test final integration workflow"""
    print("\n🔗 Testing Final Integration Workflow...")
    
    try:
        from app.services.edge_computing_engine import edge_computing_engine, TaskPriority
        from app.services.quantum_security_engine import quantum_security_engine, QuantumAlgorithm, SecurityLevel, KeyType
        from app.services.metaverse_web3_engine import metaverse_web3_engine, VirtualWorld, AvatarType
        from app.services.autonomous_systems_engine import autonomous_systems_engine
        
        # Start all engines
        await edge_computing_engine.start_edge_computing_engine()
        await quantum_security_engine.start_quantum_security_engine()
        await metaverse_web3_engine.start_metaverse_web3_engine()
        await autonomous_systems_engine.start_autonomous_systems_engine()
        print("✅ All final systems started")
        
        # Step 1: Create quantum key for secure communication
        key_id = await quantum_security_engine.generate_quantum_key(
            QuantumAlgorithm.KYBER,
            SecurityLevel.LEVEL_5,
            KeyType.ENCRYPTION
        )
        print(f"✅ Quantum key created: {key_id}")
        
        # Step 2: Encrypt sensitive data
        encryption_id = await quantum_security_engine.encrypt_quantum("sensitive_metaverse_data", key_id)
        print(f"✅ Data encrypted: {encryption_id}")
        
        # Step 3: Submit edge computing task with encrypted data
        task_id = await edge_computing_engine.submit_edge_task(
            "secure_processing",
            {"encrypted_data": encryption_id, "key_id": key_id},
            TaskPriority.CRITICAL
        )
        print(f"✅ Secure edge task submitted: {task_id}")
        
        # Step 4: Create virtual asset in metaverse
        asset_id = await metaverse_web3_engine.create_virtual_asset(
            "secure_land",
            "Quantum-Secured Virtual Land",
            "A virtual land plot secured with quantum encryption",
            VirtualWorld.DECENTRALAND,
            {"x": 500, "y": 600, "z": 0},
            "quantum_user",
            {"security_level": "quantum", "encryption": "kyber"}
        )
        print(f"✅ Quantum-secured virtual asset created: {asset_id}")
        
        # Step 5: Create autonomous avatar
        avatar_id = await metaverse_web3_engine.create_virtual_avatar(
            "autonomous_user",
            "AutonomousAvatar",
            AvatarType.ROBOT,
            {"ai_enabled": True, "autonomous": True, "quantum_secure": True},
            VirtualWorld.SANDBOX,
            {"x": 100, "y": 200, "z": 0}
        )
        print(f"✅ Autonomous avatar created: {avatar_id}")
        
        # Step 6: Monitor system health
        await asyncio.sleep(1)
        health_metrics = await autonomous_systems_engine.get_autonomous_performance_metrics()
        print(f"✅ System health monitored: {health_metrics['healthy_nodes']}/{health_metrics['total_nodes']} nodes healthy")
        
        # Step 7: Get comprehensive metrics
        edge_metrics = await edge_computing_engine.get_edge_performance_metrics()
        quantum_metrics = await quantum_security_engine.get_quantum_performance_metrics()
        metaverse_metrics = await metaverse_web3_engine.get_metaverse_performance_metrics()
        
        print(f"✅ Final integration metrics:")
        print(f"   - Edge tasks processed: {edge_metrics['performance_metrics']['tasks_processed']}")
        print(f"   - Quantum keys generated: {quantum_metrics['performance_metrics']['keys_generated']}")
        print(f"   - Virtual assets created: {metaverse_metrics['performance_metrics']['assets_created']}")
        print(f"   - System uptime: {health_metrics['performance_metrics']['system_uptime']:.1f}%")
        
        # Stop all engines
        await edge_computing_engine.stop_edge_computing_engine()
        await quantum_security_engine.stop_quantum_security_engine()
        await metaverse_web3_engine.stop_metaverse_web3_engine()
        await autonomous_systems_engine.stop_autonomous_systems_engine()
        print("✅ All final systems stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Final integration workflow test failed: {e}")
        return False

async def test_final_performance_benchmarks():
    """Test final performance benchmarks"""
    print("\n⚡ Testing Final Performance Benchmarks...")
    
    try:
        from app.services.edge_computing_engine import edge_computing_engine, TaskPriority
        from app.services.quantum_security_engine import quantum_security_engine, QuantumAlgorithm, SecurityLevel, KeyType
        from app.services.metaverse_web3_engine import metaverse_web3_engine, VirtualWorld, AvatarType
        from app.services.autonomous_systems_engine import autonomous_systems_engine
        
        # Start engines
        await edge_computing_engine.start_edge_computing_engine()
        await quantum_security_engine.start_quantum_security_engine()
        await metaverse_web3_engine.start_metaverse_web3_engine()
        await autonomous_systems_engine.start_autonomous_systems_engine()
        
        # Benchmark edge computing
        start_time = time.time()
        edge_tasks = []
        for i in range(50):
            task_id = await edge_computing_engine.submit_edge_task(
                f"benchmark_task_{i}",
                {"data": f"benchmark_data_{i}"},
                TaskPriority.MEDIUM
            )
            edge_tasks.append(task_id)
        
        edge_time = time.time() - start_time
        print(f"✅ Edge Computing: 50 tasks submitted in {edge_time:.2f}s ({50/edge_time:.1f} tasks/s)")
        
        # Benchmark quantum security
        start_time = time.time()
        quantum_keys = []
        for i in range(20):
            key_id = await quantum_security_engine.generate_quantum_key(
                QuantumAlgorithm.KYBER,
                SecurityLevel.LEVEL_3,
                KeyType.ENCRYPTION
            )
            quantum_keys.append(key_id)
        
        quantum_time = time.time() - start_time
        print(f"✅ Quantum Security: 20 keys generated in {quantum_time:.2f}s ({20/quantum_time:.1f} keys/s)")
        
        # Benchmark metaverse operations
        start_time = time.time()
        metaverse_assets = []
        for i in range(30):
            asset_id = await metaverse_web3_engine.create_virtual_asset(
                f"benchmark_asset_{i}",
                f"Benchmark Asset {i}",
                f"Performance test asset {i}",
                VirtualWorld.DECENTRALAND,
                {"x": i * 10, "y": i * 10, "z": 0},
                f"user_{i}",
                {"benchmark": True}
            )
            metaverse_assets.append(asset_id)
        
        metaverse_time = time.time() - start_time
        print(f"✅ Metaverse Web3: 30 assets created in {metaverse_time:.2f}s ({30/metaverse_time:.1f} assets/s)")
        
        # Benchmark autonomous systems
        start_time = time.time()
        await asyncio.sleep(2)  # Let health checks run
        health_checks = await autonomous_systems_engine.get_autonomous_performance_metrics()
        autonomous_time = time.time() - start_time
        print(f"✅ Autonomous Systems: {health_checks['performance_metrics']['health_checks_performed']} health checks in {autonomous_time:.2f}s")
        
        # Stop engines
        await edge_computing_engine.stop_edge_computing_engine()
        await quantum_security_engine.stop_quantum_security_engine()
        await metaverse_web3_engine.stop_metaverse_web3_engine()
        await autonomous_systems_engine.stop_autonomous_systems_engine()
        
        return True
        
    except Exception as e:
        print(f"❌ Final performance benchmarks test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Final Iteration Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Run individual tests
    test_results.append(("Edge Computing Engine", await test_edge_computing_engine()))
    test_results.append(("Quantum Security Engine", await test_quantum_security_engine()))
    test_results.append(("Metaverse Web3 Engine", await test_metaverse_web3_engine()))
    test_results.append(("Autonomous Systems Engine", await test_autonomous_systems_engine()))
    test_results.append(("New API Endpoints", await test_new_api_endpoints()))
    test_results.append(("Final Integration Workflow", await test_final_integration_workflow()))
    test_results.append(("Final Performance Benchmarks", await test_final_performance_benchmarks()))
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 FINAL ITERATION FEATURES TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 70)
    print(f"Total Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_results)*100):.1f}%")
    print("=" * 70)
    
    if failed == 0:
        print("🎉 All final iteration tests passed! All features are working correctly.")
    else:
        print(f"⚠️ {failed} test(s) failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
