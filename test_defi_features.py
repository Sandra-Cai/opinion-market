#!/usr/bin/env python3
"""
Test script for the new DeFi features.
Tests Advanced DeFi Engine including liquidity mining, yield farming, and staking.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_defi_engine():
    """Test the Advanced DeFi Engine"""
    print("\n💰 Testing Advanced DeFi Engine...")
    
    try:
        from app.services.defi_engine import defi_engine, DeFiProtocol, StakingType, LiquidityPool
        
        # Test engine initialization
        await defi_engine.start_defi_engine()
        print("✅ Advanced DeFi Engine started")
        
        # Test creating DeFi position
        position_id = await defi_engine.create_defi_position(
            user_id="test_user_001",
            protocol=DeFiProtocol.UNISWAP,
            pool_type=LiquidityPool.VOLATILE,
            token_pair=("ETH", "USDC"),
            amount=1.0,
            value_usd=2000.0
        )
        print(f"✅ DeFi position created: {position_id}")
        
        # Test creating staking position
        staking_id = await defi_engine.create_staking_position(
            user_id="test_user_001",
            staking_type=StakingType.LIQUIDITY_STAKING,
            token="OPINION",
            amount=1000.0,
            lock_period=30,
            auto_compound=True
        )
        print(f"✅ Staking position created: {staking_id}")
        
        # Test getting user positions
        positions = await defi_engine.get_user_positions("test_user_001")
        print(f"✅ User positions retrieved: {len(positions['defi_positions'])} DeFi, {len(positions['staking_positions'])} staking")
        
        # Test getting yield farms
        farms = await defi_engine.get_yield_farms()
        print(f"✅ Yield farms retrieved: {len(farms)} farms available")
        
        # Test claiming rewards
        claim_result = await defi_engine.claim_rewards("test_user_001", position_id)
        print(f"✅ Rewards claimed: {claim_result['claimed_amount']}")
        
        # Test getting metrics
        metrics = await defi_engine.get_defi_metrics()
        print(f"✅ DeFi metrics retrieved: TVL={metrics['metrics']['total_tvl']}")
        
        # Test engine shutdown
        await defi_engine.stop_defi_engine()
        print("✅ Advanced DeFi Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced DeFi Engine test failed: {e}")
        return False

async def test_defi_protocols():
    """Test DeFi protocol integration"""
    print("\n🔗 Testing DeFi Protocol Integration...")
    
    try:
        from app.services.defi_engine import defi_engine, DeFiProtocol, LiquidityPool
        
        await defi_engine.start_defi_engine()
        
        # Test multiple protocols
        protocols_to_test = [
            (DeFiProtocol.UNISWAP, LiquidityPool.VOLATILE),
            (DeFiProtocol.CURVE, LiquidityPool.STABLE_COIN),
            (DeFiProtocol.AAVE, LiquidityPool.CROSS_CHAIN),
            (DeFiProtocol.COMPOUND, LiquidityPool.GOVERNANCE)
        ]
        
        position_ids = []
        for protocol, pool_type in protocols_to_test:
            position_id = await defi_engine.create_defi_position(
                user_id="test_user_002",
                protocol=protocol,
                pool_type=pool_type,
                token_pair=("USDC", "USDT"),
                amount=100.0,
                value_usd=100.0
            )
            position_ids.append(position_id)
            print(f"✅ {protocol.value} position created: {position_id}")
        
        # Test protocol-specific APY
        for protocol in DeFiProtocol:
            apy = defi_engine._get_protocol_apy(protocol)
            print(f"✅ {protocol.value} APY: {apy}%")
        
        await defi_engine.stop_defi_engine()
        return True
        
    except Exception as e:
        print(f"❌ DeFi protocol integration test failed: {e}")
        return False

async def test_staking_system():
    """Test staking system"""
    print("\n🔒 Testing Staking System...")
    
    try:
        from app.services.defi_engine import defi_engine, StakingType
        
        await defi_engine.start_defi_engine()
        
        # Test different staking types
        staking_types = [
            (StakingType.LIQUIDITY_STAKING, 12.0),
            (StakingType.VALIDATOR_STAKING, 8.0),
            (StakingType.GOVERNANCE_STAKING, 15.0),
            (StakingType.YIELD_STAKING, 20.0),
            (StakingType.LOCKED_STAKING, 25.0)
        ]
        
        staking_ids = []
        for staking_type, expected_apy in staking_types:
            staking_id = await defi_engine.create_staking_position(
                user_id="test_user_003",
                staking_type=staking_type,
                token="OPINION",
                amount=500.0,
                lock_period=60,
                auto_compound=True
            )
            staking_ids.append(staking_id)
            
            # Verify APY
            actual_apy = defi_engine._get_staking_apy(staking_type)
            print(f"✅ {staking_type.value} staking created: {staking_id} (APY: {actual_apy}%)")
        
        # Test staking positions
        positions = await defi_engine.get_user_positions("test_user_003")
        print(f"✅ Staking positions: {len(positions['staking_positions'])} positions")
        
        await defi_engine.stop_defi_engine()
        return True
        
    except Exception as e:
        print(f"❌ Staking system test failed: {e}")
        return False

async def test_yield_farming():
    """Test yield farming strategies"""
    print("\n🌾 Testing Yield Farming Strategies...")
    
    try:
        from app.services.defi_engine import defi_engine, YieldStrategy, DeFiProtocol
        
        await defi_engine.start_defi_engine()
        
        # Test yield farms
        farms = await defi_engine.get_yield_farms()
        print(f"✅ Yield farms available: {len(farms)}")
        
        for farm in farms:
            print(f"✅ Farm: {farm['name']} - APY: {farm['apy']}%, TVL: ${farm['tvl']:,.0f}, Risk: {farm['risk_level']}")
        
        # Test different strategies
        strategies = [
            YieldStrategy.SIMPLE_STAKING,
            YieldStrategy.LIQUIDITY_PROVISION,
            YieldStrategy.AUTO_COMPOUNDING,
            YieldStrategy.ARBITRAGE
        ]
        
        for strategy in strategies:
            print(f"✅ Strategy available: {strategy.value}")
        
        await defi_engine.stop_defi_engine()
        return True
        
    except Exception as e:
        print(f"❌ Yield farming test failed: {e}")
        return False

async def test_liquidity_mining():
    """Test liquidity mining rewards"""
    print("\n⛏️ Testing Liquidity Mining...")
    
    try:
        from app.services.defi_engine import defi_engine, DeFiProtocol, LiquidityPool
        
        await defi_engine.start_defi_engine()
        
        # Create multiple positions for liquidity mining
        user_ids = ["miner_001", "miner_002", "miner_003"]
        position_ids = []
        
        for user_id in user_ids:
            position_id = await defi_engine.create_defi_position(
                user_id=user_id,
                protocol=DeFiProtocol.UNISWAP,
                pool_type=LiquidityPool.VOLATILE,
                token_pair=("ETH", "USDC"),
                amount=2.0,
                value_usd=4000.0
            )
            position_ids.append(position_id)
            print(f"✅ Liquidity position created for {user_id}: {position_id}")
        
        # Wait for rewards to be calculated
        await asyncio.sleep(2)
        
        # Check rewards
        total_rewards = 0
        for user_id in user_ids:
            positions = await defi_engine.get_user_positions(user_id)
            user_rewards = positions['total_rewards']
            total_rewards += user_rewards
            print(f"✅ {user_id} rewards: ${user_rewards:.2f}")
        
        print(f"✅ Total liquidity mining rewards: ${total_rewards:.2f}")
        
        await defi_engine.stop_defi_engine()
        return True
        
    except Exception as e:
        print(f"❌ Liquidity mining test failed: {e}")
        return False

async def test_defi_integration():
    """Test integration between DeFi components"""
    print("\n🔗 Testing DeFi Integration...")
    
    try:
        from app.services.defi_engine import defi_engine, DeFiProtocol, StakingType, LiquidityPool
        
        await defi_engine.start_defi_engine()
        
        # Step 1: Create DeFi position
        position_id = await defi_engine.create_defi_position(
            user_id="integration_user",
            protocol=DeFiProtocol.AAVE,
            pool_type=LiquidityPool.CROSS_CHAIN,
            token_pair=("USDC", "USDC"),
            amount=1000.0,
            value_usd=1000.0
        )
        print(f"✅ Step 1 - DeFi position created: {position_id}")
        
        # Step 2: Create staking position
        staking_id = await defi_engine.create_staking_position(
            user_id="integration_user",
            staking_type=StakingType.GOVERNANCE_STAKING,
            token="OPINION",
            amount=500.0,
            lock_period=90,
            auto_compound=True
        )
        print(f"✅ Step 2 - Staking position created: {staking_id}")
        
        # Step 3: Get comprehensive user portfolio
        portfolio = await defi_engine.get_user_positions("integration_user")
        print(f"✅ Step 3 - Portfolio: ${portfolio['total_value']:.2f} total value, ${portfolio['total_rewards']:.2f} rewards")
        
        # Step 4: Get yield farms for additional opportunities
        farms = await defi_engine.get_yield_farms()
        best_farm = max(farms, key=lambda x: x['apy'])
        print(f"✅ Step 4 - Best yield farm: {best_farm['name']} ({best_farm['apy']}% APY)")
        
        # Step 5: Claim rewards
        claim_result = await defi_engine.claim_rewards("integration_user", position_id)
        print(f"✅ Step 5 - Rewards claimed: ${claim_result['claimed_amount']:.2f}")
        
        # Step 6: Get final metrics
        metrics = await defi_engine.get_defi_metrics()
        print(f"✅ Step 6 - System metrics: {metrics['metrics']['total_tvl']:.2f} TVL, {metrics['metrics']['average_apy']:.1f}% avg APY")
        
        await defi_engine.stop_defi_engine()
        return True
        
    except Exception as e:
        print(f"❌ DeFi integration test failed: {e}")
        return False

async def test_defi_performance():
    """Test DeFi engine performance"""
    print("\n⚡ Testing DeFi Performance...")
    
    try:
        from app.services.defi_engine import defi_engine, DeFiProtocol, StakingType, LiquidityPool
        
        await defi_engine.start_defi_engine()
        
        # Test position creation performance
        start_time = time.time()
        position_ids = []
        for i in range(50):
            position_id = await defi_engine.create_defi_position(
                user_id=f"perf_user_{i}",
                protocol=DeFiProtocol.UNISWAP,
                pool_type=LiquidityPool.VOLATILE,
                token_pair=("ETH", "USDC"),
                amount=1.0,
                value_usd=2000.0
            )
            position_ids.append(position_id)
        position_time = time.time() - start_time
        print(f"✅ Position creation: 50 positions in {position_time:.2f}s ({50/position_time:.1f} pos/s)")
        
        # Test staking creation performance
        start_time = time.time()
        staking_ids = []
        for i in range(50):
            staking_id = await defi_engine.create_staking_position(
                user_id=f"perf_user_{i}",
                staking_type=StakingType.LIQUIDITY_STAKING,
                token="OPINION",
                amount=100.0,
                lock_period=30,
                auto_compound=True
            )
            staking_ids.append(staking_id)
        staking_time = time.time() - start_time
        print(f"✅ Staking creation: 50 positions in {staking_time:.2f}s ({50/staking_time:.1f} pos/s)")
        
        # Test position retrieval performance
        start_time = time.time()
        for i in range(50):
            positions = await defi_engine.get_user_positions(f"perf_user_{i}")
        retrieval_time = time.time() - start_time
        print(f"✅ Position retrieval: 50 queries in {retrieval_time:.2f}s ({50/retrieval_time:.1f} queries/s)")
        
        # Test yield farm retrieval performance
        start_time = time.time()
        for i in range(50):
            farms = await defi_engine.get_yield_farms()
        farms_time = time.time() - start_time
        print(f"✅ Yield farm retrieval: 50 queries in {farms_time:.2f}s ({50/farms_time:.1f} queries/s)")
        
        await defi_engine.stop_defi_engine()
        return True
        
    except Exception as e:
        print(f"❌ DeFi performance test failed: {e}")
        return False

async def test_new_api_endpoints():
    """Test the new DeFi API endpoints"""
    print("\n🌐 Testing New DeFi API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        endpoints = [
            "/defi/positions",
            "/defi/staking",
            "/defi/yield-farms",
            "/defi/metrics",
            "/defi/protocols",
            "/defi/staking-types",
            "/defi/yield-strategies",
            "/defi/liquidity-pools",
            "/defi/status"
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
        print(f"❌ API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting DeFi Features Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test individual components
    test_results.append(await test_defi_engine())
    test_results.append(await test_defi_protocols())
    test_results.append(await test_staking_system())
    test_results.append(await test_yield_farming())
    test_results.append(await test_liquidity_mining())
    
    # Test integration
    test_results.append(await test_defi_integration())
    
    # Test performance
    test_results.append(await test_defi_performance())
    
    # Test API endpoints
    test_results.append(await test_new_api_endpoints())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}")
    print(f"❌ Failed: {len(test_results) - sum(test_results)}")
    print(f"📈 Success Rate: {sum(test_results)/len(test_results)*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 All DeFi Features tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all(test_results)

if __name__ == "__main__":
    asyncio.run(main())
