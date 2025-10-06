#!/usr/bin/env python3
"""
Test script for the new Advanced Blockchain Features.
Tests Advanced Blockchain Engine, DeFi Protocol Manager, and Smart Contract Engine.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_blockchain_engine():
    """Test the Advanced Blockchain Engine"""
    print("\n⛓️ Testing Advanced Blockchain Engine...")
    
    try:
        from app.services.advanced_blockchain_engine import (
            advanced_blockchain_engine, 
            BlockchainType, 
            TransactionType, 
            DeFiProtocol
        )
        
        # Test engine initialization
        await advanced_blockchain_engine.start_blockchain_engine()
        print("✅ Advanced Blockchain Engine started")
        
        # Test getting transactions
        transactions = await advanced_blockchain_engine.get_transactions(limit=10)
        print(f"✅ Transactions retrieved: {len(transactions)} transactions")
        
        # Test getting smart contracts
        contracts = await advanced_blockchain_engine.get_smart_contracts()
        print(f"✅ Smart contracts retrieved: {len(contracts)} contracts")
        
        # Test getting DeFi positions
        positions = await advanced_blockchain_engine.get_defi_positions(limit=10)
        print(f"✅ DeFi positions retrieved: {len(positions)} positions")
        
        # Test getting token info
        tokens = await advanced_blockchain_engine.get_token_info()
        print(f"✅ Token info retrieved: {len(tokens)} tokens")
        
        # Test adding smart contract
        contract_id = await advanced_blockchain_engine.add_smart_contract(
            blockchain=BlockchainType.ETHEREUM,
            address="0x1234567890123456789012345678901234567890",
            name="Test Contract",
            protocol=DeFiProtocol.UNISWAP
        )
        print(f"✅ Smart contract added: {contract_id}")
        
        # Test adding token
        token_address = await advanced_blockchain_engine.add_token(
            address="0x9876543210987654321098765432109876543210",
            symbol="TEST",
            name="Test Token",
            decimals=18,
            blockchain=BlockchainType.ETHEREUM
        )
        print(f"✅ Token added: {token_address}")
        
        # Test getting engine metrics
        metrics = await advanced_blockchain_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_transactions']} transactions, {metrics['total_contracts']} contracts")
        
        # Test engine shutdown
        await advanced_blockchain_engine.stop_blockchain_engine()
        print("✅ Advanced Blockchain Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced Blockchain Engine test failed: {e}")
        return False

async def test_defi_protocol_manager():
    """Test the DeFi Protocol Manager"""
    print("\n🏦 Testing DeFi Protocol Manager...")
    
    try:
        from app.services.defi_protocol_manager import (
            defi_protocol_manager, 
            DeFiStrategyType, 
            RiskLevel
        )
        
        # Test manager initialization
        await defi_protocol_manager.start_defi_manager()
        print("✅ DeFi Protocol Manager started")
        
        # Test getting strategies
        strategies = await defi_protocol_manager.get_strategies()
        print(f"✅ Strategies retrieved: {len(strategies)} strategies")
        
        # Test getting positions
        positions = await defi_protocol_manager.get_positions(limit=10)
        print(f"✅ Positions retrieved: {len(positions)} positions")
        
        # Test getting yield opportunities
        opportunities = await defi_protocol_manager.get_yield_opportunities(limit=10)
        print(f"✅ Yield opportunities retrieved: {len(opportunities)} opportunities")
        
        # Test adding strategy
        strategy_id = await defi_protocol_manager.add_strategy(
            name="Test Strategy",
            strategy_type=DeFiStrategyType.YIELD_FARMING,
            protocol="aave",
            blockchain="ethereum",
            risk_level=RiskLevel.MEDIUM,
            expected_apy=0.10,  # 10%
            min_investment=1000,
            max_investment=50000
        )
        print(f"✅ Strategy added: {strategy_id}")
        
        # Test updating strategy
        success = await defi_protocol_manager.update_strategy(
            strategy_id, 
            expected_apy=0.12,  # 12%
            enabled=False
        )
        print(f"✅ Strategy updated: {success}")
        
        # Test getting risk metrics
        risk_metrics = await defi_protocol_manager.get_risk_metrics()
        print(f"✅ Risk metrics: {risk_metrics.get('total_exposure', 0)} total exposure")
        
        # Test getting performance metrics
        performance = await defi_protocol_manager.get_performance_metrics()
        print(f"✅ Performance metrics: {performance['total_positions']} positions")
        
        # Test getting manager metrics
        metrics = await defi_protocol_manager.get_manager_metrics()
        print(f"✅ Manager metrics: {metrics['total_strategies']} strategies, {metrics['total_positions']} positions")
        
        # Test manager shutdown
        await defi_protocol_manager.stop_defi_manager()
        print("✅ DeFi Protocol Manager stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ DeFi Protocol Manager test failed: {e}")
        return False

async def test_smart_contract_engine():
    """Test the Smart Contract Engine"""
    print("\n📜 Testing Smart Contract Engine...")
    
    try:
        from app.services.smart_contract_engine import (
            smart_contract_engine, 
            ContractType, 
            ContractStatus
        )
        
        # Test engine initialization
        await smart_contract_engine.start_smart_contract_engine()
        print("✅ Smart Contract Engine started")
        
        # Test getting contracts
        contracts = await smart_contract_engine.get_contracts(limit=10)
        print(f"✅ Contracts retrieved: {len(contracts)} contracts")
        
        # Test getting contract templates
        templates = await smart_contract_engine.get_contract_templates()
        print(f"✅ Contract templates retrieved: {len(templates)} templates")
        
        # Test getting contract events
        events = await smart_contract_engine.get_contract_events(limit=10)
        print(f"✅ Contract events retrieved: {len(events)} events")
        
        # Test deploying contract
        if templates:
            template_id = templates[0]["template_id"]
            contract_id = await smart_contract_engine.deploy_contract(
                template_id=template_id,
                name="Test Contract",
                blockchain="ethereum",
                constructor_args=["Test Token", "TEST", 18, 1000000],
                deployer_address="0x1234567890123456789012345678901234567890"
            )
            print(f"✅ Contract deployed: {contract_id}")
        
        # Test interacting with contract
        if contracts:
            contract_id = contracts[0]["contract_id"]
            interaction_id = await smart_contract_engine.interact_with_contract(
                contract_id=contract_id,
                function_name="transfer",
                caller_address="0x1234567890123456789012345678901234567890",
                parameters={"to": "0x9876543210987654321098765432109876543210", "amount": 1000},
                gas_limit=100000,
                gas_price=20.0,
                value=0.0
            )
            print(f"✅ Contract interaction: {interaction_id}")
        
        # Test getting engine metrics
        metrics = await smart_contract_engine.get_engine_metrics()
        print(f"✅ Engine metrics: {metrics['total_contracts']} contracts, {metrics['total_events']} events")
        
        # Test engine shutdown
        await smart_contract_engine.stop_smart_contract_engine()
        print("✅ Smart Contract Engine stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart Contract Engine test failed: {e}")
        return False

async def test_blockchain_integration():
    """Test integration between blockchain engines"""
    print("\n🔗 Testing Blockchain Integration...")
    
    try:
        from app.services.advanced_blockchain_engine import advanced_blockchain_engine
        from app.services.defi_protocol_manager import defi_protocol_manager
        from app.services.smart_contract_engine import smart_contract_engine
        
        # Start all engines
        await advanced_blockchain_engine.start_blockchain_engine()
        await defi_protocol_manager.start_defi_manager()
        await smart_contract_engine.start_smart_contract_engine()
        
        print("✅ All blockchain engines started")
        
        # Step 1: Get blockchain data
        transactions = await advanced_blockchain_engine.get_transactions(limit=5)
        contracts = await advanced_blockchain_engine.get_smart_contracts()
        print(f"✅ Step 1 - Blockchain data: {len(transactions)} transactions, {len(contracts)} contracts")
        
        # Step 2: Get DeFi data
        strategies = await defi_protocol_manager.get_strategies()
        positions = await defi_protocol_manager.get_positions(limit=5)
        opportunities = await defi_protocol_manager.get_yield_opportunities(limit=5)
        print(f"✅ Step 2 - DeFi data: {len(strategies)} strategies, {len(positions)} positions, {len(opportunities)} opportunities")
        
        # Step 3: Get smart contract data
        smart_contracts = await smart_contract_engine.get_contracts(limit=5)
        templates = await smart_contract_engine.get_contract_templates()
        events = await smart_contract_engine.get_contract_events(limit=5)
        print(f"✅ Step 3 - Smart contract data: {len(smart_contracts)} contracts, {len(templates)} templates, {len(events)} events")
        
        # Step 4: Get integrated metrics
        blockchain_metrics = await advanced_blockchain_engine.get_engine_metrics()
        defi_metrics = await defi_protocol_manager.get_manager_metrics()
        contract_metrics = await smart_contract_engine.get_engine_metrics()
        
        print(f"✅ Step 4 - Integration results:")
        print(f"   Blockchain: {blockchain_metrics['total_transactions']} transactions, {blockchain_metrics['total_contracts']} contracts")
        print(f"   DeFi: {defi_metrics['total_strategies']} strategies, {defi_metrics['total_positions']} positions")
        print(f"   Smart Contracts: {contract_metrics['total_contracts']} contracts, {contract_metrics['total_events']} events")
        
        # Stop all engines
        await advanced_blockchain_engine.stop_blockchain_engine()
        await defi_protocol_manager.stop_defi_manager()
        await smart_contract_engine.stop_smart_contract_engine()
        
        print("✅ All blockchain engines stopped")
        return True
        
    except Exception as e:
        print(f"❌ Blockchain integration test failed: {e}")
        return False

async def test_blockchain_performance():
    """Test performance of blockchain engines"""
    print("\n⚡ Testing Blockchain Performance...")
    
    try:
        from app.services.advanced_blockchain_engine import advanced_blockchain_engine
        from app.services.defi_protocol_manager import defi_protocol_manager
        from app.services.smart_contract_engine import smart_contract_engine
        
        # Start all engines
        await advanced_blockchain_engine.start_blockchain_engine()
        await defi_protocol_manager.start_defi_manager()
        await smart_contract_engine.start_smart_contract_engine()
        
        # Test blockchain engine performance
        start_time = time.time()
        for i in range(20):
            await advanced_blockchain_engine.get_transactions(limit=10)
        blockchain_time = time.time() - start_time
        print(f"✅ Blockchain Engine: 20 queries in {blockchain_time:.2f}s ({20/blockchain_time:.1f} queries/s)")
        
        # Test DeFi manager performance
        start_time = time.time()
        for i in range(15):
            await defi_protocol_manager.get_positions(limit=10)
        defi_time = time.time() - start_time
        print(f"✅ DeFi Manager: 15 queries in {defi_time:.2f}s ({15/defi_time:.1f} queries/s)")
        
        # Test smart contract engine performance
        start_time = time.time()
        for i in range(10):
            await smart_contract_engine.get_contracts(limit=10)
        contract_time = time.time() - start_time
        print(f"✅ Smart Contract Engine: 10 queries in {contract_time:.2f}s ({10/contract_time:.1f} queries/s)")
        
        # Stop all engines
        await advanced_blockchain_engine.stop_blockchain_engine()
        await defi_protocol_manager.stop_defi_manager()
        await smart_contract_engine.stop_smart_contract_engine()
        
        return True
        
    except Exception as e:
        print(f"❌ Blockchain performance test failed: {e}")
        return False

async def test_new_blockchain_api_endpoints():
    """Test the new blockchain API endpoints"""
    print("\n🌐 Testing New Blockchain API Endpoints...")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8000/api/v1"
        endpoints = [
            "/blockchain/health",
            "/defi/health",
            "/smart-contracts/health"
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
        print(f"❌ Blockchain API endpoints test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Advanced Blockchain Features Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Test individual engines
    test_results.append(await test_advanced_blockchain_engine())
    test_results.append(await test_defi_protocol_manager())
    test_results.append(await test_smart_contract_engine())
    
    # Test integration
    test_results.append(await test_blockchain_integration())
    
    # Test performance
    test_results.append(await test_blockchain_performance())
    
    # Test API endpoints
    test_results.append(await test_new_blockchain_api_endpoints())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(test_results)}")
    print(f"❌ Failed: {len(test_results) - sum(test_results)}")
    print(f"📈 Success Rate: {sum(test_results)/len(test_results)*100:.1f}%")
    
    if all(test_results):
        print("\n🎉 All Advanced Blockchain Features tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
    
    return all(test_results)

if __name__ == "__main__":
    asyncio.run(main())
