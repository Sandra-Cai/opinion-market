"""
Blockchain Service for Opinion Market
Handles blockchain integration, smart contracts, and decentralized features
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
from fastapi import HTTPException, status

from app.core.database import get_db, get_redis_client
from app.core.config import settings
from app.core.cache import cache
from app.core.logging import log_system_metric
from app.models.user import User
from app.models.market import Market
from app.models.trade import Trade


class BlockchainNetwork(str, Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    BSC = "bsc"


class ContractType(str, Enum):
    """Types of smart contracts"""
    MARKET_CONTRACT = "market_contract"
    TRADE_CONTRACT = "trade_contract"
    TOKEN_CONTRACT = "token_contract"
    GOVERNANCE_CONTRACT = "governance_contract"


@dataclass
class BlockchainTransaction:
    """Blockchain transaction data structure"""
    transaction_id: str
    network: BlockchainNetwork
    contract_type: ContractType
    contract_address: str
    function_name: str
    parameters: Dict[str, Any]
    gas_used: int
    gas_price: int
    transaction_hash: str
    block_number: int
    status: str
    created_at: datetime
    confirmed_at: Optional[datetime] = None


@dataclass
class SmartContract:
    """Smart contract data structure"""
    contract_id: str
    contract_type: ContractType
    network: BlockchainNetwork
    address: str
    abi: Dict[str, Any]
    bytecode: str
    deployed_at: datetime
    is_active: bool
    version: str


class BlockchainService:
    """Service for blockchain operations"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = 300  # 5 minutes
        self.networks = {}
        self.contracts = {}
        
        # Initialize blockchain connections
        self._initialize_blockchain_connections()
    
    def _initialize_blockchain_connections(self):
        """Initialize connections to blockchain networks"""
        try:
            if settings.BLOCKCHAIN_ENABLED:
                # Initialize Ethereum connection
                if settings.ETHEREUM_RPC_URL:
                    self.networks[BlockchainNetwork.ETHEREUM] = {
                        "rpc_url": settings.ETHEREUM_RPC_URL,
                        "chain_id": 1,
                        "is_active": True
                    }
                
                # Initialize Polygon connection
                if settings.POLYGON_RPC_URL:
                    self.networks[BlockchainNetwork.POLYGON] = {
                        "rpc_url": settings.POLYGON_RPC_URL,
                        "chain_id": 137,
                        "is_active": True
                    }
                
                # Initialize Arbitrum connection
                if settings.ARBITRUM_RPC_URL:
                    self.networks[BlockchainNetwork.ARBITRUM] = {
                        "rpc_url": settings.ARBITRUM_RPC_URL,
                        "chain_id": 42161,
                        "is_active": True
                    }
                
                # Load smart contracts
                self._load_smart_contracts()
                
                log_system_metric("blockchain_initialized", 1, {
                    "networks": list(self.networks.keys()),
                    "contracts": list(self.contracts.keys())
                })
            else:
                log_system_metric("blockchain_disabled", 1, {})
                
        except Exception as e:
            log_system_metric("blockchain_initialization_error", 1, {"error": str(e)})
    
    def _load_smart_contracts(self):
        """Load smart contract configurations"""
        try:
            # Load market contract
            self.contracts[ContractType.MARKET_CONTRACT] = SmartContract(
                contract_id="market_contract_v1",
                contract_type=ContractType.MARKET_CONTRACT,
                network=BlockchainNetwork.POLYGON,
                address="0x0000000000000000000000000000000000000000",  # Mock address
                abi={},  # Mock ABI
                bytecode="0x",  # Mock bytecode
                deployed_at=datetime.utcnow(),
                is_active=True,
                version="1.0.0"
            )
            
            # Load trade contract
            self.contracts[ContractType.TRADE_CONTRACT] = SmartContract(
                contract_id="trade_contract_v1",
                contract_type=ContractType.TRADE_CONTRACT,
                network=BlockchainNetwork.POLYGON,
                address="0x0000000000000000000000000000000000000001",  # Mock address
                abi={},  # Mock ABI
                bytecode="0x",  # Mock bytecode
                deployed_at=datetime.utcnow(),
                is_active=True,
                version="1.0.0"
            )
            
        except Exception as e:
            log_system_metric("smart_contract_loading_error", 1, {"error": str(e)})
    
    async def create_market_on_blockchain(self, market_data: Dict[str, Any]) -> BlockchainTransaction:
        """Create a market on the blockchain"""
        try:
            if not settings.BLOCKCHAIN_ENABLED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Blockchain integration is disabled"
                )
            
            # Get market contract
            market_contract = self.contracts.get(ContractType.MARKET_CONTRACT)
            if not market_contract:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Market contract not available"
                )
            
            # Prepare transaction parameters
            parameters = {
                "market_id": market_data["market_id"],
                "title": market_data["title"],
                "description": market_data["description"],
                "outcome_a": market_data["outcome_a"],
                "outcome_b": market_data["outcome_b"],
                "closes_at": int(market_data["closes_at"].timestamp()),
                "creator": market_data["creator_address"]
            }
            
            # Create blockchain transaction
            transaction = await self._create_blockchain_transaction(
                network=market_contract.network,
                contract_type=ContractType.MARKET_CONTRACT,
                contract_address=market_contract.address,
                function_name="createMarket",
                parameters=parameters
            )
            
            # Log transaction
            log_system_metric("blockchain_market_created", 1, {
                "market_id": market_data["market_id"],
                "transaction_id": transaction.transaction_id,
                "network": transaction.network.value
            })
            
            return transaction
            
        except Exception as e:
            log_system_metric("blockchain_market_creation_error", 1, {"error": str(e)})
            raise
    
    async def execute_trade_on_blockchain(self, trade_data: Dict[str, Any]) -> BlockchainTransaction:
        """Execute a trade on the blockchain"""
        try:
            if not settings.BLOCKCHAIN_ENABLED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Blockchain integration is disabled"
                )
            
            # Get trade contract
            trade_contract = self.contracts.get(ContractType.TRADE_CONTRACT)
            if not trade_contract:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Trade contract not available"
                )
            
            # Prepare transaction parameters
            parameters = {
                "trade_id": trade_data["trade_id"],
                "market_id": trade_data["market_id"],
                "user_address": trade_data["user_address"],
                "trade_type": trade_data["trade_type"],
                "outcome": trade_data["outcome"],
                "amount": trade_data["amount"],
                "price": trade_data["price"],
                "total_value": trade_data["total_value"]
            }
            
            # Create blockchain transaction
            transaction = await self._create_blockchain_transaction(
                network=trade_contract.network,
                contract_type=ContractType.TRADE_CONTRACT,
                contract_address=trade_contract.address,
                function_name="executeTrade",
                parameters=parameters
            )
            
            # Log transaction
            log_system_metric("blockchain_trade_executed", 1, {
                "trade_id": trade_data["trade_id"],
                "transaction_id": transaction.transaction_id,
                "network": transaction.network.value
            })
            
            return transaction
            
        except Exception as e:
            log_system_metric("blockchain_trade_execution_error", 1, {"error": str(e)})
            raise
    
    async def resolve_market_on_blockchain(self, market_id: int, outcome: str) -> BlockchainTransaction:
        """Resolve a market on the blockchain"""
        try:
            if not settings.BLOCKCHAIN_ENABLED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Blockchain integration is disabled"
                )
            
            # Get market contract
            market_contract = self.contracts.get(ContractType.MARKET_CONTRACT)
            if not market_contract:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Market contract not available"
                )
            
            # Prepare transaction parameters
            parameters = {
                "market_id": market_id,
                "outcome": outcome,
                "resolver": "0x0000000000000000000000000000000000000000"  # Mock resolver address
            }
            
            # Create blockchain transaction
            transaction = await self._create_blockchain_transaction(
                network=market_contract.network,
                contract_type=ContractType.MARKET_CONTRACT,
                contract_address=market_contract.address,
                function_name="resolveMarket",
                parameters=parameters
            )
            
            # Log transaction
            log_system_metric("blockchain_market_resolved", 1, {
                "market_id": market_id,
                "outcome": outcome,
                "transaction_id": transaction.transaction_id,
                "network": transaction.network.value
            })
            
            return transaction
            
        except Exception as e:
            log_system_metric("blockchain_market_resolution_error", 1, {"error": str(e)})
            raise
    
    async def get_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
        """Get blockchain transaction status"""
        try:
            # Get transaction from cache
            cache_key = f"blockchain_transaction:{transaction_id}"
            transaction = cache.get(cache_key)
            
            if not transaction:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Transaction not found"
                )
            
            # In a real implementation, you'd check the blockchain for confirmation
            # For now, we'll return mock status
            status_info = {
                "transaction_id": transaction_id,
                "status": "confirmed",
                "confirmations": 12,
                "block_number": 12345678,
                "gas_used": 21000,
                "gas_price": 20000000000,
                "transaction_hash": transaction.transaction_hash,
                "created_at": transaction.created_at.isoformat(),
                "confirmed_at": datetime.utcnow().isoformat()
            }
            
            return status_info
            
        except Exception as e:
            log_system_metric("blockchain_transaction_status_error", 1, {"error": str(e)})
            raise
    
    async def get_network_status(self, network: BlockchainNetwork) -> Dict[str, Any]:
        """Get blockchain network status"""
        try:
            if network not in self.networks:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Network {network.value} not configured"
                )
            
            network_info = self.networks[network]
            
            # In a real implementation, you'd check the actual network status
            # For now, we'll return mock status
            status_info = {
                "network": network.value,
                "chain_id": network_info["chain_id"],
                "is_active": network_info["is_active"],
                "latest_block": 12345678,
                "gas_price": 20000000000,
                "network_hash_rate": "500 TH/s",
                "difficulty": "1500000000000000",
                "status": "healthy"
            }
            
            return status_info
            
        except Exception as e:
            log_system_metric("blockchain_network_status_error", 1, {"error": str(e)})
            raise
    
    async def get_contract_info(self, contract_type: ContractType) -> Dict[str, Any]:
        """Get smart contract information"""
        try:
            if contract_type not in self.contracts:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Contract {contract_type.value} not found"
                )
            
            contract = self.contracts[contract_type]
            
            return {
                "contract_id": contract.contract_id,
                "contract_type": contract.contract_type.value,
                "network": contract.network.value,
                "address": contract.address,
                "version": contract.version,
                "deployed_at": contract.deployed_at.isoformat(),
                "is_active": contract.is_active
            }
            
        except Exception as e:
            log_system_metric("blockchain_contract_info_error", 1, {"error": str(e)})
            raise
    
    async def _create_blockchain_transaction(
        self,
        network: BlockchainNetwork,
        contract_type: ContractType,
        contract_address: str,
        function_name: str,
        parameters: Dict[str, Any]
    ) -> BlockchainTransaction:
        """Create a blockchain transaction"""
        try:
            # Generate transaction ID
            transaction_id = str(uuid.uuid4())
            
            # Generate mock transaction hash
            transaction_hash = hashlib.sha256(
                f"{transaction_id}{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
            
            # Create transaction object
            transaction = BlockchainTransaction(
                transaction_id=transaction_id,
                network=network,
                contract_type=contract_type,
                contract_address=contract_address,
                function_name=function_name,
                parameters=parameters,
                gas_used=21000,  # Mock gas usage
                gas_price=20000000000,  # Mock gas price
                transaction_hash=transaction_hash,
                block_number=12345678,  # Mock block number
                status="pending",
                created_at=datetime.utcnow()
            )
            
            # Store transaction in cache
            cache_key = f"blockchain_transaction:{transaction_id}"
            cache.set(cache_key, transaction, ttl=24 * 3600)  # 24 hours
            
            # In a real implementation, you'd submit the transaction to the blockchain
            # For now, we'll simulate confirmation after a delay
            asyncio.create_task(self._simulate_transaction_confirmation(transaction))
            
            return transaction
            
        except Exception as e:
            log_system_metric("blockchain_transaction_creation_error", 1, {"error": str(e)})
            raise
    
    async def _simulate_transaction_confirmation(self, transaction: BlockchainTransaction):
        """Simulate transaction confirmation (for testing)"""
        try:
            # Wait for 5 seconds to simulate blockchain confirmation
            await asyncio.sleep(5)
            
            # Update transaction status
            transaction.status = "confirmed"
            transaction.confirmed_at = datetime.utcnow()
            
            # Update cache
            cache_key = f"blockchain_transaction:{transaction.transaction_id}"
            cache.set(cache_key, transaction, ttl=24 * 3600)
            
            # Log confirmation
            log_system_metric("blockchain_transaction_confirmed", 1, {
                "transaction_id": transaction.transaction_id,
                "network": transaction.network.value,
                "contract_type": transaction.contract_type.value
            })
            
        except Exception as e:
            log_system_metric("blockchain_transaction_confirmation_error", 1, {"error": str(e)})
    
    async def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        try:
            stats = {
                "blockchain_enabled": settings.BLOCKCHAIN_ENABLED,
                "networks": {},
                "contracts": {},
                "total_transactions": 0,
                "active_transactions": 0
            }
            
            if settings.BLOCKCHAIN_ENABLED:
                # Get network stats
                for network, info in self.networks.items():
                    stats["networks"][network.value] = {
                        "chain_id": info["chain_id"],
                        "is_active": info["is_active"]
                    }
                
                # Get contract stats
                for contract_type, contract in self.contracts.items():
                    stats["contracts"][contract_type.value] = {
                        "version": contract.version,
                        "network": contract.network.value,
                        "is_active": contract.is_active
                    }
                
                # Get transaction stats (mock data)
                stats["total_transactions"] = 1000
                stats["active_transactions"] = 5
            
            return stats
            
        except Exception as e:
            log_system_metric("blockchain_stats_error", 1, {"error": str(e)})
            raise


# Global blockchain service instance
blockchain_service = BlockchainService()
