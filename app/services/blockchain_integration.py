"""
Blockchain Integration Service
Provides smart contract management, transaction monitoring, and decentralized features
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import aiohttp
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
import redis.asyncio as redis
from sqlalchemy.orm import Session
import os

logger = logging.getLogger(__name__)


@dataclass
class SmartContract:
    """Smart contract information"""

    contract_address: str
    contract_name: str
    contract_type: str  # 'market', 'token', 'governance', 'oracle'
    abi: List[Dict]
    bytecode: str
    deployed_at: datetime
    network: str  # 'ethereum', 'polygon', 'arbitrum'
    owner_address: str
    gas_used: int
    transaction_hash: str


@dataclass
class BlockchainTransaction:
    """Blockchain transaction information"""

    transaction_hash: str
    block_number: int
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: int
    status: str  # 'pending', 'confirmed', 'failed'
    timestamp: datetime
    network: str
    contract_interaction: bool
    method_name: Optional[str] = None
    input_data: Optional[str] = None


@dataclass
class MarketToken:
    """Market-specific token information"""

    token_address: str
    market_id: int
    token_name: str
    token_symbol: str
    total_supply: int
    decimals: int
    current_price: float
    market_cap: float
    holders_count: int
    network: str
    created_at: datetime


@dataclass
class OracleData:
    """Oracle data for market resolution"""

    oracle_address: str
    market_id: int
    data_source: str  # 'chainlink', 'custom', 'api'
    data_type: str  # 'price', 'event', 'sport', 'politics'
    data_value: str
    confidence: float
    timestamp: datetime
    transaction_hash: str
    network: str


class BlockchainIntegrationService:
    """Comprehensive blockchain integration service"""

    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.web3_connections: Dict[str, Web3] = {}
        self.contracts: Dict[str, SmartContract] = {}
        self.tokens: Dict[str, MarketToken] = {}
        self.oracles: Dict[str, OracleData] = {}

        # Network configurations
        self.networks = {
            "ethereum": {
                "rpc_url": os.getenv(
                    "ETHEREUM_RPC_URL", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
                ),
                "chain_id": 1,
                "explorer": "https://etherscan.io",
            },
            "polygon": {
                "rpc_url": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
                "chain_id": 137,
                "explorer": "https://polygonscan.com",
            },
            "arbitrum": {
                "rpc_url": os.getenv(
                    "ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"
                ),
                "chain_id": 42161,
                "explorer": "https://arbiscan.io",
            },
        }

        # Smart contract ABIs (simplified versions)
        self.contract_abis = {
            "market": self._get_market_contract_abi(),
            "token": self._get_token_contract_abi(),
            "oracle": self._get_oracle_contract_abi(),
            "governance": self._get_governance_contract_abi(),
        }

    async def initialize(self):
        """Initialize the blockchain integration service"""
        logger.info("Initializing Blockchain Integration Service")

        # Initialize Web3 connections
        await self._initialize_web3_connections()

        # Deploy or load smart contracts
        await self._initialize_smart_contracts()

        # Start monitoring tasks
        asyncio.create_task(self._monitor_transactions())
        asyncio.create_task(self._monitor_gas_prices())
        asyncio.create_task(self._update_token_prices())
        asyncio.create_task(self._monitor_oracle_data())

        logger.info("Blockchain Integration Service initialized successfully")

    async def _initialize_web3_connections(self):
        """Initialize Web3 connections for different networks"""
        try:
            for network_name, config in self.networks.items():
                web3 = Web3(Web3.HTTPProvider(config["rpc_url"]))

                # Add middleware for PoA networks
                if network_name in ["polygon", "arbitrum"]:
                    web3.middleware_onion.inject(geth_poa_middleware, layer=0)

                # Test connection
                if web3.is_connected():
                    self.web3_connections[network_name] = web3
                    logger.info(f"Connected to {network_name} network")
                else:
                    logger.warning(f"Failed to connect to {network_name} network")

        except Exception as e:
            logger.error(f"Error initializing Web3 connections: {e}")

    async def _initialize_smart_contracts(self):
        """Initialize smart contracts"""
        try:
            # Load existing contracts from database or deploy new ones
            existing_contracts = await self._load_existing_contracts()

            for contract in existing_contracts:
                self.contracts[contract.contract_address] = contract

            # Deploy missing contracts
            await self._deploy_missing_contracts()

        except Exception as e:
            logger.error(f"Error initializing smart contracts: {e}")

    def _get_market_contract_abi(self) -> List[Dict]:
        """Get market contract ABI"""
        return [
            {
                "inputs": [
                    {"name": "marketId", "type": "uint256"},
                    {"name": "outcome", "type": "string"},
                    {"name": "shares", "type": "uint256"},
                ],
                "name": "placeTrade",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [{"name": "marketId", "type": "uint256"}],
                "name": "getMarketInfo",
                "outputs": [
                    {"name": "title", "type": "string"},
                    {"name": "totalVolume", "type": "uint256"},
                    {"name": "participantCount", "type": "uint256"},
                    {"name": "isActive", "type": "bool"},
                ],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "marketId", "type": "uint256"},
                    {"indexed": True, "name": "trader", "type": "address"},
                    {"indexed": False, "name": "outcome", "type": "string"},
                    {"indexed": False, "name": "shares", "type": "uint256"},
                ],
                "name": "TradePlaced",
                "type": "event",
            },
        ]

    def _get_token_contract_abi(self) -> List[Dict]:
        """Get token contract ABI"""
        return [
            {
                "inputs": [
                    {"name": "to", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                "name": "transfer",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function",
            },
        ]

    def _get_oracle_contract_abi(self) -> List[Dict]:
        """Get oracle contract ABI"""
        return [
            {
                "inputs": [
                    {"name": "marketId", "type": "uint256"},
                    {"name": "data", "type": "string"},
                ],
                "name": "submitData",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [{"name": "marketId", "type": "uint256"}],
                "name": "getData",
                "outputs": [{"name": "", "type": "string"}],
                "stateMutability": "view",
                "type": "function",
            },
        ]

    def _get_governance_contract_abi(self) -> List[Dict]:
        """Get governance contract ABI"""
        return [
            {
                "inputs": [
                    {"name": "proposalId", "type": "uint256"},
                    {"name": "support", "type": "bool"},
                ],
                "name": "vote",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [{"name": "proposalId", "type": "uint256"}],
                "name": "getProposal",
                "outputs": [
                    {"name": "description", "type": "string"},
                    {"name": "yesVotes", "type": "uint256"},
                    {"name": "noVotes", "type": "uint256"},
                    {"name": "isActive", "type": "bool"},
                ],
                "stateMutability": "view",
                "type": "function",
            },
        ]

    async def create_market_on_blockchain(
        self, market_data: Dict[str, Any], network: str = "ethereum"
    ) -> SmartContract:
        """Create a new market on the blockchain"""
        try:
            if network not in self.web3_connections:
                raise ValueError(f"Network {network} not supported")

            web3 = self.web3_connections[network]

            # Generate market contract
            contract_address = await self._deploy_market_contract(
                market_data, web3, network
            )

            # Create smart contract object
            contract = SmartContract(
                contract_address=contract_address,
                contract_name=f"Market_{market_data['id']}",
                contract_type="market",
                abi=self.contract_abis["market"],
                bytecode=self._get_market_bytecode(),
                deployed_at=datetime.utcnow(),
                network=network,
                owner_address=market_data.get("owner_address", ""),
                gas_used=0,  # Will be updated after deployment
                transaction_hash="",  # Will be updated after deployment
            )

            self.contracts[contract_address] = contract
            await self._cache_contract(contract)

            logger.info(f"Created market contract on {network}: {contract_address}")
            return contract

        except Exception as e:
            logger.error(f"Error creating market on blockchain: {e}")
            raise

    async def _deploy_market_contract(
        self, market_data: Dict[str, Any], web3: Web3, network: str
    ) -> str:
        """Deploy a market contract"""
        try:
            # This would be the actual deployment logic
            # For now, we'll simulate deployment

            # Generate a mock contract address
            contract_address = f"0x{hashlib.md5(f'market_{market_data['id']}_{network}'.encode()).hexdigest()[:40]}"

            # Simulate deployment transaction
            deployment_tx = {
                "from": market_data.get(
                    "owner_address", "0x0000000000000000000000000000000000000000"
                ),
                "to": None,
                "data": self._get_market_bytecode(),
                "gas": 2000000,
                "gasPrice": web3.eth.gas_price,
            }

            # In a real implementation, you would:
            # 1. Sign the transaction with a private key
            # 2. Send the transaction to the network
            # 3. Wait for confirmation
            # 4. Get the deployed contract address

            logger.info(f"Simulated deployment of market contract: {contract_address}")
            return contract_address

        except Exception as e:
            logger.error(f"Error deploying market contract: {e}")
            raise

    def _get_market_bytecode(self) -> str:
        """Get market contract bytecode (simplified)"""
        # This would be the actual compiled bytecode
        return "0x608060405234801561001057600080fd5b506040516101e83803806101e88339818101604052602081101561003357600080fd5b810190808051906020019092919050505080600081905550506101918061005c6000396000f3fe608060405234801561001057600080fd5b50600436106100365760003560e01c80632e1a7d4d1461003b578063a9059cbb14610069575b600080fd5b6100676004803603602081101561005157600080fd5b8101908080359060200190929190505050610097565b005b6100956004803603604081101561007f57600080fd5b8101908080359060200190929190803590602001909291905050506100a1565b005b8060008190555050565b80820190509291505056fea2646970667358221220..."

    async def place_trade_on_blockchain(
        self,
        market_id: int,
        outcome: str,
        shares: int,
        user_address: str,
        network: str = "ethereum",
    ) -> BlockchainTransaction:
        """Place a trade on the blockchain"""
        try:
            if network not in self.web3_connections:
                raise ValueError(f"Network {network} not supported")

            web3 = self.web3_connections[network]

            # Get market contract
            contract_address = await self._get_market_contract_address(
                market_id, network
            )
            if not contract_address:
                raise ValueError(f"Market {market_id} not found on blockchain")

            # Create transaction
            contract = web3.eth.contract(
                address=contract_address, abi=self.contract_abis["market"]
            )

            # Build transaction
            transaction = contract.functions.placeTrade(
                market_id, outcome, shares
            ).build_transaction(
                {
                    "from": user_address,
                    "gas": 200000,
                    "gasPrice": web3.eth.gas_price,
                    "nonce": web3.eth.get_transaction_count(user_address),
                }
            )

            # In a real implementation, you would:
            # 1. Sign the transaction with user's private key
            # 2. Send the transaction to the network
            # 3. Wait for confirmation

            # Simulate transaction
            tx_hash = f"0x{hashlib.md5(f'trade_{market_id}_{user_address}_{datetime.utcnow().isoformat()}'.encode()).hexdigest()[:64]}"

            blockchain_tx = BlockchainTransaction(
                transaction_hash=tx_hash,
                block_number=web3.eth.block_number + 1,
                from_address=user_address,
                to_address=contract_address,
                value=0.0,
                gas_used=150000,
                gas_price=web3.eth.gas_price,
                status="confirmed",
                timestamp=datetime.utcnow(),
                network=network,
                contract_interaction=True,
                method_name="placeTrade",
                input_data=transaction["data"].hex(),
            )

            await self._cache_transaction(blockchain_tx)
            logger.info(f"Placed trade on blockchain: {tx_hash}")

            return blockchain_tx

        except Exception as e:
            logger.error(f"Error placing trade on blockchain: {e}")
            raise

    async def create_market_token(
        self,
        market_id: int,
        token_name: str,
        token_symbol: str,
        total_supply: int,
        network: str = "ethereum",
    ) -> MarketToken:
        """Create a token for a market"""
        try:
            if network not in self.web3_connections:
                raise ValueError(f"Network {network} not supported")

            web3 = self.web3_connections[network]

            # Deploy token contract
            token_address = await self._deploy_token_contract(
                token_name, token_symbol, total_supply, web3, network
            )

            # Create token object
            token = MarketToken(
                token_address=token_address,
                market_id=market_id,
                token_name=token_name,
                token_symbol=token_symbol,
                total_supply=total_supply,
                decimals=18,
                current_price=0.0,
                market_cap=0.0,
                holders_count=0,
                network=network,
                created_at=datetime.utcnow(),
            )

            self.tokens[token_address] = token
            await self._cache_token(token)

            logger.info(f"Created market token: {token_address}")
            return token

        except Exception as e:
            logger.error(f"Error creating market token: {e}")
            raise

    async def _deploy_token_contract(
        self,
        token_name: str,
        token_symbol: str,
        total_supply: int,
        web3: Web3,
        network: str,
    ) -> str:
        """Deploy a token contract"""
        try:
            # Generate mock token address
            token_address = f"0x{hashlib.md5(f'token_{token_name}_{token_symbol}_{network}'.encode()).hexdigest()[:40]}"

            logger.info(f"Simulated deployment of token contract: {token_address}")
            return token_address

        except Exception as e:
            logger.error(f"Error deploying token contract: {e}")
            raise

    async def submit_oracle_data(
        self,
        market_id: int,
        data_source: str,
        data_type: str,
        data_value: str,
        network: str = "ethereum",
    ) -> OracleData:
        """Submit oracle data for market resolution"""
        try:
            if network not in self.web3_connections:
                raise ValueError(f"Network {network} not supported")

            web3 = self.web3_connections[network]

            # Get oracle contract
            oracle_address = await self._get_oracle_contract_address(network)

            # Submit data to oracle
            tx_hash = await self._submit_oracle_data_transaction(
                oracle_address, market_id, data_value, web3, network
            )

            # Create oracle data object
            oracle_data = OracleData(
                oracle_address=oracle_address,
                market_id=market_id,
                data_source=data_source,
                data_type=data_type,
                data_value=data_value,
                confidence=0.95,  # Would be calculated based on data source
                timestamp=datetime.utcnow(),
                transaction_hash=tx_hash,
                network=network,
            )

            self.oracles[f"{market_id}_{network}"] = oracle_data
            await self._cache_oracle_data(oracle_data)

            logger.info(f"Submitted oracle data for market {market_id}: {data_value}")
            return oracle_data

        except Exception as e:
            logger.error(f"Error submitting oracle data: {e}")
            raise

    async def _submit_oracle_data_transaction(
        self,
        oracle_address: str,
        market_id: int,
        data_value: str,
        web3: Web3,
        network: str,
    ) -> str:
        """Submit oracle data transaction"""
        try:
            # Simulate transaction
            tx_hash = f"0x{hashlib.md5(f'oracle_{market_id}_{data_value}_{datetime.utcnow().isoformat()}'.encode()).hexdigest()[:64]}"

            logger.info(f"Simulated oracle data submission: {tx_hash}")
            return tx_hash

        except Exception as e:
            logger.error(f"Error submitting oracle data transaction: {e}")
            raise

    async def _monitor_transactions(self):
        """Monitor blockchain transactions"""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while True:
            try:
                for network_name, web3 in self.web3_connections.items():
                    # Get latest block
                    latest_block = web3.eth.block_number

                    # Monitor recent transactions
                    for block_number in range(latest_block - 10, latest_block + 1):
                        block = web3.eth.get_block(block_number, full_transactions=True)

                        for tx in block.transactions:
                            if tx.to and tx.to in self.contracts:
                                # Process contract transaction
                                await self._process_contract_transaction(
                                    tx, network_name
                                )

                consecutive_errors = 0  # Reset error counter on success
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Error monitoring transactions (attempt {consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Too many consecutive errors ({consecutive_errors}), stopping transaction monitoring"
                    )
                    break

                await asyncio.sleep(
                    min(60 * consecutive_errors, 300)
                )  # Exponential backoff, max 5 minutes

    async def _monitor_gas_prices(self):
        """Monitor gas prices across networks"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            try:
                for network_name, web3 in self.web3_connections.items():
                    gas_price = web3.eth.gas_price
                    await self._record_metric(
                        f"gas_price_{network_name}",
                        web3.from_wei(gas_price, "gwei"),
                        "gwei",
                    )

                consecutive_errors = 0  # Reset error counter on success
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Error monitoring gas prices (attempt {consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Too many consecutive errors ({consecutive_errors}), stopping gas price monitoring"
                    )
                    break

                await asyncio.sleep(
                    min(600 * consecutive_errors, 1800)
                )  # Exponential backoff, max 30 minutes

    async def _update_token_prices(self):
        """Update token prices"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            try:
                for token_address, token in self.tokens.items():
                    # Get token price from DEX or price feed
                    price = await self._get_token_price(token_address, token.network)
                    token.current_price = price
                    token.market_cap = price * token.total_supply / (10**token.decimals)

                    # Update holders count
                    holders = await self._get_token_holders(
                        token_address, token.network
                    )
                    token.holders_count = holders

                    await self._cache_token(token)

                consecutive_errors = 0  # Reset error counter on success
                await asyncio.sleep(600)  # Update every 10 minutes

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Error updating token prices (attempt {consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Too many consecutive errors ({consecutive_errors}), stopping token price updates"
                    )
                    break

                await asyncio.sleep(
                    min(1200 * consecutive_errors, 3600)
                )  # Exponential backoff, max 1 hour

    async def _monitor_oracle_data(self):
        """Monitor oracle data updates"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while True:
            try:
                for oracle_key, oracle_data in self.oracles.items():
                    # Check for new oracle data
                    latest_data = await self._get_latest_oracle_data(
                        oracle_data.oracle_address,
                        oracle_data.market_id,
                        oracle_data.network,
                    )

                    if latest_data and latest_data != oracle_data.data_value:
                        # Update oracle data
                        oracle_data.data_value = latest_data
                        oracle_data.timestamp = datetime.utcnow()
                        await self._cache_oracle_data(oracle_data)

                        logger.info(
                            f"Updated oracle data for market {oracle_data.market_id}"
                        )

                consecutive_errors = 0  # Reset error counter on success
                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Error monitoring oracle data (attempt {consecutive_errors}): {e}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(
                        f"Too many consecutive errors ({consecutive_errors}), stopping oracle data monitoring"
                    )
                    break

                await asyncio.sleep(
                    min(600 * consecutive_errors, 1800)
                )  # Exponential backoff, max 30 minutes

    async def _process_contract_transaction(self, tx, network_name: str):
        """Process a contract transaction"""
        try:
            # Parse transaction data
            contract_address = tx.to
            contract = self.contracts.get(contract_address)

            if contract:
                # Get web3 instance for the network
                web3 = self.web3_connections.get(network_name)
                if not web3:
                    logger.error(f"No web3 connection for network {network_name}")
                    return

                # Decode transaction input
                decoded_input = self._decode_transaction_input(tx.input, contract.abi)

                if decoded_input:
                    # Create transaction record
                    blockchain_tx = BlockchainTransaction(
                        transaction_hash=tx.hash.hex(),
                        block_number=tx.blockNumber,
                        from_address=(
                            tx["from"]
                            if hasattr(tx, "__getitem__")
                            else getattr(tx, "from_address", "")
                        ),
                        to_address=tx.to,
                        value=web3.from_wei(tx.value, "ether"),
                        gas_used=tx.gas,
                        gas_price=tx.gasPrice,
                        status="confirmed",
                        timestamp=datetime.utcnow(),
                        network=network_name,
                        contract_interaction=True,
                        method_name=decoded_input.get("method_name"),
                        input_data=tx.input.hex(),
                    )

                    await self._cache_transaction(blockchain_tx)

        except Exception as e:
            logger.error(f"Error processing contract transaction: {e}")

    def _decode_transaction_input(
        self, input_data: bytes, abi: List[Dict]
    ) -> Optional[Dict]:
        """Decode transaction input data"""
        try:
            # This would use web3.py's contract.decode_function_input
            # For now, return a simplified structure
            return {
                "method_name": "placeTrade",
                "params": ["market_id", "outcome", "shares"],
            }
        except Exception as e:
            logger.error(f"Error decoding transaction input: {e}")
            return None

    # Helper methods
    async def _get_market_contract_address(
        self, market_id: int, network: str
    ) -> Optional[str]:
        """Get market contract address"""
        # Implementation would query database or cache
        return f"0x{hashlib.md5(f'market_{market_id}_{network}'.encode()).hexdigest()[:40]}"

    async def _get_oracle_contract_address(self, network: str) -> str:
        """Get oracle contract address"""
        return f"0x{hashlib.md5(f'oracle_{network}'.encode()).hexdigest()[:40]}"

    async def _get_token_price(self, token_address: str, network: str) -> float:
        """Get token price from DEX or price feed"""
        # Implementation would query DEX APIs or price feeds
        # Use hashlib for deterministic hash instead of built-in hash()
        deterministic_hash = int(
            hashlib.md5(token_address.encode()).hexdigest()[:8], 16
        )
        return 0.5 + (deterministic_hash % 100) / 1000  # Simulated price

    async def _get_token_holders(self, token_address: str, network: str) -> int:
        """Get number of token holders"""
        # Implementation would query blockchain or indexer
        # Use hashlib for deterministic hash instead of built-in hash()
        deterministic_hash = int(
            hashlib.md5(token_address.encode()).hexdigest()[:8], 16
        )
        return deterministic_hash % 1000  # Simulated holder count

    async def _get_latest_oracle_data(
        self, oracle_address: str, market_id: int, network: str
    ) -> Optional[str]:
        """Get latest oracle data"""
        # Implementation would query oracle contract
        return f"data_{market_id}_{datetime.utcnow().timestamp()}"

    async def _load_existing_contracts(self) -> List[SmartContract]:
        """Load existing contracts from database"""
        # Implementation would query database
        return []

    async def _deploy_missing_contracts(self):
        """Deploy missing contracts"""
        # Implementation would deploy contracts that don't exist
        pass

    async def _record_metric(self, metric_name: str, value: float, unit: str):
        """Record a metric"""
        try:
            cache_key = f"blockchain_metric:{metric_name}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "value": value,
                        "unit": unit,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error recording metric: {e}")

    # Caching methods
    async def _cache_contract(self, contract: SmartContract):
        """Cache contract in Redis"""
        try:
            cache_key = f"contract:{contract.contract_address}"
            await self.redis.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps(
                    {
                        "contract_name": contract.contract_name,
                        "contract_type": contract.contract_type,
                        "network": contract.network,
                        "owner_address": contract.owner_address,
                        "deployed_at": contract.deployed_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching contract: {e}")

    async def _cache_transaction(self, transaction: BlockchainTransaction):
        """Cache transaction in Redis"""
        try:
            cache_key = f"transaction:{transaction.transaction_hash}"
            await self.redis.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps(
                    {
                        "block_number": transaction.block_number,
                        "from_address": transaction.from_address,
                        "to_address": transaction.to_address,
                        "value": transaction.value,
                        "status": transaction.status,
                        "network": transaction.network,
                        "timestamp": transaction.timestamp.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching transaction: {e}")

    async def _cache_token(self, token: MarketToken):
        """Cache token in Redis"""
        try:
            cache_key = f"token:{token.token_address}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "market_id": token.market_id,
                        "token_name": token.token_name,
                        "token_symbol": token.token_symbol,
                        "current_price": token.current_price,
                        "market_cap": token.market_cap,
                        "holders_count": token.holders_count,
                        "network": token.network,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching token: {e}")

    async def _cache_oracle_data(self, oracle_data: OracleData):
        """Cache oracle data in Redis"""
        try:
            cache_key = f"oracle:{oracle_data.market_id}:{oracle_data.network}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "data_source": oracle_data.data_source,
                        "data_type": oracle_data.data_type,
                        "data_value": oracle_data.data_value,
                        "confidence": oracle_data.confidence,
                        "timestamp": oracle_data.timestamp.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching oracle data: {e}")


# Factory function
async def get_blockchain_integration_service(
    redis_client: redis.Redis, db_session: Session
) -> BlockchainIntegrationService:
    """Get blockchain integration service instance"""
    service = BlockchainIntegrationService(redis_client, db_session)
    await service.initialize()
    return service
