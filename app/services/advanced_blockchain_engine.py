"""
Advanced Blockchain Integration Engine
Comprehensive blockchain and DeFi integration with multi-chain support
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import aiohttp
import asyncio
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BlockchainType(Enum):
    """Blockchain types"""
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "binance_smart_chain"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"

class TransactionType(Enum):
    """Transaction types"""
    TRANSFER = "transfer"
    SWAP = "swap"
    STAKE = "stake"
    UNSTAKE = "unstake"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"
    BORROW = "borrow"
    REPAY = "repay"
    YIELD_FARM = "yield_farm"
    NFT_MINT = "nft_mint"
    NFT_TRANSFER = "nft_transfer"
    SMART_CONTRACT = "smart_contract"

class DeFiProtocol(Enum):
    """DeFi protocols"""
    UNISWAP = "uniswap"
    SUSHISWAP = "sushiswap"
    PANCAKESWAP = "pancakeswap"
    AAVE = "aave"
    COMPOUND = "compound"
    MAKERDAO = "makerdao"
    CURVE = "curve"
    BALANCER = "balancer"
    YEARN = "yearn"
    CONVEX = "convex"
    FRAX = "frax"
    LIDO = "lido"

class TransactionStatus(Enum):
    """Transaction status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BlockchainTransaction:
    """Blockchain transaction"""
    tx_id: str
    blockchain: BlockchainType
    from_address: str
    to_address: str
    amount: Decimal
    token_address: str
    token_symbol: str
    transaction_type: TransactionType
    protocol: Optional[DeFiProtocol] = None
    gas_used: Optional[int] = None
    gas_price: Optional[Decimal] = None
    block_number: Optional[int] = None
    status: TransactionStatus = TransactionStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SmartContract:
    """Smart contract"""
    contract_id: str
    blockchain: BlockchainType
    address: str
    name: str
    protocol: DeFiProtocol
    abi: Dict[str, Any]
    functions: List[str]
    events: List[str]
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeFiPosition:
    """DeFi position"""
    position_id: str
    user_address: str
    protocol: DeFiProtocol
    blockchain: BlockchainType
    position_type: str  # "liquidity", "lending", "borrowing", "staking"
    token_address: str
    token_symbol: str
    amount: Decimal
    value_usd: Decimal
    apy: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TokenInfo:
    """Token information"""
    address: str
    symbol: str
    name: str
    decimals: int
    blockchain: BlockchainType
    total_supply: Optional[Decimal] = None
    price_usd: Optional[Decimal] = None
    market_cap: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedBlockchainEngine:
    """Advanced Blockchain Integration Engine"""
    
    def __init__(self):
        self.engine_id = f"blockchain_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Blockchain data
        self.transactions: List[BlockchainTransaction] = []
        self.smart_contracts: Dict[str, SmartContract] = {}
        self.defi_positions: List[DeFiPosition] = []
        self.token_info: Dict[str, TokenInfo] = {}
        
        # Blockchain configurations
        self.blockchain_configs = {
            BlockchainType.ETHEREUM: {
                "rpc_url": "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
                "chain_id": 1,
                "native_token": "ETH",
                "gas_token": "ETH"
            },
            BlockchainType.BINANCE_SMART_CHAIN: {
                "rpc_url": "https://bsc-dataseed.binance.org/",
                "chain_id": 56,
                "native_token": "BNB",
                "gas_token": "BNB"
            },
            BlockchainType.POLYGON: {
                "rpc_url": "https://polygon-rpc.com/",
                "chain_id": 137,
                "native_token": "MATIC",
                "gas_token": "MATIC"
            },
            BlockchainType.AVALANCHE: {
                "rpc_url": "https://api.avax.network/ext/bc/C/rpc",
                "chain_id": 43114,
                "native_token": "AVAX",
                "gas_token": "AVAX"
            }
        }
        
        # DeFi protocol configurations
        self.defi_configs = {
            DeFiProtocol.UNISWAP: {
                "router_address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "factory_address": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "supported_chains": [BlockchainType.ETHEREUM, BlockchainType.POLYGON]
            },
            DeFiProtocol.AAVE: {
                "lending_pool_address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
                "supported_chains": [BlockchainType.ETHEREUM, BlockchainType.POLYGON, BlockchainType.AVALANCHE]
            },
            DeFiProtocol.COMPOUND: {
                "comptroller_address": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",
                "supported_chains": [BlockchainType.ETHEREUM]
            }
        }
        
        # Processing tasks
        self.transaction_monitoring_task: Optional[asyncio.Task] = None
        self.defi_monitoring_task: Optional[asyncio.Task] = None
        self.price_feed_task: Optional[asyncio.Task] = None
        self.contract_verification_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.transaction_throughput: List[float] = []
        self.defi_position_updates: List[float] = []
        self.price_update_latency: List[float] = []
        
        logger.info(f"Advanced Blockchain Engine {self.engine_id} initialized")

    async def start_blockchain_engine(self):
        """Start the blockchain engine"""
        if self.is_running:
            return
        
        logger.info("Starting Advanced Blockchain Engine...")
        
        # Initialize blockchain data
        await self._initialize_smart_contracts()
        await self._initialize_token_info()
        await self._initialize_defi_positions()
        
        # Start processing tasks
        self.is_running = True
        
        self.transaction_monitoring_task = asyncio.create_task(self._transaction_monitoring_loop())
        self.defi_monitoring_task = asyncio.create_task(self._defi_monitoring_loop())
        self.price_feed_task = asyncio.create_task(self._price_feed_loop())
        self.contract_verification_task = asyncio.create_task(self._contract_verification_loop())
        
        logger.info("Advanced Blockchain Engine started")

    async def stop_blockchain_engine(self):
        """Stop the blockchain engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Advanced Blockchain Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.transaction_monitoring_task,
            self.defi_monitoring_task,
            self.price_feed_task,
            self.contract_verification_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Advanced Blockchain Engine stopped")

    async def _initialize_smart_contracts(self):
        """Initialize smart contracts"""
        try:
            # Initialize major DeFi contracts
            contracts = [
                {
                    "name": "Uniswap V2 Router",
                    "address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                    "protocol": DeFiProtocol.UNISWAP,
                    "blockchain": BlockchainType.ETHEREUM
                },
                {
                    "name": "Aave Lending Pool",
                    "address": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
                    "protocol": DeFiProtocol.AAVE,
                    "blockchain": BlockchainType.ETHEREUM
                },
                {
                    "name": "Compound Comptroller",
                    "address": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",
                    "protocol": DeFiProtocol.COMPOUND,
                    "blockchain": BlockchainType.ETHEREUM
                }
            ]
            
            for contract_data in contracts:
                contract = SmartContract(
                    contract_id=f"contract_{secrets.token_hex(8)}",
                    blockchain=contract_data["blockchain"],
                    address=contract_data["address"],
                    name=contract_data["name"],
                    protocol=contract_data["protocol"],
                    abi=self._get_contract_abi(contract_data["protocol"]),
                    functions=self._get_contract_functions(contract_data["protocol"]),
                    events=self._get_contract_events(contract_data["protocol"]),
                    is_verified=True
                )
                
                self.smart_contracts[contract.contract_id] = contract
            
            logger.info(f"Initialized {len(self.smart_contracts)} smart contracts")
            
        except Exception as e:
            logger.error(f"Error initializing smart contracts: {e}")

    def _get_contract_abi(self, protocol: DeFiProtocol) -> Dict[str, Any]:
        """Get contract ABI for protocol"""
        # Simplified ABI structures for demonstration
        abis = {
            DeFiProtocol.UNISWAP: {
                "functions": [
                    {"name": "swapExactTokensForTokens", "type": "function"},
                    {"name": "addLiquidity", "type": "function"},
                    {"name": "removeLiquidity", "type": "function"}
                ],
                "events": [
                    {"name": "Swap", "type": "event"},
                    {"name": "Mint", "type": "event"},
                    {"name": "Burn", "type": "event"}
                ]
            },
            DeFiProtocol.AAVE: {
                "functions": [
                    {"name": "deposit", "type": "function"},
                    {"name": "withdraw", "type": "function"},
                    {"name": "borrow", "type": "function"},
                    {"name": "repay", "type": "function"}
                ],
                "events": [
                    {"name": "Deposit", "type": "event"},
                    {"name": "Withdraw", "type": "event"},
                    {"name": "Borrow", "type": "event"},
                    {"name": "Repay", "type": "event"}
                ]
            },
            DeFiProtocol.COMPOUND: {
                "functions": [
                    {"name": "mint", "type": "function"},
                    {"name": "redeem", "type": "function"},
                    {"name": "borrow", "type": "function"},
                    {"name": "repayBorrow", "type": "function"}
                ],
                "events": [
                    {"name": "Mint", "type": "event"},
                    {"name": "Redeem", "type": "event"},
                    {"name": "Borrow", "type": "event"},
                    {"name": "RepayBorrow", "type": "event"}
                ]
            }
        }
        
        return abis.get(protocol, {})

    def _get_contract_functions(self, protocol: DeFiProtocol) -> List[str]:
        """Get contract functions for protocol"""
        functions = {
            DeFiProtocol.UNISWAP: [
                "swapExactTokensForTokens",
                "swapTokensForExactTokens",
                "addLiquidity",
                "removeLiquidity",
                "getAmountsOut",
                "getAmountsIn"
            ],
            DeFiProtocol.AAVE: [
                "deposit",
                "withdraw",
                "borrow",
                "repay",
                "liquidationCall",
                "flashLoan"
            ],
            DeFiProtocol.COMPOUND: [
                "mint",
                "redeem",
                "borrow",
                "repayBorrow",
                "liquidateBorrow",
                "enterMarkets"
            ]
        }
        
        return functions.get(protocol, [])

    def _get_contract_events(self, protocol: DeFiProtocol) -> List[str]:
        """Get contract events for protocol"""
        events = {
            DeFiProtocol.UNISWAP: [
                "Swap",
                "Mint",
                "Burn",
                "Sync"
            ],
            DeFiProtocol.AAVE: [
                "Deposit",
                "Withdraw",
                "Borrow",
                "Repay",
                "LiquidationCall",
                "FlashLoan"
            ],
            DeFiProtocol.COMPOUND: [
                "Mint",
                "Redeem",
                "Borrow",
                "RepayBorrow",
                "LiquidateBorrow"
            ]
        }
        
        return events.get(protocol, [])

    async def _initialize_token_info(self):
        """Initialize token information"""
        try:
            # Initialize major tokens
            tokens = [
                {
                    "address": "0x0000000000000000000000000000000000000000",
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "decimals": 18,
                    "blockchain": BlockchainType.ETHEREUM
                },
                {
                    "address": "0xA0b86a33E6441b8c4C8C0e4b8b8c8c8c8c8c8c8c",
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "decimals": 6,
                    "blockchain": BlockchainType.ETHEREUM
                },
                {
                    "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                    "symbol": "USDT",
                    "name": "Tether USD",
                    "decimals": 6,
                    "blockchain": BlockchainType.ETHEREUM
                },
                {
                    "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                    "symbol": "DAI",
                    "name": "Dai Stablecoin",
                    "decimals": 18,
                    "blockchain": BlockchainType.ETHEREUM
                },
                {
                    "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
                    "symbol": "WBTC",
                    "name": "Wrapped BTC",
                    "decimals": 8,
                    "blockchain": BlockchainType.ETHEREUM
                }
            ]
            
            for token_data in tokens:
                token = TokenInfo(
                    address=token_data["address"],
                    symbol=token_data["symbol"],
                    name=token_data["name"],
                    decimals=token_data["decimals"],
                    blockchain=token_data["blockchain"],
                    price_usd=Decimal(str(secrets.randbelow(1000) + 1)),  # Mock price
                    total_supply=Decimal(str(secrets.randbelow(1000000000) + 1000000))
                )
                
                self.token_info[token.address] = token
            
            logger.info(f"Initialized {len(self.token_info)} tokens")
            
        except Exception as e:
            logger.error(f"Error initializing token info: {e}")

    async def _initialize_defi_positions(self):
        """Initialize DeFi positions"""
        try:
            # Generate mock DeFi positions
            position_types = ["liquidity", "lending", "borrowing", "staking"]
            protocols = list(DeFiProtocol)
            blockchains = list(BlockchainType)
            
            for i in range(50):  # Generate 50 mock positions
                position = DeFiPosition(
                    position_id=f"position_{secrets.token_hex(8)}",
                    user_address=f"0x{secrets.token_hex(20)}",
                    protocol=secrets.choice(protocols),
                    blockchain=secrets.choice(blockchains),
                    position_type=secrets.choice(position_types),
                    token_address=secrets.choice(list(self.token_info.keys())),
                    token_symbol=self.token_info[secrets.choice(list(self.token_info.keys()))].symbol,
                    amount=Decimal(str(secrets.randbelow(10000) + 100)),
                    value_usd=Decimal(str(secrets.randbelow(50000) + 1000)),
                    apy=Decimal(str(secrets.randbelow(50) + 1))  # 1-50% APY
                )
                
                self.defi_positions.append(position)
            
            logger.info(f"Initialized {len(self.defi_positions)} DeFi positions")
            
        except Exception as e:
            logger.error(f"Error initializing DeFi positions: {e}")

    async def _transaction_monitoring_loop(self):
        """Transaction monitoring loop"""
        while self.is_running:
            try:
                # Simulate transaction monitoring
                await self._monitor_new_transactions()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in transaction monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _defi_monitoring_loop(self):
        """DeFi monitoring loop"""
        while self.is_running:
            try:
                # Update DeFi positions
                await self._update_defi_positions()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in DeFi monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _price_feed_loop(self):
        """Price feed loop"""
        while self.is_running:
            try:
                # Update token prices
                await self._update_token_prices()
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in price feed loop: {e}")
                await asyncio.sleep(300)

    async def _contract_verification_loop(self):
        """Contract verification loop"""
        while self.is_running:
            try:
                # Verify smart contracts
                await self._verify_smart_contracts()
                
                await asyncio.sleep(3600)  # Verify every hour
                
            except Exception as e:
                logger.error(f"Error in contract verification loop: {e}")
                await asyncio.sleep(3600)

    async def _monitor_new_transactions(self):
        """Monitor new transactions"""
        try:
            # Generate mock transactions
            transaction_types = list(TransactionType)
            protocols = list(DeFiProtocol)
            blockchains = list(BlockchainType)
            
            # Generate 1-5 new transactions
            num_transactions = secrets.randbelow(5) + 1
            
            for _ in range(num_transactions):
                transaction = BlockchainTransaction(
                    tx_id=f"0x{secrets.token_hex(32)}",
                    blockchain=secrets.choice(blockchains),
                    from_address=f"0x{secrets.token_hex(20)}",
                    to_address=f"0x{secrets.token_hex(20)}",
                    amount=Decimal(str(secrets.randbelow(1000) + 1)),
                    token_address=secrets.choice(list(self.token_info.keys())),
                    token_symbol=self.token_info[secrets.choice(list(self.token_info.keys()))].symbol,
                    transaction_type=secrets.choice(transaction_types),
                    protocol=secrets.choice(protocols) if secrets.choice([True, False]) else None,
                    gas_used=secrets.randbelow(100000) + 21000,
                    gas_price=Decimal(str(secrets.randbelow(100) + 1)),
                    block_number=secrets.randbelow(1000000) + 1000000,
                    status=TransactionStatus.CONFIRMED
                )
                
                self.transactions.append(transaction)
            
            # Keep only last 10000 transactions
            if len(self.transactions) > 10000:
                self.transactions = self.transactions[-10000:]
            
            logger.info(f"Monitored {num_transactions} new transactions")
            
        except Exception as e:
            logger.error(f"Error monitoring transactions: {e}")

    async def _update_defi_positions(self):
        """Update DeFi positions"""
        try:
            # Update position values and APYs
            for position in self.defi_positions:
                # Simulate value changes
                value_change = Decimal(str(secrets.randbelow(20) - 10)) / 100  # -10% to +10%
                position.value_usd *= (1 + value_change)
                
                # Simulate APY changes
                apy_change = Decimal(str(secrets.randbelow(10) - 5)) / 100  # -5% to +5%
                if position.apy:
                    position.apy += apy_change
                    position.apy = max(Decimal("0"), position.apy)  # Don't go negative
                
                position.updated_at = datetime.now()
            
            logger.info(f"Updated {len(self.defi_positions)} DeFi positions")
            
        except Exception as e:
            logger.error(f"Error updating DeFi positions: {e}")

    async def _update_token_prices(self):
        """Update token prices"""
        try:
            # Update token prices with mock data
            for token in self.token_info.values():
                # Simulate price changes
                price_change = Decimal(str(secrets.randbelow(20) - 10)) / 100  # -10% to +10%
                if token.price_usd:
                    token.price_usd *= (1 + price_change)
                    token.price_usd = max(Decimal("0.01"), token.price_usd)  # Minimum price
                
                # Update market cap and volume
                if token.total_supply and token.price_usd:
                    token.market_cap = token.total_supply * token.price_usd
                
                token.volume_24h = Decimal(str(secrets.randbelow(10000000) + 100000))
            
            logger.info(f"Updated prices for {len(self.token_info)} tokens")
            
        except Exception as e:
            logger.error(f"Error updating token prices: {e}")

    async def _verify_smart_contracts(self):
        """Verify smart contracts"""
        try:
            # Simulate contract verification
            for contract in self.smart_contracts.values():
                # Simulate verification process
                if not contract.is_verified:
                    contract.is_verified = secrets.choice([True, False])
                    if contract.is_verified:
                        logger.info(f"Verified contract: {contract.name}")
            
        except Exception as e:
            logger.error(f"Error verifying smart contracts: {e}")

    # Public API methods
    async def get_transactions(self, blockchain: Optional[BlockchainType] = None,
                             transaction_type: Optional[TransactionType] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get transactions"""
        try:
            transactions = self.transactions
            
            # Filter by blockchain
            if blockchain:
                transactions = [t for t in transactions if t.blockchain == blockchain]
            
            # Filter by transaction type
            if transaction_type:
                transactions = [t for t in transactions if t.transaction_type == transaction_type]
            
            # Sort by timestamp (most recent first)
            transactions = sorted(transactions, key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            transactions = transactions[:limit]
            
            return [
                {
                    "tx_id": tx.tx_id,
                    "blockchain": tx.blockchain.value,
                    "from_address": tx.from_address,
                    "to_address": tx.to_address,
                    "amount": float(tx.amount),
                    "token_address": tx.token_address,
                    "token_symbol": tx.token_symbol,
                    "transaction_type": tx.transaction_type.value,
                    "protocol": tx.protocol.value if tx.protocol else None,
                    "gas_used": tx.gas_used,
                    "gas_price": float(tx.gas_price) if tx.gas_price else None,
                    "block_number": tx.block_number,
                    "status": tx.status.value,
                    "timestamp": tx.timestamp.isoformat(),
                    "metadata": tx.metadata
                }
                for tx in transactions
            ]
            
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return []

    async def get_smart_contracts(self) -> List[Dict[str, Any]]:
        """Get smart contracts"""
        try:
            return [
                {
                    "contract_id": contract.contract_id,
                    "blockchain": contract.blockchain.value,
                    "address": contract.address,
                    "name": contract.name,
                    "protocol": contract.protocol.value,
                    "functions": contract.functions,
                    "events": contract.events,
                    "is_verified": contract.is_verified,
                    "created_at": contract.created_at.isoformat(),
                    "metadata": contract.metadata
                }
                for contract in self.smart_contracts.values()
            ]
            
        except Exception as e:
            logger.error(f"Error getting smart contracts: {e}")
            return []

    async def get_defi_positions(self, user_address: Optional[str] = None,
                               protocol: Optional[DeFiProtocol] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get DeFi positions"""
        try:
            positions = self.defi_positions
            
            # Filter by user address
            if user_address:
                positions = [p for p in positions if p.user_address == user_address]
            
            # Filter by protocol
            if protocol:
                positions = [p for p in positions if p.protocol == protocol]
            
            # Sort by updated time (most recent first)
            positions = sorted(positions, key=lambda x: x.updated_at, reverse=True)
            
            # Limit results
            positions = positions[:limit]
            
            return [
                {
                    "position_id": pos.position_id,
                    "user_address": pos.user_address,
                    "protocol": pos.protocol.value,
                    "blockchain": pos.blockchain.value,
                    "position_type": pos.position_type,
                    "token_address": pos.token_address,
                    "token_symbol": pos.token_symbol,
                    "amount": float(pos.amount),
                    "value_usd": float(pos.value_usd),
                    "apy": float(pos.apy) if pos.apy else None,
                    "created_at": pos.created_at.isoformat(),
                    "updated_at": pos.updated_at.isoformat(),
                    "metadata": pos.metadata
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting DeFi positions: {e}")
            return []

    async def get_token_info(self, token_address: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get token information"""
        try:
            if token_address:
                token = self.token_info.get(token_address)
                if token:
                    return [{
                        "address": token.address,
                        "symbol": token.symbol,
                        "name": token.name,
                        "decimals": token.decimals,
                        "blockchain": token.blockchain.value,
                        "total_supply": float(token.total_supply) if token.total_supply else None,
                        "price_usd": float(token.price_usd) if token.price_usd else None,
                        "market_cap": float(token.market_cap) if token.market_cap else None,
                        "volume_24h": float(token.volume_24h) if token.volume_24h else None,
                        "metadata": token.metadata
                    }]
                else:
                    return []
            else:
                return [
                    {
                        "address": token.address,
                        "symbol": token.symbol,
                        "name": token.name,
                        "decimals": token.decimals,
                        "blockchain": token.blockchain.value,
                        "total_supply": float(token.total_supply) if token.total_supply else None,
                        "price_usd": float(token.price_usd) if token.price_usd else None,
                        "market_cap": float(token.market_cap) if token.market_cap else None,
                        "volume_24h": float(token.volume_24h) if token.volume_24h else None,
                        "metadata": token.metadata
                    }
                    for token in self.token_info.values()
                ]
            
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_transactions": len(self.transactions),
                "total_contracts": len(self.smart_contracts),
                "total_positions": len(self.defi_positions),
                "total_tokens": len(self.token_info),
                "supported_blockchains": [bc.value for bc in BlockchainType],
                "supported_protocols": [protocol.value for protocol in DeFiProtocol],
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def add_smart_contract(self, blockchain: BlockchainType, address: str,
                               name: str, protocol: DeFiProtocol) -> str:
        """Add a new smart contract"""
        try:
            contract = SmartContract(
                contract_id=f"contract_{secrets.token_hex(8)}",
                blockchain=blockchain,
                address=address,
                name=name,
                protocol=protocol,
                abi=self._get_contract_abi(protocol),
                functions=self._get_contract_functions(protocol),
                events=self._get_contract_events(protocol),
                is_verified=False
            )
            
            self.smart_contracts[contract.contract_id] = contract
            
            logger.info(f"Added smart contract: {contract.contract_id}")
            return contract.contract_id
            
        except Exception as e:
            logger.error(f"Error adding smart contract: {e}")
            raise

    async def add_token(self, address: str, symbol: str, name: str,
                       decimals: int, blockchain: BlockchainType) -> str:
        """Add a new token"""
        try:
            token = TokenInfo(
                address=address,
                symbol=symbol,
                name=name,
                decimals=decimals,
                blockchain=blockchain
            )
            
            self.token_info[token.address] = token
            
            logger.info(f"Added token: {token.symbol}")
            return token.address
            
        except Exception as e:
            logger.error(f"Error adding token: {e}")
            raise

# Global instance
advanced_blockchain_engine = AdvancedBlockchainEngine()
