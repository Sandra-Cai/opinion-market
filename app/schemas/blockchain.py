"""
Pydantic schemas for Blockchain Integration API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class MarketCreationRequest(BaseModel):
    """Request model for creating a market on blockchain"""

    market_id: int = Field(..., description="Market ID")
    title: str = Field(..., description="Market title")
    description: str = Field(..., description="Market description")
    outcome_a: str = Field(..., description="First outcome")
    outcome_b: str = Field(..., description="Second outcome")
    owner_address: str = Field(..., description="Owner's blockchain address")
    network: str = Field(default="ethereum", description="Blockchain network")
    end_date: Optional[datetime] = Field(None, description="Market end date")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "title": "Will Bitcoin reach $100k by end of 2024?",
                "description": "Prediction market for Bitcoin price",
                "outcome_a": "Yes",
                "outcome_b": "No",
                "owner_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                "network": "ethereum",
                "end_date": "2024-12-31T23:59:59Z",
            }
        }


class MarketCreationResponse(BaseModel):
    """Response model for market creation"""

    market_id: int = Field(..., description="Market ID")
    contract_address: str = Field(..., description="Deployed contract address")
    network: str = Field(..., description="Blockchain network")
    transaction_hash: str = Field(..., description="Deployment transaction hash")
    gas_used: int = Field(..., description="Gas used for deployment")
    deployed_at: datetime = Field(..., description="Deployment timestamp")
    status: str = Field(..., description="Deployment status")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "contract_address": "0x1234567890123456789012345678901234567890",
                "network": "ethereum",
                "transaction_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "gas_used": 1500000,
                "deployed_at": "2024-01-15T10:30:00Z",
                "status": "deployed",
            }
        }


class TradeRequest(BaseModel):
    """Request model for placing a trade on blockchain"""

    trade_id: str = Field(..., description="Trade ID")
    market_id: int = Field(..., description="Market ID")
    outcome: str = Field(..., description="Traded outcome")
    shares: int = Field(..., description="Number of shares")
    user_address: str = Field(..., description="User's blockchain address")
    network: str = Field(default="ethereum", description="Blockchain network")

    class Config:
        schema_extra = {
            "example": {
                "trade_id": "trade_123",
                "market_id": 1,
                "outcome": "Yes",
                "shares": 100,
                "user_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                "network": "ethereum",
            }
        }


class TradeResponse(BaseModel):
    """Response model for trade placement"""

    trade_id: str = Field(..., description="Trade ID")
    transaction_hash: str = Field(..., description="Transaction hash")
    block_number: int = Field(..., description="Block number")
    gas_used: int = Field(..., description="Gas used")
    gas_price: int = Field(..., description="Gas price")
    status: str = Field(..., description="Transaction status")
    network: str = Field(..., description="Blockchain network")
    timestamp: datetime = Field(..., description="Transaction timestamp")

    class Config:
        schema_extra = {
            "example": {
                "trade_id": "trade_123",
                "transaction_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "block_number": 12345678,
                "gas_used": 150000,
                "gas_price": 20000000000,
                "status": "confirmed",
                "network": "ethereum",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class TokenCreationRequest(BaseModel):
    """Request model for creating a market token"""

    market_id: int = Field(..., description="Market ID")
    token_name: str = Field(..., description="Token name")
    token_symbol: str = Field(..., description="Token symbol")
    total_supply: int = Field(..., description="Total token supply")
    network: str = Field(default="ethereum", description="Blockchain network")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "token_name": "Bitcoin Prediction Token",
                "token_symbol": "BTCPRED",
                "total_supply": 1000000,
                "network": "ethereum",
            }
        }


class TokenResponse(BaseModel):
    """Response model for token creation"""

    token_address: str = Field(..., description="Token contract address")
    market_id: int = Field(..., description="Market ID")
    token_name: str = Field(..., description="Token name")
    token_symbol: str = Field(..., description="Token symbol")
    total_supply: int = Field(..., description="Total token supply")
    decimals: int = Field(..., description="Token decimals")
    current_price: float = Field(..., description="Current token price")
    market_cap: float = Field(..., description="Market capitalization")
    holders_count: int = Field(..., description="Number of token holders")
    network: str = Field(..., description="Blockchain network")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "token_address": "0x1234567890123456789012345678901234567890",
                "market_id": 1,
                "token_name": "Bitcoin Prediction Token",
                "token_symbol": "BTCPRED",
                "total_supply": 1000000,
                "decimals": 18,
                "current_price": 0.5,
                "market_cap": 500000.0,
                "holders_count": 150,
                "network": "ethereum",
                "created_at": "2024-01-15T10:30:00Z",
            }
        }


class OracleDataRequest(BaseModel):
    """Request model for submitting oracle data"""

    market_id: int = Field(..., description="Market ID")
    data_source: str = Field(..., description="Data source (chainlink, custom, api)")
    data_type: str = Field(..., description="Data type (price, event, sport, politics)")
    data_value: str = Field(..., description="Data value")
    network: str = Field(default="ethereum", description="Blockchain network")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "data_source": "chainlink",
                "data_type": "price",
                "data_value": "95000",
                "network": "ethereum",
            }
        }


class OracleDataResponse(BaseModel):
    """Response model for oracle data submission"""

    oracle_address: str = Field(..., description="Oracle contract address")
    market_id: int = Field(..., description="Market ID")
    data_source: str = Field(..., description="Data source")
    data_type: str = Field(..., description="Data type")
    data_value: str = Field(..., description="Data value")
    confidence: float = Field(..., description="Data confidence score")
    transaction_hash: str = Field(..., description="Submission transaction hash")
    network: str = Field(..., description="Blockchain network")
    timestamp: datetime = Field(..., description="Submission timestamp")

    class Config:
        schema_extra = {
            "example": {
                "oracle_address": "0x1234567890123456789012345678901234567890",
                "market_id": 1,
                "data_source": "chainlink",
                "data_type": "price",
                "data_value": "95000",
                "confidence": 0.95,
                "transaction_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "network": "ethereum",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class TransactionStatusResponse(BaseModel):
    """Response model for transaction status"""

    transaction_hash: str = Field(..., description="Transaction hash")
    block_number: Optional[int] = Field(None, description="Block number")
    from_address: str = Field(..., description="From address")
    to_address: str = Field(..., description="To address")
    value: float = Field(..., description="Transaction value")
    gas_used: Optional[int] = Field(None, description="Gas used")
    gas_price: Optional[int] = Field(None, description="Gas price")
    status: str = Field(..., description="Transaction status")
    network: str = Field(..., description="Blockchain network")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    contract_interaction: bool = Field(
        ..., description="Whether it's a contract interaction"
    )
    method_name: Optional[str] = Field(None, description="Contract method name")

    class Config:
        schema_extra = {
            "example": {
                "transaction_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "block_number": 12345678,
                "from_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                "to_address": "0x1234567890123456789012345678901234567890",
                "value": 0.0,
                "gas_used": 150000,
                "gas_price": 20000000000,
                "status": "confirmed",
                "network": "ethereum",
                "timestamp": "2024-01-15T10:30:00Z",
                "contract_interaction": True,
                "method_name": "placeTrade",
            }
        }


class ContractInfoResponse(BaseModel):
    """Response model for contract information"""

    contract_address: str = Field(..., description="Contract address")
    contract_name: str = Field(..., description="Contract name")
    contract_type: str = Field(..., description="Contract type")
    network: str = Field(..., description="Blockchain network")
    owner_address: str = Field(..., description="Contract owner address")
    deployed_at: datetime = Field(..., description="Deployment timestamp")
    gas_used: int = Field(..., description="Gas used for deployment")
    transaction_hash: str = Field(..., description="Deployment transaction hash")

    class Config:
        schema_extra = {
            "example": {
                "contract_address": "0x1234567890123456789012345678901234567890",
                "contract_name": "Market_1",
                "contract_type": "market",
                "network": "ethereum",
                "owner_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                "deployed_at": "2024-01-15T10:30:00Z",
                "gas_used": 1500000,
                "transaction_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            }
        }


class NetworkStatusResponse(BaseModel):
    """Response model for network status"""

    network: str = Field(..., description="Network name")
    chain_id: int = Field(..., description="Chain ID")
    latest_block: int = Field(..., description="Latest block number")
    gas_price_gwei: float = Field(..., description="Current gas price in Gwei")
    is_connected: bool = Field(..., description="Connection status")
    rpc_url: str = Field(..., description="RPC URL")
    explorer_url: str = Field(..., description="Block explorer URL")
    status: str = Field(..., description="Overall status")

    class Config:
        schema_extra = {
            "example": {
                "network": "ethereum",
                "chain_id": 1,
                "latest_block": 12345678,
                "gas_price_gwei": 25.5,
                "is_connected": True,
                "rpc_url": "https://mainnet.infura.io/v3/YOUR_PROJECT_ID",
                "explorer_url": "https://etherscan.io",
                "status": "connected",
            }
        }


class BlockchainHealthResponse(BaseModel):
    """Response model for blockchain health check"""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    features: List[str] = Field(..., description="Available features")
    supported_networks: List[str] = Field(..., description="Supported networks")
    dependencies: List[str] = Field(..., description="Required dependencies")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "blockchain_integration",
                "timestamp": "2024-01-15T10:30:00Z",
                "features": [
                    "smart_contract_deployment",
                    "transaction_monitoring",
                    "token_management",
                    "oracle_integration",
                    "multi_network_support",
                    "gas_price_monitoring",
                    "websocket_updates",
                ],
                "supported_networks": ["ethereum", "polygon", "arbitrum"],
                "dependencies": ["web3", "eth_account", "redis"],
            }
        }


class WebSocketBlockchainMessage(BaseModel):
    """Base model for WebSocket blockchain messages"""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")


class WebSocketTransactionSubscription(BaseModel):
    """Request model for WebSocket transaction subscription"""

    type: str = Field("subscribe_transactions", description="Message type")
    network: str = Field(..., description="Blockchain network")


class WebSocketGasPriceSubscription(BaseModel):
    """Request model for WebSocket gas price subscription"""

    type: str = Field("subscribe_gas_prices", description="Message type")


class WebSocketContractEventsRequest(BaseModel):
    """Request model for WebSocket contract events"""

    type: str = Field("get_contract_events", description="Message type")
    contract_address: str = Field(..., description="Contract address")


class GasPriceResponse(BaseModel):
    """Response model for gas prices"""

    gas_price_wei: int = Field(..., description="Gas price in Wei")
    gas_price_gwei: float = Field(..., description="Gas price in Gwei")
    gas_price_eth: float = Field(..., description="Gas price in ETH")
    timestamp: datetime = Field(..., description="Timestamp")

    class Config:
        schema_extra = {
            "example": {
                "gas_price_wei": 25000000000,
                "gas_price_gwei": 25.0,
                "gas_price_eth": 0.000000025,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class RecentTransactionResponse(BaseModel):
    """Response model for recent transactions"""

    transaction_hash: str = Field(..., description="Transaction hash")
    block_number: int = Field(..., description="Block number")
    from_address: str = Field(..., description="From address")
    to_address: str = Field(..., description="To address")
    value: float = Field(..., description="Transaction value")
    gas_used: int = Field(..., description="Gas used")
    status: str = Field(..., description="Transaction status")
    timestamp: datetime = Field(..., description="Transaction timestamp")

    class Config:
        schema_extra = {
            "example": {
                "transaction_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                "block_number": 12345678,
                "from_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
                "to_address": "0x1234567890123456789012345678901234567890",
                "value": 0.1,
                "gas_used": 150000,
                "status": "confirmed",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class BlockchainAnalyticsResponse(BaseModel):
    """Response model for blockchain analytics"""

    network: str = Field(..., description="Blockchain network")
    timeframe: str = Field(..., description="Analytics timeframe")
    total_transactions: int = Field(..., description="Total transactions")
    total_volume: float = Field(..., description="Total volume")
    avg_gas_price: float = Field(..., description="Average gas price")
    active_contracts: int = Field(..., description="Active contracts")
    unique_addresses: int = Field(..., description="Unique addresses")
    timestamp: datetime = Field(..., description="Analytics timestamp")

    class Config:
        schema_extra = {
            "example": {
                "network": "ethereum",
                "timeframe": "24h",
                "total_transactions": 1500,
                "total_volume": 50000.0,
                "avg_gas_price": 25.5,
                "active_contracts": 50,
                "unique_addresses": 300,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class ContractDeploymentRequest(BaseModel):
    """Request model for contract deployment"""

    contract_type: str = Field(..., description="Type of contract to deploy")
    network: str = Field(..., description="Blockchain network")
    contract_data: Optional[Dict[str, Any]] = Field(
        None, description="Contract deployment data"
    )

    class Config:
        schema_extra = {
            "example": {
                "contract_type": "market",
                "network": "ethereum",
                "contract_data": {
                    "market_id": 1,
                    "title": "Sample Market",
                    "description": "Sample market description",
                },
            }
        }


class ContractDeploymentResponse(BaseModel):
    """Response model for contract deployment"""

    contract_address: str = Field(..., description="Deployed contract address")
    contract_type: str = Field(..., description="Contract type")
    network: str = Field(..., description="Blockchain network")
    deployed_at: datetime = Field(..., description="Deployment timestamp")
    status: str = Field(..., description="Deployment status")

    class Config:
        schema_extra = {
            "example": {
                "contract_address": "0x1234567890123456789012345678901234567890",
                "contract_type": "market",
                "network": "ethereum",
                "deployed_at": "2024-01-15T10:30:00Z",
                "status": "deployed",
            }
        }
