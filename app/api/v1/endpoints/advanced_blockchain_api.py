"""
Advanced Blockchain API Endpoints
REST API for the Advanced Blockchain Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.advanced_blockchain_engine import (
    advanced_blockchain_engine,
    BlockchainType,
    TransactionType,
    DeFiProtocol,
    TransactionStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class TransactionResponse(BaseModel):
    tx_id: str
    blockchain: str
    from_address: str
    to_address: str
    amount: float
    token_address: str
    token_symbol: str
    transaction_type: str
    protocol: Optional[str]
    gas_used: Optional[int]
    gas_price: Optional[float]
    block_number: Optional[int]
    status: str
    timestamp: str
    metadata: Dict[str, Any]

class SmartContractResponse(BaseModel):
    contract_id: str
    blockchain: str
    address: str
    name: str
    protocol: str
    functions: List[str]
    events: List[str]
    is_verified: bool
    created_at: str
    metadata: Dict[str, Any]

class DeFiPositionResponse(BaseModel):
    position_id: str
    user_address: str
    protocol: str
    blockchain: str
    position_type: str
    token_address: str
    token_symbol: str
    amount: float
    value_usd: float
    apy: Optional[float]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

class TokenInfoResponse(BaseModel):
    address: str
    symbol: str
    name: str
    decimals: int
    blockchain: str
    total_supply: Optional[float]
    price_usd: Optional[float]
    market_cap: Optional[float]
    volume_24h: Optional[float]
    metadata: Dict[str, Any]

@router.get("/transactions", response_model=List[TransactionResponse])
async def get_transactions(
    blockchain: Optional[str] = None,
    transaction_type: Optional[str] = None,
    limit: int = 100
):
    """Get blockchain transactions"""
    try:
        # Validate blockchain if provided
        blockchain_type = None
        if blockchain:
            try:
                blockchain_type = BlockchainType(blockchain.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid blockchain: {blockchain}")
        
        # Validate transaction type if provided
        tx_type = None
        if transaction_type:
            try:
                tx_type = TransactionType(transaction_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid transaction type: {transaction_type}")
        
        transactions = await advanced_blockchain_engine.get_transactions(
            blockchain=blockchain_type,
            transaction_type=tx_type,
            limit=limit
        )
        return transactions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/smart-contracts", response_model=List[SmartContractResponse])
async def get_smart_contracts():
    """Get smart contracts"""
    try:
        contracts = await advanced_blockchain_engine.get_smart_contracts()
        return contracts
        
    except Exception as e:
        logger.error(f"Error getting smart contracts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/defi-positions", response_model=List[DeFiPositionResponse])
async def get_defi_positions(
    user_address: Optional[str] = None,
    protocol: Optional[str] = None,
    limit: int = 100
):
    """Get DeFi positions"""
    try:
        # Validate protocol if provided
        defi_protocol = None
        if protocol:
            try:
                defi_protocol = DeFiProtocol(protocol.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid protocol: {protocol}")
        
        positions = await advanced_blockchain_engine.get_defi_positions(
            user_address=user_address,
            protocol=defi_protocol,
            limit=limit
        )
        return positions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting DeFi positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tokens", response_model=List[TokenInfoResponse])
async def get_token_info(token_address: Optional[str] = None):
    """Get token information"""
    try:
        tokens = await advanced_blockchain_engine.get_token_info(token_address=token_address)
        return tokens
        
    except Exception as e:
        logger.error(f"Error getting token info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/smart-contracts", response_model=Dict[str, str])
async def add_smart_contract(
    blockchain: str,
    address: str,
    name: str,
    protocol: str
):
    """Add a new smart contract"""
    try:
        # Validate blockchain
        try:
            blockchain_type = BlockchainType(blockchain.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid blockchain: {blockchain}")
        
        # Validate protocol
        try:
            defi_protocol = DeFiProtocol(protocol.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid protocol: {protocol}")
        
        contract_id = await advanced_blockchain_engine.add_smart_contract(
            blockchain=blockchain_type,
            address=address,
            name=name,
            protocol=defi_protocol
        )
        
        return {
            "contract_id": contract_id,
            "status": "added",
            "message": f"Smart contract '{name}' added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding smart contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tokens", response_model=Dict[str, str])
async def add_token(
    address: str,
    symbol: str,
    name: str,
    decimals: int,
    blockchain: str
):
    """Add a new token"""
    try:
        # Validate blockchain
        try:
            blockchain_type = BlockchainType(blockchain.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid blockchain: {blockchain}")
        
        token_address = await advanced_blockchain_engine.add_token(
            address=address,
            symbol=symbol,
            name=name,
            decimals=decimals,
            blockchain=blockchain_type
        )
        
        return {
            "token_address": token_address,
            "status": "added",
            "message": f"Token '{symbol}' added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await advanced_blockchain_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-blockchains")
async def get_supported_blockchains():
    """Get supported blockchains"""
    try:
        return {
            "blockchains": [
                {
                    "name": blockchain.value,
                    "display_name": blockchain.value.replace("_", " ").title(),
                    "enabled": True
                }
                for blockchain in BlockchainType
            ],
            "transaction_types": [
                {
                    "name": tx_type.value,
                    "display_name": tx_type.value.replace("_", " ").title()
                }
                for tx_type in TransactionType
            ],
            "defi_protocols": [
                {
                    "name": protocol.value,
                    "display_name": protocol.value.replace("_", " ").title()
                }
                for protocol in DeFiProtocol
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting supported blockchains: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_blockchain_health():
    """Get blockchain engine health status"""
    try:
        return {
            "engine_id": advanced_blockchain_engine.engine_id,
            "is_running": advanced_blockchain_engine.is_running,
            "total_transactions": len(advanced_blockchain_engine.transactions),
            "total_contracts": len(advanced_blockchain_engine.smart_contracts),
            "total_positions": len(advanced_blockchain_engine.defi_positions),
            "total_tokens": len(advanced_blockchain_engine.token_info),
            "supported_blockchains": [bc.value for bc in BlockchainType],
            "supported_protocols": [protocol.value for protocol in DeFiProtocol],
            "uptime": "active" if advanced_blockchain_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting blockchain health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
