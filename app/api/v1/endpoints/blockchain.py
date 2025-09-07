"""
Blockchain Integration API Endpoints
Provides smart contract management, transaction monitoring, and decentralized features
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.redis_client import get_redis_sync
from app.services.blockchain_integration import (
    get_blockchain_integration_service,
    SmartContract,
    BlockchainTransaction,
    MarketToken,
    OracleData,
)
from app.schemas.blockchain import (
    MarketCreationRequest,
    MarketCreationResponse,
    TradeRequest,
    TradeResponse,
    TokenCreationRequest,
    TokenResponse,
    OracleDataRequest,
    OracleDataResponse,
    TransactionStatusResponse,
    ContractInfoResponse,
    NetworkStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time blockchain updates
websocket_connections: Dict[str, WebSocket] = {}


@router.post("/markets/create", response_model=MarketCreationResponse)
async def create_market_on_blockchain(
    request: MarketCreationRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a new market on the blockchain
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Prepare market data
        market_data = {
            "id": request.market_id,
            "title": request.title,
            "description": request.description,
            "outcome_a": request.outcome_a,
            "outcome_b": request.outcome_b,
            "owner_address": request.owner_address,
            "end_date": request.end_date.isoformat() if request.end_date else None,
        }

        # Create market on blockchain
        contract = await blockchain_service.create_market_on_blockchain(
            market_data, request.network
        )

        return MarketCreationResponse(
            market_id=request.market_id,
            contract_address=contract.contract_address,
            network=contract.network,
            transaction_hash=contract.transaction_hash,
            gas_used=contract.gas_used,
            deployed_at=contract.deployed_at,
            status="deployed",
        )

    except Exception as e:
        logger.error(f"Error creating market on blockchain: {e}")
        raise HTTPException(
            status_code=500, detail="Error creating market on blockchain"
        )


@router.post("/trades/place", response_model=TradeResponse)
async def place_trade_on_blockchain(
    request: TradeRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Place a trade on the blockchain
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Place trade on blockchain
        transaction = await blockchain_service.place_trade_on_blockchain(
            request.market_id,
            request.outcome,
            request.shares,
            request.user_address,
            request.network,
        )

        return TradeResponse(
            trade_id=request.trade_id,
            transaction_hash=transaction.transaction_hash,
            block_number=transaction.block_number,
            gas_used=transaction.gas_used,
            gas_price=transaction.gas_price,
            status=transaction.status,
            network=transaction.network,
            timestamp=transaction.timestamp,
        )

    except Exception as e:
        logger.error(f"Error placing trade on blockchain: {e}")
        raise HTTPException(status_code=500, detail="Error placing trade on blockchain")


@router.post("/tokens/create", response_model=TokenResponse)
async def create_market_token(
    request: TokenCreationRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a token for a market
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Create token on blockchain
        token = await blockchain_service.create_market_token(
            request.market_id,
            request.token_name,
            request.token_symbol,
            request.total_supply,
            request.network,
        )

        return TokenResponse(
            token_address=token.token_address,
            market_id=token.market_id,
            token_name=token.token_name,
            token_symbol=token.token_symbol,
            total_supply=token.total_supply,
            decimals=token.decimals,
            current_price=token.current_price,
            market_cap=token.market_cap,
            holders_count=token.holders_count,
            network=token.network,
            created_at=token.created_at,
        )

    except Exception as e:
        logger.error(f"Error creating market token: {e}")
        raise HTTPException(status_code=500, detail="Error creating market token")


@router.post("/oracle/submit", response_model=OracleDataResponse)
async def submit_oracle_data(
    request: OracleDataRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Submit oracle data for market resolution
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Submit oracle data
        oracle_data = await blockchain_service.submit_oracle_data(
            request.market_id,
            request.data_source,
            request.data_type,
            request.data_value,
            request.network,
        )

        return OracleDataResponse(
            oracle_address=oracle_data.oracle_address,
            market_id=oracle_data.market_id,
            data_source=oracle_data.data_source,
            data_type=oracle_data.data_type,
            data_value=oracle_data.data_value,
            confidence=oracle_data.confidence,
            transaction_hash=oracle_data.transaction_hash,
            network=oracle_data.network,
            timestamp=oracle_data.timestamp,
        )

    except Exception as e:
        logger.error(f"Error submitting oracle data: {e}")
        raise HTTPException(status_code=500, detail="Error submitting oracle data")


@router.get(
    "/transactions/{transaction_hash}", response_model=TransactionStatusResponse
)
async def get_transaction_status(
    transaction_hash: str,
    network: str = Query(..., description="Blockchain network"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get transaction status and details
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Get transaction from cache or blockchain
        transaction_data = await blockchain_service._get_transaction_data(
            transaction_hash, network
        )

        if not transaction_data:
            raise HTTPException(status_code=404, detail="Transaction not found")

        return TransactionStatusResponse(
            transaction_hash=transaction_hash,
            block_number=transaction_data.get("block_number"),
            from_address=transaction_data.get("from_address"),
            to_address=transaction_data.get("to_address"),
            value=transaction_data.get("value", 0.0),
            gas_used=transaction_data.get("gas_used"),
            gas_price=transaction_data.get("gas_price"),
            status=transaction_data.get("status", "unknown"),
            network=network,
            timestamp=datetime.fromisoformat(transaction_data.get("timestamp")),
            contract_interaction=transaction_data.get("contract_interaction", False),
            method_name=transaction_data.get("method_name"),
        )

    except Exception as e:
        logger.error(f"Error getting transaction status: {e}")
        raise HTTPException(status_code=500, detail="Error getting transaction status")


@router.get("/contracts/{contract_address}", response_model=ContractInfoResponse)
async def get_contract_info(
    contract_address: str,
    network: str = Query(..., description="Blockchain network"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get smart contract information
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Get contract from cache or blockchain
        contract_data = await blockchain_service._get_contract_data(
            contract_address, network
        )

        if not contract_data:
            raise HTTPException(status_code=404, detail="Contract not found")

        return ContractInfoResponse(
            contract_address=contract_address,
            contract_name=contract_data.get("contract_name"),
            contract_type=contract_data.get("contract_type"),
            network=network,
            owner_address=contract_data.get("owner_address"),
            deployed_at=datetime.fromisoformat(contract_data.get("deployed_at")),
            gas_used=contract_data.get("gas_used", 0),
            transaction_hash=contract_data.get("transaction_hash", ""),
        )

    except Exception as e:
        logger.error(f"Error getting contract info: {e}")
        raise HTTPException(status_code=500, detail="Error getting contract info")


@router.get("/networks/status", response_model=List[NetworkStatusResponse])
async def get_network_status(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get status of all supported blockchain networks
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        network_statuses = []

        for network_name, web3 in blockchain_service.web3_connections.items():
            try:
                # Get network status
                latest_block = web3.eth.block_number
                gas_price = web3.eth.gas_price
                chain_id = web3.eth.chain_id

                # Get network config
                network_config = blockchain_service.networks.get(network_name, {})

                network_statuses.append(
                    NetworkStatusResponse(
                        network=network_name,
                        chain_id=chain_id,
                        latest_block=latest_block,
                        gas_price_gwei=web3.from_wei(gas_price, "gwei"),
                        is_connected=web3.is_connected(),
                        rpc_url=network_config.get("rpc_url", ""),
                        explorer_url=network_config.get("explorer", ""),
                        status="connected" if web3.is_connected() else "disconnected",
                    )
                )

            except Exception as e:
                logger.error(f"Error getting status for network {network_name}: {e}")
                network_statuses.append(
                    NetworkStatusResponse(
                        network=network_name,
                        chain_id=0,
                        latest_block=0,
                        gas_price_gwei=0.0,
                        is_connected=False,
                        rpc_url="",
                        explorer_url="",
                        status="error",
                    )
                )

        return network_statuses

    except Exception as e:
        logger.error(f"Error getting network status: {e}")
        raise HTTPException(status_code=500, detail="Error getting network status")


@router.get("/tokens/{token_address}")
async def get_token_info(
    token_address: str,
    network: str = Query(..., description="Blockchain network"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get token information
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Get token from cache or blockchain
        token_data = await blockchain_service._get_token_data(token_address, network)

        if not token_data:
            raise HTTPException(status_code=404, detail="Token not found")

        return JSONResponse(
            content={
                "token_address": token_address,
                "market_id": token_data.get("market_id"),
                "token_name": token_data.get("token_name"),
                "token_symbol": token_data.get("token_symbol"),
                "total_supply": token_data.get("total_supply"),
                "decimals": token_data.get("decimals"),
                "current_price": token_data.get("current_price"),
                "market_cap": token_data.get("market_cap"),
                "holders_count": token_data.get("holders_count"),
                "network": network,
                "created_at": token_data.get("created_at"),
            }
        )

    except Exception as e:
        logger.error(f"Error getting token info: {e}")
        raise HTTPException(status_code=500, detail="Error getting token info")


@router.get("/oracle/{market_id}")
async def get_oracle_data(
    market_id: int,
    network: str = Query(..., description="Blockchain network"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get oracle data for a market
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Get oracle data from cache or blockchain
        oracle_data = await blockchain_service._get_oracle_data(market_id, network)

        if not oracle_data:
            raise HTTPException(status_code=404, detail="Oracle data not found")

        return JSONResponse(
            content={
                "oracle_address": oracle_data.get("oracle_address"),
                "market_id": market_id,
                "data_source": oracle_data.get("data_source"),
                "data_type": oracle_data.get("data_type"),
                "data_value": oracle_data.get("data_value"),
                "confidence": oracle_data.get("confidence"),
                "transaction_hash": oracle_data.get("transaction_hash"),
                "network": network,
                "timestamp": oracle_data.get("timestamp"),
            }
        )

    except Exception as e:
        logger.error(f"Error getting oracle data: {e}")
        raise HTTPException(status_code=500, detail="Error getting oracle data")


@router.get("/gas-prices")
async def get_gas_prices(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get current gas prices for all networks
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        gas_prices = {}

        for network_name, web3 in blockchain_service.web3_connections.items():
            try:
                gas_price = web3.eth.gas_price
                gas_prices[network_name] = {
                    "gas_price_wei": gas_price,
                    "gas_price_gwei": web3.from_wei(gas_price, "gwei"),
                    "gas_price_eth": web3.from_wei(gas_price, "ether"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                logger.error(f"Error getting gas price for {network_name}: {e}")
                gas_prices[network_name] = {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }

        return JSONResponse(content=gas_prices)

    except Exception as e:
        logger.error(f"Error getting gas prices: {e}")
        raise HTTPException(status_code=500, detail="Error getting gas prices")


@router.websocket("/ws/blockchain/{client_id}")
async def websocket_blockchain_updates(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time blockchain updates
    """
    await websocket.accept()
    websocket_connections[client_id] = websocket

    try:
        # Send initial connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "type": "blockchain_connected",
                    "client_id": client_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "available_features": [
                        "transaction_monitoring",
                        "gas_price_updates",
                        "contract_events",
                        "oracle_data_updates",
                    ],
                }
            )
        )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "subscribe_transactions":
                    network = message.get("network", "ethereum")
                    # Subscribe to transaction updates
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "transaction_subscription_confirmed",
                                "network": network,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "subscribe_gas_prices":
                    # Subscribe to gas price updates
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "gas_price_subscription_confirmed",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "get_contract_events":
                    contract_address = message.get("contract_address")
                    # Get contract events
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "contract_events",
                                "contract_address": contract_address,
                                "events": [],  # Would contain actual events
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info(f"Blockchain client {client_id} disconnected")
    finally:
        if client_id in websocket_connections:
            del websocket_connections[client_id]


@router.get("/health/blockchain")
async def blockchain_health():
    """
    Health check for blockchain integration services
    """
    return {
        "status": "healthy",
        "service": "blockchain_integration",
        "timestamp": datetime.utcnow().isoformat(),
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


@router.post("/contracts/deploy")
async def deploy_contract(
    contract_type: str = Query(..., description="Type of contract to deploy"),
    network: str = Query(..., description="Blockchain network"),
    contract_data: Dict[str, Any] = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Deploy a smart contract
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Deploy contract based on type
        if contract_type == "market":
            contract = await blockchain_service.create_market_on_blockchain(
                contract_data or {}, network
            )
        elif contract_type == "token":
            token = await blockchain_service.create_market_token(
                contract_data.get("market_id", 0),
                contract_data.get("token_name", ""),
                contract_data.get("token_symbol", ""),
                contract_data.get("total_supply", 0),
                network,
            )
            contract = SmartContract(
                contract_address=token.token_address,
                contract_name=token.token_name,
                contract_type="token",
                abi=blockchain_service.contract_abis["token"],
                bytecode="",
                deployed_at=token.created_at,
                network=network,
                owner_address="",
                gas_used=0,
                transaction_hash="",
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported contract type: {contract_type}"
            )

        return JSONResponse(
            content={
                "contract_address": contract.contract_address,
                "contract_type": contract.contract_type,
                "network": contract.network,
                "deployed_at": contract.deployed_at.isoformat(),
                "status": "deployed",
            }
        )

    except Exception as e:
        logger.error(f"Error deploying contract: {e}")
        raise HTTPException(status_code=500, detail="Error deploying contract")


@router.get("/transactions/recent")
async def get_recent_transactions(
    network: str = Query(..., description="Blockchain network"),
    limit: int = Query(10, description="Number of transactions to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get recent transactions for a network
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Get recent transactions (this would be implemented in the service)
        recent_transactions = await blockchain_service._get_recent_transactions(
            network, limit
        )

        return JSONResponse(
            content={
                "network": network,
                "transactions": recent_transactions,
                "count": len(recent_transactions),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting recent transactions: {e}")
        raise HTTPException(status_code=500, detail="Error getting recent transactions")


@router.get("/analytics/blockchain")
async def get_blockchain_analytics(
    network: str = Query(..., description="Blockchain network"),
    timeframe: str = Query("24h", description="Timeframe for analytics"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get blockchain analytics
    """
    try:
        redis_client = await get_redis_sync()
        blockchain_service = await get_blockchain_integration_service(redis_client, db)

        # Get analytics data (this would be implemented in the service)
        analytics = await blockchain_service._get_blockchain_analytics(
            network, timeframe
        )

        return JSONResponse(
            content={
                "network": network,
                "timeframe": timeframe,
                "analytics": analytics,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error getting blockchain analytics: {e}")
        raise HTTPException(
            status_code=500, detail="Error getting blockchain analytics"
        )
