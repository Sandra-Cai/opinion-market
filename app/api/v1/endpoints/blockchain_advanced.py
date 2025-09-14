"""
Advanced blockchain integration endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
import logging

from app.core.blockchain_integration import (
    blockchain_manager, 
    BlockchainNetwork, 
    TransactionStatus
)
from app.api.v1.endpoints.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/blockchain/wallet/create")
async def create_blockchain_wallet(
    network: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new blockchain wallet"""
    try:
        # Validate network
        try:
            blockchain_network = BlockchainNetwork(network.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported network: {network}"
            )
        
        # Create wallet
        wallet_info = await blockchain_manager.create_wallet(
            blockchain_network, 
            current_user["user_id"]
        )
        
        return {
            "message": "Wallet created successfully",
            "wallet": wallet_info,
            "user_id": current_user["user_id"]
        }
    
    except Exception as e:
        logger.error(f"Failed to create wallet: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create wallet: {str(e)}"
        )


@router.get("/blockchain/wallet/{address}/balance")
async def get_wallet_balance(
    address: str,
    token: str = "ETH",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get wallet balance"""
    try:
        balance = await blockchain_manager.get_wallet_balance(address, token)
        
        return {
            "address": address,
            "token": token,
            "balance": balance,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get balance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get balance: {str(e)}"
        )


@router.post("/blockchain/transaction/send")
async def send_blockchain_transaction(
    from_address: str,
    to_address: str,
    amount: float,
    token: str = "ETH",
    network: str = "ethereum",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Send a blockchain transaction"""
    try:
        # Validate network
        try:
            blockchain_network = BlockchainNetwork(network.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported network: {network}"
            )
        
        # Validate amount
        if amount <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Amount must be greater than 0"
            )
        
        # Send transaction
        transaction = await blockchain_manager.send_transaction(
            from_address,
            to_address,
            amount,
            token,
            blockchain_network
        )
        
        return {
            "message": "Transaction sent successfully",
            "transaction": {
                "hash": transaction.hash,
                "from": transaction.from_address,
                "to": transaction.to_address,
                "amount": transaction.amount,
                "token": transaction.token,
                "network": transaction.network.value,
                "status": transaction.status.value,
                "timestamp": transaction.timestamp.isoformat() if transaction.timestamp else None
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to send transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send transaction: {str(e)}"
        )


@router.get("/blockchain/transaction/{tx_hash}")
async def get_transaction_status(
    tx_hash: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get transaction status"""
    try:
        transaction = await blockchain_manager.get_transaction_status(tx_hash)
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction {tx_hash} not found"
            )
        
        return {
            "transaction": {
                "hash": transaction.hash,
                "from": transaction.from_address,
                "to": transaction.to_address,
                "amount": transaction.amount,
                "token": transaction.token,
                "network": transaction.network.value,
                "status": transaction.status.value,
                "block_number": transaction.block_number,
                "gas_used": transaction.gas_used,
                "gas_price": transaction.gas_price,
                "timestamp": transaction.timestamp.isoformat() if transaction.timestamp else None
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get transaction status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transaction status: {str(e)}"
        )


@router.get("/blockchain/wallet/{address}/transactions")
async def get_transaction_history(
    address: str,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get transaction history for a wallet"""
    try:
        if limit > 100:
            limit = 100  # Cap at 100 transactions
        
        transactions = await blockchain_manager.get_transaction_history(address, limit)
        
        return {
            "address": address,
            "transactions": [
                {
                    "hash": tx.hash,
                    "from": tx.from_address,
                    "to": tx.to_address,
                    "amount": tx.amount,
                    "token": tx.token,
                    "network": tx.network.value,
                    "status": tx.status.value,
                    "block_number": tx.block_number,
                    "timestamp": tx.timestamp.isoformat() if tx.timestamp else None
                }
                for tx in transactions
            ],
            "total_transactions": len(transactions),
            "limit": limit
        }
    
    except Exception as e:
        logger.error(f"Failed to get transaction history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transaction history: {str(e)}"
        )


@router.post("/blockchain/contract/deploy")
async def deploy_smart_contract(
    contract_name: str,
    abi: Dict[str, Any],
    network: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Deploy a smart contract"""
    try:
        # Validate network
        try:
            blockchain_network = BlockchainNetwork(network.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported network: {network}"
            )
        
        # Validate ABI
        if not abi or not isinstance(abi, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Valid ABI is required"
            )
        
        # Deploy contract
        contract = await blockchain_manager.deploy_smart_contract(
            contract_name,
            abi,
            blockchain_network,
            current_user["user_id"]
        )
        
        return {
            "message": "Smart contract deployed successfully",
            "contract": {
                "address": contract.address,
                "name": contract.name,
                "network": contract.network.value,
                "version": contract.version,
                "deployed_at": contract.deployed_at.isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to deploy smart contract: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy smart contract: {str(e)}"
        )


@router.post("/blockchain/contract/{contract_address}/call")
async def call_contract_function(
    contract_address: str,
    function_name: str,
    parameters: List[Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Call a smart contract function"""
    try:
        if not function_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Function name is required"
            )
        
        # Call contract function
        result = await blockchain_manager.call_contract_function(
            contract_address,
            function_name,
            parameters,
            current_user["user_id"]
        )
        
        return {
            "contract_address": contract_address,
            "function_name": function_name,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to call contract function: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to call contract function: {str(e)}"
        )


@router.get("/blockchain/networks")
async def get_supported_networks(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get supported blockchain networks"""
    try:
        networks = blockchain_manager.get_supported_networks()
        
        return {
            "networks": networks,
            "total_networks": len(networks)
        }
    
    except Exception as e:
        logger.error(f"Failed to get supported networks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get supported networks: {str(e)}"
        )


@router.get("/blockchain/network/{network}/info")
async def get_network_info(
    network: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get blockchain network information"""
    try:
        # Validate network
        try:
            blockchain_network = BlockchainNetwork(network.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported network: {network}"
            )
        
        # Get network info
        network_info = await blockchain_manager.get_network_info(blockchain_network)
        
        return {
            "network_info": network_info
        }
    
    except Exception as e:
        logger.error(f"Failed to get network info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get network info: {str(e)}"
        )


@router.get("/blockchain/contracts")
async def get_deployed_contracts(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get list of deployed smart contracts"""
    try:
        contracts = blockchain_manager.get_contracts()
        
        return {
            "contracts": contracts,
            "total_contracts": len(contracts)
        }
    
    except Exception as e:
        logger.error(f"Failed to get deployed contracts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get deployed contracts: {str(e)}"
        )


@router.post("/blockchain/gas/estimate")
async def estimate_gas_cost(
    from_address: str,
    to_address: str,
    amount: float,
    network: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Estimate gas cost for a transaction"""
    try:
        # Validate network
        try:
            blockchain_network = BlockchainNetwork(network.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported network: {network}"
            )
        
        # Validate amount
        if amount <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Amount must be greater than 0"
            )
        
        # Estimate gas
        gas_estimate = await blockchain_manager.estimate_gas(
            from_address,
            to_address,
            amount,
            blockchain_network
        )
        
        return {
            "gas_estimate": gas_estimate,
            "from_address": from_address,
            "to_address": to_address,
            "amount": amount,
            "network": network
        }
    
    except Exception as e:
        logger.error(f"Failed to estimate gas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to estimate gas: {str(e)}"
        )


@router.get("/blockchain/analytics/overview")
async def get_blockchain_analytics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get blockchain analytics overview"""
    try:
        # Get analytics data
        networks = blockchain_manager.get_supported_networks()
        contracts = blockchain_manager.get_contracts()
        
        # Calculate statistics
        total_contracts = len(contracts)
        active_networks = len(networks)
        
        # Mock transaction statistics
        total_transactions = len(blockchain_manager.transactions)
        pending_transactions = len([
            tx for tx in blockchain_manager.transactions.values()
            if tx.status == TransactionStatus.PENDING
        ])
        confirmed_transactions = len([
            tx for tx in blockchain_manager.transactions.values()
            if tx.status == TransactionStatus.CONFIRMED
        ])
        
        return {
            "analytics": {
                "total_networks": active_networks,
                "total_contracts": total_contracts,
                "total_transactions": total_transactions,
                "pending_transactions": pending_transactions,
                "confirmed_transactions": confirmed_transactions,
                "success_rate": (confirmed_transactions / total_transactions * 100) if total_transactions > 0 else 0
            },
            "networks": networks,
            "contracts": contracts,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get blockchain analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get blockchain analytics: {str(e)}"
        )
