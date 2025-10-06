"""
Smart Contract API Endpoints
REST API for the Smart Contract Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.smart_contract_engine import (
    smart_contract_engine,
    ContractType,
    ContractStatus,
    ContractFunction
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class SmartContractResponse(BaseModel):
    contract_id: str
    name: str
    contract_type: str
    blockchain: str
    address: str
    status: str
    deployed_at: str
    gas_used: int
    gas_price: float
    deployer_address: str
    is_verified: bool
    verification_tx: Optional[str]
    functions: List[str]
    events: List[str]
    metadata: Dict[str, Any]

class ContractTemplateResponse(BaseModel):
    template_id: str
    name: str
    contract_type: str
    description: str
    gas_estimate: int
    constructor_params: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    created_at: str
    metadata: Dict[str, Any]

class ContractEventResponse(BaseModel):
    event_id: str
    contract_id: str
    event_name: str
    block_number: int
    transaction_hash: str
    log_index: int
    topics: List[str]
    data: str
    decoded_data: Dict[str, Any]
    timestamp: str
    metadata: Dict[str, Any]

class DeployContractRequest(BaseModel):
    template_id: str = Field(..., description="Template ID")
    name: str = Field(..., description="Contract name")
    blockchain: str = Field(..., description="Blockchain name")
    constructor_args: List[Any] = Field(..., description="Constructor arguments")
    deployer_address: str = Field(..., description="Deployer address")

class ContractInteractionRequest(BaseModel):
    contract_id: str = Field(..., description="Contract ID")
    function_name: str = Field(..., description="Function name")
    caller_address: str = Field(..., description="Caller address")
    parameters: Dict[str, Any] = Field(..., description="Function parameters")
    gas_limit: int = Field(..., description="Gas limit")
    gas_price: float = Field(..., description="Gas price")
    value: float = Field(0.0, description="ETH value")

@router.get("/contracts", response_model=List[SmartContractResponse])
async def get_contracts(
    contract_type: Optional[str] = None,
    blockchain: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get smart contracts"""
    try:
        # Validate contract type if provided
        contract_type_enum = None
        if contract_type:
            try:
                contract_type_enum = ContractType(contract_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid contract type: {contract_type}")
        
        # Validate status if provided
        status_enum = None
        if status:
            try:
                status_enum = ContractStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        contracts = await smart_contract_engine.get_contracts(
            contract_type=contract_type_enum,
            blockchain=blockchain,
            status=status_enum,
            limit=limit
        )
        return contracts
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contracts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates", response_model=List[ContractTemplateResponse])
async def get_contract_templates():
    """Get contract templates"""
    try:
        templates = await smart_contract_engine.get_contract_templates()
        return templates
        
    except Exception as e:
        logger.error(f"Error getting contract templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events", response_model=List[ContractEventResponse])
async def get_contract_events(
    contract_id: Optional[str] = None,
    event_name: Optional[str] = None,
    limit: int = 100
):
    """Get contract events"""
    try:
        events = await smart_contract_engine.get_contract_events(
            contract_id=contract_id,
            event_name=event_name,
            limit=limit
        )
        return events
        
    except Exception as e:
        logger.error(f"Error getting contract events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deploy", response_model=Dict[str, str])
async def deploy_contract(deploy_request: DeployContractRequest):
    """Deploy a new contract"""
    try:
        contract_id = await smart_contract_engine.deploy_contract(
            template_id=deploy_request.template_id,
            name=deploy_request.name,
            blockchain=deploy_request.blockchain,
            constructor_args=deploy_request.constructor_args,
            deployer_address=deploy_request.deployer_address
        )
        
        return {
            "contract_id": contract_id,
            "status": "deploying",
            "message": f"Contract '{deploy_request.name}' deployment initiated"
        }
        
    except Exception as e:
        logger.error(f"Error deploying contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interact", response_model=Dict[str, str])
async def interact_with_contract(interaction_request: ContractInteractionRequest):
    """Interact with a contract"""
    try:
        from decimal import Decimal
        
        interaction_id = await smart_contract_engine.interact_with_contract(
            contract_id=interaction_request.contract_id,
            function_name=interaction_request.function_name,
            caller_address=interaction_request.caller_address,
            parameters=interaction_request.parameters,
            gas_limit=interaction_request.gas_limit,
            gas_price=Decimal(str(interaction_request.gas_price)),
            value=Decimal(str(interaction_request.value))
        )
        
        return {
            "interaction_id": interaction_id,
            "status": "submitted",
            "message": f"Contract interaction '{interaction_request.function_name}' submitted"
        }
        
    except Exception as e:
        logger.error(f"Error interacting with contract: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await smart_contract_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-contract-types")
async def get_available_contract_types():
    """Get available contract types"""
    try:
        return {
            "contract_types": [
                {
                    "name": contract_type.value,
                    "display_name": contract_type.value.replace("_", " ").title(),
                    "description": f"{contract_type.value.replace('_', ' ').title()} contract type"
                }
                for contract_type in ContractType
            ],
            "contract_statuses": [
                {
                    "name": status.value,
                    "display_name": status.value.replace("_", " ").title(),
                    "description": f"{status.value.replace('_', ' ').title()} status"
                }
                for status in ContractStatus
            ],
            "contract_functions": [
                {
                    "name": function.value,
                    "display_name": function.value.replace("_", " ").title(),
                    "description": f"{function.value.replace('_', ' ').title()} function"
                }
                for function in ContractFunction
            ],
            "supported_blockchains": list(smart_contract_engine.blockchain_configs.keys())
        }
        
    except Exception as e:
        logger.error(f"Error getting available contract types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_smart_contract_health():
    """Get smart contract engine health status"""
    try:
        return {
            "engine_id": smart_contract_engine.engine_id,
            "is_running": smart_contract_engine.is_running,
            "total_contracts": len(smart_contract_engine.contracts),
            "total_templates": len(smart_contract_engine.templates),
            "total_interactions": len(smart_contract_engine.interactions),
            "total_events": len(smart_contract_engine.events),
            "verified_contracts": len([c for c in smart_contract_engine.contracts.values() if c.is_verified]),
            "supported_contract_types": [ct.value for ct in ContractType],
            "supported_blockchains": list(smart_contract_engine.blockchain_configs.keys()),
            "uptime": "active" if smart_contract_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting smart contract health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
