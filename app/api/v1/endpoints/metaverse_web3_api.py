"""
API endpoints for Metaverse Web3 Engine
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from app.services.metaverse_web3_engine import metaverse_web3_engine, VirtualWorld, AvatarType, BlockchainNetwork, NFTStandard

logger = logging.getLogger(__name__)

router = APIRouter()

class VirtualAssetRequest(BaseModel):
    asset_type: str
    name: str
    description: str
    virtual_world: str
    coordinates: Dict[str, float]
    owner: str
    metadata: Optional[Dict[str, Any]] = None

class NFTMintRequest(BaseModel):
    token_id: str
    contract_address: str
    blockchain: str
    standard: str
    name: str
    description: str
    image_url: str
    owner: str
    metadata: Optional[Dict[str, Any]] = None

class VirtualAvatarRequest(BaseModel):
    user_id: str
    name: str
    avatar_type: str
    appearance: Dict[str, Any]
    virtual_world: str
    location: Dict[str, float]

class VirtualEventRequest(BaseModel):
    name: str
    description: str
    virtual_world: str
    location: Dict[str, float]
    start_time: str
    end_time: str
    organizer: str
    max_attendees: int = 100
    ticket_price: float = 0.0

class AssetTransferRequest(BaseModel):
    asset_id: str
    from_owner: str
    to_owner: str

class EventJoinRequest(BaseModel):
    event_id: str
    user_id: str

class VirtualAssetResponse(BaseModel):
    asset_id: str
    asset_type: str
    name: str
    description: str
    virtual_world: str
    coordinates: Dict[str, float]
    owner: str
    metadata: Dict[str, Any]
    created_at: str
    last_updated: str

class NFTResponse(BaseModel):
    nft_id: str
    token_id: str
    contract_address: str
    blockchain: str
    standard: str
    name: str
    description: str
    image_url: str
    owner: str
    metadata: Dict[str, Any]
    created_at: str

class VirtualAvatarResponse(BaseModel):
    avatar_id: str
    user_id: str
    name: str
    avatar_type: str
    appearance: Dict[str, Any]
    virtual_world: str
    location: Dict[str, float]
    nft_items: List[str]
    created_at: str
    last_active: str

class VirtualEventResponse(BaseModel):
    event_id: str
    name: str
    description: str
    virtual_world: str
    location: Dict[str, float]
    start_time: str
    end_time: str
    organizer: str
    attendees: List[str]
    max_attendees: int
    ticket_price: float
    nft_ticket: Optional[str]

class MetaversePerformanceResponse(BaseModel):
    performance_metrics: Dict[str, Any]
    total_assets: int
    total_nfts: int
    total_avatars: int
    total_events: int
    active_avatars: int
    active_events: int

@router.post("/assets")
async def create_virtual_asset(request: VirtualAssetRequest):
    """Create virtual asset"""
    try:
        virtual_world = VirtualWorld(request.virtual_world)
        
        asset_id = await metaverse_web3_engine.create_virtual_asset(
            request.asset_type,
            request.name,
            request.description,
            virtual_world,
            request.coordinates,
            request.owner,
            request.metadata
        )
        
        if asset_id:
            return {"asset_id": asset_id, "status": "created"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create virtual asset")
            
    except Exception as e:
        logger.error(f"Error creating virtual asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assets")
async def get_virtual_assets():
    """Get all virtual assets"""
    try:
        assets = await metaverse_web3_engine.get_virtual_assets()
        return {"assets": assets, "total": len(assets)}
        
    except Exception as e:
        logger.error(f"Error getting virtual assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assets/{asset_id}")
async def get_virtual_asset(asset_id: str):
    """Get virtual asset by ID"""
    try:
        assets = await metaverse_web3_engine.get_virtual_assets()
        
        for asset in assets:
            if asset["asset_id"] == asset_id:
                return VirtualAssetResponse(**asset)
        
        raise HTTPException(status_code=404, detail="Asset not found")
        
    except Exception as e:
        logger.error(f"Error getting virtual asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assets/transfer")
async def transfer_virtual_asset(request: AssetTransferRequest):
    """Transfer virtual asset"""
    try:
        success = await metaverse_web3_engine.transfer_asset(
            request.asset_id,
            request.from_owner,
            request.to_owner
        )
        
        if success:
            return {"status": "transferred", "asset_id": request.asset_id}
        else:
            raise HTTPException(status_code=400, detail="Transfer failed")
            
    except Exception as e:
        logger.error(f"Error transferring virtual asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nfts/mint")
async def mint_nft(request: NFTMintRequest):
    """Mint NFT"""
    try:
        blockchain = BlockchainNetwork(request.blockchain)
        standard = NFTStandard(request.standard)
        
        nft_id = await metaverse_web3_engine.mint_nft(
            request.token_id,
            request.contract_address,
            blockchain,
            standard,
            request.name,
            request.description,
            request.image_url,
            request.owner,
            request.metadata
        )
        
        if nft_id:
            return {"nft_id": nft_id, "status": "minted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to mint NFT")
            
    except Exception as e:
        logger.error(f"Error minting NFT: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nfts")
async def get_nfts():
    """Get all NFTs"""
    try:
        nfts = await metaverse_web3_engine.get_nfts()
        return {"nfts": nfts, "total": len(nfts)}
        
    except Exception as e:
        logger.error(f"Error getting NFTs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nfts/{nft_id}")
async def get_nft(nft_id: str):
    """Get NFT by ID"""
    try:
        nfts = await metaverse_web3_engine.get_nfts()
        
        for nft in nfts:
            if nft["nft_id"] == nft_id:
                return NFTResponse(**nft)
        
        raise HTTPException(status_code=404, detail="NFT not found")
        
    except Exception as e:
        logger.error(f"Error getting NFT: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/avatars")
async def create_virtual_avatar(request: VirtualAvatarRequest):
    """Create virtual avatar"""
    try:
        avatar_type = AvatarType(request.avatar_type)
        virtual_world = VirtualWorld(request.virtual_world)
        
        avatar_id = await metaverse_web3_engine.create_virtual_avatar(
            request.user_id,
            request.name,
            avatar_type,
            request.appearance,
            virtual_world,
            request.location
        )
        
        if avatar_id:
            return {"avatar_id": avatar_id, "status": "created"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create virtual avatar")
            
    except Exception as e:
        logger.error(f"Error creating virtual avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/avatars")
async def get_virtual_avatars():
    """Get all virtual avatars"""
    try:
        avatars = await metaverse_web3_engine.get_virtual_avatars()
        return {"avatars": avatars, "total": len(avatars)}
        
    except Exception as e:
        logger.error(f"Error getting virtual avatars: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/avatars/{avatar_id}")
async def get_virtual_avatar(avatar_id: str):
    """Get virtual avatar by ID"""
    try:
        avatars = await metaverse_web3_engine.get_virtual_avatars()
        
        for avatar in avatars:
            if avatar["avatar_id"] == avatar_id:
                return VirtualAvatarResponse(**avatar)
        
        raise HTTPException(status_code=404, detail="Avatar not found")
        
    except Exception as e:
        logger.error(f"Error getting virtual avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events")
async def create_virtual_event(request: VirtualEventRequest):
    """Create virtual event"""
    try:
        virtual_world = VirtualWorld(request.virtual_world)
        start_time = datetime.fromisoformat(request.start_time)
        end_time = datetime.fromisoformat(request.end_time)
        
        event_id = await metaverse_web3_engine.create_virtual_event(
            request.name,
            request.description,
            virtual_world,
            request.location,
            start_time,
            end_time,
            request.organizer,
            request.max_attendees,
            request.ticket_price
        )
        
        if event_id:
            return {"event_id": event_id, "status": "created"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create virtual event")
            
    except Exception as e:
        logger.error(f"Error creating virtual event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events")
async def get_virtual_events():
    """Get all virtual events"""
    try:
        events = await metaverse_web3_engine.get_virtual_events()
        return {"events": events, "total": len(events)}
        
    except Exception as e:
        logger.error(f"Error getting virtual events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/{event_id}")
async def get_virtual_event(event_id: str):
    """Get virtual event by ID"""
    try:
        events = await metaverse_web3_engine.get_virtual_events()
        
        for event in events:
            if event["event_id"] == event_id:
                return VirtualEventResponse(**event)
        
        raise HTTPException(status_code=404, detail="Event not found")
        
    except Exception as e:
        logger.error(f"Error getting virtual event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events/join")
async def join_virtual_event(request: EventJoinRequest):
    """Join virtual event"""
    try:
        success = await metaverse_web3_engine.join_virtual_event(
            request.event_id,
            request.user_id
        )
        
        if success:
            return {"status": "joined", "event_id": request.event_id, "user_id": request.user_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to join event")
            
    except Exception as e:
        logger.error(f"Error joining virtual event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_metaverse_performance():
    """Get metaverse performance metrics"""
    try:
        metrics = await metaverse_web3_engine.get_metaverse_performance_metrics()
        return MetaversePerformanceResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/virtual-worlds")
async def get_virtual_worlds():
    """Get available virtual worlds"""
    try:
        worlds = [
            {"name": world.value, "description": f"Virtual world: {world.value}"}
            for world in VirtualWorld
        ]
        
        return {"virtual_worlds": worlds, "total": len(worlds)}
        
    except Exception as e:
        logger.error(f"Error getting virtual worlds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/blockchain-networks")
async def get_blockchain_networks():
    """Get available blockchain networks"""
    try:
        networks = [
            {"name": network.value, "description": f"Blockchain network: {network.value}"}
            for network in BlockchainNetwork
        ]
        
        return {"blockchain_networks": networks, "total": len(networks)}
        
    except Exception as e:
        logger.error(f"Error getting blockchain networks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nft-standards")
async def get_nft_standards():
    """Get available NFT standards"""
    try:
        standards = [
            {"name": standard.value, "description": f"NFT standard: {standard.value}"}
            for standard in NFTStandard
        ]
        
        return {"nft_standards": standards, "total": len(standards)}
        
    except Exception as e:
        logger.error(f"Error getting NFT standards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/avatar-types")
async def get_avatar_types():
    """Get available avatar types"""
    try:
        types = [
            {"name": avatar_type.value, "description": f"Avatar type: {avatar_type.value}"}
            for avatar_type in AvatarType
        ]
        
        return {"avatar_types": types, "total": len(types)}
        
    except Exception as e:
        logger.error(f"Error getting avatar types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_metaverse_health():
    """Get metaverse Web3 engine health"""
    try:
        metrics = await metaverse_web3_engine.get_metaverse_performance_metrics()
        
        health_status = "healthy"
        if metrics["total_avatars"] == 0:
            health_status = "unhealthy"
        elif metrics["active_avatars"] < metrics["total_avatars"] * 0.3:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "total_assets": metrics["total_assets"],
            "total_nfts": metrics["total_nfts"],
            "total_avatars": metrics["total_avatars"],
            "active_avatars": metrics["active_avatars"],
            "total_events": metrics["total_events"],
            "active_events": metrics["active_events"],
            "virtual_world_activity": metrics["performance_metrics"]["virtual_world_activity"]
        }
        
    except Exception as e:
        logger.error(f"Error getting metaverse health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_metaverse_stats():
    """Get comprehensive metaverse statistics"""
    try:
        metrics = await metaverse_web3_engine.get_metaverse_performance_metrics()
        assets = await metaverse_web3_engine.get_virtual_assets()
        nfts = await metaverse_web3_engine.get_nfts()
        avatars = await metaverse_web3_engine.get_virtual_avatars()
        events = await metaverse_web3_engine.get_virtual_events()
        
        # Calculate additional statistics
        world_stats = {}
        blockchain_stats = {}
        avatar_type_stats = {}
        
        for asset in assets:
            world = asset["virtual_world"]
            world_stats[world] = world_stats.get(world, 0) + 1
        
        for nft in nfts:
            blockchain = nft["blockchain"]
            blockchain_stats[blockchain] = blockchain_stats.get(blockchain, 0) + 1
        
        for avatar in avatars:
            avatar_type = avatar["avatar_type"]
            avatar_type_stats[avatar_type] = avatar_type_stats.get(avatar_type, 0) + 1
        
        return {
            "performance_metrics": metrics["performance_metrics"],
            "asset_statistics": {
                "total_assets": metrics["total_assets"],
                "world_distribution": world_stats
            },
            "nft_statistics": {
                "total_nfts": metrics["total_nfts"],
                "blockchain_distribution": blockchain_stats
            },
            "avatar_statistics": {
                "total_avatars": metrics["total_avatars"],
                "active_avatars": metrics["active_avatars"],
                "type_distribution": avatar_type_stats
            },
            "event_statistics": {
                "total_events": metrics["total_events"],
                "active_events": metrics["active_events"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metaverse stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
