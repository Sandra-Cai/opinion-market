"""
API endpoints for Quantum Security Engine
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from app.services.quantum_security_engine import quantum_security_engine, QuantumAlgorithm, SecurityLevel, KeyType

logger = logging.getLogger(__name__)

router = APIRouter()

class QuantumKeyRequest(BaseModel):
    algorithm: str
    security_level: int
    key_type: str

class QuantumSignatureRequest(BaseModel):
    data: str
    key_id: str

class QuantumEncryptionRequest(BaseModel):
    data: str
    key_id: str

class QuantumKeyResponse(BaseModel):
    key_id: str
    key_type: str
    algorithm: str
    security_level: int
    public_key: str
    created_at: str
    expires_at: Optional[str]
    metadata: Dict[str, Any]

class QuantumSignatureResponse(BaseModel):
    signature_id: str
    algorithm: str
    key_id: str
    verified: bool
    created_at: str

class QuantumEncryptionResponse(BaseModel):
    encryption_id: str
    algorithm: str
    key_id: str
    created_at: str

class QuantumPerformanceResponse(BaseModel):
    performance_metrics: Dict[str, Any]
    total_keys: int
    total_signatures: int
    total_encryptions: int
    active_algorithms: int
    security_levels: int

@router.post("/generate-key")
async def generate_quantum_key(request: QuantumKeyRequest):
    """Generate quantum-resistant key"""
    try:
        algorithm = QuantumAlgorithm(request.algorithm)
        security_level = SecurityLevel(request.security_level)
        key_type = KeyType(request.key_type)
        
        key_id = await quantum_security_engine.generate_quantum_key(
            algorithm,
            security_level,
            key_type
        )
        
        if key_id:
            return {"key_id": key_id, "status": "generated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate key")
            
    except Exception as e:
        logger.error(f"Error generating quantum key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/key/{key_id}")
async def get_quantum_key(key_id: str):
    """Get quantum key details"""
    try:
        key = await quantum_security_engine.get_quantum_key(key_id)
        
        if key:
            return QuantumKeyResponse(**key)
        else:
            raise HTTPException(status_code=404, detail="Key not found")
            
    except Exception as e:
        logger.error(f"Error getting quantum key: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keys")
async def get_all_quantum_keys():
    """Get all quantum keys"""
    try:
        keys = []
        for key_id in quantum_security_engine.quantum_keys:
            key = await quantum_security_engine.get_quantum_key(key_id)
            if key:
                keys.append(key)
        
        return {"keys": keys, "total": len(keys)}
        
    except Exception as e:
        logger.error(f"Error getting all quantum keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keys/algorithm/{algorithm}")
async def get_quantum_keys_by_algorithm(algorithm: str):
    """Get quantum keys by algorithm"""
    try:
        algo = QuantumAlgorithm(algorithm)
        keys = await quantum_security_engine.get_quantum_keys_by_algorithm(algo)
        
        return {"keys": keys, "total": len(keys)}
        
    except Exception as e:
        logger.error(f"Error getting keys by algorithm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sign")
async def create_quantum_signature(request: QuantumSignatureRequest):
    """Create quantum signature"""
    try:
        signature_id = await quantum_security_engine.create_quantum_signature(
            request.data,
            request.key_id
        )
        
        if signature_id:
            return {"signature_id": signature_id, "status": "created"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create signature")
            
    except Exception as e:
        logger.error(f"Error creating quantum signature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify/{signature_id}")
async def verify_quantum_signature(signature_id: str, data: str):
    """Verify quantum signature"""
    try:
        verified = await quantum_security_engine.verify_quantum_signature(signature_id, data)
        
        return {"signature_id": signature_id, "verified": verified}
        
    except Exception as e:
        logger.error(f"Error verifying quantum signature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signatures")
async def get_quantum_signatures():
    """Get all quantum signatures"""
    try:
        signatures = await quantum_security_engine.get_quantum_signatures()
        
        return {"signatures": signatures, "total": len(signatures)}
        
    except Exception as e:
        logger.error(f"Error getting quantum signatures: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signature/{signature_id}")
async def get_quantum_signature(signature_id: str):
    """Get quantum signature details"""
    try:
        signatures = await quantum_security_engine.get_quantum_signatures()
        
        for sig in signatures:
            if sig["signature_id"] == signature_id:
                return QuantumSignatureResponse(**sig)
        
        raise HTTPException(status_code=404, detail="Signature not found")
        
    except Exception as e:
        logger.error(f"Error getting quantum signature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/encrypt")
async def encrypt_quantum(request: QuantumEncryptionRequest):
    """Encrypt data using quantum-resistant encryption"""
    try:
        encryption_id = await quantum_security_engine.encrypt_quantum(
            request.data,
            request.key_id
        )
        
        if encryption_id:
            return {"encryption_id": encryption_id, "status": "encrypted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to encrypt data")
            
    except Exception as e:
        logger.error(f"Error encrypting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decrypt/{encryption_id}")
async def decrypt_quantum(encryption_id: str):
    """Decrypt data using quantum-resistant decryption"""
    try:
        decrypted_data = await quantum_security_engine.decrypt_quantum(encryption_id)
        
        if decrypted_data is not None:
            return {"encryption_id": encryption_id, "decrypted_data": decrypted_data}
        else:
            raise HTTPException(status_code=404, detail="Encryption not found or decryption failed")
            
    except Exception as e:
        logger.error(f"Error decrypting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/encryptions")
async def get_quantum_encryptions():
    """Get all quantum encryptions"""
    try:
        encryptions = await quantum_security_engine.get_quantum_encryptions()
        
        return {"encryptions": encryptions, "total": len(encryptions)}
        
    except Exception as e:
        logger.error(f"Error getting quantum encryptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/encryption/{encryption_id}")
async def get_quantum_encryption(encryption_id: str):
    """Get quantum encryption details"""
    try:
        encryptions = await quantum_security_engine.get_quantum_encryptions()
        
        for enc in encryptions:
            if enc["encryption_id"] == encryption_id:
                return QuantumEncryptionResponse(**enc)
        
        raise HTTPException(status_code=404, detail="Encryption not found")
        
    except Exception as e:
        logger.error(f"Error getting quantum encryption: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_quantum_performance():
    """Get quantum security performance metrics"""
    try:
        metrics = await quantum_security_engine.get_quantum_performance_metrics()
        return QuantumPerformanceResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/algorithms")
async def get_quantum_algorithms():
    """Get available quantum algorithms"""
    try:
        algorithms = [
            {"name": algo.value, "description": f"Quantum-resistant {algo.value} algorithm"}
            for algo in QuantumAlgorithm
        ]
        
        return {"algorithms": algorithms, "total": len(algorithms)}
        
    except Exception as e:
        logger.error(f"Error getting quantum algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/security-levels")
async def get_security_levels():
    """Get available security levels"""
    try:
        levels = [
            {"level": level.value, "description": f"{level.value * 64}-bit security level"}
            for level in SecurityLevel
        ]
        
        return {"security_levels": levels, "total": len(levels)}
        
    except Exception as e:
        logger.error(f"Error getting security levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/key-types")
async def get_key_types():
    """Get available key types"""
    try:
        types = [
            {"type": key_type.value, "description": f"Key for {key_type.value}"}
            for key_type in KeyType
        ]
        
        return {"key_types": types, "total": len(types)}
        
    except Exception as e:
        logger.error(f"Error getting key types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_quantum_health():
    """Get quantum security engine health"""
    try:
        metrics = await quantum_security_engine.get_quantum_performance_metrics()
        
        health_status = "healthy"
        if metrics["total_keys"] == 0:
            health_status = "unhealthy"
        elif metrics["total_keys"] < 5:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "total_keys": metrics["total_keys"],
            "active_algorithms": metrics["active_algorithms"],
            "security_levels": metrics["security_levels"],
            "keys_generated": metrics["performance_metrics"]["keys_generated"],
            "signatures_created": metrics["performance_metrics"]["signatures_created"],
            "encryptions_performed": metrics["performance_metrics"]["encryptions_performed"]
        }
        
    except Exception as e:
        logger.error(f"Error getting quantum health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_quantum_stats():
    """Get comprehensive quantum security statistics"""
    try:
        metrics = await quantum_security_engine.get_quantum_performance_metrics()
        keys = await quantum_security_engine.get_all_quantum_keys()
        signatures = await quantum_security_engine.get_quantum_signatures()
        encryptions = await quantum_security_engine.get_quantum_encryptions()
        
        # Calculate additional statistics
        algorithm_stats = {}
        security_level_stats = {}
        key_type_stats = {}
        
        for key in keys:
            algo = key["algorithm"]
            level = key["security_level"]
            key_type = key["key_type"]
            
            algorithm_stats[algo] = algorithm_stats.get(algo, 0) + 1
            security_level_stats[level] = security_level_stats.get(level, 0) + 1
            key_type_stats[key_type] = key_type_stats.get(key_type, 0) + 1
        
        return {
            "performance_metrics": metrics["performance_metrics"],
            "key_statistics": {
                "total_keys": metrics["total_keys"],
                "algorithm_distribution": algorithm_stats,
                "security_level_distribution": security_level_stats,
                "key_type_distribution": key_type_stats
            },
            "signature_statistics": {
                "total_signatures": metrics["total_signatures"],
                "verified_signatures": len([s for s in signatures if s["verified"]])
            },
            "encryption_statistics": {
                "total_encryptions": metrics["total_encryptions"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting quantum stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
