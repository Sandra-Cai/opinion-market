"""
Batch operation utilities for efficient bulk processing
"""

import logging
from typing import List, Dict, Any, Optional, TypeVar, Type, Callable
from sqlalchemy.orm import Session
from sqlalchemy import and_
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BatchOperationResult:
    """Result of a batch operation"""
    
    def __init__(self):
        self.successful: List[Any] = []
        self.failed: List[Dict[str, Any]] = []
        self.total: int = 0
    
    @property
    def success_count(self) -> int:
        return len(self.successful)
    
    @property
    def failure_count(self) -> int:
        return len(self.failed)
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.success_count / self.total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "successful": self.success_count,
            "failed": self.failure_count,
            "success_rate": self.success_rate,
            "errors": self.failed
        }


def batch_create(
    db: Session,
    model_class: Type[T],
    items: List[Dict[str, Any]],
    validate_func: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> BatchOperationResult:
    """
    Create multiple records in a batch operation.
    
    Args:
        db: Database session
        model_class: SQLAlchemy model class
        items: List of dictionaries with data for each record
        validate_func: Optional validation function
        
    Returns:
        BatchOperationResult with success/failure details
    """
    result = BatchOperationResult()
    result.total = len(items)
    
    for index, item_data in enumerate(items):
        try:
            # Validate if validation function provided
            if validate_func and not validate_func(item_data):
                result.failed.append({
                    "index": index,
                    "data": item_data,
                    "error": "Validation failed"
                })
                continue
            
            # Create instance
            instance = model_class(**item_data)
            db.add(instance)
            result.successful.append(instance)
            
        except Exception as e:
            logger.error(f"Error creating batch item {index}: {e}", exc_info=True)
            result.failed.append({
                "index": index,
                "data": item_data,
                "error": str(e)
            })
    
    # Commit all successful items
    try:
        db.commit()
        logger.info(f"Batch create: {result.success_count}/{result.total} successful")
    except Exception as e:
        logger.error(f"Error committing batch create: {e}", exc_info=True)
        db.rollback()
        # Mark all as failed
        result.failed.extend([
            {"index": i, "data": item, "error": "Commit failed"}
            for i, item in enumerate(items)
        ])
        result.successful = []
    
    return result


def batch_update(
    db: Session,
    model_class: Type[T],
    updates: List[Dict[str, Any]],
    filter_field: str = "id"
) -> BatchOperationResult:
    """
    Update multiple records in a batch operation.
    
    Args:
        db: Database session
        model_class: SQLAlchemy model class
        updates: List of dictionaries with updates (must include filter_field)
        filter_field: Field to use for filtering (default: "id")
        
    Returns:
        BatchOperationResult with success/failure details
    """
    result = BatchOperationResult()
    result.total = len(updates)
    
    for index, update_data in enumerate(updates):
        try:
            if filter_field not in update_data:
                result.failed.append({
                    "index": index,
                    "data": update_data,
                    "error": f"Missing {filter_field} field"
                })
                continue
            
            filter_value = update_data.pop(filter_field)
            update_dict = {k: v for k, v in update_data.items() if v is not None}
            
            if not update_dict:
                result.failed.append({
                    "index": index,
                    "data": update_data,
                    "error": "No fields to update"
                })
                continue
            
            # Update record
            updated = db.query(model_class).filter(
                getattr(model_class, filter_field) == filter_value
            ).update(update_dict)
            
            if updated > 0:
                result.successful.append({"id": filter_value, "updated": True})
            else:
                result.failed.append({
                    "index": index,
                    "data": update_data,
                    "error": "Record not found"
                })
            
        except Exception as e:
            logger.error(f"Error updating batch item {index}: {e}", exc_info=True)
            result.failed.append({
                "index": index,
                "data": update_data,
                "error": str(e)
            })
    
    # Commit all updates
    try:
        db.commit()
        logger.info(f"Batch update: {result.success_count}/{result.total} successful")
    except Exception as e:
        logger.error(f"Error committing batch update: {e}", exc_info=True)
        db.rollback()
        result.successful = []
    
    return result


def batch_delete(
    db: Session,
    model_class: Type[T],
    ids: List[Any],
    filter_field: str = "id",
    soft_delete: bool = False,
    soft_delete_field: str = "is_deleted"
) -> BatchOperationResult:
    """
    Delete multiple records in a batch operation.
    
    Args:
        db: Database session
        model_class: SQLAlchemy model class
        ids: List of IDs to delete
        filter_field: Field to use for filtering (default: "id")
        soft_delete: If True, perform soft delete instead of hard delete
        soft_delete_field: Field name for soft delete flag
        
    Returns:
        BatchOperationResult with success/failure details
    """
    result = BatchOperationResult()
    result.total = len(ids)
    
    for index, id_value in enumerate(ids):
        try:
            query = db.query(model_class).filter(
                getattr(model_class, filter_field) == id_value
            )
            
            if soft_delete and hasattr(model_class, soft_delete_field):
                # Soft delete
                updated = query.update({soft_delete_field: True})
                if updated > 0:
                    result.successful.append({"id": id_value, "deleted": True})
                else:
                    result.failed.append({
                        "index": index,
                        "id": id_value,
                        "error": "Record not found"
                    })
            else:
                # Hard delete
                deleted = query.delete()
                if deleted > 0:
                    result.successful.append({"id": id_value, "deleted": True})
                else:
                    result.failed.append({
                        "index": index,
                        "id": id_value,
                        "error": "Record not found"
                    })
            
        except Exception as e:
            logger.error(f"Error deleting batch item {index}: {e}", exc_info=True)
            result.failed.append({
                "index": index,
                "id": id_value,
                "error": str(e)
            })
    
    # Commit all deletions
    try:
        db.commit()
        logger.info(f"Batch delete: {result.success_count}/{result.total} successful")
    except Exception as e:
        logger.error(f"Error committing batch delete: {e}", exc_info=True)
        db.rollback()
        result.successful = []
    
    return result


def batch_get(
    db: Session,
    model_class: Type[T],
    ids: List[Any],
    filter_field: str = "id"
) -> List[T]:
    """
    Get multiple records by IDs in a single query.
    
    Args:
        db: Database session
        model_class: SQLAlchemy model class
        ids: List of IDs to retrieve
        filter_field: Field to use for filtering (default: "id")
        
    Returns:
        List of model instances
    """
    if not ids:
        return []
    
    filter_attr = getattr(model_class, filter_field)
    return db.query(model_class).filter(filter_attr.in_(ids)).all()

