"""
Database query optimization and helper utilities
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
from sqlalchemy.orm import Session, Query
from sqlalchemy import desc, asc, func
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


def paginate_query(
    query: Query,
    page: int = 1,
    page_size: int = 20,
    max_page_size: int = 100
) -> tuple[Query, int]:
    """
    Apply pagination to a SQLAlchemy query.
    
    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        page_size: Number of items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Tuple of (paginated_query, total_count)
    """
    # Validate and clamp page_size
    page_size = min(page_size, max_page_size)
    page_size = max(1, page_size)
    
    # Validate and clamp page
    page = max(1, page)
    
    # Get total count (before pagination)
    total = query.count()
    
    # Calculate skip
    skip = (page - 1) * page_size
    
    # Apply pagination
    paginated_query = query.offset(skip).limit(page_size)
    
    return paginated_query, total


def order_by_field(
    query: Query,
    order_by: Optional[str] = None,
    default_order: str = "created_at",
    order_direction: str = "desc"
) -> Query:
    """
    Apply ordering to a query based on field name.
    
    Args:
        query: SQLAlchemy query object
        order_by: Field name to order by
        default_order: Default field to order by if order_by is None
        order_direction: "asc" or "desc"
        
    Returns:
        Query with ordering applied
    """
    order_field = order_by or default_order
    
    # Get the model class from query
    model_class = query.column_descriptions[0]['entity'] if query.column_descriptions else None
    
    if model_class and hasattr(model_class, order_field):
        field = getattr(model_class, order_field)
        if order_direction.lower() == "desc":
            return query.order_by(desc(field))
        else:
            return query.order_by(asc(field))
    
    # Fallback to default ordering
    if hasattr(model_class, default_order):
        field = getattr(model_class, default_order)
        return query.order_by(desc(field))
    
    return query


def filter_by_date_range(
    query: Query,
    date_field: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Query:
    """
    Apply date range filtering to a query.
    
    Args:
        query: SQLAlchemy query object
        date_field: Name of the date field to filter on
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        
    Returns:
        Query with date filtering applied
    """
    # Get the model class
    model_class = query.column_descriptions[0]['entity'] if query.column_descriptions else None
    
    if not model_class or not hasattr(model_class, date_field):
        logger.warning(f"Date field {date_field} not found in model")
        return query
    
    field = getattr(model_class, date_field)
    
    if start_date:
        query = query.filter(field >= start_date)
    
    if end_date:
        # Make end_date inclusive by adding one day
        end_date_inclusive = end_date + timedelta(days=1)
        query = query.filter(field < end_date_inclusive)
    
    return query


def get_or_404(
    query: Query,
    error_message: str = "Resource not found"
) -> Any:
    """
    Get first result from query or raise 404 error.
    
    Args:
        query: SQLAlchemy query object
        error_message: Error message if not found
        
    Returns:
        First result from query
        
    Raises:
        HTTPException: If no result found
    """
    from fastapi import HTTPException, status
    
    result = query.first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_message
        )
    return result


def bulk_update(
    db: Session,
    model_class: Type[T],
    updates: List[Dict[str, Any]],
    filter_field: str = "id"
) -> int:
    """
    Perform bulk update operations.
    
    Args:
        db: Database session
        model_class: SQLAlchemy model class
        updates: List of dictionaries with updates (must include filter_field)
        filter_field: Field to use for filtering (default: "id")
        
    Returns:
        Number of records updated
    """
    updated_count = 0
    
    for update_data in updates:
        if filter_field not in update_data:
            logger.warning(f"Missing {filter_field} in update data, skipping")
            continue
        
        filter_value = update_data.pop(filter_field)
        update_dict = {k: v for k, v in update_data.items() if v is not None}
        
        if not update_dict:
            continue
        
        db.query(model_class).filter(
            getattr(model_class, filter_field) == filter_value
        ).update(update_dict)
        updated_count += 1
    
    db.commit()
    return updated_count


def get_aggregate_stats(
    query: Query,
    group_by_field: Optional[str] = None,
    aggregate_fields: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get aggregate statistics from a query.
    
    Args:
        query: SQLAlchemy query object
        group_by_field: Optional field to group by
        aggregate_fields: Dictionary of field_name -> aggregate_function
        
    Returns:
        List of dictionaries with aggregate results
    """
    if not aggregate_fields:
        aggregate_fields = {"count": func.count()}
    
    # Get model class
    model_class = query.column_descriptions[0]['entity'] if query.column_descriptions else None
    
    if not model_class:
        return []
    
    # Build aggregate query
    select_fields = []
    
    if group_by_field and hasattr(model_class, group_by_field):
        group_field = getattr(model_class, group_by_field)
        select_fields.append(group_field)
    
    for field_name, agg_func in aggregate_fields.items():
        if isinstance(agg_func, str):
            # String like "sum", "count", "avg"
            if hasattr(func, agg_func):
                select_fields.append(getattr(func, agg_func)(model_class.id).label(field_name))
        else:
            # Already a function
            select_fields.append(agg_func.label(field_name))
    
    if not select_fields:
        return []
    
    # Note: This function needs a session parameter
    # For now, return empty list as placeholder
    # In practice, you'd pass db: Session and use db.query(*select_fields)
    logger.warning("get_aggregate_stats needs database session parameter")
    return []


def optimize_query(query: Query, use_eager_loading: bool = True) -> Query:
    """
    Optimize a query with common performance improvements.
    
    Args:
        query: SQLAlchemy query object
        use_eager_loading: Whether to use eager loading for relationships
        
    Returns:
        Optimized query
    """
    # Note: This is a placeholder for query optimization
    # In practice, you'd add:
    # - Eager loading with joinedload/selectinload
    # - Query result caching hints
    # - Index hints
    # - etc.
    
    return query

