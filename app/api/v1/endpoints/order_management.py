"""
Order Management System API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from app.services.order_management_system import (
    OrderManagementSystem,
    Order,
    OrderFill,
    ExecutionReport,
    OrderBook,
    MarketData,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    OrderRoute,
    get_order_management_system,
)
from app.services.execution_management_system import (
    ExecutionManagementSystem,
    ExecutionOrder,
    ExecutionSlice,
    ExecutionMetrics,
    ExecutionAlgorithm,
    ExecutionStrategy,
)
from app.schemas.order_management import (
    OrderCreate,
    OrderResponse,
    OrderModify,
    OrderCancel,
    ExecutionCreate,
    ExecutionResponse,
    ExecutionMetricsResponse,
    OrderBookResponse,
    MarketDataResponse,
    ExecutionReportResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time updates
websocket_connections: Dict[str, Any] = {}


@router.post(
    "/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED
)
async def create_order(
    order_data: OrderCreate,
    oms: OrderManagementSystem = Depends(get_order_management_system),
):
    """Create a new order"""
    try:
        order = await oms.create_order(
            user_id=order_data.user_id,
            account_id=order_data.account_id,
            symbol=order_data.symbol,
            order_type=OrderType(order_data.order_type),
            side=OrderSide(order_data.side),
            quantity=order_data.quantity,
            price=order_data.price,
            stop_price=order_data.stop_price,
            time_in_force=TimeInForce(order_data.time_in_force),
            client_order_id=order_data.client_order_id,
            algo_type=order_data.algo_type,
            algo_parameters=order_data.algo_parameters,
        )

        return OrderResponse(
            order_id=order.order_id,
            user_id=order.user_id,
            account_id=order.account_id,
            symbol=order.symbol,
            order_type=order.order_type.value,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force.value,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            average_price=order.average_price,
            created_at=order.created_at,
            last_updated=order.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create order: {str(e)}",
        )


@router.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    user_id: int,
    oms: OrderManagementSystem = Depends(get_order_management_system),
):
    """Get order details"""
    try:
        order = await oms.get_order(order_id, user_id)

        if not order:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
            )

        return OrderResponse(
            order_id=order.order_id,
            user_id=order.user_id,
            account_id=order.account_id,
            symbol=order.symbol,
            order_type=order.order_type.value,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force.value,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            average_price=order.average_price,
            created_at=order.created_at,
            last_updated=order.last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get order: {str(e)}",
        )


@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    user_id: int,
    account_id: Optional[str] = None,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    oms: OrderManagementSystem = Depends(get_order_management_system),
):
    """Get orders for a user"""
    try:
        order_status = OrderStatus(status) if status else None

        orders = await oms.get_orders(
            user_id=user_id,
            account_id=account_id,
            symbol=symbol,
            status=order_status,
            limit=limit,
        )

        return [
            OrderResponse(
                order_id=order.order_id,
                user_id=order.user_id,
                account_id=order.account_id,
                symbol=order.symbol,
                order_type=order.order_type.value,
                side=order.side.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force.value,
                status=order.status.value,
                filled_quantity=order.filled_quantity,
                remaining_quantity=order.remaining_quantity,
                average_price=order.average_price,
                created_at=order.created_at,
                last_updated=order.last_updated,
            )
            for order in orders
        ]

    except Exception as e:
        logger.error(f"Error getting orders: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get orders: {str(e)}",
        )


@router.put("/orders/{order_id}/modify", response_model=OrderResponse)
async def modify_order(
    order_id: str,
    modify_data: OrderModify,
    oms: OrderManagementSystem = Depends(get_order_management_system),
):
    """Modify an existing order"""
    try:
        success = await oms.modify_order(
            order_id=order_id,
            user_id=modify_data.user_id,
            new_quantity=modify_data.new_quantity,
            new_price=modify_data.new_price,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to modify order"
            )

        # Get updated order
        order = await oms.get_order(order_id, modify_data.user_id)

        return OrderResponse(
            order_id=order.order_id,
            user_id=order.user_id,
            account_id=order.account_id,
            symbol=order.symbol,
            order_type=order.order_type.value,
            side=order.side.value,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force.value,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            average_price=order.average_price,
            created_at=order.created_at,
            last_updated=order.last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error modifying order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to modify order: {str(e)}",
        )


@router.delete("/orders/{order_id}", response_model=Dict[str, str])
async def cancel_order(
    order_id: str,
    cancel_data: OrderCancel,
    oms: OrderManagementSystem = Depends(get_order_management_system),
):
    """Cancel an order"""
    try:
        success = await oms.cancel_order(order_id, cancel_data.user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to cancel order"
            )

        return {"message": "Order cancelled successfully", "order_id": order_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel order: {str(e)}",
        )


@router.get("/orders/{order_id}/fills", response_model=List[Dict[str, Any]])
async def get_order_fills(
    order_id: str,
    user_id: int,
    oms: OrderManagementSystem = Depends(get_order_management_system),
):
    """Get fills for an order"""
    try:
        fills = await oms.get_order_fills(order_id, user_id)

        return [
            {
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "fill_value": fill.fill_value,
                "commission": fill.commission,
                "venue": fill.venue,
                "fill_time": fill.fill_time,
                "created_at": fill.created_at,
            }
            for fill in fills
        ]

    except Exception as e:
        logger.error(f"Error getting order fills: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get order fills: {str(e)}",
        )


@router.get(
    "/orders/{order_id}/execution-reports", response_model=List[ExecutionReportResponse]
)
async def get_execution_reports(
    order_id: str,
    user_id: int,
    oms: OrderManagementSystem = Depends(get_order_management_system),
):
    """Get execution reports for an order"""
    try:
        reports = await oms.get_execution_reports(order_id, user_id)

        return [
            ExecutionReportResponse(
                report_id=report.report_id,
                order_id=report.order_id,
                execution_type=report.execution_type,
                order_status=report.order_status.value,
                filled_quantity=report.filled_quantity,
                remaining_quantity=report.remaining_quantity,
                average_price=report.average_price,
                last_fill_price=report.last_fill_price,
                last_fill_quantity=report.last_fill_quantity,
                commission=report.commission,
                venue=report.venue,
                execution_time=report.execution_time,
                text=report.text,
                created_at=report.created_at,
            )
            for report in reports
        ]

    except Exception as e:
        logger.error(f"Error getting execution reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution reports: {str(e)}",
        )


@router.post(
    "/executions", response_model=ExecutionResponse, status_code=status.HTTP_201_CREATED
)
async def create_execution(
    execution_data: ExecutionCreate,
    ems: ExecutionManagementSystem = Depends(get_execution_management_system),
):
    """Create a new execution order"""
    try:
        execution = await ems.create_execution(
            parent_order_id=execution_data.parent_order_id,
            symbol=execution_data.symbol,
            side=execution_data.side,
            quantity=execution_data.quantity,
            algorithm=ExecutionAlgorithm(execution_data.algorithm),
            strategy=ExecutionStrategy(execution_data.strategy),
            parameters=execution_data.parameters,
        )

        return ExecutionResponse(
            execution_id=execution.execution_id,
            parent_order_id=execution.parent_order_id,
            symbol=execution.symbol,
            side=execution.side,
            quantity=execution.quantity,
            algorithm=execution.algorithm.value,
            strategy=execution.strategy.value,
            parameters=execution.parameters,
            status=execution.status,
            filled_quantity=execution.filled_quantity,
            remaining_quantity=execution.remaining_quantity,
            average_price=execution.average_price,
            venues=execution.venues,
            created_at=execution.created_at,
            last_updated=execution.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create execution: {str(e)}",
        )


@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(
    execution_id: str,
    ems: ExecutionManagementSystem = Depends(get_execution_management_system),
):
    """Get execution details"""
    try:
        execution = await ems.get_execution(execution_id)

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found"
            )

        return ExecutionResponse(
            execution_id=execution.execution_id,
            parent_order_id=execution.parent_order_id,
            symbol=execution.symbol,
            side=execution.side,
            quantity=execution.quantity,
            algorithm=execution.algorithm.value,
            strategy=execution.strategy.value,
            parameters=execution.parameters,
            status=execution.status,
            filled_quantity=execution.filled_quantity,
            remaining_quantity=execution.remaining_quantity,
            average_price=execution.average_price,
            venues=execution.venues,
            created_at=execution.created_at,
            last_updated=execution.last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution: {str(e)}",
        )


@router.get("/executions/{execution_id}/slices", response_model=List[Dict[str, Any]])
async def get_execution_slices(
    execution_id: str,
    ems: ExecutionManagementSystem = Depends(get_execution_management_system),
):
    """Get execution slices"""
    try:
        slices = await ems.get_execution_slices(execution_id)

        return [
            {
                "slice_id": slice_obj.slice_id,
                "execution_id": slice_obj.execution_id,
                "symbol": slice_obj.symbol,
                "side": slice_obj.side,
                "quantity": slice_obj.quantity,
                "price": slice_obj.price,
                "venue": slice_obj.venue,
                "status": slice_obj.status,
                "filled_quantity": slice_obj.filled_quantity,
                "remaining_quantity": slice_obj.remaining_quantity,
                "average_price": slice_obj.average_price,
                "start_time": slice_obj.start_time,
                "end_time": slice_obj.end_time,
                "created_at": slice_obj.created_at,
            }
            for slice_obj in slices
        ]

    except Exception as e:
        logger.error(f"Error getting execution slices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution slices: {str(e)}",
        )


@router.get(
    "/executions/{execution_id}/metrics", response_model=ExecutionMetricsResponse
)
async def get_execution_metrics(
    execution_id: str,
    ems: ExecutionManagementSystem = Depends(get_execution_management_system),
):
    """Get execution metrics"""
    try:
        metrics = await ems.get_execution_metrics(execution_id)

        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution metrics not found",
            )

        return ExecutionMetricsResponse(
            execution_id=metrics.execution_id,
            symbol=metrics.symbol,
            total_quantity=metrics.total_quantity,
            filled_quantity=metrics.filled_quantity,
            average_price=metrics.average_price,
            benchmark_price=metrics.benchmark_price,
            implementation_shortfall=metrics.implementation_shortfall,
            market_impact=metrics.market_impact,
            timing_cost=metrics.timing_cost,
            opportunity_cost=metrics.opportunity_cost,
            total_cost=metrics.total_cost,
            vwap_deviation=metrics.vwap_deviation,
            participation_rate=metrics.participation_rate,
            fill_rate=metrics.fill_rate,
            execution_time=metrics.execution_time,
            created_at=metrics.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution metrics: {str(e)}",
        )


@router.delete("/executions/{execution_id}", response_model=Dict[str, str])
async def cancel_execution(
    execution_id: str,
    ems: ExecutionManagementSystem = Depends(get_execution_management_system),
):
    """Cancel execution"""
    try:
        success = await ems.cancel_execution(execution_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel execution",
            )

        return {
            "message": "Execution cancelled successfully",
            "execution_id": execution_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}",
        )


@router.get("/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(
    symbol: str, oms: OrderManagementSystem = Depends(get_order_management_system)
):
    """Get market data for a symbol"""
    try:
        market_data = await oms.get_market_data(symbol)

        if not market_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Market data not found"
            )

        return MarketDataResponse(
            symbol=market_data.symbol,
            bid_price=market_data.bid_price,
            ask_price=market_data.ask_price,
            bid_size=market_data.bid_size,
            ask_size=market_data.ask_size,
            last_price=market_data.last_price,
            volume=market_data.volume,
            high_price=market_data.high_price,
            low_price=market_data.low_price,
            open_price=market_data.open_price,
            timestamp=market_data.timestamp,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market data: {str(e)}",
        )


@router.get("/order-book/{symbol}", response_model=OrderBookResponse)
async def get_order_book(
    symbol: str, oms: OrderManagementSystem = Depends(get_order_management_system)
):
    """Get order book for a symbol"""
    try:
        order_book = await oms.get_order_book(symbol)

        if not order_book:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Order book not found"
            )

        return OrderBookResponse(
            symbol=order_book.symbol,
            bids=order_book.bids,
            asks=order_book.asks,
            last_trade_price=order_book.last_trade_price,
            last_trade_quantity=order_book.last_trade_quantity,
            last_trade_time=order_book.last_trade_time,
            volume=order_book.volume,
            timestamp=order_book.timestamp,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order book: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get order book: {str(e)}",
        )


# Dependency injection functions
async def get_order_management_system() -> OrderManagementSystem:
    """Get Order Management System instance"""
    # This would be injected from the main app
    # For now, return a mock instance
    pass


async def get_execution_management_system() -> ExecutionManagementSystem:
    """Get Execution Management System instance"""
    # This would be injected from the main app
    # For now, return a mock instance
    pass
