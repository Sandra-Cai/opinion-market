from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import json
import logging

from app.core.database import SessionLocal
from app.models.order import Order, OrderStatus, OrderType, OrderSide
from app.models.market import Market
from app.models.trade import Trade
from app.models.position import Position

logger = logging.getLogger(__name__)


class AdvancedOrderManager:
    """Advanced order management system with complex order types"""

    def __init__(self):
        self.active_orders = {}  # order_id -> order object
        self.price_monitors = {}  # market_id -> list of orders to monitor
        self.conditional_triggers = {}  # order_id -> trigger conditions

    def create_stop_loss_order(
        self,
        user_id: int,
        market_id: int,
        outcome: str,
        shares: float,
        stop_price: float,
        order_type: str = "market",
    ) -> Dict:
        """Create a stop-loss order to limit losses"""
        db = SessionLocal()
        try:
            # Validate user has position
            position = (
                db.query(Position)
                .filter(
                    Position.user_id == user_id,
                    Position.market_id == market_id,
                    Position.outcome == outcome,
                    Position.is_active == True,
                )
                .first()
            )

            if not position or position.shares_owned < shares:
                return {"error": "Insufficient shares for stop-loss order"}

            # Create stop-loss order
            order = Order(
                user_id=user_id,
                market_id=market_id,
                order_type=OrderType.STOP,
                side=OrderSide.SELL,
                outcome=outcome,
                original_amount=shares,
                remaining_amount=shares,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                metadata={
                    "order_type": "stop_loss",
                    "trigger_price": stop_price,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            db.add(order)
            db.commit()
            db.refresh(order)

            # Add to price monitoring
            self._add_to_price_monitor(market_id, order.id, stop_price, "below")

            return {
                "success": True,
                "order_id": order.id,
                "message": "Stop-loss order created successfully",
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating stop-loss order: {e}")
            return {"error": "Failed to create stop-loss order"}
        finally:
            db.close()

    def create_take_profit_order(
        self,
        user_id: int,
        market_id: int,
        outcome: str,
        shares: float,
        take_profit_price: float,
        order_type: str = "market",
    ) -> Dict:
        """Create a take-profit order to secure gains"""
        db = SessionLocal()
        try:
            # Validate user has position
            position = (
                db.query(Position)
                .filter(
                    Position.user_id == user_id,
                    Position.market_id == market_id,
                    Position.outcome == outcome,
                    Position.is_active == True,
                )
                .first()
            )

            if not position or position.shares_owned < shares:
                return {"error": "Insufficient shares for take-profit order"}

            # Create take-profit order
            order = Order(
                user_id=user_id,
                market_id=market_id,
                order_type=OrderType.STOP,
                side=OrderSide.SELL,
                outcome=outcome,
                original_amount=shares,
                remaining_amount=shares,
                stop_price=take_profit_price,
                status=OrderStatus.PENDING,
                metadata={
                    "order_type": "take_profit",
                    "trigger_price": take_profit_price,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            db.add(order)
            db.commit()
            db.refresh(order)

            # Add to price monitoring
            self._add_to_price_monitor(market_id, order.id, take_profit_price, "above")

            return {
                "success": True,
                "order_id": order.id,
                "message": "Take-profit order created successfully",
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating take-profit order: {e}")
            return {"error": "Failed to create take-profit order"}
        finally:
            db.close()

    def create_trailing_stop_order(
        self,
        user_id: int,
        market_id: int,
        outcome: str,
        shares: float,
        trailing_percentage: float,
    ) -> Dict:
        """Create a trailing stop order that follows price movements"""
        db = SessionLocal()
        try:
            # Validate user has position
            position = (
                db.query(Position)
                .filter(
                    Position.user_id == user_id,
                    Position.market_id == market_id,
                    Position.outcome == outcome,
                    Position.is_active == True,
                )
                .first()
            )

            if not position or position.shares_owned < shares:
                return {"error": "Insufficient shares for trailing stop order"}

            # Get current market price
            market = db.query(Market).filter(Market.id == market_id).first()
            current_price = (
                market.current_price_a
                if outcome == "outcome_a"
                else market.current_price_b
            )

            # Calculate initial stop price
            initial_stop_price = current_price * (1 - trailing_percentage / 100)

            # Create trailing stop order
            order = Order(
                user_id=user_id,
                market_id=market_id,
                order_type=OrderType.STOP,
                side=OrderSide.SELL,
                outcome=outcome,
                original_amount=shares,
                remaining_amount=shares,
                stop_price=initial_stop_price,
                status=OrderStatus.PENDING,
                metadata={
                    "order_type": "trailing_stop",
                    "trailing_percentage": trailing_percentage,
                    "initial_stop_price": initial_stop_price,
                    "highest_price": current_price,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            db.add(order)
            db.commit()
            db.refresh(order)

            # Add to price monitoring
            self._add_to_price_monitor(market_id, order.id, initial_stop_price, "below")

            return {
                "success": True,
                "order_id": order.id,
                "message": "Trailing stop order created successfully",
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating trailing stop order: {e}")
            return {"error": "Failed to create trailing stop order"}
        finally:
            db.close()

    def create_conditional_order(
        self,
        user_id: int,
        market_id: int,
        outcome: str,
        shares: float,
        limit_price: float,
        condition_market_id: int,
        condition_outcome: str,
        condition_price: float,
        condition_type: str,
    ) -> Dict:
        """Create a conditional order that triggers based on another market's price"""
        db = SessionLocal()
        try:
            # Validate condition market exists
            condition_market = (
                db.query(Market).filter(Market.id == condition_market_id).first()
            )
            if not condition_market:
                return {"error": "Condition market not found"}

            # Create conditional order
            order = Order(
                user_id=user_id,
                market_id=market_id,
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY,
                outcome=outcome,
                original_amount=shares,
                remaining_amount=shares,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                metadata={
                    "order_type": "conditional",
                    "condition_market_id": condition_market_id,
                    "condition_outcome": condition_outcome,
                    "condition_price": condition_price,
                    "condition_type": condition_type,  # "above" or "below"
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            db.add(order)
            db.commit()
            db.refresh(order)

            # Add to conditional triggers
            self.conditional_triggers[order.id] = {
                "condition_market_id": condition_market_id,
                "condition_outcome": condition_outcome,
                "condition_price": condition_price,
                "condition_type": condition_type,
            }

            return {
                "success": True,
                "order_id": order.id,
                "message": "Conditional order created successfully",
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating conditional order: {e}")
            return {"error": "Failed to create conditional order"}
        finally:
            db.close()

    def create_bracket_order(
        self,
        user_id: int,
        market_id: int,
        outcome: str,
        shares: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> Dict:
        """Create a bracket order with stop-loss and take-profit"""
        db = SessionLocal()
        try:
            # Create main limit order
            main_order = Order(
                user_id=user_id,
                market_id=market_id,
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY,
                outcome=outcome,
                original_amount=shares,
                remaining_amount=shares,
                limit_price=entry_price,
                status=OrderStatus.PENDING,
                metadata={
                    "order_type": "bracket_main",
                    "bracket_id": f"bracket_{user_id}_{int(datetime.utcnow().timestamp())}",
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            db.add(main_order)
            db.flush()  # Get the ID without committing

            # Create stop-loss order
            stop_loss_order = Order(
                user_id=user_id,
                market_id=market_id,
                order_type=OrderType.STOP,
                side=OrderSide.SELL,
                outcome=outcome,
                original_amount=shares,
                remaining_amount=shares,
                stop_price=stop_loss_price,
                status=OrderStatus.PENDING,
                metadata={
                    "order_type": "bracket_stop_loss",
                    "bracket_id": main_order.metadata["bracket_id"],
                    "main_order_id": main_order.id,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            # Create take-profit order
            take_profit_order = Order(
                user_id=user_id,
                market_id=market_id,
                order_type=OrderType.STOP,
                side=OrderSide.SELL,
                outcome=outcome,
                original_amount=shares,
                remaining_amount=shares,
                stop_price=take_profit_price,
                status=OrderStatus.PENDING,
                metadata={
                    "order_type": "bracket_take_profit",
                    "bracket_id": main_order.metadata["bracket_id"],
                    "main_order_id": main_order.id,
                    "created_at": datetime.utcnow().isoformat(),
                },
            )

            db.add(stop_loss_order)
            db.add(take_profit_order)
            db.commit()

            # Add to price monitoring
            self._add_to_price_monitor(
                market_id, stop_loss_order.id, stop_loss_price, "below"
            )
            self._add_to_price_monitor(
                market_id, take_profit_order.id, take_profit_price, "above"
            )

            return {
                "success": True,
                "bracket_id": main_order.metadata["bracket_id"],
                "main_order_id": main_order.id,
                "stop_loss_order_id": stop_loss_order.id,
                "take_profit_order_id": take_profit_order.id,
                "message": "Bracket order created successfully",
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating bracket order: {e}")
            return {"error": "Failed to create bracket order"}
        finally:
            db.close()

    def check_price_triggers(self, market_id: int, current_price: float, outcome: str):
        """Check if any orders should be triggered by current price"""
        if market_id not in self.price_monitors:
            return

        triggered_orders = []
        for order_id, monitor_data in self.price_monitors[market_id].items():
            if monitor_data["outcome"] != outcome:
                continue

            trigger_price = monitor_data["trigger_price"]
            trigger_type = monitor_data["trigger_type"]

            should_trigger = False
            if trigger_type == "above" and current_price >= trigger_price:
                should_trigger = True
            elif trigger_type == "below" and current_price <= trigger_price:
                should_trigger = True

            if should_trigger:
                triggered_orders.append(order_id)

        # Execute triggered orders
        for order_id in triggered_orders:
            self._execute_triggered_order(order_id, current_price)

    def check_conditional_triggers(
        self, condition_market_id: int, current_price: float, outcome: str
    ):
        """Check if any conditional orders should be triggered"""
        triggered_orders = []

        for order_id, trigger_data in self.conditional_triggers.items():
            if trigger_data["condition_market_id"] != condition_market_id:
                continue

            if trigger_data["condition_outcome"] != outcome:
                continue

            condition_price = trigger_data["condition_price"]
            condition_type = trigger_data["condition_type"]

            should_trigger = False
            if condition_type == "above" and current_price >= condition_price:
                should_trigger = True
            elif condition_type == "below" and current_price <= condition_price:
                should_trigger = True

            if should_trigger:
                triggered_orders.append(order_id)

        # Activate conditional orders
        for order_id in triggered_orders:
            self._activate_conditional_order(order_id)

    def update_trailing_stops(self, market_id: int, current_price: float, outcome: str):
        """Update trailing stop orders based on price movement"""
        db = SessionLocal()
        try:
            trailing_orders = (
                db.query(Order)
                .filter(
                    Order.market_id == market_id,
                    Order.status == OrderStatus.PENDING,
                    Order.metadata.contains({"order_type": "trailing_stop"}),
                )
                .all()
            )

            for order in trailing_orders:
                if order.outcome != outcome:
                    continue

                metadata = order.metadata
                trailing_percentage = metadata["trailing_percentage"]
                highest_price = metadata["highest_price"]

                # Update highest price if current price is higher
                if current_price > highest_price:
                    metadata["highest_price"] = current_price

                    # Calculate new stop price
                    new_stop_price = current_price * (1 - trailing_percentage / 100)

                    # Only update if new stop price is higher than current
                    if new_stop_price > order.stop_price:
                        order.stop_price = new_stop_price
                        metadata["initial_stop_price"] = new_stop_price
                        order.metadata = metadata

                        # Update price monitor
                        self._update_price_monitor(
                            market_id, order.id, new_stop_price, "below"
                        )

            db.commit()

        except Exception as e:
            db.rollback()
            logger.error(f"Error updating trailing stops: {e}")
        finally:
            db.close()

    def _add_to_price_monitor(
        self, market_id: int, order_id: int, trigger_price: float, trigger_type: str
    ):
        """Add order to price monitoring system"""
        if market_id not in self.price_monitors:
            self.price_monitors[market_id] = {}

        self.price_monitors[market_id][order_id] = {
            "trigger_price": trigger_price,
            "trigger_type": trigger_type,
            "outcome": "outcome_a",  # This should be set based on order outcome
        }

    def _update_price_monitor(
        self, market_id: int, order_id: int, new_trigger_price: float, trigger_type: str
    ):
        """Update price monitor with new trigger price"""
        if (
            market_id in self.price_monitors
            and order_id in self.price_monitors[market_id]
        ):
            self.price_monitors[market_id][order_id][
                "trigger_price"
            ] = new_trigger_price

    def _execute_triggered_order(self, order_id: int, trigger_price: float):
        """Execute a triggered order"""
        db = SessionLocal()
        try:
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order or order.status != OrderStatus.PENDING:
                return

            # Convert stop order to market order and execute
            order.order_type = OrderType.MARKET
            order.status = OrderStatus.FILLED
            order.filled_amount = order.remaining_amount
            order.remaining_amount = 0
            order.average_fill_price = trigger_price

            # Create trade record
            trade = Trade(
                user_id=order.user_id,
                market_id=order.market_id,
                trade_type=order.side,
                outcome=order.outcome,
                amount=order.filled_amount,
                price_per_share=trigger_price,
                total_value=order.filled_amount * trigger_price,
                profit_loss=0,  # Calculate based on position
                created_at=datetime.utcnow(),
            )

            db.add(trade)
            db.commit()

            # Remove from price monitoring
            self._remove_from_price_monitor(order.market_id, order_id)

            logger.info(f"Executed triggered order {order_id} at price {trigger_price}")

        except Exception as e:
            db.rollback()
            logger.error(f"Error executing triggered order {order_id}: {e}")
        finally:
            db.close()

    def _activate_conditional_order(self, order_id: int):
        """Activate a conditional order"""
        db = SessionLocal()
        try:
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order or order.status != OrderStatus.PENDING:
                return

            # Change status to active (ready for execution)
            order.status = OrderStatus.PENDING
            order.metadata["activated_at"] = datetime.utcnow().isoformat()

            db.commit()

            # Remove from conditional triggers
            if order_id in self.conditional_triggers:
                del self.conditional_triggers[order_id]

            logger.info(f"Activated conditional order {order_id}")

        except Exception as e:
            db.rollback()
            logger.error(f"Error activating conditional order {order_id}: {e}")
        finally:
            db.close()

    def _remove_from_price_monitor(self, market_id: int, order_id: int):
        """Remove order from price monitoring"""
        if (
            market_id in self.price_monitors
            and order_id in self.price_monitors[market_id]
        ):
            del self.price_monitors[market_id][order_id]

    def get_user_advanced_orders(self, user_id: int) -> Dict:
        """Get user's advanced orders"""
        db = SessionLocal()
        try:
            orders = (
                db.query(Order)
                .filter(
                    Order.user_id == user_id,
                    Order.order_type.in_([OrderType.STOP]),
                    Order.status == OrderStatus.PENDING,
                )
                .all()
            )

            return {
                "orders": [
                    {
                        "id": order.id,
                        "market_id": order.market_id,
                        "order_type": order.metadata.get("order_type", "unknown"),
                        "outcome": order.outcome,
                        "amount": order.remaining_amount,
                        "trigger_price": order.stop_price,
                        "created_at": order.created_at,
                        "metadata": order.metadata,
                    }
                    for order in orders
                ],
                "total": len(orders),
            }

        finally:
            db.close()


# Global advanced order manager instance
advanced_order_manager = AdvancedOrderManager()


def get_advanced_order_manager() -> AdvancedOrderManager:
    """Get the global advanced order manager instance"""
    return advanced_order_manager
