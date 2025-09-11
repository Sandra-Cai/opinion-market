"""
Cryptocurrency Trading Service
Provides spot trading, futures, DeFi integration, and crypto-specific analytics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Cryptocurrency:
    """Cryptocurrency information"""

    crypto_id: str
    symbol: str
    name: str
    blockchain: str
    contract_address: Optional[str]
    decimals: int
    total_supply: float
    circulating_supply: float
    market_cap: float
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class CryptoPrice:
    """Cryptocurrency price information"""

    price_id: str
    crypto_id: str
    price_usd: float
    price_btc: float
    price_eth: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    price_change_percent_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime
    source: str


@dataclass
class CryptoPosition:
    """Cryptocurrency trading position"""

    position_id: str
    user_id: int
    crypto_id: str
    position_type: str  # 'long', 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: datetime
    last_updated: datetime


@dataclass
class DeFiProtocol:
    """DeFi protocol information"""

    protocol_id: str
    name: str
    protocol_type: str  # 'DEX', 'Lending', 'Yield', 'Derivatives'
    blockchain: str
    tvl: float
    apy: float
    risk_score: float
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class CryptoOrder:
    """Cryptocurrency trading order"""

    order_id: str
    user_id: int
    crypto_id: str
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    side: str  # 'buy', 'sell'
    size: float
    price: Optional[float]
    stop_price: Optional[float]
    limit_price: Optional[float]
    time_in_force: str  # 'GTC', 'IOC', 'FOK'
    status: str  # 'pending', 'filled', 'cancelled', 'rejected'
    filled_size: float
    filled_price: float
    commission: float
    created_at: datetime
    last_updated: datetime


class CryptoTradingService:
    """Comprehensive cryptocurrency trading service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.cryptocurrencies: Dict[str, Cryptocurrency] = {}
        self.crypto_prices: Dict[str, List[CryptoPrice]] = defaultdict(list)
        self.crypto_positions: Dict[str, CryptoPosition] = {}
        self.defi_protocols: Dict[str, DeFiProtocol] = {}
        self.crypto_orders: Dict[str, CryptoOrder] = {}

        # Market data
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volatility_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Crypto-specific data
        self.blockchain_data: Dict[str, Dict[str, Any]] = {}
        self.defi_metrics: Dict[str, Dict[str, Any]] = {}
        self.nft_market_data: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize the cryptocurrency trading service"""
        logger.info("Initializing Cryptocurrency Trading Service")

        # Load existing data
        await self._load_cryptocurrencies()
        await self._load_defi_protocols()

        # Start background tasks
        asyncio.create_task(self._update_crypto_prices())
        asyncio.create_task(self._update_positions())
        asyncio.create_task(self._update_volatility())
        asyncio.create_task(self._monitor_orders())

        logger.info("Cryptocurrency Trading Service initialized successfully")

    async def create_cryptocurrency(
        self,
        symbol: str,
        name: str,
        blockchain: str,
        contract_address: Optional[str],
        decimals: int,
        total_supply: float,
    ) -> Cryptocurrency:
        """Create a new cryptocurrency"""
        try:
            crypto_id = (
                f"crypto_{symbol.lower()}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            crypto = Cryptocurrency(
                crypto_id=crypto_id,
                symbol=symbol,
                name=name,
                blockchain=blockchain,
                contract_address=contract_address,
                decimals=decimals,
                total_supply=total_supply,
                circulating_supply=total_supply,
                market_cap=0.0,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.cryptocurrencies[crypto_id] = crypto
            await self._cache_cryptocurrency(crypto)

            logger.info(f"Created cryptocurrency {crypto_id}")
            return crypto

        except Exception as e:
            logger.error(f"Error creating cryptocurrency: {e}")
            raise

    async def add_crypto_price(
        self,
        crypto_id: str,
        price_usd: float,
        price_btc: float,
        price_eth: float,
        volume_24h: float,
        market_cap: float,
        price_change_24h: float,
        price_change_percent_24h: float,
        high_24h: float,
        low_24h: float,
        source: str,
    ) -> CryptoPrice:
        """Add a new cryptocurrency price"""
        try:
            if crypto_id not in self.cryptocurrencies:
                raise ValueError(f"Cryptocurrency {crypto_id} not found")

            price = CryptoPrice(
                price_id=f"price_{crypto_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                crypto_id=crypto_id,
                price_usd=price_usd,
                price_btc=price_btc,
                price_eth=price_eth,
                volume_24h=volume_24h,
                market_cap=market_cap,
                price_change_24h=price_change_24h,
                price_change_percent_24h=price_change_percent_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                timestamp=datetime.utcnow(),
                source=source,
            )

            self.crypto_prices[crypto_id].append(price)

            # Keep only recent prices
            if len(self.crypto_prices[crypto_id]) > 1000:
                self.crypto_prices[crypto_id] = self.crypto_prices[crypto_id][-1000:]

            # Update cryptocurrency market cap
            crypto = self.cryptocurrencies[crypto_id]
            crypto.market_cap = market_cap
            crypto.last_updated = datetime.utcnow()
            await self._cache_cryptocurrency(crypto)

            await self._cache_crypto_price(price)

            logger.info(f"Added price for {crypto_id}: ${price_usd}")
            return price

        except Exception as e:
            logger.error(f"Error adding crypto price: {e}")
            raise

    async def create_crypto_position(
        self,
        user_id: int,
        crypto_id: str,
        position_type: str,
        size: float,
        entry_price: float,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> CryptoPosition:
        """Create a new cryptocurrency position"""
        try:
            if crypto_id not in self.cryptocurrencies:
                raise ValueError(f"Cryptocurrency {crypto_id} not found")

            # Calculate margin requirement
            margin_used = size * entry_price / leverage

            position_id = f"crypto_pos_{user_id}_{crypto_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            position = CryptoPosition(
                position_id=position_id,
                user_id=user_id,
                crypto_id=crypto_id,
                position_type=position_type,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                margin_used=margin_used,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.crypto_positions[position_id] = position
            await self._cache_crypto_position(position)

            logger.info(f"Created crypto position {position_id}")
            return position

        except Exception as e:
            logger.error(f"Error creating crypto position: {e}")
            raise

    async def create_defi_protocol(
        self,
        name: str,
        protocol_type: str,
        blockchain: str,
        tvl: float,
        apy: float,
        risk_score: float,
    ) -> DeFiProtocol:
        """Create a new DeFi protocol"""
        try:
            protocol_id = f"defi_{name.lower().replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            protocol = DeFiProtocol(
                protocol_id=protocol_id,
                name=name,
                protocol_type=protocol_type,
                blockchain=blockchain,
                tvl=tvl,
                apy=apy,
                risk_score=risk_score,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.defi_protocols[protocol_id] = protocol
            await self._cache_defi_protocol(protocol)

            logger.info(f"Created DeFi protocol {protocol_id}")
            return protocol

        except Exception as e:
            logger.error(f"Error creating DeFi protocol: {e}")
            raise

    async def create_crypto_order(
        self,
        user_id: int,
        crypto_id: str,
        order_type: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        limit_price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> CryptoOrder:
        """Create a new cryptocurrency order"""
        try:
            if crypto_id not in self.cryptocurrencies:
                raise ValueError(f"Cryptocurrency {crypto_id} not found")

            order_id = f"crypto_order_{user_id}_{crypto_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            order = CryptoOrder(
                order_id=order_id,
                user_id=user_id,
                crypto_id=crypto_id,
                order_type=order_type,
                side=side,
                size=size,
                price=price,
                stop_price=stop_price,
                limit_price=limit_price,
                time_in_force=time_in_force,
                status="pending",
                filled_size=0.0,
                filled_price=0.0,
                commission=0.0,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.crypto_orders[order_id] = order
            await self._cache_crypto_order(order)

            logger.info(f"Created crypto order {order_id}")
            return order

        except Exception as e:
            logger.error(f"Error creating crypto order: {e}")
            raise

    async def calculate_crypto_metrics(self, crypto_id: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a cryptocurrency"""
        try:
            if crypto_id not in self.cryptocurrencies:
                raise ValueError(f"Cryptocurrency {crypto_id} not found")

            crypto = self.cryptocurrencies[crypto_id]
            prices = self.crypto_prices.get(crypto_id, [])

            if not prices:
                return {"crypto_id": crypto_id, "error": "No price data available"}

            # Calculate price statistics
            usd_prices = [p.price_usd for p in prices]
            btc_prices = [p.price_btc for p in prices]
            volumes = [p.volume_24h for p in prices]

            # Calculate volatility
            returns = [
                np.log(usd_prices[i] / usd_prices[i - 1])
                for i in range(1, len(usd_prices))
            ]
            volatility = (
                np.std(returns) * np.sqrt(365 * 24) if returns else 0
            )  # Annualized hourly volatility

            # Calculate moving averages
            sma_20 = np.mean(usd_prices[-20:]) if len(usd_prices) >= 20 else None
            sma_50 = np.mean(usd_prices[-50:]) if len(usd_prices) >= 50 else None
            sma_200 = np.mean(usd_prices[-200:]) if len(usd_prices) >= 200 else None

            # Calculate RSI
            rsi = self._calculate_rsi(usd_prices) if len(usd_prices) >= 14 else None

            metrics = {
                "crypto_id": crypto_id,
                "symbol": crypto.symbol,
                "name": crypto.name,
                "blockchain": crypto.blockchain,
                "current_price_usd": usd_prices[-1] if usd_prices else None,
                "current_price_btc": btc_prices[-1] if btc_prices else None,
                "market_cap": crypto.market_cap,
                "total_supply": crypto.total_supply,
                "circulating_supply": crypto.circulating_supply,
                "price_statistics": {
                    "price_mean": np.mean(usd_prices) if usd_prices else None,
                    "price_std": np.std(usd_prices) if usd_prices else None,
                    "price_min": np.min(usd_prices) if usd_prices else None,
                    "price_max": np.max(usd_prices) if usd_prices else None,
                    "volume_mean": np.mean(volumes) if volumes else None,
                },
                "technical_indicators": {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "sma_200": sma_200,
                    "rsi": rsi,
                    "volatility": volatility,
                },
                "price_trends": await self._calculate_price_trends(crypto_id),
                "blockchain_metrics": self.blockchain_data.get(crypto.blockchain, {}),
                "defi_integration": await self._get_defi_integration(crypto_id),
                "last_updated": datetime.utcnow().isoformat(),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating crypto metrics: {e}")
            raise

    async def get_defi_analytics(self) -> Dict[str, Any]:
        """Get comprehensive DeFi analytics"""
        try:
            total_tvl = sum(p.tvl for p in self.defi_protocols.values() if p.is_active)
            total_protocols = len(
                [p for p in self.defi_protocols.values() if p.is_active]
            )

            # Group by protocol type
            protocol_types = defaultdict(list)
            for protocol in self.defi_protocols.values():
                if protocol.is_active:
                    protocol_types[protocol.protocol_type].append(protocol)

            # Calculate metrics by type
            type_metrics = {}
            for protocol_type, protocols in protocol_types.items():
                type_metrics[protocol_type] = {
                    "count": len(protocols),
                    "total_tvl": sum(p.tvl for p in protocols),
                    "avg_apy": np.mean([p.apy for p in protocols]) if protocols else 0,
                    "avg_risk_score": (
                        np.mean([p.risk_score for p in protocols]) if protocols else 0
                    ),
                }

            analytics = {
                "total_tvl": total_tvl,
                "total_protocols": total_protocols,
                "protocol_types": type_metrics,
                "top_protocols": sorted(
                    [p for p in self.defi_protocols.values() if p.is_active],
                    key=lambda x: x.tvl,
                    reverse=True,
                )[:10],
                "blockchain_distribution": await self._get_blockchain_distribution(),
                "risk_analysis": await self._get_risk_analysis(),
                "last_updated": datetime.utcnow().isoformat(),
            }

            return analytics

        except Exception as e:
            logger.error(f"Error getting DeFi analytics: {e}")
            raise

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI

            # Calculate price changes
            changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

            # Separate gains and losses
            gains = [change if change > 0 else 0 for change in changes]
            losses = [-change if change < 0 else 0 for change in changes]

            # Calculate average gains and losses
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0

    async def _calculate_price_trends(self, crypto_id: str) -> Dict[str, Any]:
        """Calculate price trends for a cryptocurrency"""
        try:
            prices = self.crypto_prices.get(crypto_id, [])
            if len(prices) < 2:
                return {}

            # Get recent prices
            recent_prices = prices[-30:]  # Last 30 prices
            usd_prices = [p.price_usd for p in recent_prices]

            # Calculate trends
            if len(usd_prices) >= 2:
                price_change = usd_prices[-1] - usd_prices[0]
                price_change_percent = (
                    (price_change / usd_prices[0]) * 100 if usd_prices[0] != 0 else 0
                )

                # Simple moving averages
                sma_5 = np.mean(usd_prices[-5:]) if len(usd_prices) >= 5 else None
                sma_10 = np.mean(usd_prices[-10:]) if len(usd_prices) >= 10 else None
                sma_20 = np.mean(usd_prices[-20:]) if len(usd_prices) >= 20 else None

                return {
                    "price_change": price_change,
                    "price_change_percent": price_change_percent,
                    "trend_direction": (
                        "up"
                        if price_change > 0
                        else "down" if price_change < 0 else "flat"
                    ),
                    "sma_5": sma_5,
                    "sma_10": sma_10,
                    "sma_20": sma_20,
                    "volatility": np.std(usd_prices) if len(usd_prices) > 1 else 0,
                }

            return {}

        except Exception as e:
            logger.error(f"Error calculating price trends: {e}")
            return {}

    async def _get_defi_integration(self, crypto_id: str) -> Dict[str, Any]:
        """Get DeFi integration information for a cryptocurrency"""
        try:
            crypto = self.cryptocurrencies.get(crypto_id)
            if not crypto:
                return {}

            # Find protocols that use this cryptocurrency
            related_protocols = []
            for protocol in self.defi_protocols.values():
                if protocol.is_active and protocol.blockchain == crypto.blockchain:
                    related_protocols.append(
                        {
                            "protocol_id": protocol.protocol_id,
                            "name": protocol.name,
                            "protocol_type": protocol.protocol_type,
                            "tvl": protocol.tvl,
                            "apy": protocol.apy,
                        }
                    )

            return {
                "related_protocols": related_protocols,
                "protocol_count": len(related_protocols),
                "total_tvl_integration": sum(p["tvl"] for p in related_protocols),
            }

        except Exception as e:
            logger.error(f"Error getting DeFi integration: {e}")
            return {}

    async def _get_blockchain_distribution(self) -> Dict[str, Any]:
        """Get blockchain distribution for DeFi protocols"""
        try:
            blockchain_stats = defaultdict(lambda: {"count": 0, "tvl": 0, "avg_apy": 0})

            for protocol in self.defi_protocols.values():
                if protocol.is_active:
                    blockchain_stats[protocol.blockchain]["count"] += 1
                    blockchain_stats[protocol.blockchain]["tvl"] += protocol.tvl

            # Calculate average APY per blockchain
            for blockchain in blockchain_stats:
                protocols = [
                    p
                    for p in self.defi_protocols.values()
                    if p.is_active and p.blockchain == blockchain
                ]
                if protocols:
                    blockchain_stats[blockchain]["avg_apy"] = np.mean(
                        [p.apy for p in protocols]
                    )

            return dict(blockchain_stats)

        except Exception as e:
            logger.error(f"Error getting blockchain distribution: {e}")
            return {}

    async def _get_risk_analysis(self) -> Dict[str, Any]:
        """Get risk analysis for DeFi protocols"""
        try:
            risk_scores = [
                p.risk_score for p in self.defi_protocols.values() if p.is_active
            ]

            if not risk_scores:
                return {}

            return {
                "avg_risk_score": np.mean(risk_scores),
                "min_risk_score": np.min(risk_scores),
                "max_risk_score": np.max(risk_scores),
                "risk_distribution": {
                    "low_risk": len([r for r in risk_scores if r <= 0.3]),
                    "medium_risk": len([r for r in risk_scores if 0.3 < r <= 0.7]),
                    "high_risk": len([r for r in risk_scores if r > 0.7]),
                },
            }

        except Exception as e:
            logger.error(f"Error getting risk analysis: {e}")
            return {}

    # Background tasks
    async def _update_crypto_prices(self):
        """Update cryptocurrency prices periodically"""
        while True:
            try:
                # Update prices for all cryptocurrencies
                for crypto in self.cryptocurrencies.values():
                    if crypto.is_active:
                        # Simulate price updates (in practice, this would fetch from data providers)
                        current_price = await self._get_current_crypto_price(
                            crypto.crypto_id
                        )
                        if current_price:
                            # Add small random price movement
                            movement = np.random.normal(0, 0.02)  # 2% volatility
                            new_price = current_price.price_usd * (1 + movement)

                            await self.add_crypto_price(
                                crypto.crypto_id,
                                new_price,
                                current_price.price_btc,
                                current_price.price_eth,
                                current_price.volume_24h,
                                new_price * crypto.circulating_supply,
                                new_price - current_price.price_usd,
                                movement * 100,
                                new_price,
                                new_price * 0.95,
                                "simulated",
                            )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating crypto prices: {e}")
                await asyncio.sleep(120)

    async def _update_positions(self):
        """Update cryptocurrency positions"""
        while True:
            try:
                for position in self.crypto_positions.values():
                    # Get current price
                    current_price = await self._get_current_crypto_price(
                        position.crypto_id
                    )
                    if current_price:
                        position.current_price = current_price.price_usd

                        # Calculate unrealized P&L
                        if position.position_type == "long":
                            position.unrealized_pnl = (
                                position.current_price - position.entry_price
                            ) * position.size
                        else:  # short
                            position.unrealized_pnl = (
                                position.entry_price - position.current_price
                            ) * position.size

                        position.last_updated = datetime.utcnow()
                        await self._cache_crypto_position(position)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error updating positions: {e}")
                await asyncio.sleep(60)

    async def _update_volatility(self):
        """Update volatility calculations"""
        while True:
            try:
                # Update volatility for all cryptocurrencies
                for crypto in self.cryptocurrencies.values():
                    if crypto.is_active:
                        prices = self.crypto_prices.get(crypto.crypto_id, [])
                        if len(prices) >= 2:
                            usd_prices = [
                                p.price_usd for p in prices[-100:]
                            ]  # Last 100 prices
                            returns = [
                                np.log(usd_prices[i] / usd_prices[i - 1])
                                for i in range(1, len(usd_prices))
                            ]
                            volatility = (
                                np.std(returns) * np.sqrt(365 * 24) if returns else 0
                            )
                            self.volatility_history[crypto.crypto_id].append(volatility)

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error updating volatility: {e}")
                await asyncio.sleep(600)

    async def _monitor_orders(self):
        """Monitor cryptocurrency orders"""
        while True:
            try:
                # Placeholder for order monitoring
                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(30)

    async def _get_current_crypto_price(self, crypto_id: str) -> Optional[CryptoPrice]:
        """Get current cryptocurrency price"""
        try:
            prices = self.crypto_prices.get(crypto_id, [])
            if prices:
                return prices[-1]
            return None

        except Exception as e:
            logger.error(f"Error getting current crypto price: {e}")
            return None

    # Helper methods (implementations would depend on your data models)
    async def _load_cryptocurrencies(self):
        """Load cryptocurrencies from database"""
        pass

    async def _load_defi_protocols(self):
        """Load DeFi protocols from database"""
        pass

    # Caching methods
    async def _cache_cryptocurrency(self, crypto: Cryptocurrency):
        """Cache cryptocurrency"""
        try:
            cache_key = f"cryptocurrency:{crypto.crypto_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "symbol": crypto.symbol,
                        "name": crypto.name,
                        "blockchain": crypto.blockchain,
                        "contract_address": crypto.contract_address,
                        "decimals": crypto.decimals,
                        "total_supply": crypto.total_supply,
                        "circulating_supply": crypto.circulating_supply,
                        "market_cap": crypto.market_cap,
                        "is_active": crypto.is_active,
                        "created_at": crypto.created_at.isoformat(),
                        "last_updated": crypto.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching cryptocurrency: {e}")

    async def _cache_crypto_price(self, price: CryptoPrice):
        """Cache crypto price"""
        try:
            cache_key = f"crypto_price:{price.price_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "crypto_id": price.crypto_id,
                        "price_usd": price.price_usd,
                        "price_btc": price.price_btc,
                        "price_eth": price.price_eth,
                        "volume_24h": price.volume_24h,
                        "market_cap": price.market_cap,
                        "price_change_24h": price.price_change_24h,
                        "price_change_percent_24h": price.price_change_percent_24h,
                        "high_24h": price.high_24h,
                        "low_24h": price.low_24h,
                        "timestamp": price.timestamp.isoformat(),
                        "source": price.source,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching crypto price: {e}")

    async def _cache_crypto_position(self, position: CryptoPosition):
        """Cache crypto position"""
        try:
            cache_key = f"crypto_position:{position.position_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "user_id": position.user_id,
                        "crypto_id": position.crypto_id,
                        "position_type": position.position_type,
                        "size": position.size,
                        "entry_price": position.entry_price,
                        "current_price": position.current_price,
                        "unrealized_pnl": position.unrealized_pnl,
                        "realized_pnl": position.realized_pnl,
                        "margin_used": position.margin_used,
                        "leverage": position.leverage,
                        "stop_loss": position.stop_loss,
                        "take_profit": position.take_profit,
                        "created_at": position.created_at.isoformat(),
                        "last_updated": position.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching crypto position: {e}")

    async def _cache_defi_protocol(self, protocol: DeFiProtocol):
        """Cache DeFi protocol"""
        try:
            cache_key = f"defi_protocol:{protocol.protocol_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "name": protocol.name,
                        "protocol_type": protocol.protocol_type,
                        "blockchain": protocol.blockchain,
                        "tvl": protocol.tvl,
                        "apy": protocol.apy,
                        "risk_score": protocol.risk_score,
                        "is_active": protocol.is_active,
                        "created_at": protocol.created_at.isoformat(),
                        "last_updated": protocol.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching DeFi protocol: {e}")

    async def _cache_crypto_order(self, order: CryptoOrder):
        """Cache crypto order"""
        try:
            cache_key = f"crypto_order:{order.order_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "user_id": order.user_id,
                        "crypto_id": order.crypto_id,
                        "order_type": order.order_type,
                        "side": order.side,
                        "size": order.size,
                        "price": order.price,
                        "stop_price": order.stop_price,
                        "limit_price": order.limit_price,
                        "time_in_force": order.time_in_force,
                        "status": order.status,
                        "filled_size": order.filled_size,
                        "filled_price": order.filled_price,
                        "commission": order.commission,
                        "created_at": order.created_at.isoformat(),
                        "last_updated": order.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching crypto order: {e}")


# Factory function
async def get_crypto_trading_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> CryptoTradingService:
    """Get cryptocurrency trading service instance"""
    service = CryptoTradingService(redis_client, db_session)
    await service.initialize()
    return service
