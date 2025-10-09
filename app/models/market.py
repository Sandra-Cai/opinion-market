from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    Text,
    Enum,
    JSON,
    ForeignKey,
    Index,
    CheckConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import enum
from typing import Optional, Dict, Any, List
from app.core.database import Base


class MarketStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"
    DISPUTED = "disputed"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class MarketCategory(str, enum.Enum):
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SCIENCE = "science"
    CRYPTO = "crypto"
    WEATHER = "weather"
    HEALTH = "health"
    EDUCATION = "education"
    OTHER = "other"


class MarketType(str, enum.Enum):
    BINARY = "binary"  # Yes/No
    MULTIPLE_CHOICE = "multiple_choice"  # Multiple outcomes
    NUMERIC = "numeric"  # Numeric range
    DATE = "date"  # Date-based


class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    category = Column(Enum(MarketCategory), default=MarketCategory.OTHER)
    market_type = Column(Enum(MarketType), default=MarketType.BINARY)

    # Market details
    question = Column(String, nullable=False)
    outcome_a = Column(String, nullable=False)  # "Yes" or specific outcome
    outcome_b = Column(String, nullable=False)  # "No" or alternative outcome
    outcomes = Column(JSON, default=list)  # For multiple choice markets

    # Advanced pricing and liquidity (like Polymarket)
    current_price_a = Column(Float, default=0.5)  # Price for outcome A (0-1)
    current_price_b = Column(Float, default=0.5)  # Price for outcome B (0-1)
    total_liquidity = Column(Float, default=1000.0)
    liquidity_pool_a = Column(Float, default=500.0)  # Liquidity for outcome A
    liquidity_pool_b = Column(Float, default=500.0)  # Liquidity for outcome B

    # Market mechanics
    fee_rate = Column(Float, default=0.02)  # 2% trading fee
    min_trade_amount = Column(Float, default=1.0)
    max_trade_amount = Column(Float, default=10000.0)

    # Market status
    status = Column(Enum(MarketStatus), default=MarketStatus.PENDING_VERIFICATION)
    resolved_outcome = Column(String)  # Which outcome won
    resolution_source = Column(String)  # Source of resolution
    verification_required = Column(Boolean, default=True)
    verified_by = Column(Integer, ForeignKey("users.id"))
    verified_at = Column(DateTime)

    # Dispute resolution
    dispute_count = Column(Integer, default=0)
    dispute_reason = Column(Text)
    dispute_resolved_by = Column(Integer, ForeignKey("users.id"))
    dispute_resolved_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    closes_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime)

    # Creator
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Additional metadata
    tags = Column(JSON, default=list)  # Market tags
    image_url = Column(String)  # Market image
    volume_24h = Column(Float, default=0.0)  # 24h trading volume
    volume_total = Column(Float, default=0.0)  # Total trading volume

    # Market quality metrics
    unique_traders = Column(Integer, default=0)  # Number of unique traders
    market_quality_score = Column(Float, default=0.0)  # Quality score (0-100)
    trending_score = Column(Float, default=0.0)  # Trending algorithm score

    # Relationships
    creator = relationship(
        "User", foreign_keys=[creator_id], back_populates="markets_created"
    )
    verifier = relationship("User", foreign_keys=[verified_by], overlaps="dispute_resolver")
    dispute_resolver = relationship("User", foreign_keys=[dispute_resolved_by], overlaps="verifier")
    trades = relationship("Trade", back_populates="market")
    votes = relationship("Vote", back_populates="market")
    disputes = relationship("MarketDispute", back_populates="market")
    orders = relationship("Order", back_populates="market")
    positions = relationship("Position", back_populates="market")

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint('current_price_a >= 0 AND current_price_a <= 1', name='check_price_a_range'),
        CheckConstraint('current_price_b >= 0 AND current_price_b <= 1', name='check_price_b_range'),
        CheckConstraint('total_liquidity > 0', name='check_positive_liquidity'),
        CheckConstraint('liquidity_pool_a >= 0', name='check_positive_pool_a'),
        CheckConstraint('liquidity_pool_b >= 0', name='check_positive_pool_b'),
        CheckConstraint('fee_rate >= 0 AND fee_rate <= 0.1', name='check_fee_rate_range'),
        CheckConstraint('min_trade_amount > 0', name='check_positive_min_trade'),
        CheckConstraint('max_trade_amount > min_trade_amount', name='check_max_greater_than_min'),
        Index('idx_market_status', 'status'),
        Index('idx_market_category', 'category'),
        Index('idx_market_creator', 'creator_id'),
        Index('idx_market_created_at', 'created_at'),
        Index('idx_market_closes_at', 'closes_at'),
        Index('idx_market_quality_score', 'market_quality_score'),
        Index('idx_market_trending_score', 'trending_score'),
        Index('idx_market_volume_total', 'volume_total'),
    )

    @property
    def is_active(self) -> bool:
        return self.status == MarketStatus.OPEN and datetime.utcnow() < self.closes_at

    @property
    def total_volume(self) -> float:
        return sum(trade.total_value for trade in self.trades)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def price_impact(self) -> float:
        """Calculate price impact for large trades"""
        total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
        return 0.1 / total_liquidity if total_liquidity > 0 else 0.1

    @property
    def is_verified(self) -> bool:
        """Check if market is verified"""
        return self.verified_by is not None and self.verified_at is not None

    @property
    def time_until_close(self) -> float:
        """Time until market closes in seconds"""
        if self.closes_at:
            return (self.closes_at - datetime.utcnow()).total_seconds()
        return 0

    @property
    def is_trending(self) -> bool:
        """Check if market is trending"""
        return self.trending_score > 50.0

    def update_prices(self, trade_amount: float, outcome: str, trade_type: str):
        """Update market prices based on trade (like Polymarket's AMM)"""
        if outcome == "outcome_a":
            if trade_type == "buy":
                # Buy outcome A - price goes up
                self.liquidity_pool_a += trade_amount
                self.liquidity_pool_b -= trade_amount
                # Calculate new price using constant product formula
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity
            else:
                # Sell outcome A - price goes down
                self.liquidity_pool_a -= trade_amount
                self.liquidity_pool_b += trade_amount
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity
        else:  # outcome_b
            if trade_type == "buy":
                # Buy outcome B - price goes up
                self.liquidity_pool_b += trade_amount
                self.liquidity_pool_a -= trade_amount
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity
            else:
                # Sell outcome B - price goes down
                self.liquidity_pool_b -= trade_amount
                self.liquidity_pool_a += trade_amount
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity

    def calculate_quality_score(self):
        """Calculate market quality score based on various factors"""
        score = 0.0

        # Volume factor (30%)
        if self.volume_total > 0:
            volume_score = min(30, (self.volume_total / 10000) * 30)
            score += volume_score

        # Trader diversity factor (25%)
        if self.unique_traders > 0:
            diversity_score = min(25, (self.unique_traders / 100) * 25)
            score += diversity_score

        # Time factor (20%)
        time_remaining = self.time_until_close
        if time_remaining > 0:
            time_score = min(20, (time_remaining / (7 * 24 * 3600)) * 20)  # 7 days max
            score += time_score

        # Verification factor (15%)
        if self.is_verified:
            score += 15

        # Dispute factor (10%)
        if self.dispute_count == 0:
            score += 10
        else:
            score += max(0, 10 - (self.dispute_count * 2))

        self.market_quality_score = min(100, score)
        return self.market_quality_score

    def calculate_trending_score(self):
        """Calculate trending score based on recent activity"""
        from datetime import timedelta

        recent_trades = [
            t
            for t in self.trades
            if t.created_at >= datetime.utcnow() - timedelta(hours=24)
        ]
        recent_volume = sum(t.total_value for t in recent_trades)

        # Base score on 24h volume
        base_score = min(50, (recent_volume / 1000) * 50)

        # Bonus for high volume growth
        if self.volume_24h > self.volume_total * 0.1:  # 10% of total volume in 24h
            base_score += 25

        # Bonus for new markets
        if self.created_at >= datetime.utcnow() - timedelta(days=1):
            base_score += 15

        # Bonus for verified markets
        if self.is_verified:
            base_score += 10

        self.trending_score = min(100, base_score)
        return self.trending_score

    def validate_trade(self, amount: float, outcome: str) -> Dict[str, Any]:
        """Validate if a trade can be executed"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check if market is active
        if not self.is_active:
            validation_result["valid"] = False
            validation_result["errors"].append("Market is not active for trading")

        # Check amount constraints
        if amount < self.min_trade_amount:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Trade amount below minimum: ${self.min_trade_amount}")

        if amount > self.max_trade_amount:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Trade amount above maximum: ${self.max_trade_amount}")

        # Check liquidity
        if outcome == "outcome_a" and self.liquidity_pool_a < amount:
            validation_result["warnings"].append("Insufficient liquidity for outcome A")
        elif outcome == "outcome_b" and self.liquidity_pool_b < amount:
            validation_result["warnings"].append("Insufficient liquidity for outcome B")

        return validation_result

    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "category": self.category.value,
            "status": self.status.value,
            "current_prices": {
                "outcome_a": self.current_price_a,
                "outcome_b": self.current_price_b
            },
            "liquidity": {
                "total": self.total_liquidity,
                "pool_a": self.liquidity_pool_a,
                "pool_b": self.liquidity_pool_b
            },
            "volume": {
                "total": self.volume_total,
                "24h": self.volume_24h
            },
            "trading": {
                "total_trades": self.total_trades,
                "unique_traders": self.unique_traders,
                "min_trade": self.min_trade_amount,
                "max_trade": self.max_trade_amount,
                "fee_rate": self.fee_rate
            },
            "quality": {
                "score": self.market_quality_score,
                "trending_score": self.trending_score,
                "is_trending": self.is_trending,
                "is_verified": self.is_verified
            },
            "timing": {
                "created_at": self.created_at.isoformat(),
                "closes_at": self.closes_at.isoformat(),
                "time_until_close": self.time_until_close
            }
        }

    def calculate_implied_probability(self) -> Dict[str, float]:
        """Calculate implied probabilities from current prices"""
        return {
            "outcome_a": self.current_price_a,
            "outcome_b": self.current_price_b,
            "total": self.current_price_a + self.current_price_b
        }

    def get_trading_limits(self) -> Dict[str, float]:
        """Get trading limits and constraints"""
        return {
            "min_trade_amount": self.min_trade_amount,
            "max_trade_amount": self.max_trade_amount,
            "fee_rate": self.fee_rate,
            "available_liquidity_a": self.liquidity_pool_a,
            "available_liquidity_b": self.liquidity_pool_b
        }

    def is_ready_for_resolution(self) -> bool:
        """Check if market is ready for resolution"""
        return (
            self.status == MarketStatus.OPEN and 
            datetime.utcnow() >= self.closes_at
        )

    def can_be_disputed(self) -> bool:
        """Check if market can be disputed"""
        return (
            self.status == MarketStatus.RESOLVED and 
            self.dispute_count == 0 and
            datetime.utcnow() <= self.resolved_at + timedelta(days=7) if self.resolved_at else False
        )
