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
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
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
    creator_id = Column(Integer, nullable=False)

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
    verifier = relationship("User", foreign_keys=[verified_by])
    dispute_resolver = relationship("User", foreign_keys=[dispute_resolved_by])
    trades = relationship("Trade", back_populates="market")
    votes = relationship("Vote", back_populates="market")
    disputes = relationship("MarketDispute", back_populates="market")

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
