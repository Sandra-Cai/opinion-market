from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    bio = Column(Text)
    avatar_url = Column(String)

    # Trading stats (like Polymarket)
    total_trades = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    total_profit = Column(Float, default=0.0)
    total_volume = Column(Float, default=0.0)  # Total trading volume
    reputation_score = Column(Float, default=100.0)

    # Portfolio tracking
    portfolio_value = Column(Float, default=0.0)  # Current portfolio value
    available_balance = Column(Float, default=1000.0)  # Available cash
    total_invested = Column(Float, default=0.0)  # Total amount invested

    # Advanced stats
    win_rate = Column(Float, default=0.0)  # Win rate percentage
    avg_trade_size = Column(Float, default=0.0)  # Average trade size
    largest_win = Column(Float, default=0.0)  # Largest single win
    largest_loss = Column(Float, default=0.0)  # Largest single loss

    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)  # Premium user features

    # Preferences
    preferences = Column(JSON, default=dict)  # User preferences
    notification_settings = Column(JSON, default=dict)  # Notification settings

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    markets_created = relationship("Market", foreign_keys="Market.creator_id", back_populates="creator")
    trades = relationship("Trade", back_populates="user")
    votes = relationship("Vote", back_populates="user")

    @property
    def success_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.successful_trades / self.total_trades) * 100

    @property
    def total_balance(self) -> float:
        """Total balance including portfolio value and available cash"""
        return self.available_balance + self.portfolio_value

    @property
    def profit_loss_percentage(self) -> float:
        """Calculate profit/loss percentage"""
        if self.total_invested == 0:
            return 0.0
        return ((self.total_balance - self.total_invested) / self.total_invested) * 100

    def update_stats(self, trade_amount: float, profit: float = 0.0):
        """Update user stats after a trade"""
        self.total_trades += 1
        self.total_volume += trade_amount
        self.total_profit += profit

        if profit > 0:
            self.successful_trades += 1
            if profit > self.largest_win:
                self.largest_win = profit
        elif profit < 0 and abs(profit) > self.largest_loss:
            self.largest_loss = abs(profit)

        # Update averages
        if self.total_trades > 0:
            self.avg_trade_size = self.total_volume / self.total_trades
            self.win_rate = (self.successful_trades / self.total_trades) * 100

        # Update reputation based on performance
        if self.total_trades > 10:
            if self.win_rate > 60:
                self.reputation_score = min(1000.0, self.reputation_score + 1)
            elif self.win_rate < 40:
                self.reputation_score = max(0.0, self.reputation_score - 1)
