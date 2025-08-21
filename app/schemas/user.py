from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    avatar_url: Optional[str] = None
    total_trades: int
    successful_trades: int
    total_profit: float
    total_volume: float
    reputation_score: float
    success_rate: float
    win_rate: float
    avg_trade_size: float
    largest_win: float
    largest_loss: float
    portfolio_value: float
    available_balance: float
    total_invested: float
    total_balance: float
    profit_loss_percentage: float
    is_active: bool
    is_verified: bool
    is_premium: bool
    preferences: Dict[str, Any]
    notification_settings: Dict[str, Any]
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserStats(BaseModel):
    total_users: int
    active_users_24h: int
    total_volume_all_time: float
    top_traders: List[UserResponse]
