"""
Admin schemas for system management
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from app.schemas.user import UserResponse
from app.schemas.market import MarketResponse
from app.schemas.trade import TradeResponse
from app.schemas.order import OrderResponse


class UserStats(BaseModel):
    """User statistics for admin dashboard"""
    total: int
    active: int
    verified: int
    premium: int
    new_24h: int
    new_7d: int
    new_30d: int


class MarketStats(BaseModel):
    """Market statistics for admin dashboard"""
    total: int
    open: int
    closed: int
    resolved: int
    top_categories: List[Dict[str, Any]]
    top_markets: List[Dict[str, Any]]


class TradeStats(BaseModel):
    """Trade statistics for admin dashboard"""
    total: int
    volume_total: float
    volume_24h: float
    volume_7d: float
    trades_24h: int


class OrderStats(BaseModel):
    """Order statistics for admin dashboard"""
    pending: int
    filled: int


class AdminStats(BaseModel):
    """Comprehensive admin statistics"""
    users: UserStats
    markets: MarketStats
    trades: TradeStats
    orders: OrderStats
    top_traders: List[Dict[str, Any]]


class AdminUserList(BaseModel):
    """Paginated list of users for admin"""
    users: List[UserResponse]
    total: int
    skip: int
    limit: int


class AdminMarketList(BaseModel):
    """Paginated list of markets for admin"""
    markets: List[MarketResponse]
    total: int
    skip: int
    limit: int


class AdminTradeList(BaseModel):
    """Paginated list of trades for admin"""
    trades: List[TradeResponse]
    total: int
    skip: int
    limit: int


class AdminOrderList(BaseModel):
    """Paginated list of orders for admin"""
    orders: List[OrderResponse]
    total: int
    skip: int
    limit: int


class UserModeration(BaseModel):
    """User moderation action"""
    user_id: int
    action: str
    reason: Optional[str] = None
    duration_days: Optional[int] = None
    admin_id: int
    timestamp: datetime


class MarketModeration(BaseModel):
    """Market moderation action"""
    market_id: int
    action: str
    reason: Optional[str] = None
    resolution_outcome: Optional[str] = None
    admin_id: int
    timestamp: datetime


class SystemHealth(BaseModel):
    """System health status"""
    status: str
    database: Dict[str, Any]
    redis: Dict[str, Any]
    cache: Dict[str, Any]
    timestamp: datetime


class SystemSettings(BaseModel):
    """System settings configuration"""
    app_name: str
    app_version: str
    environment: str
    debug: bool
    rate_limit_enabled: bool
    rate_limit_requests: int
    rate_limit_window: int
    caching_enabled: bool
    cache_ttl: int
    compression_enabled: bool
    websocket_enabled: bool
    ml_enabled: bool
    blockchain_enabled: bool
    monitoring_enabled: bool


class AuditLog(BaseModel):
    """Audit log entry"""
    id: int
    event_type: str
    user_id: Optional[int] = None
    admin_id: Optional[int] = None
    details: Dict[str, Any]
    timestamp: datetime


class AdminDashboard(BaseModel):
    """Admin dashboard data"""
    stats: AdminStats
    recent_activity: List[Dict[str, Any]]
    system_alerts: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class UserAnalytics(BaseModel):
    """User analytics data"""
    user_id: int
    username: str
    total_trades: int
    total_volume: float
    win_rate: float
    average_trade_size: float
    favorite_categories: List[str]
    risk_score: float
    activity_score: float
    last_activity: datetime


class MarketAnalytics(BaseModel):
    """Market analytics data"""
    market_id: int
    title: str
    total_volume: float
    unique_traders: int
    price_volatility: float
    liquidity_score: float
    sentiment_score: float
    trending_score: float
    completion_rate: float
    average_trade_size: float


class TradingAnalytics(BaseModel):
    """Trading analytics data"""
    total_trades: int
    total_volume: float
    average_trade_size: float
    trade_distribution: Dict[str, int]
    volume_by_category: Dict[str, float]
    volume_by_time: Dict[str, float]
    top_traders: List[Dict[str, Any]]
    top_markets: List[Dict[str, Any]]


class SystemMetrics(BaseModel):
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    database_connections: int
    redis_connections: int
    active_websockets: int
    request_rate: float
    response_time: float
    error_rate: float


class SecurityMetrics(BaseModel):
    """Security metrics"""
    blocked_ips: int
    suspicious_activities: int
    failed_logins: int
    rate_limit_hits: int
    security_events: int
    threat_level: str


class AdminNotification(BaseModel):
    """Admin notification"""
    id: int
    type: str
    title: str
    message: str
    severity: str
    timestamp: datetime
    read: bool
    action_required: bool


class AdminReport(BaseModel):
    """Admin report"""
    id: int
    title: str
    description: str
    report_type: str
    generated_by: int
    generated_at: datetime
    data: Dict[str, Any]
    file_path: Optional[str] = None


class AdminAction(BaseModel):
    """Admin action log"""
    id: int
    admin_id: int
    action_type: str
    target_type: str
    target_id: int
    details: Dict[str, Any]
    timestamp: datetime
    ip_address: str
    user_agent: str


class SystemBackup(BaseModel):
    """System backup information"""
    id: int
    backup_type: str
    file_path: str
    file_size: int
    created_at: datetime
    status: str
    description: Optional[str] = None


class AdminUserCreate(BaseModel):
    """Create user from admin panel"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    is_active: bool = True
    is_verified: bool = False
    is_premium: bool = False
    is_admin: bool = False


class AdminUserUpdate(BaseModel):
    """Update user from admin panel"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    is_premium: Optional[bool] = None
    is_admin: Optional[bool] = None


class AdminMarketCreate(BaseModel):
    """Create market from admin panel"""
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    question: str = Field(..., min_length=10, max_length=500)
    category: str
    outcome_a: str = Field(..., min_length=1, max_length=100)
    outcome_b: str = Field(..., min_length=1, max_length=100)
    creator_id: int
    closes_at: datetime
    resolution_criteria: Optional[str] = Field(None, max_length=1000)
    initial_liquidity: float = Field(1000.0, ge=0)
    trading_fee: float = Field(0.02, ge=0, le=0.1)


class AdminMarketUpdate(BaseModel):
    """Update market from admin panel"""
    title: Optional[str] = Field(None, min_length=5, max_length=200)
    description: Optional[str] = Field(None, min_length=10, max_length=2000)
    question: Optional[str] = Field(None, min_length=10, max_length=500)
    category: Optional[str] = None
    outcome_a: Optional[str] = Field(None, min_length=1, max_length=100)
    outcome_b: Optional[str] = Field(None, min_length=1, max_length=100)
    closes_at: Optional[datetime] = None
    resolution_criteria: Optional[str] = Field(None, max_length=1000)
    status: Optional[str] = None
    trading_fee: Optional[float] = Field(None, ge=0, le=0.1)


class AdminBulkAction(BaseModel):
    """Bulk action for admin operations"""
    action: str
    target_ids: List[int]
    parameters: Dict[str, Any] = {}


class AdminSearchResult(BaseModel):
    """Search result for admin panel"""
    type: str
    id: int
    title: str
    description: str
    created_at: datetime
    relevance_score: float


class AdminSearchResponse(BaseModel):
    """Search response for admin panel"""
    results: List[AdminSearchResult]
    total: int
    query: str
    search_time: float


class AdminExportRequest(BaseModel):
    """Export request for admin panel"""
    export_type: str
    format: str = "csv"
    filters: Dict[str, Any] = {}
    date_range: Optional[Dict[str, datetime]] = None


class AdminExportResponse(BaseModel):
    """Export response for admin panel"""
    export_id: str
    file_path: str
    file_size: int
    created_at: datetime
    expires_at: datetime


class AdminImportRequest(BaseModel):
    """Import request for admin panel"""
    import_type: str
    file_path: str
    options: Dict[str, Any] = {}


class AdminImportResponse(BaseModel):
    """Import response for admin panel"""
    import_id: str
    status: str
    processed: int
    successful: int
    failed: int
    errors: List[str] = []
    created_at: datetime


class AdminSystemInfo(BaseModel):
    """System information for admin panel"""
    version: str
    environment: str
    uptime: float
    database_version: str
    redis_version: str
    python_version: str
    dependencies: Dict[str, str]
    last_backup: Optional[datetime] = None
    disk_usage: Dict[str, Any]
    memory_usage: Dict[str, Any]
    cpu_usage: Dict[str, Any]


class AdminMaintenanceTask(BaseModel):
    """Maintenance task for admin panel"""
    id: int
    name: str
    description: str
    status: str
    progress: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AdminAlert(BaseModel):
    """System alert for admin panel"""
    id: int
    type: str
    severity: str
    title: str
    message: str
    source: str
    timestamp: datetime
    acknowledged: bool
    resolved: bool
    metadata: Dict[str, Any] = {}
