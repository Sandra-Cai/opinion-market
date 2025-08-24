from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.user import User
from app.models.trade import Trade

class RateLimiter:
    """Rate limiting system for API endpoints"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.limits = {
            "auth": {"requests": 5, "window": 300},  # 5 requests per 5 minutes
            "trades": {"requests": 100, "window": 60},  # 100 trades per minute
            "orders": {"requests": 50, "window": 60},  # 50 orders per minute
            "default": {"requests": 1000, "window": 60}  # 1000 requests per minute
        }
    
    def is_allowed(self, client_id: str, endpoint: str) -> bool:
        """Check if request is allowed based on rate limits"""
        now = time.time()
        limit_config = self.limits.get(endpoint, self.limits["default"])
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < limit_config["window"]
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= limit_config["requests"]:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining_requests(self, client_id: str, endpoint: str) -> int:
        """Get remaining requests for client"""
        now = time.time()
        limit_config = self.limits.get(endpoint, self.limits["default"])
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < limit_config["window"]
        ]
        
        return max(0, limit_config["requests"] - len(self.requests[client_id]))

class FraudDetector:
    """Fraud detection system for suspicious trading activity"""
    
    def __init__(self):
        self.suspicious_patterns = []
        self.user_risk_scores = defaultdict(float)
    
    def analyze_trade(self, trade: Trade, user: User) -> Dict[str, any]:
        """Analyze a trade for suspicious activity"""
        risk_factors = []
        risk_score = 0.0
        
        # Check for unusual trade size
        if trade.total_value > user.avg_trade_size * 10:
            risk_factors.append("unusual_trade_size")
            risk_score += 0.3
        
        # Check for rapid trading
        recent_trades = self._get_recent_trades(user.id, minutes=5)
        if len(recent_trades) > 20:
            risk_factors.append("rapid_trading")
            risk_score += 0.4
        
        # Check for wash trading (buying and selling same market rapidly)
        if self._detect_wash_trading(user.id, trade.market_id):
            risk_factors.append("wash_trading")
            risk_score += 0.8
        
        # Check for price manipulation
        if self._detect_price_manipulation(trade):
            risk_factors.append("price_manipulation")
            risk_score += 0.6
        
        # Check user reputation
        if user.reputation_score < 100:
            risk_factors.append("low_reputation")
            risk_score += 0.2
        
        # Update user risk score
        self.user_risk_scores[user.id] = min(1.0, self.user_risk_scores[user.id] + risk_score)
        
        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "is_suspicious": risk_score > 0.5,
            "user_risk_score": self.user_risk_scores[user.id]
        }
    
    def _get_recent_trades(self, user_id: int, minutes: int) -> List[Trade]:
        """Get recent trades for a user"""
        db = SessionLocal()
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            return db.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.created_at >= cutoff_time
            ).all()
        finally:
            db.close()
    
    def _detect_wash_trading(self, user_id: int, market_id: int) -> bool:
        """Detect wash trading patterns"""
        db = SessionLocal()
        try:
            recent_trades = db.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.market_id == market_id,
                Trade.created_at >= datetime.utcnow() - timedelta(minutes=10)
            ).order_by(Trade.created_at).all()
            
            if len(recent_trades) < 4:
                return False
            
            # Check for alternating buy/sell pattern
            for i in range(len(recent_trades) - 1):
                if recent_trades[i].trade_type == recent_trades[i + 1].trade_type:
                    return False
            
            return True
        finally:
            db.close()
    
    def _detect_price_manipulation(self, trade: Trade) -> bool:
        """Detect potential price manipulation"""
        # Check if trade size is unusually large compared to market liquidity
        if trade.total_value > trade.market.total_liquidity * 0.1:  # More than 10% of liquidity
            return True
        
        # Check for unusual price impact
        if trade.price_impact > 0.05:  # More than 5% price impact
            return True
        
        return False

class SecurityMonitor:
    """Security monitoring and alerting system"""
    
    def __init__(self):
        self.alerts = []
        self.suspicious_activities = []
        self.blocked_ips = set()
        self.rate_limiter = RateLimiter()
        self.fraud_detector = FraudDetector()
    
    def monitor_request(self, request: Request, user: Optional[User] = None) -> Dict[str, any]:
        """Monitor and analyze incoming request"""
        client_ip = request.client.host
        endpoint = request.url.path.split('/')[-1]
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="IP address is blocked"
            )
        
        # Check rate limits
        if not self.rate_limiter.is_allowed(client_ip, endpoint):
            self._create_alert("rate_limit_exceeded", {
                "ip": client_ip,
                "endpoint": endpoint,
                "user_id": user.id if user else None
            })
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Check for suspicious patterns
        if user:
            suspicious_score = self._check_suspicious_patterns(request, user)
            if suspicious_score > 0.7:
                self._create_alert("suspicious_activity", {
                    "ip": client_ip,
                    "user_id": user.id,
                    "score": suspicious_score,
                    "endpoint": endpoint
                })
        
        return {
            "allowed": True,
            "remaining_requests": self.rate_limiter.get_remaining_requests(client_ip, endpoint),
            "user_risk_score": self.fraud_detector.user_risk_scores.get(user.id, 0.0) if user else 0.0
        }
    
    def monitor_trade(self, trade: Trade, user: User) -> Dict[str, any]:
        """Monitor trading activity for fraud"""
        fraud_analysis = self.fraud_detector.analyze_trade(trade, user)
        
        if fraud_analysis["is_suspicious"]:
            self._create_alert("suspicious_trade", {
                "trade_id": trade.id,
                "user_id": user.id,
                "risk_score": fraud_analysis["risk_score"],
                "risk_factors": fraud_analysis["risk_factors"]
            })
        
        return fraud_analysis
    
    def _check_suspicious_patterns(self, request: Request, user: User) -> float:
        """Check for suspicious request patterns"""
        score = 0.0
        
        # Check user agent
        user_agent = request.headers.get("user-agent", "")
        if not user_agent or "bot" in user_agent.lower():
            score += 0.3
        
        # Check for unusual request patterns
        if user.reputation_score < 50:
            score += 0.2
        
        # Check request frequency
        if self.rate_limiter.get_remaining_requests(request.client.host, "default") < 10:
            score += 0.3
        
        return score
    
    def _create_alert(self, alert_type: str, data: Dict):
        """Create a security alert"""
        alert = {
            "type": alert_type,
            "timestamp": datetime.utcnow(),
            "data": data,
            "severity": "high" if alert_type in ["suspicious_trade", "price_manipulation"] else "medium"
        }
        
        self.alerts.append(alert)
        
        # Log alert
        print(f"🚨 SECURITY ALERT: {alert_type} - {json.dumps(data)}")
    
    def block_ip(self, ip: str, reason: str):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        self._create_alert("ip_blocked", {"ip": ip, "reason": reason})
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Get recent security alerts"""
        return sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_user_risk_score(self, user_id: int) -> float:
        """Get risk score for a user"""
        return self.fraud_detector.user_risk_scores.get(user_id, 0.0)

# Global security monitor instance
security_monitor = SecurityMonitor()

def get_security_monitor() -> SecurityMonitor:
    """Get the global security monitor instance"""
    return security_monitor


