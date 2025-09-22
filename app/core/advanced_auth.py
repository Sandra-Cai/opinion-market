"""
Advanced Authentication System
Provides sophisticated authentication with JWT tokens, password security, and session management
"""

import jwt
import bcrypt
import secrets
import hashlib
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from collections import defaultdict, deque
import threading
import json

from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles"""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"
    BOT = "bot"

class TokenType(Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"
    API_KEY = "api_key"

class AuthStatus(Enum):
    """Authentication status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"
    PENDING_VERIFICATION = "pending_verification"

@dataclass
class User:
    """User model for authentication"""
    id: str
    username: str
    email: str
    password_hash: str
    roles: Set[UserRole] = field(default_factory=lambda: {UserRole.USER})
    status: AuthStatus = AuthStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    email_verified: bool = False
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Token:
    """Token model"""
    id: str
    user_id: str
    token_type: TokenType
    token_value: str
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """Session model"""
    id: str
    user_id: str
    session_token: str
    ip_address: str
    user_agent: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedAuthManager:
    """Advanced authentication manager"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", 
                 access_token_expire_minutes: int = 30, refresh_token_expire_days: int = 30):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # User management
        self.users: Dict[str, User] = {}
        self.users_by_email: Dict[str, str] = {}
        self.users_by_username: Dict[str, str] = {}
        
        # Token management
        self.tokens: Dict[str, Token] = {}
        self.user_tokens: Dict[str, Set[str]] = defaultdict(set)
        
        # Session management
        self.sessions: Dict[str, Session] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)
        
        # Security features
        self.login_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.password_history: Dict[str, List[str]] = defaultdict(list)
        self.max_password_history: int = 5
        
        # Rate limiting
        self.login_rate_limit: int = 5  # Max login attempts per minute
        self.password_reset_rate_limit: int = 3  # Max password reset attempts per hour
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_task = None
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # Event loop not running yet
            pass
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def generate_token(self, user_id: str, token_type: TokenType, 
                      expires_in: Optional[timedelta] = None) -> str:
        """Generate JWT token"""
        try:
            if expires_in is None:
                if token_type == TokenType.ACCESS:
                    expires_in = timedelta(minutes=self.access_token_expire_minutes)
                elif token_type == TokenType.REFRESH:
                    expires_in = timedelta(days=self.refresh_token_expire_days)
                else:
                    expires_in = timedelta(hours=1)
            
            payload = {
                "user_id": user_id,
                "token_type": token_type.value,
                "exp": datetime.utcnow() + expires_in,
                "iat": datetime.utcnow(),
                "jti": secrets.token_urlsafe(32)  # JWT ID
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
            
        except Exception as e:
            logger.error(f"Error generating token: {e}")
            raise
    
    def verify_token(self, token: str, token_type: Optional[TokenType] = None) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type if specified
            if token_type and payload.get("token_type") != token_type.value:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Check if token is revoked
            token_id = payload.get("jti")
            if token_id and token_id in self.tokens:
                token_obj = self.tokens[token_id]
                if token_obj.is_revoked:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Token has been revoked"
                    )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    async def register_user(self, username: str, email: str, password: str, 
                           roles: Optional[Set[UserRole]] = None) -> User:
        """Register a new user"""
        try:
            with self._lock:
                # Check if user already exists
                if email in self.users_by_email:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already registered"
                    )
                
                if username in self.users_by_username:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username already taken"
                    )
                
                # Validate password strength
                if not self._validate_password_strength(password):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Password does not meet security requirements"
                    )
                
                # Create user
                user_id = secrets.token_urlsafe(16)
                password_hash = self.hash_password(password)
                
                user = User(
                    id=user_id,
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    roles=roles or {UserRole.USER},
                    status=AuthStatus.PENDING_VERIFICATION
                )
                
                # Store user
                self.users[user_id] = user
                self.users_by_email[email] = user_id
                self.users_by_username[username] = user_id
                
                # Store password in history
                self.password_history[user_id].append(password_hash)
                
                logger.info(f"User registered: {username} ({email})")
                return user
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def authenticate_user(self, email: str, password: str, 
                               ip_address: str, user_agent: str) -> Dict[str, Any]:
        """Authenticate user and return tokens"""
        try:
            # Check rate limiting
            if not self._check_login_rate_limit(ip_address):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts"
                )
            
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="IP address is blocked"
                )
            
            with self._lock:
                # Find user by email
                if email not in self.users_by_email:
                    self._record_failed_login(ip_address)
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials"
                    )
                
                user_id = self.users_by_email[email]
                user = self.users[user_id]
                
                # Check if user is locked
                if user.locked_until and datetime.utcnow() < user.locked_until:
                    raise HTTPException(
                        status_code=status.HTTP_423_LOCKED,
                        detail="Account is temporarily locked"
                    )
                
                # Check user status
                if user.status == AuthStatus.BANNED:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Account is banned"
                    )
                
                # Verify password
                if not self.verify_password(password, user.password_hash):
                    self._record_failed_login(ip_address)
                    user.failed_login_attempts += 1
                    
                    # Lock account after too many failed attempts
                    if user.failed_login_attempts >= 5:
                        user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                        logger.warning(f"Account locked due to failed login attempts: {email}")
                    
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials"
                    )
                
                # Reset failed login attempts
                user.failed_login_attempts = 0
                user.locked_until = None
                user.last_login = datetime.utcnow()
                
                # Generate tokens
                access_token = self.generate_token(user_id, TokenType.ACCESS)
                refresh_token = self.generate_token(user_id, TokenType.REFRESH)
                
                # Create session
                session = await self._create_session(user_id, ip_address, user_agent)
                
                # Store tokens
                self._store_token(user_id, access_token, TokenType.ACCESS)
                self._store_token(user_id, refresh_token, TokenType.REFRESH)
                
                logger.info(f"User authenticated: {user.username} ({email})")
                
                return {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": self.access_token_expire_minutes * 60,
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "roles": [role.value for role in user.roles],
                        "status": user.status.value,
                        "email_verified": user.email_verified
                    },
                    "session_id": session.id
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, TokenType.REFRESH)
            user_id = payload["user_id"]
            
            with self._lock:
                if user_id not in self.users:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User not found"
                    )
                
                user = self.users[user_id]
                
                # Check user status
                if user.status in [AuthStatus.BANNED, AuthStatus.SUSPENDED]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Account is not active"
                    )
                
                # Generate new access token
                new_access_token = self.generate_token(user_id, TokenType.ACCESS)
                self._store_token(user_id, new_access_token, TokenType.ACCESS)
                
                return {
                    "access_token": new_access_token,
                    "token_type": "bearer",
                    "expires_in": self.access_token_expire_minutes * 60
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token refresh failed"
            )
    
    async def logout(self, user_id: str, session_id: Optional[str] = None):
        """Logout user and revoke tokens"""
        try:
            with self._lock:
                # Revoke all tokens for user
                if user_id in self.user_tokens:
                    for token_id in self.user_tokens[user_id].copy():
                        if token_id in self.tokens:
                            self.tokens[token_id].is_revoked = True
                
                # Deactivate session
                if session_id and session_id in self.sessions:
                    session = self.sessions[session_id]
                    if session.user_id == user_id:
                        session.is_active = False
                
                # Remove from user sessions
                if user_id in self.user_sessions:
                    for sid in self.user_sessions[user_id].copy():
                        if sid in self.sessions:
                            self.sessions[sid].is_active = False
                
                logger.info(f"User logged out: {user_id}")
                
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            with self._lock:
                if user_id not in self.users:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="User not found"
                    )
                
                user = self.users[user_id]
                
                # Verify old password
                if not self.verify_password(old_password, user.password_hash):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid old password"
                    )
                
                # Validate new password strength
                if not self._validate_password_strength(new_password):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="New password does not meet security requirements"
                    )
                
                # Check if new password is in history
                new_password_hash = self.hash_password(new_password)
                if new_password_hash in self.password_history[user_id]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="New password cannot be the same as a recent password"
                    )
                
                # Update password
                user.password_hash = new_password_hash
                
                # Add to password history
                self.password_history[user_id].append(new_password_hash)
                if len(self.password_history[user_id]) > self.max_password_history:
                    self.password_history[user_id] = self.password_history[user_id][-self.max_password_history:]
                
                # Revoke all existing tokens
                if user_id in self.user_tokens:
                    for token_id in self.user_tokens[user_id].copy():
                        if token_id in self.tokens:
                            self.tokens[token_id].is_revoked = True
                
                logger.info(f"Password changed for user: {user_id}")
                return True
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password change failed"
            )
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False
        
        if not any(c.isupper() for c in password):
            return False
        
        if not any(c.islower() for c in password):
            return False
        
        if not any(c.isdigit() for c in password):
            return False
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False
        
        return True
    
    def _check_login_rate_limit(self, ip_address: str) -> bool:
        """Check login rate limit for IP address"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old attempts
        self.login_attempts[ip_address] = [
            attempt for attempt in self.login_attempts[ip_address]
            if attempt > minute_ago
        ]
        
        # Check rate limit
        return len(self.login_attempts[ip_address]) < self.login_rate_limit
    
    def _record_failed_login(self, ip_address: str):
        """Record failed login attempt"""
        self.login_attempts[ip_address].append(datetime.utcnow())
        
        # Block IP after too many failed attempts
        if len(self.login_attempts[ip_address]) >= 10:
            self.blocked_ips.add(ip_address)
            logger.warning(f"IP address blocked due to failed login attempts: {ip_address}")
    
    def _store_token(self, user_id: str, token: str, token_type: TokenType):
        """Store token in memory"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            token_id = payload["jti"]
            
            token_obj = Token(
                id=token_id,
                user_id=user_id,
                token_type=token_type,
                token_value=token,
                expires_at=datetime.fromtimestamp(payload["exp"])
            )
            
            self.tokens[token_id] = token_obj
            self.user_tokens[user_id].add(token_id)
            
        except Exception as e:
            logger.error(f"Error storing token: {e}")
    
    async def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> Session:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        session_token = secrets.token_urlsafe(64)
        
        session = Session(
            id=session_id,
            user_id=user_id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        self.sessions[session_id] = session
        self.user_sessions[user_id].add(session_id)
        
        return session
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.utcnow()
                
                # Clean up expired tokens
                with self._lock:
                    expired_tokens = [
                        token_id for token_id, token in self.tokens.items()
                        if token.expires_at < now
                    ]
                    
                    for token_id in expired_tokens:
                        token = self.tokens[token_id]
                        self.user_tokens[token.user_id].discard(token_id)
                        del self.tokens[token_id]
                
                # Clean up expired sessions
                with self._lock:
                    expired_sessions = [
                        session_id for session_id, session in self.sessions.items()
                        if session.expires_at < now
                    ]
                    
                    for session_id in expired_sessions:
                        session = self.sessions[session_id]
                        self.user_sessions[session.user_id].discard(session_id)
                        del self.sessions[session_id]
                
                # Clean up old login attempts
                hour_ago = now - timedelta(hours=1)
                for ip_address in list(self.login_attempts.keys()):
                    self.login_attempts[ip_address] = [
                        attempt for attempt in self.login_attempts[ip_address]
                        if attempt > hour_ago
                    ]
                    
                    if not self.login_attempts[ip_address]:
                        del self.login_attempts[ip_address]
                
                logger.info("Authentication cleanup completed")
                
            except Exception as e:
                logger.error(f"Error in authentication cleanup: {e}")
                await asyncio.sleep(3600)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        user_id = self.users_by_email.get(email)
        return self.users.get(user_id) if user_id else None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        user_id = self.users_by_username.get(username)
        return self.users.get(user_id) if user_id else None
    
    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get user sessions"""
        with self._lock:
            if user_id not in self.user_sessions:
                return []
            
            return [
                self.sessions[session_id] for session_id in self.user_sessions[user_id]
                if session_id in self.sessions
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        with self._lock:
            return {
                "users": {
                    "total": len(self.users),
                    "active": len([u for u in self.users.values() if u.status == AuthStatus.ACTIVE]),
                    "pending_verification": len([u for u in self.users.values() if u.status == AuthStatus.PENDING_VERIFICATION]),
                    "suspended": len([u for u in self.users.values() if u.status == AuthStatus.SUSPENDED]),
                    "banned": len([u for u in self.users.values() if u.status == AuthStatus.BANNED])
                },
                "tokens": {
                    "total": len(self.tokens),
                    "active": len([t for t in self.tokens.values() if not t.is_revoked]),
                    "revoked": len([t for t in self.tokens.values() if t.is_revoked])
                },
                "sessions": {
                    "total": len(self.sessions),
                    "active": len([s for s in self.sessions.values() if s.is_active])
                },
                "security": {
                    "blocked_ips": len(self.blocked_ips),
                    "failed_login_attempts": sum(len(attempts) for attempts in self.login_attempts.values())
                }
            }

# Global authentication manager instance
auth_manager = AdvancedAuthManager(
    secret_key="your-secret-key-here",  # In production, use environment variable
    access_token_expire_minutes=30,
    refresh_token_expire_days=30
)

# Security scheme for FastAPI
security = HTTPBearer()

# Convenience functions
async def register_user(username: str, email: str, password: str, 
                       roles: Optional[Set[UserRole]] = None) -> User:
    """Register a new user"""
    return await auth_manager.register_user(username, email, password, roles)

async def authenticate_user(email: str, password: str, ip_address: str, user_agent: str) -> Dict[str, Any]:
    """Authenticate user"""
    return await auth_manager.authenticate_user(email, password, ip_address, user_agent)

async def refresh_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh access token"""
    return await auth_manager.refresh_token(refresh_token)

async def logout_user(user_id: str, session_id: Optional[str] = None):
    """Logout user"""
    await auth_manager.logout(user_id, session_id)

async def change_password(user_id: str, old_password: str, new_password: str) -> bool:
    """Change user password"""
    return await auth_manager.change_password(user_id, old_password, new_password)

def verify_token(token: str, token_type: Optional[TokenType] = None) -> Dict[str, Any]:
    """Verify JWT token"""
    return auth_manager.verify_token(token, token_type)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from token"""
    try:
        payload = verify_token(credentials.credentials, TokenType.ACCESS)
        user_id = payload["user_id"]
        user = auth_manager.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
