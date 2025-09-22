"""
Role-Based Access Control (RBAC) System
Provides sophisticated permission management and access control
"""

from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import defaultdict

from fastapi import HTTPException, status, Depends
from app.core.advanced_auth import User, UserRole, get_current_user

logger = logging.getLogger(__name__)

class Permission(Enum):
    """System permissions"""
    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    LIST_USERS = "list_users"
    
    # Market management
    CREATE_MARKET = "create_market"
    READ_MARKET = "read_market"
    UPDATE_MARKET = "update_market"
    DELETE_MARKET = "delete_market"
    LIST_MARKETS = "list_markets"
    TRADE_MARKET = "trade_market"
    
    # Order management
    CREATE_ORDER = "create_order"
    READ_ORDER = "read_order"
    UPDATE_ORDER = "update_order"
    DELETE_ORDER = "delete_order"
    LIST_ORDERS = "list_orders"
    
    # Analytics and reporting
    READ_ANALYTICS = "read_analytics"
    EXPORT_DATA = "export_data"
    VIEW_REPORTS = "view_reports"
    
    # System administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_LOGS = "view_logs"
    MANAGE_CONFIG = "manage_config"
    MANAGE_ROLES = "manage_roles"
    
    # API management
    MANAGE_API_KEYS = "manage_api_keys"
    VIEW_API_USAGE = "view_api_usage"
    
    # Security
    MANAGE_SECURITY = "manage_security"
    VIEW_SECURITY_LOGS = "view_security_logs"
    MANAGE_BLOCKED_IPS = "manage_blocked_ips"

class Resource(Enum):
    """System resources"""
    USER = "user"
    MARKET = "market"
    ORDER = "order"
    TRADE = "trade"
    ANALYTICS = "analytics"
    SYSTEM = "system"
    LOGS = "logs"
    CONFIG = "config"
    ROLE = "role"
    API_KEY = "api_key"
    SECURITY = "security"

class Action(Enum):
    """Actions that can be performed on resources"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    EXECUTE = "execute"
    MANAGE = "manage"

@dataclass
class Role:
    """Role definition"""
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    resource_permissions: Dict[Resource, Set[Action]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_system_role: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Policy:
    """Access control policy"""
    id: str
    name: str
    description: str
    rules: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    effect: str = "allow"  # allow or deny
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessRequest:
    """Access request context"""
    user_id: str
    user_roles: Set[str]
    resource: str
    action: str
    resource_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class RBACManager:
    """Role-Based Access Control Manager"""
    
    def __init__(self):
        # Role management
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = defaultdict(set)
        
        # Permission management
        self.permissions: Dict[Permission, Dict[str, Any]] = {}
        self.resource_permissions: Dict[Resource, Dict[Action, Set[Permission]]] = defaultdict(lambda: defaultdict(set))
        
        # Policy management
        self.policies: Dict[str, Policy] = {}
        self.resource_policies: Dict[str, List[Policy]] = defaultdict(list)
        
        # Access control
        self.access_cache: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self.cache_ttl: timedelta = timedelta(minutes=5)
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default roles and permissions
        self._initialize_default_roles()
        self._initialize_default_permissions()
        self._initialize_default_policies()
    
    def _initialize_default_roles(self):
        """Initialize default system roles"""
        # Admin role
        admin_role = Role(
            name="admin",
            description="System administrator with full access",
            permissions=set(Permission),
            is_system_role=True
        )
        self.roles["admin"] = admin_role
        
        # Moderator role
        moderator_permissions = {
            Permission.READ_USER, Permission.UPDATE_USER, Permission.LIST_USERS,
            Permission.CREATE_MARKET, Permission.READ_MARKET, Permission.UPDATE_MARKET,
            Permission.DELETE_MARKET, Permission.LIST_MARKETS,
            Permission.READ_ORDER, Permission.LIST_ORDERS,
            Permission.READ_ANALYTICS, Permission.VIEW_REPORTS,
            Permission.VIEW_LOGS, Permission.MANAGE_SECURITY
        }
        moderator_role = Role(
            name="moderator",
            description="Moderator with limited administrative access",
            permissions=moderator_permissions,
            is_system_role=True
        )
        self.roles["moderator"] = moderator_role
        
        # User role
        user_permissions = {
            Permission.READ_USER, Permission.UPDATE_USER,
            Permission.CREATE_MARKET, Permission.READ_MARKET, Permission.LIST_MARKETS,
            Permission.TRADE_MARKET,
            Permission.CREATE_ORDER, Permission.READ_ORDER, Permission.UPDATE_ORDER,
            Permission.DELETE_ORDER, Permission.LIST_ORDERS,
            Permission.READ_ANALYTICS
        }
        user_role = Role(
            name="user",
            description="Regular user with standard access",
            permissions=user_permissions,
            is_system_role=True
        )
        self.roles["user"] = user_role
        
        # Guest role
        guest_permissions = {
            Permission.READ_MARKET, Permission.LIST_MARKETS,
            Permission.READ_ANALYTICS
        }
        guest_role = Role(
            name="guest",
            description="Guest user with read-only access",
            permissions=guest_permissions,
            is_system_role=True
        )
        self.roles["guest"] = guest_role
    
    def _initialize_default_permissions(self):
        """Initialize default permissions"""
        for permission in Permission:
            self.permissions[permission] = {
                "name": permission.value,
                "description": self._get_permission_description(permission),
                "resource": self._get_permission_resource(permission),
                "action": self._get_permission_action(permission)
            }
    
    def _initialize_default_policies(self):
        """Initialize default access control policies"""
        # Time-based access policy
        time_policy = Policy(
            id="time_based_access",
            name="Time-based Access Control",
            description="Restrict access during maintenance hours",
            rules=[
                {
                    "condition": "time.hour >= 2 and time.hour <= 4",
                    "effect": "deny",
                    "message": "Access restricted during maintenance hours"
                }
            ],
            conditions={
                "time_restriction": True,
                "maintenance_hours": [2, 3, 4]
            },
            effect="deny",
            priority=100
        )
        self.policies["time_based_access"] = time_policy
        
        # Rate limiting policy
        rate_limit_policy = Policy(
            id="rate_limiting",
            name="Rate Limiting Policy",
            description="Limit API access based on user role",
            rules=[
                {
                    "condition": "user.role == 'guest' and request.count > 100",
                    "effect": "deny",
                    "message": "Rate limit exceeded for guest users"
                }
            ],
            conditions={
                "rate_limiting": True,
                "guest_limit": 100,
                "user_limit": 1000,
                "admin_limit": 10000
            },
            effect="deny",
            priority=50
        )
        self.policies["rate_limiting"] = rate_limit_policy
    
    def _get_permission_description(self, permission: Permission) -> str:
        """Get permission description"""
        descriptions = {
            Permission.CREATE_USER: "Create new users",
            Permission.READ_USER: "Read user information",
            Permission.UPDATE_USER: "Update user information",
            Permission.DELETE_USER: "Delete users",
            Permission.LIST_USERS: "List all users",
            Permission.CREATE_MARKET: "Create new markets",
            Permission.READ_MARKET: "Read market information",
            Permission.UPDATE_MARKET: "Update market information",
            Permission.DELETE_MARKET: "Delete markets",
            Permission.LIST_MARKETS: "List all markets",
            Permission.TRADE_MARKET: "Trade in markets",
            Permission.CREATE_ORDER: "Create new orders",
            Permission.READ_ORDER: "Read order information",
            Permission.UPDATE_ORDER: "Update order information",
            Permission.DELETE_ORDER: "Delete orders",
            Permission.LIST_ORDERS: "List all orders",
            Permission.READ_ANALYTICS: "Read analytics data",
            Permission.EXPORT_DATA: "Export data",
            Permission.VIEW_REPORTS: "View reports",
            Permission.MANAGE_SYSTEM: "Manage system settings",
            Permission.VIEW_LOGS: "View system logs",
            Permission.MANAGE_CONFIG: "Manage configuration",
            Permission.MANAGE_ROLES: "Manage user roles",
            Permission.MANAGE_API_KEYS: "Manage API keys",
            Permission.VIEW_API_USAGE: "View API usage statistics",
            Permission.MANAGE_SECURITY: "Manage security settings",
            Permission.VIEW_SECURITY_LOGS: "View security logs",
            Permission.MANAGE_BLOCKED_IPS: "Manage blocked IP addresses"
        }
        return descriptions.get(permission, "Unknown permission")
    
    def _get_permission_resource(self, permission: Permission) -> Resource:
        """Get resource for permission"""
        resource_mapping = {
            Permission.CREATE_USER: Resource.USER,
            Permission.READ_USER: Resource.USER,
            Permission.UPDATE_USER: Resource.USER,
            Permission.DELETE_USER: Resource.USER,
            Permission.LIST_USERS: Resource.USER,
            Permission.CREATE_MARKET: Resource.MARKET,
            Permission.READ_MARKET: Resource.MARKET,
            Permission.UPDATE_MARKET: Resource.MARKET,
            Permission.DELETE_MARKET: Resource.MARKET,
            Permission.LIST_MARKETS: Resource.MARKET,
            Permission.TRADE_MARKET: Resource.MARKET,
            Permission.CREATE_ORDER: Resource.ORDER,
            Permission.READ_ORDER: Resource.ORDER,
            Permission.UPDATE_ORDER: Resource.ORDER,
            Permission.DELETE_ORDER: Resource.ORDER,
            Permission.LIST_ORDERS: Resource.ORDER,
            Permission.READ_ANALYTICS: Resource.ANALYTICS,
            Permission.EXPORT_DATA: Resource.ANALYTICS,
            Permission.VIEW_REPORTS: Resource.ANALYTICS,
            Permission.MANAGE_SYSTEM: Resource.SYSTEM,
            Permission.VIEW_LOGS: Resource.LOGS,
            Permission.MANAGE_CONFIG: Resource.CONFIG,
            Permission.MANAGE_ROLES: Resource.ROLE,
            Permission.MANAGE_API_KEYS: Resource.API_KEY,
            Permission.VIEW_API_USAGE: Resource.API_KEY,
            Permission.MANAGE_SECURITY: Resource.SECURITY,
            Permission.VIEW_SECURITY_LOGS: Resource.SECURITY,
            Permission.MANAGE_BLOCKED_IPS: Resource.SECURITY
        }
        return resource_mapping.get(permission, Resource.SYSTEM)
    
    def _get_permission_action(self, permission: Permission) -> Action:
        """Get action for permission"""
        action_mapping = {
            Permission.CREATE_USER: Action.CREATE,
            Permission.READ_USER: Action.READ,
            Permission.UPDATE_USER: Action.UPDATE,
            Permission.DELETE_USER: Action.DELETE,
            Permission.LIST_USERS: Action.LIST,
            Permission.CREATE_MARKET: Action.CREATE,
            Permission.READ_MARKET: Action.READ,
            Permission.UPDATE_MARKET: Action.UPDATE,
            Permission.DELETE_MARKET: Action.DELETE,
            Permission.LIST_MARKETS: Action.LIST,
            Permission.TRADE_MARKET: Action.EXECUTE,
            Permission.CREATE_ORDER: Action.CREATE,
            Permission.READ_ORDER: Action.READ,
            Permission.UPDATE_ORDER: Action.UPDATE,
            Permission.DELETE_ORDER: Action.DELETE,
            Permission.LIST_ORDERS: Action.LIST,
            Permission.READ_ANALYTICS: Action.READ,
            Permission.EXPORT_DATA: Action.EXECUTE,
            Permission.VIEW_REPORTS: Action.READ,
            Permission.MANAGE_SYSTEM: Action.MANAGE,
            Permission.VIEW_LOGS: Action.READ,
            Permission.MANAGE_CONFIG: Action.MANAGE,
            Permission.MANAGE_ROLES: Action.MANAGE,
            Permission.MANAGE_API_KEYS: Action.MANAGE,
            Permission.VIEW_API_USAGE: Action.READ,
            Permission.MANAGE_SECURITY: Action.MANAGE,
            Permission.VIEW_SECURITY_LOGS: Action.READ,
            Permission.MANAGE_BLOCKED_IPS: Action.MANAGE
        }
        return action_mapping.get(permission, Action.READ)
    
    def create_role(self, name: str, description: str, permissions: Set[Permission], 
                   is_system_role: bool = False) -> Role:
        """Create a new role"""
        with self._lock:
            if name in self.roles:
                raise ValueError(f"Role '{name}' already exists")
            
            role = Role(
                name=name,
                description=description,
                permissions=permissions,
                is_system_role=is_system_role
            )
            
            self.roles[name] = role
            logger.info(f"Role created: {name}")
            return role
    
    def update_role(self, name: str, description: Optional[str] = None, 
                   permissions: Optional[Set[Permission]] = None) -> Role:
        """Update an existing role"""
        with self._lock:
            if name not in self.roles:
                raise ValueError(f"Role '{name}' not found")
            
            role = self.roles[name]
            
            if description is not None:
                role.description = description
            
            if permissions is not None:
                role.permissions = permissions
            
            role.updated_at = datetime.utcnow()
            
            # Clear cache for users with this role
            self._clear_role_cache(name)
            
            logger.info(f"Role updated: {name}")
            return role
    
    def delete_role(self, name: str) -> bool:
        """Delete a role"""
        with self._lock:
            if name not in self.roles:
                return False
            
            role = self.roles[name]
            
            if role.is_system_role:
                raise ValueError(f"Cannot delete system role '{name}'")
            
            # Check if role is assigned to users
            if any(name in roles for roles in self.user_roles.values()):
                raise ValueError(f"Cannot delete role '{name}' - it is assigned to users")
            
            del self.roles[name]
            self._clear_role_cache(name)
            
            logger.info(f"Role deleted: {name}")
            return True
    
    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign role to user"""
        with self._lock:
            if role_name not in self.roles:
                raise ValueError(f"Role '{role_name}' not found")
            
            self.user_roles[user_id].add(role_name)
            self._clear_user_cache(user_id)
            
            logger.info(f"Role '{role_name}' assigned to user {user_id}")
            return True
    
    def remove_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Remove role from user"""
        with self._lock:
            if user_id in self.user_roles:
                self.user_roles[user_id].discard(role_name)
                self._clear_user_cache(user_id)
                
                logger.info(f"Role '{role_name}' removed from user {user_id}")
                return True
            
            return False
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get user roles"""
        return self.user_roles.get(user_id, set())
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get user permissions"""
        user_roles = self.get_user_roles(user_id)
        permissions = set()
        
        for role_name in user_roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                permissions.update(role.permissions)
        
        return permissions
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        cache_key = f"{user_id}:{permission.value}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.access_cache[user_id][permission.value]
        
        # Check permissions
        user_permissions = self.get_user_permissions(user_id)
        has_permission = permission in user_permissions
        
        # Cache result
        self._cache_access_result(user_id, permission.value, has_permission)
        
        return has_permission
    
    def can_access_resource(self, user_id: str, resource: Resource, action: Action, 
                           resource_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user can access resource with specific action"""
        # Check basic permissions first
        required_permission = self._get_permission_for_resource_action(resource, action)
        if required_permission and not self.has_permission(user_id, required_permission):
            return False
        
        # Check policies
        access_request = AccessRequest(
            user_id=user_id,
            user_roles=self.get_user_roles(user_id),
            resource=resource.value,
            action=action.value,
            resource_id=resource_id,
            context=context or {}
        )
        
        return self._evaluate_policies(access_request)
    
    def _get_permission_for_resource_action(self, resource: Resource, action: Action) -> Optional[Permission]:
        """Get permission required for resource action"""
        for permission, perm_info in self.permissions.items():
            if (perm_info["resource"] == resource and 
                perm_info["action"] == action):
                return permission
        return None
    
    def _evaluate_policies(self, request: AccessRequest) -> bool:
        """Evaluate access control policies"""
        # Get relevant policies
        relevant_policies = []
        
        for policy in self.policies.values():
            if not policy.is_active:
                continue
            
            # Check if policy applies to this request
            if self._policy_applies(policy, request):
                relevant_policies.append(policy)
        
        # Sort by priority (higher priority first)
        relevant_policies.sort(key=lambda p: p.priority, reverse=True)
        
        # Evaluate policies
        for policy in relevant_policies:
            if self._evaluate_policy(policy, request):
                return policy.effect == "allow"
        
        # Default to deny if no policies match
        return False
    
    def _policy_applies(self, policy: Policy, request: AccessRequest) -> bool:
        """Check if policy applies to request"""
        # Simple implementation - in production, you'd have more sophisticated matching
        return True
    
    def _evaluate_policy(self, policy: Policy, request: AccessRequest) -> bool:
        """Evaluate a specific policy"""
        # Simple implementation - in production, you'd have more sophisticated evaluation
        for rule in policy.rules:
            if self._evaluate_rule(rule, request):
                return True
        return False
    
    def _evaluate_rule(self, rule: Dict[str, Any], request: AccessRequest) -> bool:
        """Evaluate a specific rule"""
        # Simple implementation - in production, you'd have more sophisticated rule evaluation
        condition = rule.get("condition", "")
        
        # Basic condition evaluation
        if "user.role" in condition:
            # Check if user has required role
            required_role = condition.split("==")[1].strip().strip("'\"")
            return required_role in request.user_roles
        
        return True
    
    def create_policy(self, policy_id: str, name: str, description: str, 
                     rules: List[Dict[str, Any]], effect: str = "allow", 
                     priority: int = 0) -> Policy:
        """Create a new access control policy"""
        with self._lock:
            if policy_id in self.policies:
                raise ValueError(f"Policy '{policy_id}' already exists")
            
            policy = Policy(
                id=policy_id,
                name=name,
                description=description,
                rules=rules,
                effect=effect,
                priority=priority
            )
            
            self.policies[policy_id] = policy
            logger.info(f"Policy created: {policy_id}")
            return policy
    
    def update_policy(self, policy_id: str, **kwargs) -> Policy:
        """Update an existing policy"""
        with self._lock:
            if policy_id not in self.policies:
                raise ValueError(f"Policy '{policy_id}' not found")
            
            policy = self.policies[policy_id]
            
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            policy.updated_at = datetime.utcnow()
            
            # Clear relevant caches
            self._clear_policy_cache()
            
            logger.info(f"Policy updated: {policy_id}")
            return policy
    
    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy"""
        with self._lock:
            if policy_id not in self.policies:
                return False
            
            del self.policies[policy_id]
            self._clear_policy_cache()
            
            logger.info(f"Policy deleted: {policy_id}")
            return True
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        return datetime.utcnow() - self.cache_timestamps[cache_key] < self.cache_ttl
    
    def _cache_access_result(self, user_id: str, permission: str, result: bool):
        """Cache access result"""
        self.access_cache[user_id][permission] = result
        self.cache_timestamps[f"{user_id}:{permission}"] = datetime.utcnow()
    
    def _clear_user_cache(self, user_id: str):
        """Clear cache for user"""
        if user_id in self.access_cache:
            del self.access_cache[user_id]
        
        # Clear timestamp cache
        keys_to_remove = [key for key in self.cache_timestamps.keys() if key.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.cache_timestamps[key]
    
    def _clear_role_cache(self, role_name: str):
        """Clear cache for role"""
        # Clear cache for all users with this role
        for user_id, roles in self.user_roles.items():
            if role_name in roles:
                self._clear_user_cache(user_id)
    
    def _clear_policy_cache(self):
        """Clear all policy-related caches"""
        self.access_cache.clear()
        self.cache_timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RBAC statistics"""
        with self._lock:
            return {
                "roles": {
                    "total": len(self.roles),
                    "system_roles": len([r for r in self.roles.values() if r.is_system_role]),
                    "custom_roles": len([r for r in self.roles.values() if not r.is_system_role])
                },
                "permissions": {
                    "total": len(self.permissions),
                    "by_resource": {
                        resource.value: len([p for p in self.permissions.values() if p["resource"] == resource])
                        for resource in Resource
                    }
                },
                "policies": {
                    "total": len(self.policies),
                    "active": len([p for p in self.policies.values() if p.is_active])
                },
                "users": {
                    "with_roles": len(self.user_roles),
                    "total_role_assignments": sum(len(roles) for roles in self.user_roles.values())
                },
                "cache": {
                    "entries": sum(len(cache) for cache in self.access_cache.values()),
                    "timestamps": len(self.cache_timestamps)
                }
            }

# Global RBAC manager instance
rbac_manager = RBACManager()

# Convenience functions
def has_permission(user_id: str, permission: Permission) -> bool:
    """Check if user has permission"""
    return rbac_manager.has_permission(user_id, permission)

def can_access_resource(user_id: str, resource: Resource, action: Action, 
                       resource_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> bool:
    """Check if user can access resource"""
    return rbac_manager.can_access_resource(user_id, resource, action, resource_id, context)

def get_user_permissions(user_id: str) -> Set[Permission]:
    """Get user permissions"""
    return rbac_manager.get_user_permissions(user_id)

def get_user_roles(user_id: str) -> Set[str]:
    """Get user roles"""
    return rbac_manager.get_user_roles(user_id)

# FastAPI dependency functions
def require_permission(permission: Permission):
    """FastAPI dependency to require specific permission"""
    def permission_dependency(current_user: User = Depends(get_current_user)):
        if not has_permission(current_user.id, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )
        return current_user
    return permission_dependency

def require_role(role_name: str):
    """FastAPI dependency to require specific role"""
    def role_dependency(current_user: User = Depends(get_current_user)):
        user_roles = get_user_roles(current_user.id)
        if role_name not in user_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_name}' required"
            )
        return current_user
    return role_dependency

def require_resource_access(resource: Resource, action: Action):
    """FastAPI dependency to require resource access"""
    def resource_dependency(current_user: User = Depends(get_current_user)):
        if not can_access_resource(current_user.id, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access to {resource.value} with action {action.value} denied"
            )
        return current_user
    return resource_dependency
