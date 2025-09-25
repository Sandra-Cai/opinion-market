"""
Microservices Architecture Foundation
This module provides the base classes and utilities for microservices
"""

from .base_service import BaseService
from .service_registry import ServiceRegistry
from .service_discovery import ServiceDiscovery
from .inter_service_communication import InterServiceCommunication

__all__ = [
    "BaseService",
    "ServiceRegistry", 
    "ServiceDiscovery",
    "InterServiceCommunication"
]