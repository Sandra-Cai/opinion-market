"""
Advanced Configuration Management System
Provides hot reloading, validation, and environment-specific configurations
"""

import os
import yaml
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

logger = logging.getLogger(__name__)

class ConfigEnvironment(Enum):
    """Configuration environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ConfigRule:
    """Configuration validation rule"""
    field_name: str
    required: bool = False
    data_type: type = str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    custom_validator: Optional[Callable[[Any], bool]] = None

@dataclass
class ConfigChange:
    """Represents a configuration change"""
    timestamp: datetime
    field: str
    old_value: Any
    new_value: Any
    source: str

class ConfigFileHandler(FileSystemEventHandler):
    """Handles configuration file changes"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in ['.yaml', '.yml', '.json']:
            # Debounce rapid file changes
            current_time = datetime.now()
            if file_path in self.last_modified:
                if (current_time - self.last_modified[file_path]).total_seconds() < 1:
                    return
            
            self.last_modified[file_path] = current_time
            
            # Reload configuration
            asyncio.create_task(self.config_manager.reload_config())

class AdvancedConfigManager:
    """Advanced configuration management system"""
    
    def __init__(self, config_dir: str = "config", environment: str = "development"):
        self.config_dir = Path(config_dir)
        self.environment = ConfigEnvironment(environment)
        self.config_data: Dict[str, Any] = {}
        self.config_rules: List[ConfigRule] = []
        self.change_listeners: List[Callable[[ConfigChange], None]] = []
        self.change_history: List[ConfigChange] = []
        self.observer: Optional[Observer] = None
        self._lock = threading.RLock()
        self._initialized = False
        
        # Default configuration rules
        self._setup_default_rules()
        
        # Load initial configuration
        self.load_config()
    
    def _setup_default_rules(self):
        """Setup default configuration validation rules"""
        self.config_rules = [
            ConfigRule("database.url", required=True, data_type=str),
            ConfigRule("database.pool_size", data_type=int, min_value=1, max_value=100),
            ConfigRule("redis.url", required=True, data_type=str),
            ConfigRule("security.secret_key", required=True, data_type=str),
            ConfigRule("security.token_expire_minutes", data_type=int, min_value=1, max_value=1440),
            ConfigRule("api.rate_limit", data_type=int, min_value=1, max_value=10000),
            ConfigRule("logging.level", data_type=str, allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
            ConfigRule("monitoring.enabled", data_type=bool),
            ConfigRule("monitoring.interval_seconds", data_type=int, min_value=1, max_value=3600),
        ]
    
    def load_config(self) -> bool:
        """Load configuration from files"""
        try:
            with self._lock:
                # Load base configuration
                base_config = self._load_config_file("config.yaml") or {}
                
                # Load environment-specific configuration
                env_config = self._load_config_file(f"config.{self.environment.value}.yaml") or {}
                
                # Merge configurations (environment overrides base)
                self.config_data = {**base_config, **env_config}
                
                # Validate configuration
                if not self._validate_config():
                    logger.error("Configuration validation failed")
                    return False
                
                logger.info(f"Configuration loaded successfully for environment: {self.environment.value}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def _load_config_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a single configuration file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {e}")
            return None
    
    def _validate_config(self) -> bool:
        """Validate configuration against rules"""
        for rule in self.config_rules:
            value = self._get_nested_value(self.config_data, rule.field_name)
            
            # Check required fields
            if rule.required and value is None:
                logger.error(f"Required configuration field missing: {rule.field_name}")
                return False
            
            if value is not None:
                # Check data type
                if not isinstance(value, rule.data_type):
                    logger.error(f"Invalid data type for {rule.field_name}: expected {rule.data_type}, got {type(value)}")
                    return False
                
                # Check numeric ranges
                if rule.min_value is not None and isinstance(value, (int, float)) and value < rule.min_value:
                    logger.error(f"Value too small for {rule.field_name}: {value} < {rule.min_value}")
                    return False
                
                if rule.max_value is not None and isinstance(value, (int, float)) and value > rule.max_value:
                    logger.error(f"Value too large for {rule.field_name}: {value} > {rule.max_value}")
                    return False
                
                # Check allowed values
                if rule.allowed_values is not None and value not in rule.allowed_values:
                    logger.error(f"Invalid value for {rule.field_name}: {value} not in {rule.allowed_values}")
                    return False
                
                # Check regex pattern
                if rule.regex_pattern is not None and not re.match(rule.regex_pattern, str(value)):
                    logger.error(f"Value does not match pattern for {rule.field_name}: {value}")
                    return False
                
                # Check custom validator
                if rule.custom_validator is not None and not rule.custom_validator(value):
                    logger.error(f"Custom validation failed for {rule.field_name}: {value}")
                    return False
        
        return True
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = key_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        with self._lock:
            return self._get_nested_value(self.config_data, key) or default
    
    def set(self, key: str, value: Any, source: str = "programmatic") -> bool:
        """Set configuration value"""
        try:
            with self._lock:
                old_value = self._get_nested_value(self.config_data, key)
                
                # Set the value
                self._set_nested_value(self.config_data, key, value)
                
                # Validate the change
                if not self._validate_config():
                    # Revert the change
                    self._set_nested_value(self.config_data, key, old_value)
                    logger.error(f"Configuration validation failed for {key}: {value}")
                    return False
                
                # Record the change
                change = ConfigChange(
                    timestamp=datetime.now(),
                    field=key,
                    old_value=old_value,
                    new_value=value,
                    source=source
                )
                self.change_history.append(change)
                
                # Notify listeners
                for listener in self.change_listeners:
                    try:
                        listener(change)
                    except Exception as e:
                        logger.error(f"Error in configuration change listener: {e}")
                
                logger.info(f"Configuration updated: {key} = {value}")
                return True
                
        except Exception as e:
            logger.error(f"Error setting configuration {key}: {e}")
            return False
    
    def _set_nested_value(self, data: Dict[str, Any], key_path: str, value: Any):
        """Set nested value in dictionary using dot notation"""
        keys = key_path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def add_rule(self, rule: ConfigRule):
        """Add a configuration validation rule"""
        with self._lock:
            self.config_rules.append(rule)
    
    def add_change_listener(self, listener: Callable[[ConfigChange], None]):
        """Add a configuration change listener"""
        with self._lock:
            self.change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[ConfigChange], None]):
        """Remove a configuration change listener"""
        with self._lock:
            if listener in self.change_listeners:
                self.change_listeners.remove(listener)
    
    def start_file_watching(self):
        """Start watching configuration files for changes"""
        if self.observer is not None:
            return
        
        try:
            self.observer = Observer()
            event_handler = ConfigFileHandler(self)
            self.observer.schedule(event_handler, str(self.config_dir), recursive=False)
            self.observer.start()
            self._initialized = True
            logger.info("Configuration file watching started")
        except Exception as e:
            logger.error(f"Error starting configuration file watching: {e}")
    
    def stop_file_watching(self):
        """Stop watching configuration files"""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Configuration file watching stopped")
    
    async def reload_config(self):
        """Reload configuration from files"""
        logger.info("Reloading configuration...")
        if self.load_config():
            logger.info("Configuration reloaded successfully")
        else:
            logger.error("Configuration reload failed")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data"""
        with self._lock:
            return self.config_data.copy()
    
    def export_config(self, file_path: str, format: str = "yaml"):
        """Export current configuration to file"""
        try:
            with self._lock:
                with open(file_path, 'w') as f:
                    if format.lower() == 'json':
                        json.dump(self.config_data, f, indent=2)
                    else:
                        yaml.dump(self.config_data, f, default_flow_style=False)
                
                logger.info(f"Configuration exported to {file_path}")
                return True
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history"""
        with self._lock:
            return self.change_history[-limit:]
    
    def get_environment(self) -> ConfigEnvironment:
        """Get current environment"""
        return self.environment
    
    def set_environment(self, environment: str):
        """Set environment and reload configuration"""
        self.environment = ConfigEnvironment(environment)
        self.load_config()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        with self._lock:
            return {
                "environment": self.environment.value,
                "config_files": [f.name for f in self.config_dir.glob("*.yaml")],
                "total_fields": len(self._flatten_dict(self.config_data)),
                "validation_rules": len(self.config_rules),
                "change_listeners": len(self.change_listeners),
                "change_history_count": len(self.change_history),
                "file_watching_enabled": self.observer is not None
            }
    
    def _flatten_dict(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary"""
        result = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten_dict(value, new_key))
            else:
                result[new_key] = value
        return result

# Global configuration manager instance
advanced_config_manager = AdvancedConfigManager()

# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return advanced_config_manager.get(key, default)

def set_config(key: str, value: Any, source: str = "programmatic") -> bool:
    """Set configuration value"""
    return advanced_config_manager.set(key, value, source)

def add_config_rule(rule: ConfigRule):
    """Add configuration validation rule"""
    advanced_config_manager.add_rule(rule)

def add_config_listener(listener: Callable[[ConfigChange], None]):
    """Add configuration change listener"""
    advanced_config_manager.add_change_listener(listener)

def start_config_watching():
    """Start configuration file watching"""
    advanced_config_manager.start_file_watching()

def stop_config_watching():
    """Stop configuration file watching"""
    advanced_config_manager.stop_file_watching()
