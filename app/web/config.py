"""
Configuration Manager
Handles system-wide configuration with persistence
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import structlog

logger = structlog.get_logger()


class SystemConfig:
    """System configuration container"""

    def __init__(self):
        self.config: Dict[str, Any] = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "system": {
                "name": "MEMSHADOW",
                "version": "1.0.0",
                "environment": "production"
            },

            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "reload": False
            },

            "security": {
                "secret_key": "CHANGE_ME_IN_PRODUCTION",
                "token_expiry_hours": 24,
                "require_authentication": True,
                "tempest_class": "C",  # TEMPEST security classification
                "audit_logging": True
            },

            "federated": {
                "enabled": False,
                "node_id": "memshadow_001",
                "privacy_budget": 1.0,
                "gossip_fanout": 3,
                "sync_interval_seconds": 60
            },

            "meta_learning": {
                "enabled": True,
                "inner_lr": 0.01,
                "outer_lr": 0.001,
                "num_inner_steps": 5,
                "enable_continual_learning": True,
                "ewc_lambda": 5000.0
            },

            "consciousness": {
                "enabled": False,
                "workspace_capacity": 7,  # Miller's Law: 7±2
                "attention_heads": 8,
                "enable_metacognition": True,
                "default_processing_mode": "hybrid",
                "confidence_threshold": 0.6
            },

            "self_modifying": {
                "enabled": False,
                "safety_level": "read_only",  # read_only, documentation, low_risk, medium_risk, full_access
                "enable_auto_apply": False,
                "require_tests": True,
                "minimum_test_coverage": 0.8
            },

            "ui": {
                "theme": "tempest_class_c",
                "show_advanced_options": True,
                "refresh_interval_ms": 1000,
                "show_notifications": True
            },

            "logging": {
                "level": "INFO",
                "format": "json",
                "output": "stdout"
            },

            "auto_start": {
                "federated": False,
                "consciousness": False,
                "self_modifying": False
            }
        }


class ConfigManager:
    """
    Configuration manager for MEMSHADOW web interface.

    Handles loading, saving, and updating system configuration.
    """

    def __init__(self, config_file: str = "config/memshadow.json"):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.system_config = SystemConfig()

        logger.info("Configuration manager initialized", config_file=str(self.config_file))

    def load_config(self):
        """Load configuration from file"""
        if not self.config_file.exists():
            logger.warning("Config file not found, using defaults", file=str(self.config_file))
            self.save_config()  # Create default config file
            return

        try:
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)

            # Merge with defaults (preserve defaults for missing keys)
            self._merge_config(self.system_config.config, loaded_config)

            logger.info("Configuration loaded", file=str(self.config_file))

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise

    def save_config(self):
        """Save configuration to file"""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w') as f:
                json.dump(self.system_config.config, f, indent=2)

            logger.info("Configuration saved", file=str(self.config_file))

        except Exception as e:
            logger.error("Failed to save config", error=str(e))
            raise

    def get_config(self, section: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration section.

        Args:
            section: Section name (e.g., "federated", "consciousness")

        Returns:
            Configuration dict or None if not found
        """
        return self.system_config.config.get(section)

    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.system_config.config.copy()

    def update_config(self, section: str, updates: Dict[str, Any]):
        """
        Update configuration section.

        Args:
            section: Section to update
            updates: New values
        """
        if section not in self.system_config.config:
            logger.warning(f"Creating new config section: {section}")
            self.system_config.config[section] = {}

        self.system_config.config[section].update(updates)

        logger.info("Configuration updated", section=section, updates=list(updates.keys()))

    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]):
        """Recursively merge loaded config into default, preserving structure"""
        for key, value in loaded.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
