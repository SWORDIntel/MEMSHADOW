#!/usr/bin/env python3
"""
Ingest Plugin Framework for DSMIL Brain

Extensible system for data ingestion:
- Auto-discovery in ai/brain/plugins/ingest/
- Hot-reload without restart
- Schema validation
- Plugin lifecycle management
"""

import os
import importlib
import importlib.util
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Type
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PluginConfig:
    """Configuration for a plugin"""
    name: str
    enabled: bool = True
    priority: int = 100  # Lower = higher priority
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    """Result of data ingestion"""
    success: bool
    plugin_name: str

    # Data
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Stats
    items_ingested: int = 0
    bytes_processed: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IngestPlugin(ABC):
    """
    Base class for ingest plugins

    Subclass this to create new ingest plugins.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @property
    def description(self) -> str:
        """Plugin description"""
        return ""

    @property
    def supported_types(self) -> List[str]:
        """Data types this plugin can ingest"""
        return []

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin with config. Return True on success."""
        return True

    def shutdown(self):
        """Clean shutdown"""
        pass

    @abstractmethod
    def ingest(self, source: Any, **kwargs) -> IngestResult:
        """
        Ingest data from source

        Args:
            source: Data source (file path, URL, stream, etc.)
            **kwargs: Additional arguments

        Returns:
            IngestResult with ingested data
        """
        pass

    def validate(self, data: Any) -> bool:
        """Validate ingested data"""
        return True


class PluginManager:
    """
    Plugin Manager

    Manages plugin discovery, loading, and lifecycle.

    Usage:
        manager = PluginManager()

        # Discover plugins
        manager.discover_plugins()

        # Load specific plugin
        manager.load_plugin("file_ingest")

        # Ingest data
        result = manager.ingest("path/to/file", plugin="file_ingest")

        # Hot reload
        manager.reload_plugin("file_ingest")
    """

    def __init__(self, plugin_dir: Optional[str] = None):
        self.plugin_dir = plugin_dir or os.path.join(
            os.path.dirname(__file__), "ingest"
        )

        self._plugins: Dict[str, IngestPlugin] = {}
        self._plugin_configs: Dict[str, PluginConfig] = {}
        self._lock = threading.RLock()

        logger.info(f"PluginManager initialized (dir={self.plugin_dir})")

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directory"""
        discovered = []

        plugin_path = Path(self.plugin_dir)
        if not plugin_path.exists():
            logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            return discovered

        for file in plugin_path.glob("*.py"):
            if file.name.startswith("_"):
                continue

            plugin_name = file.stem
            discovered.append(plugin_name)
            logger.info(f"Discovered plugin: {plugin_name}")

        return discovered

    def load_plugin(self, plugin_name: str,
                   config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin by name"""
        with self._lock:
            try:
                plugin_path = os.path.join(self.plugin_dir, f"{plugin_name}.py")

                if not os.path.exists(plugin_path):
                    logger.error(f"Plugin file not found: {plugin_path}")
                    return False

                # Load module
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                if not spec or not spec.loader:
                    return False

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find plugin class
                plugin_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, IngestPlugin) and
                        attr is not IngestPlugin):
                        plugin_class = attr
                        break

                if not plugin_class:
                    logger.error(f"No IngestPlugin subclass found in {plugin_name}")
                    return False

                # Instantiate and initialize
                plugin = plugin_class()
                if not plugin.initialize(config or {}):
                    logger.error(f"Plugin initialization failed: {plugin_name}")
                    return False

                self._plugins[plugin_name] = plugin
                self._plugin_configs[plugin_name] = PluginConfig(
                    name=plugin_name,
                    config=config or {},
                )

                logger.info(f"Loaded plugin: {plugin_name} v{plugin.version}")
                return True

            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
                return False

    def unload_plugin(self, plugin_name: str):
        """Unload a plugin"""
        with self._lock:
            if plugin_name in self._plugins:
                try:
                    self._plugins[plugin_name].shutdown()
                except:
                    pass
                del self._plugins[plugin_name]
                self._plugin_configs.pop(plugin_name, None)
                logger.info(f"Unloaded plugin: {plugin_name}")

    def reload_plugin(self, plugin_name: str) -> bool:
        """Hot-reload a plugin"""
        with self._lock:
            config = self._plugin_configs.get(plugin_name, PluginConfig(name=plugin_name)).config
            self.unload_plugin(plugin_name)
            return self.load_plugin(plugin_name, config)

    def ingest(self, source: Any,
              plugin: Optional[str] = None,
              **kwargs) -> IngestResult:
        """
        Ingest data using appropriate plugin

        Args:
            source: Data source
            plugin: Specific plugin to use (or auto-detect)
            **kwargs: Additional arguments for plugin
        """
        with self._lock:
            if plugin:
                if plugin not in self._plugins:
                    return IngestResult(
                        success=False,
                        plugin_name=plugin,
                        errors=[f"Plugin not loaded: {plugin}"],
                    )
                return self._plugins[plugin].ingest(source, **kwargs)

            # Auto-detect plugin
            for name, p in sorted(
                self._plugins.items(),
                key=lambda x: self._plugin_configs.get(x[0], PluginConfig(name=x[0])).priority
            ):
                if self._plugin_configs.get(name, PluginConfig(name=name)).enabled:
                    try:
                        result = p.ingest(source, **kwargs)
                        if result.success:
                            return result
                    except:
                        continue

            return IngestResult(
                success=False,
                plugin_name="none",
                errors=["No suitable plugin found"],
            )

    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugins"""
        with self._lock:
            return list(self._plugins.keys())

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict]:
        """Get plugin information"""
        with self._lock:
            plugin = self._plugins.get(plugin_name)
            if not plugin:
                return None

            return {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "supported_types": plugin.supported_types,
            }

    def get_stats(self) -> Dict:
        """Get manager statistics"""
        with self._lock:
            return {
                "loaded_plugins": len(self._plugins),
                "plugin_names": list(self._plugins.keys()),
            }


# Built-in simple file plugin (for testing)
class SimpleFilePlugin(IngestPlugin):
    """Simple file ingest plugin"""

    @property
    def name(self) -> str:
        return "simple_file"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Simple file ingestion"

    @property
    def supported_types(self) -> List[str]:
        return ["file"]

    def ingest(self, source: Any, **kwargs) -> IngestResult:
        try:
            if isinstance(source, str) and os.path.isfile(source):
                with open(source, "rb") as f:
                    data = f.read()

                return IngestResult(
                    success=True,
                    plugin_name=self.name,
                    data=data,
                    metadata={
                        "path": source,
                        "size": len(data),
                    },
                    items_ingested=1,
                    bytes_processed=len(data),
                )

            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=["Invalid source"],
            )
        except Exception as e:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(e)],
            )


if __name__ == "__main__":
    print("Ingest Plugin Framework Self-Test")
    print("=" * 50)

    manager = PluginManager()

    print("\n[1] Discover Plugins")
    discovered = manager.discover_plugins()
    print(f"    Discovered: {discovered}")

    print("\n[2] Register Built-in Plugin")
    # Manually register for testing
    manager._plugins["simple_file"] = SimpleFilePlugin()
    manager._plugin_configs["simple_file"] = PluginConfig(name="simple_file")
    print("    Registered simple_file plugin")

    print("\n[3] Get Plugin Info")
    info = manager.get_plugin_info("simple_file")
    if info:
        print(f"    Name: {info['name']}")
        print(f"    Version: {info['version']}")
        print(f"    Types: {info['supported_types']}")

    print("\n[4] Test Ingest (this file)")
    result = manager.ingest(__file__, plugin="simple_file")
    print(f"    Success: {result.success}")
    print(f"    Bytes: {result.bytes_processed}")

    print("\n[5] Get Stats")
    stats = manager.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Ingest Plugin Framework test complete")

