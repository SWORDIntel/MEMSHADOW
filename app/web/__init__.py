"""
MEMSHADOW Web Interface
Comprehensive web UI for accessing and configuring all MEMSHADOW components
"""

from app.web.api import app
from app.web.config import ConfigManager

__all__ = ["app", "ConfigManager"]
