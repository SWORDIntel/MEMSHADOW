"""
Sentinel - Real-time Security Monitoring and Alerting

Monitors SWARM operations, HYDRA testing, and system security events.
Provides real-time alerting and dashboarding capabilities.
"""

from .monitor import SentinelMonitor
from .alerts import AlertManager
from .metrics import MetricsCollector

__all__ = ['SentinelMonitor', 'AlertManager', 'MetricsCollector']
