"""
Sentinel Security Monitor

Real-time monitoring of SWARM operations and security events.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog

from app.services.swarm.blackboard import Blackboard
from app.services.vanta_blackwidow.tempest_logger import tempest_logger

logger = structlog.get_logger()


class SentinelMonitor:
    """
    Real-time security event monitor for MEMSHADOW
    """

    def __init__(self, blackboard: Blackboard = None):
        self.blackboard = blackboard or Blackboard()
        self.monitoring = False
        self.alert_thresholds = {
            'critical_vulns': 1,  # Alert on any critical vulnerability
            'failed_stages': 2,   # Alert if 2+ stages fail
            'high_threat_iocs': 5  # Alert on 5+ high-threat IoCs
        }

        logger.info("Sentinel monitor initialized")

    async def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring = True

        logger.info("Sentinel monitoring started")

        while self.monitoring:
            try:
                # Check for new reports
                report = self.blackboard.get_report(timeout=1)

                if report:
                    await self._process_report(report)

                # Check system health
                await self._check_system_health()

                await asyncio.sleep(5)

            except Exception as e:
                logger.error("Monitoring error", error=str(e))

    async def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("Sentinel monitoring stopped")

    async def _process_report(self, report: Dict[str, Any]):
        """
        Process an agent report

        Args:
            report: Agent report from blackboard
        """
        agent_id = report.get('agent_id')
        task_id = report.get('task_id')
        status = report.get('status')
        data = report.get('data', {})

        logger.info(
            "Processing agent report",
            agent_id=agent_id,
            task_id=task_id,
            status=status
        )

        # Check for critical findings
        if 'critical' in str(data).lower():
            await self._raise_alert(
                severity='critical',
                title='Critical Finding Detected',
                description=f"Agent {agent_id} reported critical findings",
                details=data
            )

        # Check for vulnerabilities
        vulns = data.get('vulnerabilities', [])
        if len(vulns) >= self.alert_thresholds['critical_vulns']:
            await self._raise_alert(
                severity='high',
                title='Vulnerabilities Detected',
                description=f"Found {len(vulns)} vulnerabilities",
                details={'vulnerabilities': vulns}
            )

        # Audit log
        tempest_logger.audit(
            event_type='agent_report_processed',
            action='process_swarm_report',
            resource=agent_id,
            status=status,
            details={'task_id': task_id},
            severity='info'
        )

    async def _check_system_health(self):
        """Check overall system health"""
        # Check blackboard connection
        try:
            self.blackboard.redis_client.ping()
        except Exception as e:
            await self._raise_alert(
                severity='critical',
                title='Blackboard Connection Failed',
                description='Redis connection lost',
                details={'error': str(e)}
            )

    async def _raise_alert(
        self,
        severity: str,
        title: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Raise a security alert

        Args:
            severity: Alert severity
            title: Alert title
            description: Alert description
            details: Additional details
        """
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'severity': severity,
            'title': title,
            'description': description,
            'details': details or {},
            'source': 'sentinel_monitor'
        }

        # Store alert
        self.blackboard.append_to_list('alerts', alert)

        # Log
        logger.warning(
            f"SENTINEL ALERT [{severity.upper()}]",
            title=title,
            description=description
        )

        # Audit
        tempest_logger.audit(
            event_type='security_alert',
            action='raise_alert',
            resource='system',
            status='alert_raised',
            details=alert,
            severity=severity
        )

    async def get_active_alerts(self, time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get active alerts within time window

        Args:
            time_window_hours: Time window in hours

        Returns:
            List of active alerts
        """
        all_alerts = self.blackboard.get_list('alerts')

        # Filter by time window
        cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
        active_alerts = []

        for alert in all_alerts:
            if isinstance(alert, dict):
                alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
                if alert_time >= cutoff:
                    active_alerts.append(alert)

        return active_alerts

    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for dashboard display

        Returns:
            Dashboard metrics
        """
        alerts = await self.get_active_alerts(time_window_hours=24)

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'alerts': {
                'total': len(alerts),
                'critical': len([a for a in alerts if a.get('severity') == 'critical']),
                'high': len([a for a in alerts if a.get('severity') == 'high']),
                'medium': len([a for a in alerts if a.get('severity') == 'medium']),
                'low': len([a for a in alerts if a.get('severity') == 'low'])
            },
            'system_health': {
                'blackboard': 'healthy',  # Check actual health
                'agents': self._get_agent_health(),
                'coordinator': 'active'
            },
            'recent_events': alerts[:10]  # Last 10 alerts
        }

    def _get_agent_health(self) -> Dict[str, str]:
        """Get health status of all agents"""
        # Check task queues
        agent_types = ['recon', 'apimapper', 'authtest', 'bgp', 'blockchain', 'crawler']

        health = {}
        for agent_type in agent_types:
            pending = self.blackboard.get_task_count(agent_type)
            health[agent_type] = 'active' if pending >= 0 else 'unknown'

        return health


# Global instance
sentinel_monitor = SentinelMonitor()
