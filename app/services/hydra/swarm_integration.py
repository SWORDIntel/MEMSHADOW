"""
HYDRA-SWARM Integration Module

Integrates the SWARM autonomous agent system with HYDRA's phased adversarial testing.

HYDRA Phases:
- Phase 1 (CRAWL): Static analysis and scanning
- Phase 2 (WALK): Scripted adversarial testing
- Phase 3 (RUN): Autonomous SWARM deployment

This module provides the bridge between HYDRA's testing framework and SWARM orchestration.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from app.services.swarm.coordinator import SwarmCoordinator
from app.services.swarm.blackboard import Blackboard
from app.services.swarm.mission import Mission, MissionLoader
from app.services.vanta_blackwidow.tempest_logger import tempest_logger

logger = structlog.get_logger()


class HYDRASwarmBridge:
    """
    Bridge between HYDRA adversarial suite and SWARM orchestrator
    """

    def __init__(self):
        self.blackboard = Blackboard()
        self.coordinator = SwarmCoordinator(blackboard=self.blackboard)

        self.phase_results = {
            'phase1': None,
            'phase2': None,
            'phase3': None
        }

        logger.info("HYDRA-SWARM bridge initialized")

    async def execute_phase3(
        self,
        target_system: str,
        mission: Optional[Mission] = None
    ) -> Dict[str, Any]:
        """
        Execute HYDRA Phase 3 (RUN) using SWARM autonomous agents

        Args:
            target_system: Target system identifier
            mission: Mission to execute (defaults to comprehensive mission)

        Returns:
            Phase 3 execution results
        """
        logger.info(
            "Executing HYDRA Phase 3 (RUN) - Autonomous SWARM",
            target=target_system
        )

        # Audit log
        tempest_logger.audit(
            event_type='hydra_phase3_start',
            action='execute_autonomous_swarm',
            resource=target_system,
            severity='info'
        )

        try:
            # Load mission if not provided
            if not mission:
                mission = MissionLoader.create_advanced_security_mission()
                logger.info("Using default advanced security mission")

            # Load mission into coordinator
            self.coordinator.load_mission(mission)

            # Execute mission
            report = await self.coordinator.execute_mission(timeout=7200)  # 2 hour timeout

            # Enrich report with HYDRA context
            report['hydra_phase'] = 'PHASE_3_RUN'
            report['target_system'] = target_system
            report['autonomous_execution'] = True

            # Store results
            self.phase_results['phase3'] = report

            # Audit log
            tempest_logger.audit(
                event_type='hydra_phase3_complete',
                action='autonomous_swarm_completed',
                resource=target_system,
                status='success' if report['status'] == 'completed' else 'partial',
                details={
                    'mission_id': mission.mission_id,
                    'completed_stages': len(report.get('completed_stages', [])),
                    'duration': report.get('duration_seconds')
                },
                severity='info'
            )

            return report

        except Exception as e:
            logger.error(
                "HYDRA Phase 3 execution failed",
                target=target_system,
                error=str(e)
            )

            tempest_logger.audit(
                event_type='hydra_phase3_error',
                action='autonomous_swarm_failed',
                resource=target_system,
                status='failure',
                details={'error': str(e)},
                severity='error'
            )

            raise

    async def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive HYDRA report combining all phases

        Returns:
            Complete HYDRA assessment report
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'phases': {
                'phase1_crawl': self.phase_results.get('phase1'),
                'phase2_walk': self.phase_results.get('phase2'),
                'phase3_run': self.phase_results.get('phase3')
            },
            'overall_status': self._calculate_overall_status(),
            'security_score': await self._calculate_security_score(),
            'recommendations': await self._generate_recommendations()
        }

    def _calculate_overall_status(self) -> str:
        """Calculate overall HYDRA assessment status"""
        phase3 = self.phase_results.get('phase3')

        if not phase3:
            return 'incomplete'

        if phase3.get('status') == 'completed':
            return 'passed'
        elif phase3.get('status') == 'failed':
            return 'failed'

        return 'partial'

    async def _calculate_security_score(self) -> Dict[str, Any]:
        """
        Calculate security score based on findings

        Returns:
            Security score details
        """
        phase3 = self.phase_results.get('phase3')

        if not phase3:
            return {'score': 0, 'grade': 'F', 'note': 'No results'}

        # Extract findings from blackboard
        discovered_data = phase3.get('discovered_data', {})

        # Count vulnerabilities
        num_vulns = len(discovered_data.get('vulnerabilities', []))
        num_critical = len(discovered_data.get('critical_issues', []))

        # Calculate score (100 base, deduct for findings)
        score = 100
        score -= num_critical * 20
        score -= num_vulns * 5

        score = max(0, min(100, score))

        # Grade
        if score >= 90:
            grade = 'A'
        elif score >= 80:
            grade = 'B'
        elif score >= 70:
            grade = 'C'
        elif score >= 60:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'score': score,
            'grade': grade,
            'vulnerabilities_found': num_vulns,
            'critical_issues': num_critical
        }

    async def _generate_recommendations(self) -> List[str]:
        """
        Generate security recommendations based on findings

        Returns:
            List of recommendations
        """
        recommendations = []
        phase3 = self.phase_results.get('phase3')

        if not phase3:
            return ['Complete HYDRA Phase 3 assessment']

        discovered_data = phase3.get('discovered_data', {})

        # Generic recommendations based on findings
        if discovered_data.get('open_ports'):
            recommendations.append("Review and restrict open network ports")

        if discovered_data.get('discovered_endpoints'):
            recommendations.append("Implement authentication on all API endpoints")

        if discovered_data.get('potential_vulnerabilities'):
            recommendations.append("Remediate identified vulnerabilities immediately")

        if not recommendations:
            recommendations.append("Maintain current security posture")

        return recommendations


# Global instance
hydra_swarm_bridge = HYDRASwarmBridge()
