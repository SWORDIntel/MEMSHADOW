"""
Workflow Engine for ENUMERATE > PLAN > EXECUTE
FLUSTERCUCKER-inspired mission orchestration

Three-phase pentesting workflow:
- ENUMERATE: Network discovery, service detection, vulnerability scanning
- PLAN: Attack chain analysis, exploit selection, target prioritization
- EXECUTE: Automated exploitation, lateral movement, persistence
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

from app.services.swarm.coordinator import Coordinator
from app.services.swarm.mission import Mission
from app.services.swarm.blackboard import Blackboard
from app.services.vanta_blackwidow.tempest_logger import tempest_logger

logger = structlog.get_logger()


class WorkflowPhase:
    """Base class for workflow phases"""

    def __init__(self, name: str, blackboard: Blackboard):
        self.name = name
        self.blackboard = blackboard
        self.results = {}

    async def execute(self) -> Dict[str, Any]:
        """Execute phase - to be implemented by subclasses"""
        raise NotImplementedError


class EnumeratePhase(WorkflowPhase):
    """
    ENUMERATE Phase
    Network discovery, service detection, vulnerability scanning
    """

    def __init__(self, blackboard: Blackboard, targets: List[str]):
        super().__init__("ENUMERATE", blackboard)
        self.targets = targets

    async def execute(self) -> Dict[str, Any]:
        """
        Execute ENUMERATE phase

        Tasks:
        1. Network scanning (recon agent)
        2. WiFi discovery (wifi agent)
        3. Service fingerprinting
        4. Initial vulnerability detection
        """
        logger.info("Starting ENUMERATE phase", targets=len(self.targets))

        tempest_logger.audit(
            event_type="workflow_phase_start",
            action="enumerate_begin",
            resource="workflow_engine",
            status="started",
            details={"phase": "ENUMERATE", "targets": self.targets},
            severity="info"
        )

        results = {
            "phase": "ENUMERATE",
            "start_time": datetime.utcnow().isoformat(),
            "targets": self.targets,
            "discoveries": {
                "hosts": [],
                "services": [],
                "wifi_networks": [],
                "web_applications": []
            }
        }

        # Task 1: Network scanning
        for target in self.targets:
            task_id = f"enum_recon_{target}"
            self.blackboard.publish_task("recon", {
                "task_id": task_id,
                "agent_type": "recon",
                "params": {
                    "target_cidr": target,
                    "scan_type": "comprehensive",
                    "ports": "1-1000"
                }
            })

        # Task 2: WiFi discovery (if available)
        self.blackboard.publish_task("wifi", {
            "task_id": "enum_wifi_scan",
            "agent_type": "wifi",
            "params": {
                "task_type": "scan",
                "interface": "wlan0",
                "scan_time": 30,
                "show_hidden": True
            }
        })

        # Wait for results (with timeout)
        await self._collect_results(timeout=300)

        # Aggregate discoveries
        results["discoveries"]["hosts"] = self.blackboard.get_shared_state("known_hosts") or []
        results["discoveries"]["wifi_networks"] = self.blackboard.get_shared_state("wifi_networks") or []

        results["end_time"] = datetime.utcnow().isoformat()
        results["status"] = "completed"
        results["summary"] = {
            "hosts_found": len(results["discoveries"]["hosts"]),
            "wifi_networks_found": len(results["discoveries"]["wifi_networks"])
        }

        self.results = results

        tempest_logger.audit(
            event_type="workflow_phase_complete",
            action="enumerate_complete",
            resource="workflow_engine",
            status="success",
            details=results["summary"],
            severity="info"
        )

        logger.info("ENUMERATE phase completed", summary=results["summary"])
        return results

    async def _collect_results(self, timeout: int = 300):
        """Collect results from agents"""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            report = self.blackboard.get_report(timeout=5)

            if report:
                logger.debug("Received report in ENUMERATE", agent_id=report.get("agent_id"))

                # Process report
                if report.get("agent_type") == "recon":
                    # Store discovered hosts
                    hosts = report.get("data", {}).get("hosts", [])
                    if hosts:
                        existing = self.blackboard.get_shared_state("known_hosts") or []
                        existing.extend(hosts)
                        self.blackboard.update_shared_state("known_hosts", existing)

                elif report.get("agent_type") == "wifi":
                    # Store WiFi networks
                    networks = report.get("data", {}).get("networks", [])
                    if networks:
                        self.blackboard.update_shared_state("wifi_networks", networks)

            await asyncio.sleep(1)


class PlanPhase(WorkflowPhase):
    """
    PLAN Phase
    Attack chain analysis, exploit selection, target prioritization
    """

    def __init__(self, blackboard: Blackboard, enumerate_results: Dict[str, Any]):
        super().__init__("PLAN", blackboard)
        self.enumerate_results = enumerate_results

    async def execute(self) -> Dict[str, Any]:
        """
        Execute PLAN phase

        Tasks:
        1. Analyze discovered hosts and services
        2. Map vulnerabilities to exploits
        3. Prioritize targets by value/exposure
        4. Generate attack chains
        """
        logger.info("Starting PLAN phase")

        tempest_logger.audit(
            event_type="workflow_phase_start",
            action="plan_begin",
            resource="workflow_engine",
            status="started",
            details={"phase": "PLAN"},
            severity="info"
        )

        results = {
            "phase": "PLAN",
            "start_time": datetime.utcnow().isoformat(),
            "attack_chains": [],
            "priority_targets": [],
            "exploit_recommendations": []
        }

        # Analyze discovered hosts
        hosts = self.enumerate_results.get("discoveries", {}).get("hosts", [])

        for host in hosts:
            # Create attack chain for each host
            attack_chain = {
                "target": host,
                "priority": self._calculate_priority(host),
                "attack_vector": self._determine_attack_vector(host),
                "exploits": self._map_exploits(host)
            }
            results["attack_chains"].append(attack_chain)

        # Sort by priority
        results["attack_chains"].sort(key=lambda x: x["priority"], reverse=True)
        results["priority_targets"] = results["attack_chains"][:5]  # Top 5

        # Generate web scanning tasks for high-priority targets
        for target in results["priority_targets"]:
            if "http" in str(target.get("attack_vector", "")).lower():
                self.blackboard.publish_task("webscan", {
                    "task_id": f"plan_webscan_{target['target']}",
                    "agent_type": "webscan",
                    "params": {
                        "task_type": "scan",
                        "url": f"http://{target['target']}",
                        "scan_depth": "medium",
                        "cve_analysis": True
                    }
                })

        # Wait for web scan results
        await self._collect_web_scan_results(timeout=600)

        results["end_time"] = datetime.utcnow().isoformat()
        results["status"] = "completed"
        results["summary"] = {
            "attack_chains_generated": len(results["attack_chains"]),
            "priority_targets": len(results["priority_targets"]),
            "web_vulnerabilities_found": len(self.blackboard.get_list("web_vulnerabilities"))
        }

        self.results = results

        tempest_logger.audit(
            event_type="workflow_phase_complete",
            action="plan_complete",
            resource="workflow_engine",
            status="success",
            details=results["summary"],
            severity="info"
        )

        logger.info("PLAN phase completed", summary=results["summary"])
        return results

    def _calculate_priority(self, host: Dict[str, Any]) -> int:
        """Calculate target priority score"""
        score = 0

        # More open ports = higher priority
        score += len(host.get("ports", [])) * 10

        # Critical services = higher priority
        services = host.get("services", [])
        critical_services = ["ssh", "rdp", "mysql", "postgresql", "mssql", "oracle"]
        for service in services:
            if any(crit in service.lower() for crit in critical_services):
                score += 50

        # Web services = medium priority
        web_services = ["http", "https", "apache", "nginx"]
        for service in services:
            if any(web in service.lower() for web in web_services):
                score += 30

        return score

    def _determine_attack_vector(self, host: Dict[str, Any]) -> str:
        """Determine primary attack vector"""
        services = host.get("services", [])

        # Check for web services
        if any("http" in s.lower() for s in services):
            return "web_application"

        # Check for SSH
        if any("ssh" in s.lower() for s in services):
            return "ssh_bruteforce"

        # Check for SMB
        if any("smb" in s.lower() or "445" in str(s) for s in services):
            return "smb_exploitation"

        return "generic_network"

    def _map_exploits(self, host: Dict[str, Any]) -> List[str]:
        """Map potential exploits for host"""
        exploits = []

        services = host.get("services", [])

        # Example exploit mapping (simplified)
        for service in services:
            if "apache" in service.lower():
                exploits.append("apache_path_traversal")
            if "openssh" in service.lower():
                exploits.append("ssh_user_enumeration")
            if "mysql" in service.lower():
                exploits.append("mysql_udf_exploit")

        return exploits

    async def _collect_web_scan_results(self, timeout: int = 600):
        """Collect web scan results"""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            report = self.blackboard.get_report(timeout=5)

            if report and report.get("agent_type") == "webscan":
                vulnerabilities = report.get("data", {}).get("vulnerabilities", [])
                for vuln in vulnerabilities:
                    self.blackboard.append_to_list("web_vulnerabilities", vuln)

            await asyncio.sleep(1)


class ExecutePhase(WorkflowPhase):
    """
    EXECUTE Phase
    Automated exploitation, lateral movement, persistence
    """

    def __init__(self, blackboard: Blackboard, plan_results: Dict[str, Any], auto_execute: bool = False):
        super().__init__("EXECUTE", blackboard)
        self.plan_results = plan_results
        self.auto_execute = auto_execute

    async def execute(self) -> Dict[str, Any]:
        """
        Execute EXECUTE phase

        Tasks:
        1. Execute planned attack chains
        2. Attempt exploitation
        3. Lateral movement
        4. Post-exploitation
        """
        logger.info("Starting EXECUTE phase", auto_execute=self.auto_execute)

        tempest_logger.audit(
            event_type="workflow_phase_start",
            action="execute_begin",
            resource="workflow_engine",
            status="started",
            details={"phase": "EXECUTE", "auto": self.auto_execute},
            severity="warning"
        )

        results = {
            "phase": "EXECUTE",
            "start_time": datetime.utcnow().isoformat(),
            "exploits_attempted": [],
            "successful_compromises": [],
            "failed_attempts": []
        }

        # Safety check - require explicit confirmation unless auto_execute
        if not self.auto_execute:
            logger.warning("EXECUTE phase requires manual confirmation - skipping auto-execution")
            results["status"] = "skipped"
            results["message"] = "Manual confirmation required for EXECUTE phase"
            return results

        # Execute attack chains
        priority_targets = self.plan_results.get("priority_targets", [])

        for target in priority_targets:
            attack_vector = target.get("attack_vector")

            if attack_vector == "web_application":
                # Web exploitation
                task_id = f"exec_web_{target['target']}"
                self.blackboard.publish_task("authtest", {
                    "task_id": task_id,
                    "agent_type": "authtest",
                    "params": {
                        "target_host": target["target"],
                        "target_port": 80
                    }
                })

                results["exploits_attempted"].append({
                    "target": target["target"],
                    "method": "web_auth_bypass",
                    "task_id": task_id
                })

        # Wait for exploitation results
        await self._collect_exploitation_results(timeout=600)

        results["end_time"] = datetime.utcnow().isoformat()
        results["status"] = "completed"
        results["summary"] = {
            "exploits_attempted": len(results["exploits_attempted"]),
            "successful": len(results["successful_compromises"]),
            "failed": len(results["failed_attempts"])
        }

        self.results = results

        tempest_logger.audit(
            event_type="workflow_phase_complete",
            action="execute_complete",
            resource="workflow_engine",
            status="success",
            details=results["summary"],
            severity="warning"
        )

        logger.info("EXECUTE phase completed", summary=results["summary"])
        return results

    async def _collect_exploitation_results(self, timeout: int = 600):
        """Collect exploitation results"""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            report = self.blackboard.get_report(timeout=5)

            if report:
                status = report.get("status")
                if status == "completed":
                    self.results["successful_compromises"].append(report)
                elif status == "failed":
                    self.results["failed_attempts"].append(report)

            await asyncio.sleep(1)


class WorkflowEngine:
    """
    Main workflow engine orchestrating ENUMERATE > PLAN > EXECUTE
    """

    def __init__(self, targets: List[str], auto_execute: bool = False):
        self.targets = targets
        self.auto_execute = auto_execute
        self.blackboard = Blackboard()
        self.results = {
            "enumerate": {},
            "plan": {},
            "execute": {}
        }

    async def run_workflow(self) -> Dict[str, Any]:
        """
        Run complete ENUMERATE > PLAN > EXECUTE workflow
        """
        logger.info("Starting workflow engine", targets=self.targets, auto_execute=self.auto_execute)

        tempest_logger.audit(
            event_type="workflow_start",
            action="workflow_begin",
            resource="workflow_engine",
            status="started",
            details={
                "targets": self.targets,
                "auto_execute": self.auto_execute
            },
            severity="info"
        )

        # Phase 1: ENUMERATE
        enumerate_phase = EnumeratePhase(self.blackboard, self.targets)
        self.results["enumerate"] = await enumerate_phase.execute()

        # Phase 2: PLAN
        plan_phase = PlanPhase(self.blackboard, self.results["enumerate"])
        self.results["plan"] = await plan_phase.execute()

        # Phase 3: EXECUTE
        execute_phase = ExecutePhase(self.blackboard, self.results["plan"], self.auto_execute)
        self.results["execute"] = await execute_phase.execute()

        # Generate final report
        final_report = {
            "workflow": "ENUMERATE > PLAN > EXECUTE",
            "targets": self.targets,
            "auto_execute": self.auto_execute,
            "phases": self.results,
            "summary": {
                "hosts_discovered": self.results["enumerate"].get("summary", {}).get("hosts_found", 0),
                "attack_chains_generated": self.results["plan"].get("summary", {}).get("attack_chains_generated", 0),
                "exploits_attempted": self.results["execute"].get("summary", {}).get("exploits_attempted", 0),
                "successful_compromises": self.results["execute"].get("summary", {}).get("successful", 0)
            }
        }

        tempest_logger.audit(
            event_type="workflow_complete",
            action="workflow_end",
            resource="workflow_engine",
            status="success",
            details=final_report["summary"],
            severity="info"
        )

        logger.info("Workflow completed", summary=final_report["summary"])
        return final_report
