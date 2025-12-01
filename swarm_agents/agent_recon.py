"""
Network Reconnaissance Agent

Performs network scanning and service discovery using nmap.
Identifies hosts, open ports, and running services.
"""

import nmap
import uuid
from typing import Dict, Any, List
from base_agent import BaseAgent
import structlog

logger = structlog.get_logger()


class AgentRecon(BaseAgent):
    """
    Network reconnaissance agent for host and service discovery
    """

    def __init__(self, agent_id: str = None):
        agent_id = agent_id or f"recon-{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id=agent_id, agent_type="recon")

        self.nm = nmap.PortScanner()

        logger.info("Network reconnaissance agent initialized", agent_id=self.agent_id)

    async def execute_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute network reconnaissance

        Task Payload:
            - target_cidr: CIDR block to scan (e.g., "172.19.0.0/24")
            - scan_type: "quick" or "comprehensive" (default: "quick")
            - ports: Port range to scan (default: "1-1000")

        Returns:
            Dictionary with discovered hosts and services
        """
        target_cidr = task_payload.get('target_cidr', '127.0.0.1')
        scan_type = task_payload.get('scan_type', 'quick')
        ports = task_payload.get('ports', '1-1000')

        logger.info(
            "Starting network scan",
            agent_id=self.agent_id,
            target=target_cidr,
            scan_type=scan_type
        )

        try:
            # Choose scan arguments based on type
            if scan_type == "comprehensive":
                arguments = f"-sS -sV -O -p {ports}"  # SYN scan, version detection, OS detection
            else:
                arguments = f"-sT -p {ports}"  # TCP connect scan (doesn't require root)

            # Perform scan
            self.nm.scan(hosts=target_cidr, arguments=arguments)

            # Process results
            discovered_hosts = []
            known_hosts = []

            for host in self.nm.all_hosts():
                host_info = {
                    "host": host,
                    "hostname": self.nm[host].hostname(),
                    "state": self.nm[host].state(),
                    "open_ports": [],
                    "services": []
                }

                # Get open ports and services
                for proto in self.nm[host].all_protocols():
                    ports_list = self.nm[host][proto].keys()

                    for port in ports_list:
                        port_info = self.nm[host][proto][port]

                        if port_info['state'] == 'open':
                            host_info['open_ports'].append(port)

                            service_info = {
                                "port": port,
                                "protocol": proto,
                                "state": port_info['state'],
                                "service": port_info.get('name', 'unknown'),
                                "product": port_info.get('product', ''),
                                "version": port_info.get('version', '')
                            }

                            host_info['services'].append(service_info)

                discovered_hosts.append(host_info)

                # Add to known_hosts list
                known_hosts.append({
                    "host": host,
                    "ports": host_info['open_ports']
                })

            logger.info(
                "Network scan completed",
                agent_id=self.agent_id,
                hosts_discovered=len(discovered_hosts)
            )

            # Identify primary web service (host with port 80 or 8000)
            primary_web_service = None
            for host_info in discovered_hosts:
                if 80 in host_info['open_ports']:
                    primary_web_service = {"host": host_info['host'], "port": 80}
                    break
                elif 8000 in host_info['open_ports']:
                    primary_web_service = {"host": host_info['host'], "port": 8000}
                    break

            return {
                "discovered_hosts": discovered_hosts,
                "known_hosts": known_hosts,
                "num_hosts_found": len(discovered_hosts),
                "primary_web_service_host": primary_web_service['host'] if primary_web_service else None,
                "primary_web_service_port": primary_web_service['port'] if primary_web_service else None,
                "scan_summary": f"Scanned {target_cidr}, found {len(discovered_hosts)} hosts"
            }

        except Exception as e:
            logger.error(
                "Network scan failed",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            raise


if __name__ == "__main__":
    agent = AgentRecon()
    agent.run()
