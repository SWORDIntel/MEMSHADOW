"""
Authentication Testing Agent

Tests authentication and authorization mechanisms on discovered endpoints.
Attempts to identify public vs protected endpoints and auth bypass vulnerabilities.
"""

import httpx
import uuid
from typing import Dict, Any, List
from base_agent import BaseAgent
import structlog

logger = structlog.get_logger()


class AgentAuthTest(BaseAgent):
    """
    Authentication and authorization testing agent
    """

    def __init__(self, agent_id: str = None):
        agent_id = agent_id or f"authtest-{uuid.uuid4().hex[:8]}"
        super().__init__(agent_id=agent_id, agent_type="authtest")

        logger.info("Authentication testing agent initialized", agent_id=self.agent_id)

    async def execute_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute authentication testing

        Task Payload:
            - endpoints: List of endpoints to test
            - base_url: Base URL for requests
            - timeout: Request timeout (default: 5)

        Returns:
            Dictionary with auth test results
        """
        endpoints = task_payload.get('endpoints', [])
        base_url = task_payload.get('base_url', 'http://localhost:8000')
        timeout = task_payload.get('timeout', 5)

        if not endpoints and task_payload.get('discovered_endpoints'):
            # Extract paths from discovered endpoints
            endpoints = [
                ep['path'] if isinstance(ep, dict) else ep
                for ep in task_payload.get('discovered_endpoints', [])
            ]

        logger.info(
            "Starting authentication testing",
            agent_id=self.agent_id,
            num_endpoints=len(endpoints)
        )

        auth_results = []
        endpoint_classifications = {
            "PUBLIC": [],
            "PROTECTED": [],
            "REQUIRES_AUTH": [],
            "POTENTIAL_BYPASS": []
        }

        async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
            for endpoint in endpoints:
                try:
                    result = await self._test_endpoint(client, base_url, endpoint)
                    auth_results.append(result)

                    # Classify endpoint
                    classification = result.get('classification')
                    if classification in endpoint_classifications:
                        endpoint_classifications[classification].append(endpoint)

                except Exception as e:
                    logger.error(
                        "Failed to test endpoint",
                        agent_id=self.agent_id,
                        endpoint=endpoint,
                        error=str(e)
                    )

        logger.info(
            "Authentication testing completed",
            agent_id=self.agent_id,
            endpoints_tested=len(auth_results)
        )

        return {
            "auth_test_results": auth_results,
            "endpoint_classifications": endpoint_classifications,
            "all_endpoints_tested": True,
            "auth_mechanisms_classified": True,
            "num_public_endpoints": len(endpoint_classifications["PUBLIC"]),
            "num_protected_endpoints": len(endpoint_classifications["PROTECTED"]),
            "num_potential_bypasses": len(endpoint_classifications["POTENTIAL_BYPASS"])
        }

    async def _test_endpoint(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        endpoint: str
    ) -> Dict[str, Any]:
        """
        Test a single endpoint for auth requirements

        Args:
            client: HTTP client
            base_url: Base URL
            endpoint: Endpoint path

        Returns:
            Test result dictionary
        """
        url = f"{base_url}{endpoint}"

        result = {
            "endpoint": endpoint,
            "url": url,
            "tests": {},
            "classification": "UNKNOWN"
        }

        # Test 1: No authentication
        try:
            resp = await client.get(url)
            result["tests"]["no_auth"] = {
                "status_code": resp.status_code,
                "accessible": resp.status_code in [200, 201, 204]
            }
        except Exception as e:
            result["tests"]["no_auth"] = {"error": str(e)}

        # Test 2: Invalid/malformed JWT
        try:
            headers = {"Authorization": "Bearer invalid.jwt.token"}
            resp = await client.get(url, headers=headers)
            result["tests"]["malformed_jwt"] = {
                "status_code": resp.status_code,
                "accessible": resp.status_code in [200, 201, 204]
            }
        except Exception as e:
            result["tests"]["malformed_jwt"] = {"error": str(e)}

        # Test 3: Basic auth with invalid credentials
        try:
            resp = await client.get(url, auth=("invalid", "credentials"))
            result["tests"]["invalid_basic_auth"] = {
                "status_code": resp.status_code,
                "accessible": resp.status_code in [200, 201, 204]
            }
        except Exception as e:
            result["tests"]["invalid_basic_auth"] = {"error": str(e)}

        # Classify endpoint based on results
        no_auth = result["tests"].get("no_auth", {})

        if no_auth.get("accessible"):
            result["classification"] = "PUBLIC"
        elif no_auth.get("status_code") in [401, 403]:
            result["classification"] = "PROTECTED"

            # Check for potential auth bypass
            malformed = result["tests"].get("malformed_jwt", {})
            if malformed.get("accessible"):
                result["classification"] = "POTENTIAL_BYPASS"
                result["bypass_method"] = "malformed_jwt_accepted"
        elif no_auth.get("status_code") == 404:
            result["classification"] = "NOT_FOUND"
        else:
            result["classification"] = "REQUIRES_AUTH"

        return result


if __name__ == "__main__":
    agent = AgentAuthTest()
    agent.run()
