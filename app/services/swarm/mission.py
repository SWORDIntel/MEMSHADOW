"""
Mission Definition and Loading System for SWARM

Missions are defined in YAML format and describe objectives, stages, and success criteria
for autonomous agent swarms.
"""

import yaml
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path
import structlog

logger = structlog.get_logger()


class MissionTask(BaseModel):
    """
    Individual task within a mission stage
    """
    agent_type: str
    params: Optional[Dict[str, Any]] = {}
    params_from_blackboard: Optional[Dict[str, str]] = {}


class MissionStage(BaseModel):
    """
    A single stage in a mission
    """
    stage_id: str
    description: str
    depends_on: Optional[str] = None  # ID of stage that must complete first
    tasks: List[MissionTask]
    success_criteria: List[str]

    @validator('success_criteria')
    def validate_criteria(cls, v):
        if not v:
            raise ValueError("success_criteria cannot be empty")
        return v


class Mission(BaseModel):
    """
    Complete mission definition
    """
    mission_id: str
    mission_name: str
    description: str
    objective_stages: List[MissionStage]
    overall_success_condition: str

    @validator('objective_stages')
    def validate_stages(cls, v):
        if not v:
            raise ValueError("Mission must have at least one stage")

        # Check for circular dependencies
        stage_ids = {stage.stage_id for stage in v}
        for stage in v:
            if stage.depends_on and stage.depends_on not in stage_ids:
                raise ValueError(
                    f"Stage '{stage.stage_id}' depends on non-existent stage '{stage.depends_on}'"
                )

        return v

    def get_stage(self, stage_id: str) -> Optional[MissionStage]:
        """
        Get a stage by ID

        Args:
            stage_id: Stage ID

        Returns:
            MissionStage or None
        """
        for stage in self.objective_stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def get_independent_stages(self) -> List[MissionStage]:
        """
        Get all stages that don't depend on other stages

        Returns:
            List of independent stages
        """
        return [stage for stage in self.objective_stages if not stage.depends_on]

    def get_dependent_stages(self, completed_stage_id: str) -> List[MissionStage]:
        """
        Get all stages that depend on a given completed stage

        Args:
            completed_stage_id: ID of completed stage

        Returns:
            List of dependent stages
        """
        return [
            stage for stage in self.objective_stages
            if stage.depends_on == completed_stage_id
        ]

    def is_complete(self, completed_stages: set) -> bool:
        """
        Check if mission is complete

        Args:
            completed_stages: Set of completed stage IDs

        Returns:
            True if all stages are complete
        """
        all_stage_ids = {stage.stage_id for stage in self.objective_stages}
        return completed_stages == all_stage_ids


class MissionLoader:
    """
    Loads and validates mission definitions from YAML files
    """

    @staticmethod
    def load_from_file(file_path: str) -> Mission:
        """
        Load a mission from a YAML file

        Args:
            file_path: Path to YAML mission file

        Returns:
            Mission object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Mission file not found: {file_path}")

        logger.info("Loading mission from file", file_path=file_path)

        try:
            with open(path, 'r') as f:
                mission_data = yaml.safe_load(f)

            mission = Mission(**mission_data)

            logger.info(
                "Mission loaded successfully",
                mission_id=mission.mission_id,
                mission_name=mission.mission_name,
                num_stages=len(mission.objective_stages)
            )

            return mission

        except yaml.YAMLError as e:
            logger.error("Failed to parse mission YAML", error=str(e))
            raise ValueError(f"Invalid YAML format: {str(e)}")

        except Exception as e:
            logger.error("Failed to load mission", error=str(e), error_type=type(e).__name__)
            raise

    @staticmethod
    def load_from_dict(mission_data: Dict[str, Any]) -> Mission:
        """
        Load a mission from a dictionary

        Args:
            mission_data: Mission data as dict

        Returns:
            Mission object
        """
        try:
            mission = Mission(**mission_data)

            logger.info(
                "Mission loaded from dict",
                mission_id=mission.mission_id,
                mission_name=mission.mission_name
            )

            return mission

        except Exception as e:
            logger.error("Failed to load mission from dict", error=str(e))
            raise ValueError(f"Invalid mission data: {str(e)}")

    @staticmethod
    def create_default_reconnaissance_mission() -> Mission:
        """
        Create a default reconnaissance mission for testing

        Returns:
            Mission object
        """
        mission_data = {
            "mission_id": "default_recon_001",
            "mission_name": "Default Reconnaissance Mission",
            "description": "Basic reconnaissance and vulnerability mapping of the target environment",
            "objective_stages": [
                {
                    "stage_id": "network_discovery",
                    "description": "Discover all hosts and services in the target network",
                    "tasks": [
                        {
                            "agent_type": "recon",
                            "params": {
                                "target_cidr": "172.19.0.0/24"
                            }
                        }
                    ],
                    "success_criteria": [
                        "blackboard_key_exists:known_hosts",
                        "num_known_hosts >= 1"
                    ]
                },
                {
                    "stage_id": "service_enumeration",
                    "description": "Enumerate services on discovered hosts",
                    "depends_on": "network_discovery",
                    "tasks": [
                        {
                            "agent_type": "apimapper",
                            "params_from_blackboard": {
                                "target_host_key": "primary_web_service_host",
                                "target_port_key": "primary_web_service_port"
                            }
                        }
                    ],
                    "success_criteria": [
                        "blackboard_key_exists:discovered_endpoints",
                        "num_discovered_endpoints >= 1"
                    ]
                },
                {
                    "stage_id": "authentication_testing",
                    "description": "Test authentication mechanisms on discovered endpoints",
                    "depends_on": "service_enumeration",
                    "tasks": [
                        {
                            "agent_type": "authtest",
                            "params_from_blackboard": {
                                "endpoints_key": "discovered_endpoints"
                            }
                        }
                    ],
                    "success_criteria": [
                        "all_endpoints_tested"
                    ]
                }
            ],
            "overall_success_condition": "All stages completed successfully and security report generated"
        }

        return Mission(**mission_data)

    @staticmethod
    def create_advanced_security_mission() -> Mission:
        """
        Create an advanced security testing mission with BGP and blockchain analysis

        Returns:
            Mission object
        """
        mission_data = {
            "mission_id": "advanced_sec_001",
            "mission_name": "Advanced Security Assessment",
            "description": "Comprehensive security assessment including network, BGP, and blockchain analysis",
            "objective_stages": [
                {
                    "stage_id": "initial_recon",
                    "description": "Initial reconnaissance of all network assets",
                    "tasks": [
                        {
                            "agent_type": "recon",
                            "params": {
                                "target_cidr": "172.19.0.0/24",
                                "scan_type": "comprehensive"
                            }
                        }
                    ],
                    "success_criteria": [
                        "blackboard_key_exists:network_topology",
                        "num_discovered_services >= 1"
                    ]
                },
                {
                    "stage_id": "api_discovery",
                    "description": "Discover and map all API endpoints",
                    "depends_on": "initial_recon",
                    "tasks": [
                        {
                            "agent_type": "apimapper",
                            "params_from_blackboard": {
                                "discovered_hosts": "network_topology"
                            }
                        }
                    ],
                    "success_criteria": [
                        "api_endpoints_mapped"
                    ]
                },
                {
                    "stage_id": "auth_analysis",
                    "description": "Analyze authentication and authorization mechanisms",
                    "depends_on": "api_discovery",
                    "tasks": [
                        {
                            "agent_type": "authtest",
                            "params_from_blackboard": {
                                "api_endpoints": "discovered_api_endpoints"
                            }
                        }
                    ],
                    "success_criteria": [
                        "auth_mechanisms_classified"
                    ]
                },
                {
                    "stage_id": "bgp_analysis",
                    "description": "Analyze BGP routing for potential hijack vulnerabilities",
                    "tasks": [
                        {
                            "agent_type": "bgp",
                            "params": {
                                "monitor_asns": [],
                                "check_rpki": True
                            }
                        }
                    ],
                    "success_criteria": [
                        "bgp_routes_analyzed"
                    ]
                },
                {
                    "stage_id": "blockchain_analysis",
                    "description": "Analyze blockchain transactions for fraud patterns",
                    "tasks": [
                        {
                            "agent_type": "blockchain",
                            "params": {
                                "chains": ["ethereum", "bitcoin"],
                                "analysis_depth": "medium"
                            }
                        }
                    ],
                    "success_criteria": [
                        "blockchain_transactions_analyzed"
                    ]
                }
            ],
            "overall_success_condition": "All stages completed and comprehensive security report generated"
        }

        return Mission(**mission_data)
