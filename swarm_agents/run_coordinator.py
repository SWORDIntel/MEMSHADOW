"""
SWARM Coordinator Startup Script

Loads and executes missions in the Arena environment.
"""

import asyncio
import sys
import os
import structlog

sys.path.insert(0, '/app')

from swarm.coordinator import SwarmCoordinator
from swarm.mission import MissionLoader
from swarm.blackboard import Blackboard

logger = structlog.get_logger()


async def main():
    """Main coordinator execution"""

    logger.info("Starting SWARM Coordinator")

    # Initialize blackboard
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    blackboard = Blackboard(redis_url=redis_url)

    # Initialize coordinator
    coordinator = SwarmCoordinator(blackboard=blackboard)

    # Load mission
    mission_file = os.getenv('MISSION_FILE', '/missions/default_recon.yaml')

    logger.info(f"Loading mission from {mission_file}")

    try:
        if os.path.exists(mission_file):
            mission = MissionLoader.load_from_file(mission_file)
        else:
            logger.warning(f"Mission file not found, using default reconnaissance mission")
            mission = MissionLoader.create_default_reconnaissance_mission()

        # Load mission into coordinator
        coordinator.load_mission(mission)

        # Execute mission
        logger.info("Executing mission", mission_id=mission.mission_id)

        report = await coordinator.execute_mission(timeout=3600)

        # Save report
        report_file = f"/reports/{mission.mission_id}_report.json"

        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(
            "Mission completed",
            mission_id=mission.mission_id,
            status=report.get('status'),
            report_file=report_file
        )

        # Print summary
        print("\n" + "="*80)
        print("MISSION EXECUTION SUMMARY")
        print("="*80)
        print(f"Mission ID: {report.get('mission_id')}")
        print(f"Mission Name: {report.get('mission_name')}")
        print(f"Status: {report.get('status')}")
        print(f"Duration: {report.get('duration_seconds')} seconds")
        print(f"\n{report.get('summary')}")
        print("="*80 + "\n")

    except Exception as e:
        logger.error("Mission execution failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
