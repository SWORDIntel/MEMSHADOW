"""
Project-Level Memory Organization
Phase 5: Claude Deep Integration - Organize memories, code, and sessions by project

Features:
- Project creation and management
- Memory association with projects
- Milestone tracking
- Objective management
- Project-wide search and analytics
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import structlog
from dataclasses import dataclass, field
import uuid

logger = structlog.get_logger()


class ProjectStatus(str, Enum):
    """Project status"""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MilestoneStatus(str, Enum):
    """Milestone status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class ProjectMilestone:
    """Project milestone"""
    milestone_id: str
    title: str
    description: str
    status: MilestoneStatus
    created_at: datetime
    target_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)


@dataclass
class Project:
    """Project with memories, code, and sessions"""
    project_id: str
    name: str
    description: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime

    # Objectives
    objectives: List[str] = field(default_factory=list)
    current_objective: Optional[str] = None

    # Associations
    memory_ids: List[str] = field(default_factory=list)
    code_artifact_ids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)

    # Milestones
    milestones: List[ProjectMilestone] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectMemoryOrganizer:
    """
    Organizes memories, code, and sessions at project level.

    Features:
    - Project CRUD operations
    - Associate memories/code/sessions with projects
    - Track milestones and objectives
    - Project-wide search
    - Analytics and reporting

    Example:
        organizer = ProjectMemoryOrganizer()

        # Create project
        project = await organizer.create_project(
            name="E-commerce Platform",
            description="Building a full-stack e-commerce solution",
            objectives=[
                "User authentication",
                "Product catalog",
                "Shopping cart",
                "Payment integration"
            ]
        )

        # Add milestone
        milestone = await organizer.add_milestone(
            project_id=project["project_id"],
            title="User Authentication MVP",
            description="Basic login/logout with JWT",
            target_date=datetime(2025, 2, 1)
        )

        # Associate memory
        await organizer.associate_memory(
            project_id=project["project_id"],
            memory_id="mem_123"
        )
    """

    def __init__(self):
        # In-memory storage (would be database in production)
        self.projects: Dict[str, Project] = {}
        self.user_projects: Dict[str, List[str]] = {}  # user_id -> [project_ids]

        logger.info("Project memory organizer initialized")

    async def create_project(
        self,
        name: str,
        description: str,
        objectives: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        user_id: str = "default_user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create new project.

        Args:
            name: Project name
            description: Project description
            objectives: List of objectives
            tags: Project tags
            user_id: User ID
            metadata: Additional metadata

        Returns:
            Project metadata
        """
        project_id = str(uuid.uuid4())

        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            status=ProjectStatus.PLANNING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            objectives=objectives or [],
            tags=tags or [],
            metadata=metadata or {}
        )

        self.projects[project_id] = project

        # Update user index
        if user_id not in self.user_projects:
            self.user_projects[user_id] = []
        self.user_projects[user_id].append(project_id)

        logger.info(
            "Project created",
            project_id=project_id,
            name=name,
            objectives_count=len(project.objectives)
        )

        return {
            "project_id": project_id,
            "name": name,
            "description": description,
            "status": project.status,
            "objectives": project.objectives,
            "created_at": project.created_at.isoformat()
        }

    async def add_milestone(
        self,
        project_id: str,
        title: str,
        description: str,
        target_date: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add milestone to project.

        Args:
            project_id: Project ID
            title: Milestone title
            description: Milestone description
            target_date: Optional target date
            dependencies: Milestone dependencies

        Returns:
            Milestone metadata
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        milestone_id = str(uuid.uuid4())

        milestone = ProjectMilestone(
            milestone_id=milestone_id,
            title=title,
            description=description,
            status=MilestoneStatus.NOT_STARTED,
            created_at=datetime.utcnow(),
            target_date=target_date,
            dependencies=dependencies or []
        )

        project = self.projects[project_id]
        project.milestones.append(milestone)
        project.updated_at = datetime.utcnow()

        logger.info(
            "Milestone added",
            project_id=project_id,
            milestone_id=milestone_id,
            title=title
        )

        return {
            "milestone_id": milestone_id,
            "project_id": project_id,
            "title": title,
            "status": milestone.status,
            "target_date": target_date.isoformat() if target_date else None,
            "created_at": milestone.created_at.isoformat()
        }

    async def associate_memory(
        self,
        project_id: str,
        memory_id: str
    ):
        """Associate memory with project"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        project = self.projects[project_id]

        if memory_id not in project.memory_ids:
            project.memory_ids.append(memory_id)
            project.updated_at = datetime.utcnow()

            logger.info(
                "Memory associated with project",
                project_id=project_id,
                memory_id=memory_id
            )

    async def associate_code(
        self,
        project_id: str,
        artifact_id: str
    ):
        """Associate code artifact with project"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        project = self.projects[project_id]

        if artifact_id not in project.code_artifact_ids:
            project.code_artifact_ids.append(artifact_id)
            project.updated_at = datetime.utcnow()

            logger.info(
                "Code artifact associated with project",
                project_id=project_id,
                artifact_id=artifact_id
            )

    async def associate_session(
        self,
        project_id: str,
        session_id: str
    ):
        """Associate session with project"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        project = self.projects[project_id]

        if session_id not in project.session_ids:
            project.session_ids.append(session_id)
            project.updated_at = datetime.utcnow()

            logger.info(
                "Session associated with project",
                project_id=project_id,
                session_id=session_id
            )

    async def update_milestone_status(
        self,
        project_id: str,
        milestone_id: str,
        status: MilestoneStatus,
        deliverables: Optional[List[str]] = None
    ):
        """
        Update milestone status.

        Args:
            project_id: Project ID
            milestone_id: Milestone ID
            status: New status
            deliverables: Optional deliverables
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        project = self.projects[project_id]

        milestone = next(
            (m for m in project.milestones if m.milestone_id == milestone_id),
            None
        )

        if not milestone:
            raise ValueError(f"Milestone {milestone_id} not found")

        milestone.status = status

        if status == MilestoneStatus.COMPLETED:
            milestone.completed_at = datetime.utcnow()

        if deliverables:
            milestone.deliverables.extend(deliverables)

        project.updated_at = datetime.utcnow()

        logger.info(
            "Milestone status updated",
            project_id=project_id,
            milestone_id=milestone_id,
            status=status
        )

    async def get_project(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get project details.

        Args:
            project_id: Project ID

        Returns:
            Project details
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        project = self.projects[project_id]

        return {
            "project_id": project_id,
            "name": project.name,
            "description": project.description,
            "status": project.status,
            "objectives": project.objectives,
            "current_objective": project.current_objective,
            "milestones": [
                {
                    "milestone_id": m.milestone_id,
                    "title": m.title,
                    "status": m.status,
                    "target_date": m.target_date.isoformat() if m.target_date else None,
                    "completed_at": m.completed_at.isoformat() if m.completed_at else None
                }
                for m in project.milestones
            ],
            "associations": {
                "memories": len(project.memory_ids),
                "code_artifacts": len(project.code_artifact_ids),
                "sessions": len(project.session_ids)
            },
            "tags": project.tags,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat()
        }

    async def get_project_analytics(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Get project analytics.

        Args:
            project_id: Project ID

        Returns:
            Analytics data
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")

        project = self.projects[project_id]

        # Calculate milestone completion
        total_milestones = len(project.milestones)
        completed_milestones = sum(
            1 for m in project.milestones
            if m.status == MilestoneStatus.COMPLETED
        )
        in_progress_milestones = sum(
            1 for m in project.milestones
            if m.status == MilestoneStatus.IN_PROGRESS
        )

        completion_rate = (
            completed_milestones / total_milestones * 100
            if total_milestones > 0
            else 0
        )

        return {
            "project_id": project_id,
            "status": project.status,
            "milestones": {
                "total": total_milestones,
                "completed": completed_milestones,
                "in_progress": in_progress_milestones,
                "completion_rate": completion_rate
            },
            "associations": {
                "memories": len(project.memory_ids),
                "code_artifacts": len(project.code_artifact_ids),
                "sessions": len(project.session_ids)
            },
            "activity": {
                "created_at": project.created_at.isoformat(),
                "last_updated": project.updated_at.isoformat(),
                "age_days": (datetime.utcnow() - project.created_at).days
            }
        }

    async def list_user_projects(
        self,
        user_id: str = "default_user",
        status_filter: Optional[ProjectStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List projects for user.

        Args:
            user_id: User ID
            status_filter: Optional status filter

        Returns:
            List of projects
        """
        project_ids = self.user_projects.get(user_id, [])

        projects = []
        for project_id in project_ids:
            project = self.projects[project_id]

            if status_filter and project.status != status_filter:
                continue

            projects.append({
                "project_id": project_id,
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "milestones_count": len(project.milestones),
                "memories_count": len(project.memory_ids),
                "updated_at": project.updated_at.isoformat()
            })

        return sorted(projects, key=lambda x: x["updated_at"], reverse=True)

    async def search_projects(
        self,
        query: str,
        user_id: str = "default_user"
    ) -> List[Dict[str, Any]]:
        """
        Search projects by name, description, or tags.

        Args:
            query: Search query
            user_id: User ID

        Returns:
            Matching projects
        """
        query_lower = query.lower()
        project_ids = self.user_projects.get(user_id, [])

        results = []
        for project_id in project_ids:
            project = self.projects[project_id]

            # Search in name, description, tags
            if (
                query_lower in project.name.lower() or
                query_lower in project.description.lower() or
                any(query_lower in tag.lower() for tag in project.tags)
            ):
                results.append({
                    "project_id": project_id,
                    "name": project.name,
                    "description": project.description,
                    "status": project.status,
                    "tags": project.tags
                })

        return results


# Global instance
project_organizer = ProjectMemoryOrganizer()
