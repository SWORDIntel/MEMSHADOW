"""
Policy & Compartment Engine - Multi-AI Access Control

Controls which edges/nodes are visible or shareable between AIs:
- Per-AI namespaces and compartments
- Clearance levels for access control
- "Shadow links" (global knowledge, local views)
- Sharing rules and policy enforcement
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from uuid import UUID
import structlog

from .core_abstractions import MemoryObject, ShareScope, RelationType

logger = structlog.get_logger()


class AccessLevel(int, Enum):
    """Access levels for clearance"""
    PUBLIC = 0        # Anyone can access
    INTERNAL = 1      # Registered AIs only
    CONFIDENTIAL = 2  # Specific groups
    SECRET = 3        # Specific agents only
    TOP_SECRET = 4    # Owner only


@dataclass
class PolicyRule:
    """A policy rule for access control"""
    rule_id: str
    name: str
    description: str = ""
    # Conditions
    source_agents: Optional[Set[str]] = None  # None = all
    target_compartments: Optional[Set[str]] = None
    relation_types: Optional[Set[RelationType]] = None
    min_clearance: AccessLevel = AccessLevel.PUBLIC
    # Actions
    allow: bool = True
    transform: Optional[Callable] = None  # Transform data before access
    audit: bool = False
    # Metadata
    priority: int = 0  # Higher = evaluated first
    enabled: bool = True


@dataclass
class Compartment:
    """A compartment for organizing memories"""
    compartment_id: str
    name: str
    description: str = ""
    owner_agent: Optional[str] = None
    allowed_agents: Set[str] = field(default_factory=set)
    allowed_groups: Set[str] = field(default_factory=set)
    min_clearance: AccessLevel = AccessLevel.INTERNAL
    tags: Set[str] = field(default_factory=set)
    is_shadow: bool = False  # Shadow compartment for local views


@dataclass
class AgentProfile:
    """Profile for an AI agent"""
    agent_id: str
    name: str
    groups: Set[str] = field(default_factory=set)
    clearance_level: AccessLevel = AccessLevel.INTERNAL
    home_compartment: Optional[str] = None
    allowed_compartments: Set[str] = field(default_factory=set)
    private_namespace: str = ""
    custom_embeddings_model: Optional[str] = None
    # Quotas
    max_memories: int = 100000
    max_connections_per_memory: int = 100
    # Audit
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessDecision:
    """Result of an access check"""
    allowed: bool
    reason: str
    rule_applied: Optional[str] = None
    transformed_data: Optional[Any] = None
    should_audit: bool = False


class PolicyEngine:
    """
    Policy engine for multi-AI access control.

    Manages:
    - Agent registration and profiles
    - Compartment creation and management
    - Access rules and enforcement
    - Shadow links for local views of global data
    """

    def __init__(self):
        # Agent profiles
        self.agents: Dict[str, AgentProfile] = {}

        # Compartments
        self.compartments: Dict[str, Compartment] = {}

        # Policy rules
        self.rules: List[PolicyRule] = []

        # Group definitions
        self.groups: Dict[str, Set[str]] = {}  # group_id → agent_ids

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_entries = 10000

        # Default rules
        self._setup_default_rules()

        logger.info("PolicyEngine initialized")

    def _setup_default_rules(self):
        """Setup default policy rules"""
        # Global read access for public data
        self.add_rule(PolicyRule(
            rule_id="default_public_read",
            name="Public Read Access",
            description="Allow read access to public memories",
            min_clearance=AccessLevel.PUBLIC,
            allow=True,
            priority=0
        ))

        # Deny access to higher clearance data
        self.add_rule(PolicyRule(
            rule_id="default_clearance_check",
            name="Clearance Check",
            description="Deny access to data above agent clearance",
            allow=False,
            priority=100
        ))

    def register_agent(
        self,
        agent_id: str,
        name: str,
        groups: Optional[Set[str]] = None,
        clearance: AccessLevel = AccessLevel.INTERNAL
    ) -> AgentProfile:
        """Register a new AI agent"""
        profile = AgentProfile(
            agent_id=agent_id,
            name=name,
            groups=groups or set(),
            clearance_level=clearance,
            private_namespace=f"ns_{agent_id}",
        )
        self.agents[agent_id] = profile

        # Add to groups
        for group in profile.groups:
            if group not in self.groups:
                self.groups[group] = set()
            self.groups[group].add(agent_id)

        logger.info("Agent registered", agent_id=agent_id, clearance=clearance.name)
        return profile

    def create_compartment(
        self,
        compartment_id: str,
        name: str,
        owner_agent: Optional[str] = None,
        min_clearance: AccessLevel = AccessLevel.INTERNAL,
        is_shadow: bool = False
    ) -> Compartment:
        """Create a new compartment"""
        compartment = Compartment(
            compartment_id=compartment_id,
            name=name,
            owner_agent=owner_agent,
            min_clearance=min_clearance,
            is_shadow=is_shadow
        )
        self.compartments[compartment_id] = compartment

        logger.info("Compartment created",
                   compartment_id=compartment_id,
                   is_shadow=is_shadow)
        return compartment

    def add_rule(self, rule: PolicyRule):
        """Add a policy rule"""
        self.rules.append(rule)
        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def check_access(
        self,
        agent_id: str,
        memory: MemoryObject,
        access_type: str = "read"  # "read", "write", "delete", "link"
    ) -> AccessDecision:
        """
        Check if an agent can access a memory.

        Returns AccessDecision with allow/deny and reason.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return AccessDecision(
                allowed=False,
                reason="Agent not registered"
            )

        # Check basic access based on ShareScope
        policy = memory.policy

        # Owner always has access
        if policy.owner_agent_id == agent_id:
            return AccessDecision(allowed=True, reason="Owner access")

        # Check share scope
        if policy.share_scope == ShareScope.AGENT_LOCAL:
            if agent_id not in policy.allowed_agents:
                return AccessDecision(
                    allowed=False,
                    reason="Memory is agent-local and not shared"
                )

        elif policy.share_scope == ShareScope.GROUP:
            if not (agent.groups & policy.allowed_groups):
                return AccessDecision(
                    allowed=False,
                    reason="Memory is group-restricted"
                )

        # Check clearance
        if policy.clearance_level > agent.clearance_level.value:
            return AccessDecision(
                allowed=False,
                reason=f"Insufficient clearance (need {policy.clearance_level}, have {agent.clearance_level.value})"
            )

        # Check compartment
        for tag in policy.compartment_tags:
            if tag in self.compartments:
                comp = self.compartments[tag]
                if agent_id not in comp.allowed_agents and not (agent.groups & comp.allowed_groups):
                    return AccessDecision(
                        allowed=False,
                        reason=f"Not allowed in compartment {tag}"
                    )

        # Apply policy rules
        for rule in self.rules:
            if not rule.enabled:
                continue

            # Check if rule applies
            if rule.source_agents and agent_id not in rule.source_agents:
                continue

            if rule.target_compartments:
                if not (policy.compartment_tags & rule.target_compartments):
                    continue

            if rule.min_clearance.value > agent.clearance_level.value:
                return AccessDecision(
                    allowed=False,
                    reason=f"Rule {rule.rule_id}: insufficient clearance",
                    rule_applied=rule.rule_id,
                    should_audit=rule.audit
                )

            # Rule matches
            if not rule.allow:
                decision = AccessDecision(
                    allowed=False,
                    reason=f"Denied by rule: {rule.name}",
                    rule_applied=rule.rule_id,
                    should_audit=rule.audit
                )
                if rule.audit:
                    self._audit(agent_id, memory.id, access_type, decision)
                return decision

        # Default allow
        return AccessDecision(
            allowed=True,
            reason="Access granted"
        )

    def check_link_access(
        self,
        agent_id: str,
        source_memory: MemoryObject,
        target_memory: MemoryObject,
        relation_type: RelationType
    ) -> AccessDecision:
        """Check if an agent can create a link between two memories"""
        # Must have access to both memories
        source_access = self.check_access(agent_id, source_memory, "link")
        if not source_access.allowed:
            return source_access

        target_access = self.check_access(agent_id, target_memory, "link")
        if not target_access.allowed:
            return target_access

        # Check relation-specific rules
        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.relation_types and relation_type not in rule.relation_types:
                continue

            if not rule.allow:
                return AccessDecision(
                    allowed=False,
                    reason=f"Relation type {relation_type.value} denied by rule: {rule.name}",
                    rule_applied=rule.rule_id
                )

        return AccessDecision(
            allowed=True,
            reason="Link access granted"
        )

    def create_shadow_link(
        self,
        global_memory_id: UUID,
        agent_id: str,
        local_view: Dict[str, Any]
    ) -> str:
        """
        Create a shadow link for local view of global data.

        Shadow links allow agents to have their own view/annotations
        of globally shared memories.
        """
        shadow_id = f"shadow_{agent_id}_{global_memory_id}"

        # Create shadow compartment if needed
        shadow_comp_id = f"shadow_{agent_id}"
        if shadow_comp_id not in self.compartments:
            self.create_compartment(
                compartment_id=shadow_comp_id,
                name=f"Shadow compartment for {agent_id}",
                owner_agent=agent_id,
                is_shadow=True
            )

        logger.debug("Shadow link created",
                    shadow_id=shadow_id,
                    agent_id=agent_id)

        return shadow_id

    def get_agent_view_filter(
        self,
        agent_id: str
    ) -> Callable[[MemoryObject], bool]:
        """Get a filter function for agent's view of the graph"""
        agent = self.agents.get(agent_id)

        def filter_func(memory: MemoryObject) -> bool:
            if not agent:
                return memory.policy.share_scope == ShareScope.GLOBAL

            decision = self.check_access(agent_id, memory)
            return decision.allowed

        return filter_func

    def _audit(
        self,
        agent_id: str,
        memory_id: UUID,
        access_type: str,
        decision: AccessDecision
    ):
        """Log an audit entry"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "memory_id": str(memory_id),
            "access_type": access_type,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "rule_applied": decision.rule_applied,
        }
        self.audit_log.append(entry)

        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log.pop(0)

    def get_agent_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return None

        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "groups": list(agent.groups),
            "clearance": agent.clearance_level.name,
            "allowed_compartments": list(agent.allowed_compartments),
            "created_at": agent.created_at.isoformat(),
            "last_active": agent.last_active.isoformat(),
        }

    def get_compartment_stats(self) -> Dict[str, Any]:
        """Get compartment statistics"""
        return {
            "total_compartments": len(self.compartments),
            "shadow_compartments": sum(1 for c in self.compartments.values() if c.is_shadow),
            "compartments": [
                {
                    "id": c.compartment_id,
                    "name": c.name,
                    "is_shadow": c.is_shadow,
                    "min_clearance": c.min_clearance.name,
                }
                for c in self.compartments.values()
            ]
        }

    def export_policy(self) -> Dict[str, Any]:
        """Export policy configuration"""
        return {
            "agents": {
                aid: {
                    "name": a.name,
                    "groups": list(a.groups),
                    "clearance": a.clearance_level.name,
                }
                for aid, a in self.agents.items()
            },
            "compartments": {
                cid: {
                    "name": c.name,
                    "is_shadow": c.is_shadow,
                    "min_clearance": c.min_clearance.name,
                }
                for cid, c in self.compartments.items()
            },
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "allow": r.allow,
                    "priority": r.priority,
                }
                for r in self.rules
            ],
            "groups": {gid: list(agents) for gid, agents in self.groups.items()},
        }
