#!/usr/bin/env python3
"""
Canary System for DSMIL Brain

Honeypot knowledge for leak detection:
- Unique canary facts per node/accessor
- Track external surfacing of canaries
- Honeypot knowledge branches (convincing but false)
- Automatic leak source identification
- Canary generation and management
"""

import hashlib
import secrets
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timezone, timedelta
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CanaryType(Enum):
    """Types of canary tokens"""
    FACT = auto()          # False fact embedded in knowledge
    DOCUMENT = auto()      # Fake document with tracking
    CREDENTIAL = auto()    # Honeypot credentials
    ENDPOINT = auto()      # Fake API endpoint
    CONTACT = auto()       # Fake contact information
    LINK = auto()          # Tracking link
    IDENTIFIER = auto()    # Unique identifier
    QUERY = auto()         # Specific query pattern


class CanaryStatus(Enum):
    """Status of a canary"""
    ACTIVE = auto()
    TRIGGERED = auto()
    EXPIRED = auto()
    DISABLED = auto()


@dataclass
class CanaryToken:
    """A canary token for leak detection"""
    canary_id: str
    canary_type: CanaryType
    status: CanaryStatus = CanaryStatus.ACTIVE

    # Content
    content: Any = None
    description: str = ""

    # Targeting
    target_node: Optional[str] = None
    target_accessor: Optional[str] = None

    # Tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    trigger_source: Optional[str] = None
    trigger_context: Optional[Dict] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_triggered(self) -> bool:
        return self.status == CanaryStatus.TRIGGERED

    def is_active(self) -> bool:
        if self.status != CanaryStatus.ACTIVE:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def trigger(self, source: str, context: Optional[Dict] = None):
        """Mark canary as triggered"""
        self.status = CanaryStatus.TRIGGERED
        self.triggered_at = datetime.now(timezone.utc)
        self.trigger_source = source
        self.trigger_context = context


@dataclass
class HoneypotKnowledge:
    """A branch of honeypot knowledge"""
    honeypot_id: str
    name: str
    description: str

    # Content
    facts: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[Dict[str, Any]] = field(default_factory=list)

    # Embedded canaries
    canary_ids: List[str] = field(default_factory=list)

    # Plausibility
    plausibility_score: float = 0.8  # How convincing is this honeypot

    # Access tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    accessors: Set[str] = field(default_factory=set)

    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LeakDetection:
    """A detected leak"""
    detection_id: str
    canary_id: str

    # Source
    detected_at: datetime
    detected_source: str  # Where canary was found
    detection_context: Dict[str, Any] = field(default_factory=dict)

    # Attribution
    suspected_accessor: Optional[str] = None
    suspected_node: Optional[str] = None
    attribution_confidence: float = 0.0

    # Investigation
    investigated: bool = False
    investigation_notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "detection_id": self.detection_id,
            "canary_id": self.canary_id,
            "detected_at": self.detected_at.isoformat(),
            "detected_source": self.detected_source,
            "suspected_accessor": self.suspected_accessor,
            "attribution_confidence": self.attribution_confidence,
        }


class CanaryGenerator:
    """Generates various types of canary tokens"""

    @staticmethod
    def generate_fact_canary(topic: str, node_id: str = "", accessor_id: str = "") -> CanaryToken:
        """Generate a false fact canary"""
        # Create unique but plausible-sounding false fact
        unique_id = secrets.token_hex(4)

        # Templates for false facts
        templates = [
            f"Project {unique_id.upper()} was initiated in Q3 2024",
            f"Internal codename for the operation is EAGLE-{unique_id.upper()}",
            f"Reference document ID: DOC-{unique_id.upper()}-INTERNAL",
            f"Contact person for {topic}: agent-{unique_id}@internal.example",
            f"Budget allocation code: BUD-{unique_id.upper()}",
        ]

        import random
        content = random.choice(templates)

        return CanaryToken(
            canary_id=f"canary-fact-{unique_id}",
            canary_type=CanaryType.FACT,
            content=content,
            description=f"False fact canary for {topic}",
            target_node=node_id or None,
            target_accessor=accessor_id or None,
            metadata={"topic": topic, "unique_marker": unique_id},
        )

    @staticmethod
    def generate_credential_canary(service: str) -> CanaryToken:
        """Generate honeypot credentials"""
        unique_id = secrets.token_hex(4)

        username = f"svc-{service.lower()}-{unique_id[:4]}"
        password = secrets.token_urlsafe(16)

        return CanaryToken(
            canary_id=f"canary-cred-{unique_id}",
            canary_type=CanaryType.CREDENTIAL,
            content={
                "username": username,
                "password": password,
                "service": service,
            },
            description=f"Honeypot credentials for {service}",
            metadata={"service": service, "unique_marker": unique_id},
        )

    @staticmethod
    def generate_link_canary(description: str) -> CanaryToken:
        """Generate tracking link canary"""
        unique_id = secrets.token_hex(8)

        # Would use actual canary service in production
        tracking_url = f"https://track.internal.example/c/{unique_id}"

        return CanaryToken(
            canary_id=f"canary-link-{unique_id}",
            canary_type=CanaryType.LINK,
            content=tracking_url,
            description=description,
            metadata={"tracking_id": unique_id},
        )

    @staticmethod
    def generate_identifier_canary(context: str, node_id: str = "", accessor_id: str = "") -> CanaryToken:
        """Generate unique identifier canary"""
        # Create identifier unique to node/accessor combination
        seed = f"{context}:{node_id}:{accessor_id}:{secrets.token_hex(4)}"
        unique_id = hashlib.sha256(seed.encode()).hexdigest()[:12]

        return CanaryToken(
            canary_id=f"canary-id-{unique_id}",
            canary_type=CanaryType.IDENTIFIER,
            content=f"ID-{unique_id.upper()}",
            description=f"Unique identifier for {context}",
            target_node=node_id or None,
            target_accessor=accessor_id or None,
            metadata={"context": context},
        )


class CanarySystem:
    """
    Canary Management System

    Creates, tracks, and monitors canary tokens for leak detection.

    Usage:
        canary_system = CanarySystem()

        # Generate canaries
        canary = canary_system.create_canary(CanaryType.FACT, "sensitive topic")

        # Create honeypot knowledge
        honeypot = canary_system.create_honeypot("Project X", [...])

        # Check for triggered canaries
        canary_system.check_canary(canary_id, "external source")

        # Get leak detections
        leaks = canary_system.get_detections()
    """

    def __init__(self):
        self._canaries: Dict[str, CanaryToken] = {}
        self._honeypots: Dict[str, HoneypotKnowledge] = {}
        self._detections: List[LeakDetection] = []

        # Index by content for detection
        self._content_index: Dict[str, str] = {}  # content_hash -> canary_id

        self._lock = threading.RLock()

        # Callbacks
        self.on_canary_triggered: Optional[callable] = None
        self.on_leak_detected: Optional[callable] = None

        logger.info("CanarySystem initialized")

    def create_canary(self, canary_type: CanaryType,
                     topic: str = "",
                     node_id: str = "",
                     accessor_id: str = "",
                     expires_hours: Optional[int] = None) -> CanaryToken:
        """
        Create a new canary token

        Args:
            canary_type: Type of canary to create
            topic: Topic/context for the canary
            node_id: Target node (for attribution)
            accessor_id: Target accessor (for attribution)
            expires_hours: Hours until expiration

        Returns:
            Created CanaryToken
        """
        # Generate appropriate canary
        if canary_type == CanaryType.FACT:
            canary = CanaryGenerator.generate_fact_canary(topic, node_id, accessor_id)
        elif canary_type == CanaryType.CREDENTIAL:
            canary = CanaryGenerator.generate_credential_canary(topic or "default")
        elif canary_type == CanaryType.LINK:
            canary = CanaryGenerator.generate_link_canary(topic)
        elif canary_type == CanaryType.IDENTIFIER:
            canary = CanaryGenerator.generate_identifier_canary(topic, node_id, accessor_id)
        else:
            # Generic canary
            canary = CanaryToken(
                canary_id=f"canary-{canary_type.name.lower()}-{secrets.token_hex(4)}",
                canary_type=canary_type,
                content=f"{topic}-{secrets.token_hex(8)}",
                description=f"{canary_type.name} canary for {topic}",
            )

        # Set expiration
        if expires_hours:
            canary.expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)

        # Store
        with self._lock:
            self._canaries[canary.canary_id] = canary

            # Index content for detection
            if canary.content:
                content_str = str(canary.content)
                content_hash = hashlib.sha256(content_str.encode()).hexdigest()
                self._content_index[content_hash] = canary.canary_id

        logger.info(f"Created canary: {canary.canary_id}")
        return canary

    def create_honeypot(self, name: str,
                       facts: List[Dict],
                       embed_canaries: bool = True) -> HoneypotKnowledge:
        """
        Create a honeypot knowledge branch

        Args:
            name: Name of honeypot
            facts: False facts to include
            embed_canaries: Whether to embed canaries in facts

        Returns:
            Created HoneypotKnowledge
        """
        honeypot_id = f"honeypot-{secrets.token_hex(4)}"

        canary_ids = []

        # Embed canaries in facts
        if embed_canaries:
            for i, fact in enumerate(facts):
                canary = self.create_canary(
                    CanaryType.IDENTIFIER,
                    topic=f"{name}-fact-{i}",
                )
                canary_ids.append(canary.canary_id)

                # Embed canary ID in fact
                fact["_tracking_id"] = canary.content

        honeypot = HoneypotKnowledge(
            honeypot_id=honeypot_id,
            name=name,
            description=f"Honeypot knowledge branch: {name}",
            facts=facts,
            canary_ids=canary_ids,
        )

        with self._lock:
            self._honeypots[honeypot_id] = honeypot

        logger.info(f"Created honeypot: {honeypot_id} with {len(facts)} facts")
        return honeypot

    def check_canary(self, canary_id: str, source: str,
                    context: Optional[Dict] = None) -> bool:
        """
        Check/trigger a canary

        Args:
            canary_id: ID of canary to check
            source: Where canary was found
            context: Additional context

        Returns:
            True if canary was triggered (new detection)
        """
        with self._lock:
            if canary_id not in self._canaries:
                return False

            canary = self._canaries[canary_id]

            if not canary.is_active():
                return False

            # Trigger the canary
            canary.trigger(source, context)

            # Create leak detection
            detection = LeakDetection(
                detection_id=f"leak-{secrets.token_hex(4)}",
                canary_id=canary_id,
                detected_at=datetime.now(timezone.utc),
                detected_source=source,
                detection_context=context or {},
                suspected_accessor=canary.target_accessor,
                suspected_node=canary.target_node,
                attribution_confidence=0.8 if canary.target_accessor else 0.3,
            )

            self._detections.append(detection)

            # Callbacks
            if self.on_canary_triggered:
                self.on_canary_triggered(canary)
            if self.on_leak_detected:
                self.on_leak_detected(detection)

            logger.warning(f"CANARY TRIGGERED: {canary_id} detected at {source}")

            return True

    def check_content(self, content: str, source: str,
                     context: Optional[Dict] = None) -> List[LeakDetection]:
        """
        Check content for any embedded canaries

        Args:
            content: Content to scan
            source: Source of content
            context: Additional context

        Returns:
            List of leak detections
        """
        detections = []

        with self._lock:
            # Check against all active canaries
            for canary in self._canaries.values():
                if not canary.is_active():
                    continue

                canary_content = str(canary.content)

                if canary_content in content:
                    if self.check_canary(canary.canary_id, source, context):
                        # Get the detection we just created
                        if self._detections:
                            detections.append(self._detections[-1])

        return detections

    def get_canary(self, canary_id: str) -> Optional[CanaryToken]:
        """Get canary by ID"""
        return self._canaries.get(canary_id)

    def get_honeypot(self, honeypot_id: str) -> Optional[HoneypotKnowledge]:
        """Get honeypot by ID"""
        return self._honeypots.get(honeypot_id)

    def get_detections(self, since: Optional[datetime] = None) -> List[LeakDetection]:
        """Get leak detections"""
        with self._lock:
            if since:
                return [d for d in self._detections if d.detected_at >= since]
            return list(self._detections)

    def get_canaries_for_node(self, node_id: str) -> List[CanaryToken]:
        """Get canaries targeted at a specific node"""
        with self._lock:
            return [c for c in self._canaries.values() if c.target_node == node_id]

    def get_canaries_for_accessor(self, accessor_id: str) -> List[CanaryToken]:
        """Get canaries targeted at a specific accessor"""
        with self._lock:
            return [c for c in self._canaries.values() if c.target_accessor == accessor_id]

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            active = sum(1 for c in self._canaries.values() if c.is_active())
            triggered = sum(1 for c in self._canaries.values() if c.is_triggered())

            return {
                "total_canaries": len(self._canaries),
                "active_canaries": active,
                "triggered_canaries": triggered,
                "honeypots": len(self._honeypots),
                "leak_detections": len(self._detections),
            }


if __name__ == "__main__":
    print("Canary System Self-Test")
    print("=" * 50)

    system = CanarySystem()

    # Set up callback
    def on_leak(detection):
        print(f"    [ALERT] Leak detected: {detection.canary_id}")

    system.on_leak_detected = on_leak

    print("\n[1] Create Canaries")
    canary1 = system.create_canary(CanaryType.FACT, "Project Alpha", node_id="node-001")
    print(f"    Fact canary: {canary1.canary_id}")
    print(f"    Content: {canary1.content}")

    canary2 = system.create_canary(CanaryType.CREDENTIAL, "database")
    print(f"    Credential canary: {canary2.canary_id}")
    print(f"    Credentials: {canary2.content}")

    canary3 = system.create_canary(CanaryType.IDENTIFIER, "api_key", accessor_id="user-123")
    print(f"    Identifier canary: {canary3.canary_id}")
    print(f"    Identifier: {canary3.content}")

    print("\n[2] Create Honeypot")
    honeypot = system.create_honeypot("Project X", [
        {"fact": "Project X budget is $10M", "classification": "secret"},
        {"fact": "Project X deadline is Q4 2025", "classification": "internal"},
    ])
    print(f"    Honeypot: {honeypot.honeypot_id}")
    print(f"    Embedded canaries: {len(honeypot.canary_ids)}")

    print("\n[3] Simulate Leak Detection")
    # Simulate finding canary content externally
    leaked_text = f"According to internal sources, {canary1.content} This was leaked."

    detections = system.check_content(leaked_text, "external_website", {"url": "https://leak.example.com"})
    print(f"    Detections from content scan: {len(detections)}")

    # Direct canary check
    system.check_canary(canary2.canary_id, "honeypot_login_attempt", {"ip": "192.168.1.100"})

    print("\n[4] Get Detections")
    all_detections = system.get_detections()
    for detection in all_detections:
        print(f"    - {detection.canary_id}: {detection.detected_source}")
        print(f"      Suspected: {detection.suspected_accessor or detection.suspected_node or 'unknown'}")

    print("\n[5] Statistics")
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Canary System test complete")

