"""
MEMSHADOW Phase-0 SIGINT/GEOINT Database Models

SQLAlchemy models for storing network observations, geographic intelligence,
and OSINT-enriched data from server logs.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer,
    ForeignKey, Index, CheckConstraint, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.hybrid import hybrid_property
from pgvector.sqlalchemy import Vector

from app.db.postgres import Base


# ============================================================================
# OBSERVATION - Core Intelligence Event
# ============================================================================

class Observation(Base):
    """
    A single intelligence observation from network/server logs.

    This is the core entity in Phase-0 SIGINT/GEOINT.
    Each observation represents a discrete event (HTTP request, network flow, etc.)
    enriched with GeoIP data and linked to subjects (devices, identities, services).
    """
    __tablename__ = "observations"
    __table_args__ = (
        # Indexes for common query patterns
        Index("idx_obs_timestamp", "timestamp"),
        Index("idx_obs_modality_timestamp", "modality", "timestamp"),
        Index("idx_obs_host_timestamp", "host", "timestamp"),
        Index("idx_obs_node_id", "node_id", unique=True),
        Index("idx_obs_user_time", "user_id", "timestamp"),
        Index("idx_obs_payload_gin", "payload", postgresql_using="gin"),
        Index("idx_obs_labels_gin", "labels", postgresql_using="gin"),
        Index("idx_obs_aoi_memberships_gin", "aoi_memberships", postgresql_using="gin"),
        # Check constraints
        CheckConstraint("char_length(source) >= 1", name="check_source_not_empty"),
        CheckConstraint("char_length(host) >= 1", name="check_host_not_empty"),
    )

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Node identifier (e.g., "obs:2025-12-02T03:22:01Z:web01:HTTP_ACCESS:203.0.113.10")
    node_id = Column(String(512), nullable=False, unique=True)

    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Core observation metadata
    modality = Column(
        SQLEnum("SIGINT", "GEOINT", "OSINT", "HUMINT", "MASINT", name="observation_modality"),
        nullable=False,
        index=True
    )
    source = Column(String(128), nullable=False)  # e.g., "sigint-net-agent"
    timestamp = Column(DateTime, nullable=False, index=True)  # When the observation occurred
    host = Column(String(256), nullable=False, index=True)  # Hostname where observation was made
    sensor_id = Column(String(256), nullable=False)  # Unique sensor identifier

    # Labels (arbitrary key-value pairs)
    labels = Column(JSONB, nullable=False, default={})

    # Subject references (list of {type, id} dicts)
    subjects = Column(JSONB, nullable=False, default=[])

    # Geographic location (can be null if no location data)
    location = Column(JSONB, nullable=True)  # Stores GeographicLocation schema

    # Raw payload data (HTTP headers, network flow stats, etc.)
    payload = Column(JSONB, nullable=False, default={})

    # Derived signals and risk scoring
    signals = Column(JSONB, nullable=False, default={})  # Stores SignalMetadata schema

    # Semantic embedding (2048d for MEMSHADOW v2.0)
    embedding = Column(Vector(2048), nullable=True)

    # AOI memberships (list of AOI node IDs this observation falls within)
    aoi_memberships = Column(ARRAY(String), nullable=False, default=[])

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    @hybrid_property
    def age_hours(self):
        """Hours since observation occurred"""
        return (datetime.utcnow() - self.timestamp).total_seconds() / 3600

    @hybrid_property
    def risk_score(self):
        """Extract risk score from signals metadata"""
        if not self.signals:
            return 0.0
        return self.signals.get("risk_score", 0.0)


# ============================================================================
# DEVICE Node
# ============================================================================

class Device(Base):
    """
    A device/endpoint observed in network traffic.

    Primarily represents IP addresses, but can also track MACs, hostnames, etc.
    Linked to observations via the subjects field.
    """
    __tablename__ = "devices"
    __table_args__ = (
        Index("idx_device_node_id", "node_id", unique=True),
        Index("idx_device_user", "user_id"),
        Index("idx_device_type_identifier", "device_type", "identifier"),
        Index("idx_device_labels_gin", "labels", postgresql_using="gin"),
        Index("idx_device_last_seen", "last_seen"),
        CheckConstraint("char_length(identifier) >= 1", name="check_identifier_not_empty"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(512), nullable=False, unique=True)  # e.g., "device:ip:203.0.113.10"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Device metadata
    device_type = Column(
        SQLEnum("IP", "MAC", "IMEI", "HOSTNAME", "FINGERPRINT", name="device_type"),
        nullable=False
    )
    identifier = Column(String(256), nullable=False)  # IP address, MAC, etc.
    labels = Column(JSONB, nullable=False, default={})

    # Temporal tracking
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True, index=True)

    # Risk assessment
    risk_score = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# IDENTITY Node
# ============================================================================

class Identity(Base):
    """
    A user identity/account observed in authentication events.

    Represents accounts, usernames, emails, OAuth identities, etc.
    Linked to observations via the subjects field.
    """
    __tablename__ = "identities"
    __table_args__ = (
        Index("idx_identity_node_id", "node_id", unique=True),
        Index("idx_identity_user", "user_id"),
        Index("idx_identity_type_identifier", "identity_type", "identifier"),
        Index("idx_identity_labels_gin", "labels", postgresql_using="gin"),
        Index("idx_identity_last_seen", "last_seen"),
        CheckConstraint("char_length(identifier) >= 1", name="check_identity_identifier_not_empty"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(512), nullable=False, unique=True)  # e.g., "identity:acct:john@example.com"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Identity metadata
    identity_type = Column(
        SQLEnum("ACCOUNT", "USERNAME", "EMAIL", "OAUTH", "CERTIFICATE", name="identity_type"),
        nullable=False
    )
    identifier = Column(String(256), nullable=False)  # email, username, etc.
    labels = Column(JSONB, nullable=False, default={})

    # Temporal tracking
    first_seen = Column(DateTime, nullable=True)
    last_seen = Column(DateTime, nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# SERVICE Node (implicit from observations, but useful to track explicitly)
# ============================================================================

class Service(Base):
    """
    A service/application running on infrastructure.

    Examples: "matrix-web", "gitlab-api", "caddy-gateway"
    Linked to observations via the subjects field.
    """
    __tablename__ = "services"
    __table_args__ = (
        Index("idx_service_node_id", "node_id", unique=True),
        Index("idx_service_user", "user_id"),
        Index("idx_service_identifier", "identifier"),
        Index("idx_service_labels_gin", "labels", postgresql_using="gin"),
        CheckConstraint("char_length(identifier) >= 1", name="check_service_identifier_not_empty"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(512), nullable=False, unique=True)  # e.g., "svc:matrix-web"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Service metadata
    identifier = Column(String(256), nullable=False)  # Service name
    labels = Column(JSONB, nullable=False, default={})
    description = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# AOI (Area of Interest) Node
# ============================================================================

class AreaOfInterest(Base):
    """
    A geographic area of interest (country, region, city, facility, etc.)

    Defined by a GeoJSON polygon. Observations are matched against AOIs
    using point-in-polygon checks in the geo-enricher service.
    """
    __tablename__ = "areas_of_interest"
    __table_args__ = (
        Index("idx_aoi_node_id", "node_id", unique=True),
        Index("idx_aoi_user", "user_id"),
        Index("idx_aoi_category", "category"),
        Index("idx_aoi_name", "name"),
        Index("idx_aoi_labels_gin", "labels", postgresql_using="gin"),
        # GiST index for geometry (requires PostGIS extension, optional)
        # Index("idx_aoi_geometry_gist", "geometry", postgresql_using="gist"),
        CheckConstraint("char_length(name) >= 1", name="check_aoi_name_not_empty"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(512), nullable=False, unique=True)  # e.g., "aoi:lebanon-beirut-city"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # AOI metadata
    name = Column(String(256), nullable=False, index=True)
    category = Column(
        SQLEnum("COUNTRY", "REGION", "CITY", "FACILITY", "CUSTOM", name="aoi_category"),
        nullable=False
    )
    description = Column(Text, nullable=True)
    labels = Column(JSONB, nullable=False, default={})

    # GeoJSON geometry
    # Stored as JSONB for flexibility. For production with PostGIS, could use Geometry type.
    geometry = Column(JSONB, nullable=False)  # Stores GeoJSONGeometry schema

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    @hybrid_property
    def geometry_type(self):
        """Extract geometry type from GeoJSON"""
        if not self.geometry:
            return None
        return self.geometry.get("type")


# ============================================================================
# Edge / Relationship Tables (optional, for explicit graph modeling)
# ============================================================================

class ObservationLink(Base):
    """
    Explicit edges between observations and other nodes.

    While subjects are stored as JSONB in observations, this table
    provides a normalized way to query relationships.
    """
    __tablename__ = "observation_links"
    __table_args__ = (
        Index("idx_obslink_from", "from_observation_id"),
        Index("idx_obslink_to", "to_node_id"),
        Index("idx_obslink_type", "link_type"),
        Index("idx_obslink_composite", "from_observation_id", "to_node_id", "link_type"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_observation_id = Column(UUID(as_uuid=True), ForeignKey("observations.id"), nullable=False)
    to_node_id = Column(String(512), nullable=False)  # Can reference any node_id
    link_type = Column(String(64), nullable=False)  # "SUBJECT", "ENRICHED_BY", "RELATED_TO", etc.
    metadata = Column(JSONB, nullable=False, default={})

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
