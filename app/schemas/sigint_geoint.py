"""
MEMSHADOW Phase-0 SIGINT/GEOINT Schema Definitions

Unified schema for network observations, geographic intelligence,
and OSINT-enriched threat intelligence from server logs.

Based on "servers + OSINT" Phase-0 architecture.
"""

from pydantic import BaseModel, Field, UUID4, field_validator
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# Enumerations
# ============================================================================

class ObservationModality(str, Enum):
    """Type of intelligence observation"""
    SIGINT = "SIGINT"  # Signals Intelligence
    GEOINT = "GEOINT"  # Geospatial Intelligence
    OSINT = "OSINT"  # Open Source Intelligence
    HUMINT = "HUMINT"  # Human Intelligence (future)
    MASINT = "MASINT"  # Measurement and Signature Intelligence (future)


class ObservationChannel(str, Enum):
    """Channel/source of the observation"""
    HTTP_ACCESS = "HTTP_ACCESS"  # HTTP/HTTPS access logs
    NETFLOW = "NETFLOW"  # Network flow data (conntrack/ss)
    AUTH_EVENT = "AUTH_EVENT"  # SSH/sudo/login events
    DNS_QUERY = "DNS_QUERY"  # DNS queries
    TLS_HANDSHAKE = "TLS_HANDSHAKE"  # TLS connection metadata
    API_CALL = "API_CALL"  # API endpoint calls
    CUSTOM = "CUSTOM"  # Custom observation type


class LocationRole(str, Enum):
    """Role of a location in an observation"""
    REMOTE_GEOIP = "REMOTE_GEOIP"  # Location derived from remote IP via GeoIP
    LOCAL_SERVER = "LOCAL_SERVER"  # Location of local server
    TRANSIT_HOP = "TRANSIT_HOP"  # Intermediate hop in network path
    DECLARED = "DECLARED"  # Location declared by client/user


class NodeKind(str, Enum):
    """Type of node in the intelligence graph"""
    OBSERVATION = "OBSERVATION"  # A single intelligence observation
    DEVICE = "DEVICE"  # A device/IP/endpoint
    IDENTITY = "IDENTITY"  # A user/account/persona
    SERVICE = "SERVICE"  # A service/application
    AOI = "AOI"  # Area of Interest (geographic region)
    THREAT_ACTOR = "THREAT_ACTOR"  # Known threat actor
    CAMPAIGN = "CAMPAIGN"  # Attack campaign


# ============================================================================
# Sub-schemas
# ============================================================================

class GeographicLocation(BaseModel):
    """Geographic location with uncertainty"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    accuracy_m: Optional[float] = Field(None, description="Accuracy radius in meters")
    crs: str = Field("EPSG:4326", description="Coordinate Reference System")
    role: LocationRole = Field(..., description="Role of this location")
    city: Optional[str] = Field(None, description="City name from GeoIP")
    country: Optional[str] = Field(None, description="ISO country code")
    asn: Optional[int] = Field(None, description="Autonomous System Number")
    as_org: Optional[str] = Field(None, description="AS Organization name")

    class Config:
        json_schema_extra = {
            "example": {
                "lat": 40.7128,
                "lon": -74.0060,
                "accuracy_m": 50000,
                "crs": "EPSG:4326",
                "role": "REMOTE_GEOIP",
                "city": "New York",
                "country": "US",
                "asn": 15169,
                "as_org": "GOOGLE"
            }
        }


class GeoJSONGeometry(BaseModel):
    """GeoJSON geometry for AOIs"""
    type: Literal["Point", "Polygon", "MultiPolygon", "LineString"]
    crs: str = Field("EPSG:4326", description="Coordinate Reference System")
    coordinates: List[Any] = Field(..., description="GeoJSON coordinates array")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "Polygon",
                "crs": "EPSG:4326",
                "coordinates": [
                    [[35.45, 33.84], [35.62, 33.84], [35.62, 33.96], [35.45, 33.96], [35.45, 33.84]]
                ]
            }
        }


class SubjectReference(BaseModel):
    """Reference to a subject (DEVICE, IDENTITY, SERVICE) in an observation"""
    type: Literal["DEVICE", "IDENTITY", "SERVICE", "THREAT_ACTOR"]
    id: str = Field(..., description="Node ID of the subject")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "DEVICE",
                "id": "ip:203.0.113.10"
            }
        }


class SignalMetadata(BaseModel):
    """Derived signals and risk scoring"""
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Risk score (0-1)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in assessment")
    threat_indicators: List[str] = Field(default_factory=list, description="Matched threat indicators")
    anomaly_scores: Dict[str, float] = Field(default_factory=dict, description="Anomaly detection scores")
    tags: List[str] = Field(default_factory=list, description="Analyst tags")

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 0.18,
                "confidence": 0.9,
                "threat_indicators": [],
                "anomaly_scores": {"geo_distance": 0.05},
                "tags": ["routine", "web-client"]
            }
        }


# ============================================================================
# OBSERVATION - Core Intelligence Event
# ============================================================================

class ObservationBase(BaseModel):
    """Base properties for an observation"""
    modality: ObservationModality = Field(..., description="Type of intelligence")
    source: str = Field(..., description="Source system (e.g., 'sigint-net-agent')")
    timestamp: datetime = Field(..., description="When the observation occurred")
    host: str = Field(..., description="Hostname where observation was made")
    sensor_id: str = Field(..., description="Unique sensor identifier")
    labels: Dict[str, str] = Field(default_factory=dict, description="Arbitrary labels")
    subjects: List[SubjectReference] = Field(default_factory=list, description="Related subjects")
    location: Optional[GeographicLocation] = Field(None, description="Geographic location")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Raw observation data")
    signals: SignalMetadata = Field(default_factory=SignalMetadata, description="Derived signals")
    embedding: Optional[List[float]] = Field(None, description="Semantic embedding vector")


class ObservationCreate(ObservationBase):
    """Schema for creating a new observation"""
    node_id: Optional[str] = Field(None, description="Optional custom node ID")


class ObservationUpdate(BaseModel):
    """Schema for updating an observation (rare - mostly append-only)"""
    labels: Optional[Dict[str, str]] = None
    signals: Optional[SignalMetadata] = None


class ObservationInDB(ObservationBase):
    """Observation as stored in database"""
    id: UUID4
    node_id: str = Field(..., description="Unique node identifier")
    user_id: UUID4 = Field(..., description="Owner user ID")
    created_at: datetime
    updated_at: datetime
    aoi_memberships: List[str] = Field(default_factory=list, description="AOI node IDs this obs belongs to")

    class Config:
        from_attributes = True


class ObservationResponse(ObservationInDB):
    """Response schema for observation queries"""
    pass


# ============================================================================
# DEVICE Node
# ============================================================================

class DeviceBase(BaseModel):
    """Base properties for a device"""
    device_type: Literal["IP", "MAC", "IMEI", "HOSTNAME", "FINGERPRINT"] = "IP"
    identifier: str = Field(..., description="Device identifier (e.g., IP address)")
    labels: Dict[str, str] = Field(default_factory=dict, description="Device labels")
    first_seen: Optional[datetime] = Field(None, description="First observation time")
    last_seen: Optional[datetime] = Field(None, description="Most recent observation")
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class DeviceCreate(DeviceBase):
    """Schema for creating a device node"""
    node_id: Optional[str] = Field(None, description="Optional custom node ID (e.g., 'ip:1.2.3.4')")


class DeviceInDB(DeviceBase):
    """Device as stored in database"""
    id: UUID4
    node_id: str
    user_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DeviceResponse(DeviceInDB):
    """Response schema for device queries"""
    observation_count: Optional[int] = Field(None, description="Number of observations")


# ============================================================================
# IDENTITY Node
# ============================================================================

class IdentityBase(BaseModel):
    """Base properties for an identity"""
    identity_type: Literal["ACCOUNT", "USERNAME", "EMAIL", "OAUTH", "CERTIFICATE"] = "ACCOUNT"
    identifier: str = Field(..., description="Identity identifier")
    labels: Dict[str, str] = Field(default_factory=dict)
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


class IdentityCreate(IdentityBase):
    """Schema for creating an identity node"""
    node_id: Optional[str] = Field(None, description="Optional custom node ID")


class IdentityInDB(IdentityBase):
    """Identity as stored in database"""
    id: UUID4
    node_id: str
    user_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class IdentityResponse(IdentityInDB):
    """Response schema for identity queries"""
    observation_count: Optional[int] = None


# ============================================================================
# AOI (Area of Interest) Node
# ============================================================================

class AOIBase(BaseModel):
    """Base properties for an Area of Interest"""
    name: str = Field(..., description="Human-readable AOI name")
    category: Literal["COUNTRY", "REGION", "CITY", "FACILITY", "CUSTOM"] = "CUSTOM"
    geometry: GeoJSONGeometry = Field(..., description="GeoJSON geometry")
    labels: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = Field(None, description="AOI description")


class AOICreate(AOIBase):
    """Schema for creating an AOI"""
    node_id: Optional[str] = Field(None, description="Optional custom node ID (e.g., 'aoi:beirut')")


class AOIInDB(AOIBase):
    """AOI as stored in database"""
    id: UUID4
    node_id: str
    user_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AOIResponse(AOIInDB):
    """Response schema for AOI queries"""
    observation_count: Optional[int] = Field(None, description="Observations in this AOI")


# ============================================================================
# Query Schemas
# ============================================================================

class ObservationQuery(BaseModel):
    """Query parameters for searching observations"""
    modality: Optional[ObservationModality] = None
    channel: Optional[ObservationChannel] = None
    source: Optional[str] = None
    host: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    aoi_ids: Optional[List[str]] = Field(None, description="Filter by AOI memberships")
    subject_ids: Optional[List[str]] = Field(None, description="Filter by subject node IDs")
    min_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    labels: Optional[Dict[str, str]] = None
    limit: int = Field(100, ge=1, le=10000)
    offset: int = Field(0, ge=0)


class TimelineQuery(BaseModel):
    """Query for time-series analysis of observations"""
    entity_id: str = Field(..., description="Device, Identity, or Service node ID")
    start_time: datetime
    end_time: datetime
    granularity: Literal["hour", "day", "week"] = "day"
    metrics: List[str] = Field(
        default=["count", "unique_sources", "avg_risk"],
        description="Metrics to compute"
    )


class GeoClusterQuery(BaseModel):
    """Query for geographic clustering of observations"""
    modality: Optional[ObservationModality] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_cluster_size: int = Field(5, ge=1)
    max_distance_km: float = Field(100.0, gt=0)


# ============================================================================
# Enrichment Schemas
# ============================================================================

class GeoIPEnrichmentRequest(BaseModel):
    """Request to enrich an IP address with GeoIP data"""
    ip_address: str = Field(..., description="IP address to enrich")
    observation_id: Optional[str] = Field(None, description="Associated observation ID")


class GeoIPEnrichmentResponse(BaseModel):
    """GeoIP enrichment result"""
    ip_address: str
    location: Optional[GeographicLocation]
    success: bool
    error: Optional[str] = None


class AOIMembershipUpdate(BaseModel):
    """Update AOI memberships for observations"""
    observation_ids: List[str] = Field(..., description="Observation node IDs to update")
    force_recompute: bool = Field(False, description="Force recomputation of point-in-polygon")
