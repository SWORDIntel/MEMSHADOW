"""
KP14 Schemas for MEMSHADOW

Pydantic schemas for KP14 analysis data received via mesh sync.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ThreatSeverity(str, Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IOCType(str, Enum):
    """IOC types"""
    IP = "ip"
    DOMAIN = "domain"
    URL = "url"
    HASH_MD5 = "hash_md5"
    HASH_SHA1 = "hash_sha1"
    HASH_SHA256 = "hash_sha256"
    EMAIL = "email"
    FILE_PATH = "file_path"
    REGISTRY_KEY = "registry_key"
    MUTEX = "mutex"
    USER_AGENT = "user_agent"


# ============ Sample Schemas ============

class SampleInfo(BaseModel):
    """Information about analyzed sample"""
    hash_sha256: str = Field(..., description="SHA256 hash")
    hash_md5: Optional[str] = Field(None, description="MD5 hash")
    hash_sha1: Optional[str] = Field(None, description="SHA1 hash")
    file_type: Optional[str] = Field(None, description="File type")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_name: Optional[str] = Field(None, description="Original filename")


class AnalysisSummary(BaseModel):
    """Analysis summary"""
    threat_score: int = Field(..., ge=0, le=100, description="Threat score 0-100")
    severity: ThreatSeverity = Field(..., description="Threat severity")
    malware_family: Optional[str] = Field(None, description="Identified malware family")
    threat_actor: Optional[str] = Field(None, description="Attributed threat actor")
    campaign: Optional[str] = Field(None, description="Associated campaign")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============ IOC Schemas ============

class IOC(BaseModel):
    """Indicator of Compromise"""
    type: IOCType = Field(..., description="IOC type")
    value: str = Field(..., description="IOC value")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence score")
    context: Optional[str] = Field(None, description="Context/description")
    first_seen: Optional[datetime] = Field(None)
    last_seen: Optional[datetime] = Field(None)
    tags: List[str] = Field(default_factory=list)


class IOCBatch(BaseModel):
    """Batch of IOCs from KP14"""
    iocs: List[IOC] = Field(..., description="List of IOCs")
    source_analysis_id: Optional[str] = Field(None, description="Source analysis ID")
    campaign_id: Optional[str] = Field(None, description="Associated campaign")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============ ATT&CK Technique Schemas ============

class AttackTechnique(BaseModel):
    """ATT&CK technique mapping"""
    technique_id: str = Field(..., description="MITRE ATT&CK ID (e.g., T1059.001)")
    name: str = Field(..., description="Technique name")
    tactic: Optional[str] = Field(None, description="Associated tactic")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    evidence: Optional[str] = Field(None, description="Evidence for mapping")


class TechniqueBatch(BaseModel):
    """Batch of techniques from KP14"""
    techniques: List[AttackTechnique] = Field(..., description="List of techniques")
    sample_hash: str = Field(..., description="Associated sample hash")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============ C2 Endpoint Schemas ============

class C2Endpoint(BaseModel):
    """Command and Control endpoint"""
    value: str = Field(..., description="C2 address/URL")
    type: str = Field("url", description="Endpoint type (url, ip, domain)")
    protocol: Optional[str] = Field(None, description="Protocol (http, https, tcp)")
    port: Optional[int] = Field(None, description="Port number")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    active: Optional[bool] = Field(None, description="Whether C2 is active")


# ============ Full Analysis Schemas ============

class KP14AnalysisCreate(BaseModel):
    """Schema for creating KP14 analysis record in MEMSHADOW"""
    analysis_id: str = Field(..., description="KP14 analysis ID")
    sample: SampleInfo = Field(..., description="Sample information")
    analysis: AnalysisSummary = Field(..., description="Analysis summary")
    iocs: List[IOC] = Field(default_factory=list, description="Extracted IOCs")
    techniques: List[AttackTechnique] = Field(default_factory=list, description="ATT&CK mappings")
    c2_endpoints: List[C2Endpoint] = Field(default_factory=list, description="C2 endpoints")
    source_node: str = Field(..., description="Source mesh node ID")
    raw_analysis: Optional[Dict[str, Any]] = Field(None, description="Full raw analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "kp14_abc123_20240101120000",
                "sample": {
                    "hash_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "file_type": "PE32",
                    "file_size": 102400
                },
                "analysis": {
                    "threat_score": 85,
                    "severity": "high",
                    "malware_family": "Cobalt Strike"
                },
                "iocs": [
                    {"type": "ip", "value": "192.168.1.100", "confidence": 0.9}
                ],
                "techniques": [
                    {"technique_id": "T1059.001", "name": "PowerShell", "confidence": 0.8}
                ],
                "source_node": "kp14-analyzer"
            }
        }


class KP14AnalysisResponse(BaseModel):
    """Response schema for KP14 analysis"""
    id: str = Field(..., description="MEMSHADOW record ID")
    analysis_id: str = Field(..., description="KP14 analysis ID")
    sample: SampleInfo
    analysis: AnalysisSummary
    ioc_count: int = Field(0, description="Number of IOCs")
    technique_count: int = Field(0, description="Number of techniques")
    created_at: datetime
    updated_at: datetime
    confidence_score: Optional[float] = Field(None, description="MEMSHADOW confidence")

    class Config:
        from_attributes = True


class KP14AnalysisSearch(BaseModel):
    """Search schema for KP14 analyses"""
    query: Optional[str] = Field(None, description="Text search query")
    hash: Optional[str] = Field(None, description="Sample hash to search")
    malware_family: Optional[str] = Field(None, description="Filter by malware family")
    threat_actor: Optional[str] = Field(None, description="Filter by threat actor")
    min_threat_score: Optional[int] = Field(None, ge=0, le=100)
    max_threat_score: Optional[int] = Field(None, ge=0, le=100)
    technique_ids: Optional[List[str]] = Field(None, description="Filter by technique IDs")
    ioc_value: Optional[str] = Field(None, description="Search by IOC value")
    date_from: Optional[datetime] = Field(None, description="Filter from date")
    date_to: Optional[datetime] = Field(None, description="Filter to date")
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


# ============ Mesh Message Schemas ============

class MeshThreatIntel(BaseModel):
    """Threat intelligence message received via mesh"""
    type: str = Field(..., description="Message type (kp14_analysis, kp14_iocs, etc)")
    analysis_id: Optional[str] = Field(None)
    sample: Optional[SampleInfo] = Field(None)
    analysis: Optional[AnalysisSummary] = Field(None)
    iocs: List[IOC] = Field(default_factory=list)
    techniques: List[AttackTechnique] = Field(default_factory=list)
    c2_endpoints: List[C2Endpoint] = Field(default_factory=list)
    source_node: str = Field(..., description="Source mesh node")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MeshIOCBroadcast(BaseModel):
    """IOC broadcast message received via mesh"""
    type: str = Field("kp14_iocs")
    iocs: List[IOC] = Field(..., description="IOCs being broadcast")
    campaign_id: Optional[str] = Field(None)
    count: int = Field(0)
    source_node: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

