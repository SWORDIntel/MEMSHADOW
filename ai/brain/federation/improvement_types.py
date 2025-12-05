#!/usr/bin/env python3
"""
Self-Improvement Data Types for DSMIL Brain Federation

Defines data structures for propagating improvements across nodes:
- Model weight deltas (compressed updates)
- Configuration tuning values
- Learned patterns and correlations

All improvements are measured for effectiveness before propagation.
"""

import json
import hashlib
import gzip
import base64
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum, IntEnum


class ImprovementType(IntEnum):
    """Types of improvements that can be propagated"""
    MODEL_WEIGHTS = 1      # Neural network weight updates
    CONFIG_TUNING = 2      # Configuration parameter changes
    LEARNED_PATTERN = 3    # Discovered patterns/correlations
    HYPERPARAMETER = 4     # Hyperparameter optimization results
    FEATURE_ENGINEERING = 5  # Feature extraction improvements
    ARCHITECTURE = 6       # Model architecture changes


class ImprovementPriority(IntEnum):
    """Priority for improvement propagation"""
    LOW = 0        # Minor improvement, propagate via hub
    NORMAL = 1     # Standard improvement, hybrid (hub preferred)
    HIGH = 2       # Significant improvement, direct P2P preferred
    CRITICAL = 3   # Major breakthrough, direct P2P required


class CompatibilityLevel(IntEnum):
    """Compatibility level for improvements"""
    UNIVERSAL = 0      # Works on all nodes
    VERSION_SPECIFIC = 1  # Requires specific version
    ARCHITECTURE_SPECIFIC = 2  # Requires specific hardware/architecture
    CUSTOM = 3        # Requires custom validation


@dataclass
class PerformanceMetrics:
    """Performance metrics before/after improvement"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    latency_ms: float = 0.0
    throughput: float = 0.0
    confidence: float = 0.0
    error_rate: float = 0.0

    # Domain-specific metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def improvement_percentage(self, baseline: "PerformanceMetrics") -> float:
        """Calculate overall improvement percentage"""
        if baseline.accuracy == 0:
            return 0.0

        # Weighted improvement calculation
        accuracy_improvement = (self.accuracy - baseline.accuracy) / baseline.accuracy
        latency_improvement = (baseline.latency_ms - self.latency_ms) / baseline.latency_ms if baseline.latency_ms > 0 else 0

        # Combined score (accuracy improvement + latency reduction)
        return (accuracy_improvement * 0.7 + latency_improvement * 0.3) * 100


@dataclass
class ModelWeightDelta:
    """
    Compressed model weight updates

    Stores only the differences from baseline weights,
    compressed for efficient transmission.
    """
    model_name: str
    layer_name: str
    weight_shape: List[int]
    delta_data: bytes  # Compressed delta values
    baseline_hash: str  # Hash of baseline weights for validation
    
    # Metadata (with defaults)
    compression_method: str = "gzip"  # "gzip", "zstd", "none"

    # Metadata
    total_parameters: int = 0
    changed_parameters: int = 0
    sparsity: float = 0.0  # Percentage of zero deltas

    def pack(self) -> bytes:
        """Pack to binary format"""
        data = {
            "model_name": self.model_name,
            "layer_name": self.layer_name,
            "weight_shape": self.weight_shape,
            "delta_data": base64.b64encode(self.delta_data).decode(),
            "compression_method": self.compression_method,
            "baseline_hash": self.baseline_hash,
            "total_parameters": self.total_parameters,
            "changed_parameters": self.changed_parameters,
            "sparsity": self.sparsity,
        }
        return json.dumps(data).encode()

    @classmethod
    def unpack(cls, data: bytes) -> "ModelWeightDelta":
        """Unpack from binary format"""
        obj = json.loads(data.decode())
        return cls(
            model_name=obj["model_name"],
            layer_name=obj["layer_name"],
            weight_shape=obj["weight_shape"],
            delta_data=base64.b64decode(obj["delta_data"]),
            baseline_hash=obj["baseline_hash"],
            compression_method=obj.get("compression_method", "gzip"),
            total_parameters=obj.get("total_parameters", 0),
            changed_parameters=obj.get("changed_parameters", 0),
            sparsity=obj.get("sparsity", 0.0),
        )

    def get_size_bytes(self) -> int:
        """Get size in bytes"""
        return len(self.delta_data)


@dataclass
class ConfigTuning:
    """
    Configuration parameter tuning values

    Stores changes to thresholds, intervals, weights, etc.
    """
    config_category: str  # e.g., "risk_assessment", "memory_management"
    parameter_name: str
    old_value: Any
    new_value: Any
    value_type: str  # "float", "int", "bool", "str", "dict"

    # Context
    tuning_method: str = "manual"  # "manual", "grid_search", "bayesian", "auto"
    validation_score: float = 0.0

    def pack(self) -> bytes:
        """Pack to binary format"""
        data = {
            "config_category": self.config_category,
            "parameter_name": self.parameter_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "value_type": self.value_type,
            "tuning_method": self.tuning_method,
            "validation_score": self.validation_score,
        }
        return json.dumps(data, default=str).encode()

    @classmethod
    def unpack(cls, data: bytes) -> "ConfigTuning":
        """Unpack from binary format"""
        obj = json.loads(data.decode())
        return cls(
            config_category=obj["config_category"],
            parameter_name=obj["parameter_name"],
            old_value=obj["old_value"],
            new_value=obj["new_value"],
            value_type=obj["value_type"],
            tuning_method=obj.get("tuning_method", "manual"),
            validation_score=obj.get("validation_score", 0.0),
        )


@dataclass
class LearnedPattern:
    """
    Discovered pattern or correlation

    Stores patterns learned from local data that may be useful
    to other nodes.
    """
    pattern_id: str
    pattern_type: str  # "correlation", "anomaly", "sequence", "cluster", "rule"
    description: str
    pattern_data: Dict[str, Any]  # Pattern-specific data

    # Context
    domain: str  # e.g., "threat_intel", "user_behavior", "network_traffic"
    confidence: float = 0.0
    support_count: int = 0  # Number of examples supporting this pattern
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Applicability
    applicable_domains: List[str] = field(default_factory=list)
    prerequisites: Dict[str, Any] = field(default_factory=dict)

    def pack(self) -> bytes:
        """Pack to binary format"""
        data = {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "pattern_data": self.pattern_data,
            "domain": self.domain,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "discovered_at": self.discovered_at.isoformat(),
            "applicable_domains": self.applicable_domains,
            "prerequisites": self.prerequisites,
        }
        return json.dumps(data, default=str).encode()

    @classmethod
    def unpack(cls, data: bytes) -> "LearnedPattern":
        """Unpack from binary format"""
        obj = json.loads(data.decode())
        return cls(
            pattern_id=obj["pattern_id"],
            pattern_type=obj["pattern_type"],
            description=obj["description"],
            pattern_data=obj["pattern_data"],
            domain=obj["domain"],
            confidence=obj.get("confidence", 0.0),
            support_count=obj.get("support_count", 0),
            discovered_at=datetime.fromisoformat(obj["discovered_at"]),
            applicable_domains=obj.get("applicable_domains", []),
            prerequisites=obj.get("prerequisites", {}),
        )


@dataclass
class ImprovementPackage:
    """
    Complete improvement package ready for propagation

    Contains one or more improvement types with metadata.
    """
    improvement_id: str
    improvement_type: ImprovementType
    priority: ImprovementPriority
    compatibility: CompatibilityLevel

    # Source information
    source_node_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0"

    # Performance metrics
    baseline_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    improved_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    improvement_percentage: float = 0.0

    # Improvement data (one of these)
    weight_delta: Optional[ModelWeightDelta] = None
    config_tuning: Optional[ConfigTuning] = None
    learned_pattern: Optional[LearnedPattern] = None

    # Additional data for other improvement types
    raw_data: Optional[bytes] = None
    raw_data_type: Optional[str] = None

    # Compatibility requirements
    requires_version: Optional[str] = None
    requires_architecture: Optional[str] = None
    requires_capabilities: List[str] = field(default_factory=list)

    # Validation
    validation_hash: str = ""
    signature: Optional[bytes] = None  # Cryptographic signature

    def __post_init__(self):
        """Calculate improvement percentage after initialization"""
        if self.baseline_metrics and self.improved_metrics:
            self.improvement_percentage = self.improved_metrics.improvement_percentage(
                self.baseline_metrics
            )

    def pack(self) -> bytes:
        """Pack to binary format for transmission"""
        data = {
            "improvement_id": self.improvement_id,
            "improvement_type": self.improvement_type.value,
            "priority": self.priority.value,
            "compatibility": self.compatibility.value,
            "source_node_id": self.source_node_id,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "baseline_metrics": asdict(self.baseline_metrics),
            "improved_metrics": asdict(self.improved_metrics),
            "improvement_percentage": self.improvement_percentage,
            "requires_version": self.requires_version,
            "requires_architecture": self.requires_architecture,
            "requires_capabilities": self.requires_capabilities,
            "validation_hash": self.validation_hash,
        }

        # Pack improvement data
        if self.weight_delta:
            data["weight_delta"] = base64.b64encode(self.weight_delta.pack()).decode()
        if self.config_tuning:
            data["config_tuning"] = base64.b64encode(self.config_tuning.pack()).decode()
        if self.learned_pattern:
            data["learned_pattern"] = base64.b64encode(self.learned_pattern.pack()).decode()
        if self.raw_data:
            data["raw_data"] = base64.b64encode(self.raw_data).decode()
            data["raw_data_type"] = self.raw_data_type

        if self.signature:
            data["signature"] = base64.b64encode(self.signature).decode()

        return json.dumps(data, default=str).encode()

    @classmethod
    def unpack(cls, data: bytes) -> "ImprovementPackage":
        """Unpack from binary format"""
        obj = json.loads(data.decode())

        package = cls(
            improvement_id=obj["improvement_id"],
            improvement_type=ImprovementType(obj["improvement_type"]),
            priority=ImprovementPriority(obj["priority"]),
            compatibility=CompatibilityLevel(obj["compatibility"]),
            source_node_id=obj["source_node_id"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            version=obj.get("version", "1.0"),
            baseline_metrics=PerformanceMetrics(**obj.get("baseline_metrics", {})),
            improved_metrics=PerformanceMetrics(**obj.get("improved_metrics", {})),
            improvement_percentage=obj.get("improvement_percentage", 0.0),
            requires_version=obj.get("requires_version"),
            requires_architecture=obj.get("requires_architecture"),
            requires_capabilities=obj.get("requires_capabilities", []),
            validation_hash=obj.get("validation_hash", ""),
        )

        # Unpack improvement data
        if "weight_delta" in obj:
            package.weight_delta = ModelWeightDelta.unpack(
                base64.b64decode(obj["weight_delta"])
            )
        if "config_tuning" in obj:
            package.config_tuning = ConfigTuning.unpack(
                base64.b64decode(obj["config_tuning"])
            )
        if "learned_pattern" in obj:
            package.learned_pattern = LearnedPattern.unpack(
                base64.b64decode(obj["learned_pattern"])
            )
        if "raw_data" in obj:
            package.raw_data = base64.b64decode(obj["raw_data"])
            package.raw_data_type = obj.get("raw_data_type")

        if "signature" in obj:
            package.signature = base64.b64decode(obj["signature"])

        return package

    def compute_hash(self) -> str:
        """Compute validation hash of improvement data"""
        hash_data = {
            "improvement_id": self.improvement_id,
            "improvement_type": self.improvement_type.value,
            "source_node_id": self.source_node_id,
            "created_at": self.improved_metrics.created_at.isoformat() if hasattr(self.improved_metrics, 'created_at') else self.created_at.isoformat(),
        }

        # Include improvement-specific data
        if self.weight_delta:
            hash_data["weight_delta"] = self.weight_delta.baseline_hash
        if self.config_tuning:
            hash_data["config"] = f"{self.config_tuning.config_category}:{self.config_tuning.parameter_name}"
        if self.learned_pattern:
            hash_data["pattern"] = self.learned_pattern.pattern_id

        hash_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def is_critical(self) -> bool:
        """Check if improvement is critical (requires direct P2P)"""
        return (self.priority == ImprovementPriority.CRITICAL or
                self.improvement_percentage >= 15.0)  # 15%+ improvement is critical

    def get_size_bytes(self) -> int:
        """Get total size in bytes"""
        size = len(self.pack())
        if self.weight_delta:
            size += self.weight_delta.get_size_bytes()
        return size


@dataclass
class ImprovementAnnouncement:
    """
    Announcement of available improvement

    Sent to broadcast availability, contains summary only.
    """
    improvement_id: str
    improvement_type: ImprovementType
    priority: ImprovementPriority
    source_node_id: str
    improvement_percentage: float
    size_bytes: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Compatibility hints
    compatibility: CompatibilityLevel = CompatibilityLevel.UNIVERSAL
    requires_version: Optional[str] = None

    def pack(self) -> bytes:
        """Pack to binary format"""
        data = {
            "improvement_id": self.improvement_id,
            "improvement_type": self.improvement_type.value,
            "priority": self.priority.value,
            "source_node_id": self.source_node_id,
            "improvement_percentage": self.improvement_percentage,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "compatibility": self.compatibility.value,
            "requires_version": self.requires_version,
        }
        return json.dumps(data).encode()

    @classmethod
    def unpack(cls, data: bytes) -> "ImprovementAnnouncement":
        """Unpack from binary format"""
        obj = json.loads(data.decode())
        return cls(
            improvement_id=obj["improvement_id"],
            improvement_type=ImprovementType(obj["improvement_type"]),
            priority=ImprovementPriority(obj["priority"]),
            source_node_id=obj["source_node_id"],
            improvement_percentage=obj["improvement_percentage"],
            size_bytes=obj["size_bytes"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            compatibility=CompatibilityLevel(obj.get("compatibility", 0)),
            requires_version=obj.get("requires_version"),
        )


@dataclass
class ImprovementAck:
    """Acknowledgment of improvement application"""
    improvement_id: str
    node_id: str
    success: bool
    applied_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Results
    new_metrics: Optional[PerformanceMetrics] = None
    error_message: Optional[str] = None

    def pack(self) -> bytes:
        """Pack to binary format"""
        data = {
            "improvement_id": self.improvement_id,
            "node_id": self.node_id,
            "success": self.success,
            "applied_at": self.applied_at.isoformat(),
        }
        if self.new_metrics:
            data["new_metrics"] = asdict(self.new_metrics)
        if self.error_message:
            data["error_message"] = self.error_message
        return json.dumps(data).encode()

    @classmethod
    def unpack(cls, data: bytes) -> "ImprovementAck":
        """Unpack from binary format"""
        obj = json.loads(data.decode())
        ack = cls(
            improvement_id=obj["improvement_id"],
            node_id=obj["node_id"],
            success=obj["success"],
            applied_at=datetime.fromisoformat(obj["applied_at"]),
        )
        if "new_metrics" in obj:
            ack.new_metrics = PerformanceMetrics(**obj["new_metrics"])
        if "error_message" in obj:
            ack.error_message = obj["error_message"]
        return ack


@dataclass
class ImprovementReject:
    """Rejection of improvement (incompatible or degraded performance)"""
    improvement_id: str
    node_id: str
    reason: str
    rejected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # If tested, include metrics
    tested_metrics: Optional[PerformanceMetrics] = None

    def pack(self) -> bytes:
        """Pack to binary format"""
        data = {
            "improvement_id": self.improvement_id,
            "node_id": self.node_id,
            "reason": self.reason,
            "rejected_at": self.rejected_at.isoformat(),
        }
        if self.tested_metrics:
            data["tested_metrics"] = asdict(self.tested_metrics)
        return json.dumps(data).encode()

    @classmethod
    def unpack(cls, data: bytes) -> "ImprovementReject":
        """Unpack from binary format"""
        obj = json.loads(data.decode())
        reject = cls(
            improvement_id=obj["improvement_id"],
            node_id=obj["node_id"],
            reason=obj["reason"],
            rejected_at=datetime.fromisoformat(obj["rejected_at"]),
        )
        if "tested_metrics" in obj:
            reject.tested_metrics = PerformanceMetrics(**obj["tested_metrics"])
        return reject

