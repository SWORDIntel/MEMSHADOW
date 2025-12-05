"""
Improvement Types

Defines types for self-improvement propagation in the DSMIL Brain Federation.

Based on: HUB_DOCS/DSMIL Brain Federation.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class ImprovementType(Enum):
    """Types of improvements that can be propagated"""
    MODEL_WEIGHTS = "model_weights"      # Compressed neural network updates
    CONFIG_TUNING = "config_tuning"      # Threshold and parameter changes
    LEARNED_PATTERNS = "learned_patterns"  # Discovered correlations and patterns
    EMBEDDING_UPDATE = "embedding_update"  # Updated embeddings
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"


class ImprovementPriority(Enum):
    """Improvement propagation priority"""
    CRITICAL = "critical"    # >20% gain, direct P2P
    NORMAL = "normal"        # 10-20% gain, hub relay
    MINOR = "minor"          # <10% gain, background


@dataclass
class ImprovementPackage:
    """Package containing an improvement to propagate"""
    improvement_id: str = field(default_factory=lambda: str(uuid4()))
    improvement_type: ImprovementType = ImprovementType.LEARNED_PATTERNS
    source_node: str = ""
    version: str = "1.0.0"
    gain_percent: float = 0.0
    priority: ImprovementPriority = ImprovementPriority.NORMAL
    
    # Payload
    data: bytes = b""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    checksum: str = ""
    signature: Optional[str] = None  # PQC signature if applicable
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Compatibility
    required_version: str = "1.0.0"
    compatible_types: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.checksum and self.data:
            import hashlib
            self.checksum = hashlib.sha256(self.data).hexdigest()[:16]
    
    def is_compatible(self, target_version: str) -> bool:
        """Check if improvement is compatible with target version"""
        # Simple version comparison
        try:
            req_parts = [int(x) for x in self.required_version.split(".")]
            tgt_parts = [int(x) for x in target_version.split(".")]
            return tgt_parts >= req_parts
        except:
            return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImprovementPackage":
        """Create from dictionary"""
        return cls(
            improvement_id=data.get("improvement_id", str(uuid4())),
            improvement_type=ImprovementType(data.get("improvement_type", "learned_patterns")),
            source_node=data.get("source_node", ""),
            version=data.get("version", "1.0.0"),
            gain_percent=data.get("gain_percent", 0.0),
            priority=ImprovementPriority(data.get("priority", "normal")),
            data=bytes.fromhex(data.get("data", "")),
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
            signature=data.get("signature"),
            required_version=data.get("required_version", "1.0.0"),
            compatible_types=data.get("compatible_types", []),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "improvement_id": self.improvement_id,
            "improvement_type": self.improvement_type.value,
            "source_node": self.source_node,
            "version": self.version,
            "gain_percent": self.gain_percent,
            "priority": self.priority.value,
            "data": self.data.hex(),
            "metadata": self.metadata,
            "checksum": self.checksum,
            "signature": self.signature,
            "created_at": self.created_at.isoformat(),
            "required_version": self.required_version,
            "compatible_types": self.compatible_types,
        }


@dataclass
class ImprovementMetrics:
    """Metrics for tracking improvement effectiveness"""
    improvement_id: str
    source_node: str
    applied_at: datetime = field(default_factory=datetime.utcnow)
    
    # Before/after metrics
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    latency_before_ms: float = 0.0
    latency_after_ms: float = 0.0
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    
    # Calculated
    effectiveness: float = 0.0
    
    def calculate_effectiveness(self):
        """Calculate overall effectiveness score"""
        acc_delta = self.accuracy_after - self.accuracy_before
        lat_delta = self.latency_before_ms - self.latency_after_ms  # Lower is better
        conf_delta = self.confidence_after - self.confidence_before
        
        # Weighted average (accuracy most important)
        self.effectiveness = (
            0.5 * acc_delta +
            0.2 * (lat_delta / max(self.latency_before_ms, 1)) +
            0.3 * conf_delta
        )


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "ImprovementType",
    "ImprovementPriority",
    "ImprovementPackage",
    "ImprovementMetrics",
]
