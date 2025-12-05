"""
FVOAS Voice Intelligence Ingest Plugin

Ingests voice telemetry from FVOAS into DSMILBrain memory fabric.

Classification: SECRET
Device: 9 (Audio) | Layer: 3 | Clearance: 0x03030303

This plugin provides:
- Voice telemetry ingestion from FVOAS kernel driver
- Voiceprint storage in semantic memory
- Threat pattern correlation
- Continuous learning from voice features
- Intel propagation for detected threats
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import numpy as np

from ..ingest_framework import IngestPlugin, IngestResult

logger = logging.getLogger(__name__)


@dataclass
class VoiceprintRecord:
    """Stored voiceprint in semantic memory"""
    hash: str                           # SHA-384 of embedding
    embedding: Optional[np.ndarray]     # 512-dim speaker embedding
    first_seen: datetime
    last_seen: datetime
    encounter_count: int = 1
    threat_count: int = 0
    threat_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hash': self.hash,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'encounter_count': self.encounter_count,
            'threat_count': self.threat_count,
            'threat_types': self.threat_types,
            'metadata': self.metadata,
        }


class VoiceIntelligencePlugin(IngestPlugin):
    """
    Ingests voice telemetry into brain memory fabric.
    
    Memory Integration:
    - Working Memory: Current session voice features
    - Episodic Memory: Voice events with timestamps
    - Semantic Memory: Voiceprint database, speaker relationships
    
    Federation:
    - Propagates threat intel to Hub/Spoke network
    - Shares voiceprint hashes for threat correlation
    """
    
    PLUGIN_NAME = "voice_intelligence"
    PLUGIN_VERSION = "1.0.0"
    SUPPORTED_TYPES = ["voice_telemetry", "voice_threat", "voice_threat_alert"]
    
    @property
    def name(self) -> str:
        return self.PLUGIN_NAME
    
    @property
    def version(self) -> str:
        return self.PLUGIN_VERSION
    
    @property
    def description(self) -> str:
        return "FVOAS Voice Intelligence - ingests voice telemetry and threat data"
    
    @property
    def supported_types(self) -> List[str]:
        return self.SUPPORTED_TYPES
    
    def __init__(self, brain=None):
        """
        Initialize voice intelligence plugin.
        
        Args:
            brain: DSMILBrain instance
        """
        super().__init__()
        self.brain = brain
        
        # Voiceprint cache (in-memory)
        self._voiceprint_cache: Dict[str, VoiceprintRecord] = {}
        
        # Known threat patterns
        self._threat_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'telemetry_ingested': 0,
            'voiceprints_stored': 0,
            'threats_correlated': 0,
            'intel_propagated': 0,
        }
        
        logger.info(f"VoiceIntelligencePlugin initialized (v{self.PLUGIN_VERSION})")
    
    async def initialize(self, brain) -> bool:
        """Initialize plugin with brain instance"""
        self.brain = brain
        
        # Load existing voiceprints from semantic memory
        await self._load_voiceprints()
        
        # Load threat patterns
        await self._load_threat_patterns()
        
        logger.info(f"Loaded {len(self._voiceprint_cache)} voiceprints, "
                   f"{len(self._threat_patterns)} threat patterns")
        
        return True
    
    async def _load_voiceprints(self):
        """Load voiceprints from semantic memory"""
        if self.brain is None:
            return
        
        try:
            # Query semantic memory for voiceprints
            results = await self.brain.query(
                "voice voiceprint speaker",
                filters={'type': 'voiceprint'}
            )
            
            for item in results.get('items', []):
                record = VoiceprintRecord(
                    hash=item.get('hash', ''),
                    embedding=None,  # Don't load full embeddings into memory
                    first_seen=datetime.fromisoformat(item.get('first_seen', '')),
                    last_seen=datetime.fromisoformat(item.get('last_seen', '')),
                    encounter_count=item.get('encounter_count', 1),
                    threat_count=item.get('threat_count', 0),
                    threat_types=item.get('threat_types', []),
                    metadata=item.get('metadata', {}),
                )
                self._voiceprint_cache[record.hash] = record
                
        except Exception as e:
            logger.warning(f"Failed to load voiceprints: {e}")
    
    async def _load_threat_patterns(self):
        """Load known threat patterns from semantic memory"""
        if self.brain is None:
            return
        
        try:
            results = await self.brain.query(
                "voice threat pattern deepfake tts",
                filters={'type': 'voice_threat_pattern'}
            )
            
            for item in results.get('items', []):
                pattern_id = item.get('pattern_id', '')
                self._threat_patterns[pattern_id] = item
                
        except Exception as e:
            logger.warning(f"Failed to load threat patterns: {e}")
    
    async def ingest(self, data: Dict[str, Any], 
                    data_type: str = "voice_telemetry") -> IngestResult:
        """
        Ingest voice data into brain.
        
        Args:
            data: Voice telemetry/threat data
            data_type: Type of data (voice_telemetry, voice_threat, voice_threat_alert)
            
        Returns:
            IngestResult with status and metadata
        """
            if data_type not in self.SUPPORTED_TYPES:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[f"Unsupported data type: {data_type}"]
            )
        
        try:
            if data_type == "voice_telemetry":
                return await self._ingest_telemetry(data)
            elif data_type in ("voice_threat", "voice_threat_alert"):
                return await self._ingest_threat(data)
            else:
                return IngestResult(success=False, plugin_name=self.name, errors=["Unknown data type"])
                
        except Exception as e:
            logger.error(f"Ingest error: {e}")
            return IngestResult(success=False, plugin_name=self.name, errors=[str(e)])
    
    async def _ingest_telemetry(self, data: Dict[str, Any]) -> IngestResult:
        """Process voice telemetry"""
        self.stats['telemetry_ingested'] += 1
        
        # Extract key fields
        voiceprint_hash = data.get('voiceprint_hash', '')
        timestamp = data.get('timestamp', datetime.now(timezone.utc).isoformat())
        
        # Store in working memory (current session)
        if self.brain:
            self.brain._working_memory.store(
                key=f"voice_session:{timestamp}",
                value={
                    'f0': data.get('f0_median', 0),
                    'formants': data.get('formants', []),
                    'manipulation_confidence': data.get('manipulation_confidence', 0),
                    'ai_voice_probability': data.get('ai_voice_probability', 0),
                    'voiceprint_hash': voiceprint_hash,
                },
                metadata={
                    'type': 'voice_telemetry',
                    'device': 9,
                    'classification': 'SECRET',
                }
            )
        
        # Update voiceprint cache
        if voiceprint_hash:
            await self._update_voiceprint(voiceprint_hash, data)
        
        # Check for manipulation/AI voice
        if data.get('manipulation_confidence', 0) > 0.5:
            await self._check_manipulation(data)
        
        if data.get('ai_voice_probability', 0) > 0.5:
            await self._check_ai_voice(data)
        
        return IngestResult(
            success=True,
            plugin_name=self.name,
            items_ingested=1,
            metadata={'voiceprint_hash': voiceprint_hash}
        )
    
    async def _ingest_threat(self, data: Dict[str, Any]) -> IngestResult:
        """Process voice threat"""
        self.stats['threats_correlated'] += 1
        
        threat_type = data.get('threat_type', 'unknown')
        confidence = data.get('confidence', data.get('threat_confidence', 0))
        voiceprint_hash = data.get('voiceprint_hash', '')
        
        # Store in episodic memory (permanent record)
        if self.brain:
            await self.brain._episodic_memory.record_event(
                event_type='voice_threat_detected',
                data={
                    'threat_type': threat_type,
                    'confidence': confidence,
                    'voiceprint_hash': voiceprint_hash,
                    'f0': data.get('f0', 0),
                    'formants': data.get('formants', []),
                    'artifacts': data.get('artifacts', []),
                },
                metadata={
                    'source': 'fvoas',
                    'device': 9,
                    'classification': 'SECRET',
                    'priority': 'high' if confidence > 0.8 else 'medium',
                }
            )
        
        # Update voiceprint threat count
        if voiceprint_hash and voiceprint_hash in self._voiceprint_cache:
            record = self._voiceprint_cache[voiceprint_hash]
            record.threat_count += 1
            if threat_type not in record.threat_types:
                record.threat_types.append(threat_type)
        
        # Correlate with known patterns
        correlation = await self._correlate_threat_pattern(data)
        
        # Propagate intel if critical
        if confidence > 0.8 and self.brain:
            self.brain.propagate_intel({
                'type': 'voice_threat_intel',
                'threat_type': threat_type,
                'confidence': confidence,
                'voiceprint_hash': voiceprint_hash,
                'correlation': correlation,
                'source': 'fvoas',
            }, priority='critical' if confidence > 0.9 else 'high')
            
            self.stats['intel_propagated'] += 1
        
        return IngestResult(
            success=True,
            plugin_name=self.name,
            items_ingested=1,
            metadata={
                'threat_type': threat_type,
                'confidence': confidence,
                'correlation': correlation,
            }
        )
    
    async def _update_voiceprint(self, hash: str, data: Dict[str, Any]):
        """Update or create voiceprint record"""
        now = datetime.now(timezone.utc)
        
        if hash in self._voiceprint_cache:
            record = self._voiceprint_cache[hash]
            record.last_seen = now
            record.encounter_count += 1
        else:
            record = VoiceprintRecord(
                hash=hash,
                embedding=np.array(data.get('speaker_embedding', [])) if data.get('speaker_embedding') else None,
                first_seen=now,
                last_seen=now,
                encounter_count=1,
            )
            self._voiceprint_cache[hash] = record
            self.stats['voiceprints_stored'] += 1
        
        # Store/update in semantic memory
        if self.brain:
            self.brain.add_knowledge(
                subject=f"voiceprint:{hash}",
                predicate="is_speaker",
                obj=hash,
                confidence=0.95
            )
    
    async def _check_manipulation(self, data: Dict[str, Any]):
        """Check for voice manipulation patterns"""
        confidence = data.get('manipulation_confidence', 0)
        
        # Log detection
        logger.info(f"Voice manipulation detected: {confidence:.0%} confidence")
        
        # Check specific artifacts
        if data.get('artifact_signatures'):
            for artifact in data['artifact_signatures']:
                logger.debug(f"  Artifact: {artifact}")
    
    async def _check_ai_voice(self, data: Dict[str, Any]):
        """Check for AI-generated voice"""
        probability = data.get('ai_voice_probability', 0)
        
        logger.warning(f"Potential AI voice detected: {probability:.0%} probability")
    
    async def _correlate_threat_pattern(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Correlate threat with known patterns"""
        threat_type = data.get('threat_type', '')
        voiceprint_hash = data.get('voiceprint_hash', '')
        
        # Check if voiceprint has been seen with threats before
        if voiceprint_hash in self._voiceprint_cache:
            record = self._voiceprint_cache[voiceprint_hash]
            if record.threat_count > 1:
                return {
                    'known_actor': True,
                    'previous_threats': record.threat_types,
                    'encounter_count': record.encounter_count,
                    'threat_count': record.threat_count,
                }
        
        # Check against threat patterns
        for pattern_id, pattern in self._threat_patterns.items():
            if pattern.get('threat_type') == threat_type:
                return {
                    'pattern_match': True,
                    'pattern_id': pattern_id,
                    'pattern_name': pattern.get('name', 'Unknown'),
                }
        
        return None
    
    async def process_batch(self, items: List[Dict[str, Any]], 
                           data_type: str = "voice_telemetry") -> IngestResult:
        """Process batch of voice data"""
        results = []
        for item in items:
            result = await self.ingest(item, data_type)
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        
        return IngestResult(
            success=success_count == len(items),
            plugin_name=self.name,
            items_ingested=success_count,
            metadata={'total': len(items), 'success': success_count}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return {
            'plugin_name': self.PLUGIN_NAME,
            'plugin_version': self.PLUGIN_VERSION,
            **self.stats,
            'cached_voiceprints': len(self._voiceprint_cache),
            'known_threat_patterns': len(self._threat_patterns),
        }
    
    async def cleanup(self):
        """Cleanup plugin resources"""
        # Persist voiceprint cache to semantic memory
        if self.brain:
            for hash, record in self._voiceprint_cache.items():
                try:
                    self.brain.add_knowledge(
                        subject=f"voiceprint:{hash}",
                        predicate="record",
                        obj=str(record.to_dict()),
                        confidence=0.99
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist voiceprint {hash}: {e}")
        
        logger.info(f"VoiceIntelligencePlugin cleanup complete")


# Register plugin
def register_plugin():
    """Register with ingest framework"""
    return VoiceIntelligencePlugin

