"""
KP14 Enrichment Service

Pre and post-analysis enrichment for KP14 data using MEMSHADOW's
intelligence stores and knowledge graph.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class KP14EnrichmentService:
    """
    Enrichment service for KP14 analysis data.

    Provides:
    - Pre-analysis enrichment: Check for prior analyses, related samples
    - Post-analysis enrichment: Add context from knowledge graph, confidence scoring
    """

    def __init__(
        self,
        knowledge_graph=None,
        memory_service=None,
        enable_pre_enrichment: bool = True,
        enable_post_enrichment: bool = True
    ):
        """
        Initialize enrichment service.

        Args:
            knowledge_graph: KP14KnowledgeGraph instance
            memory_service: MEMSHADOW MemoryService instance
            enable_pre_enrichment: Enable pre-analysis enrichment
            enable_post_enrichment: Enable post-analysis enrichment
        """
        self.knowledge_graph = knowledge_graph
        self.memory_service = memory_service
        self.enable_pre_enrichment = enable_pre_enrichment
        self.enable_post_enrichment = enable_post_enrichment

        # Enrichment statistics
        self.stats = {
            'pre_enrichments': 0,
            'post_enrichments': 0,
            'related_samples_found': 0,
            'prior_analyses_found': 0,
        }

        logger.info("KP14 Enrichment Service initialized")

    def pre_enrich(
        self,
        sample_hash: str,
        sample_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Pre-analysis enrichment.

        Check for prior knowledge about the sample before analysis.

        Args:
            sample_hash: Sample SHA256 hash
            sample_info: Optional sample metadata

        Returns:
            Enrichment context for analysis
        """
        if not self.enable_pre_enrichment:
            return {}

        context = {
            'sample_hash': sample_hash,
            'prior_analysis': None,
            'related_samples': [],
            'known_family': None,
            'known_actor': None,
            'known_techniques': [],
            'known_iocs': [],
            'enrichment_timestamp': datetime.now(timezone.utc).isoformat(),
        }

        # Check knowledge graph for prior info
        if self.knowledge_graph:
            try:
                # Find related samples
                related = self.knowledge_graph.find_related_samples(
                    sample_hash, max_depth=2, limit=5
                )
                if related:
                    context['related_samples'] = related
                    self.stats['related_samples_found'] += len(related)

                # Check if sample exists in graph
                sample_node_id = f"sample:{sample_hash[:16]}"
                if sample_node_id in self.knowledge_graph._nodes:
                    node = self.knowledge_graph._nodes[sample_node_id]
                    context['prior_analysis'] = {
                        'exists': True,
                        'threat_score': node.attributes.get('threat_score'),
                        'first_seen': node.first_seen.isoformat(),
                        'last_seen': node.last_seen.isoformat(),
                    }
                    self.stats['prior_analyses_found'] += 1

                    # Get associated family/actor from edges
                    for (src, tgt, etype), edge in self.knowledge_graph._edges.items():
                        if src == sample_node_id:
                            target_node = self.knowledge_graph._nodes.get(tgt)
                            if target_node:
                                if target_node.node_type.value == 'malware_family':
                                    context['known_family'] = target_node.name
                                elif target_node.node_type.value == 'threat_actor':
                                    context['known_actor'] = target_node.name
                                elif target_node.node_type.value == 'technique':
                                    context['known_techniques'].append({
                                        'id': target_node.attributes.get('technique_id'),
                                        'name': target_node.name,
                                    })

            except Exception as e:
                logger.warning(f"Pre-enrichment from knowledge graph failed: {e}")

        self.stats['pre_enrichments'] += 1
        logger.debug(f"Pre-enrichment completed for {sample_hash[:16]}")

        return context

    def post_enrich(
        self,
        analysis_result: Dict[str, Any],
        pre_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Post-analysis enrichment.

        Enhance analysis result with additional context and confidence scoring.

        Args:
            analysis_result: KP14 analysis result
            pre_context: Optional pre-enrichment context

        Returns:
            Enriched analysis result
        """
        if not self.enable_post_enrichment:
            return analysis_result

        enriched = analysis_result.copy()

        # Add pre-context if available
        if pre_context:
            enriched['pre_analysis_context'] = pre_context

        # Calculate enrichment scores
        enriched['enrichment'] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'confidence_adjustments': [],
            'additional_context': {},
        }

        sample_hash = analysis_result.get('sample', {}).get('hash_sha256', '')
        analysis = analysis_result.get('analysis', {})

        # Enrich from knowledge graph
        if self.knowledge_graph and sample_hash:
            try:
                # Find related samples for correlation
                related = self.knowledge_graph.find_related_samples(
                    sample_hash, max_depth=2, limit=10
                )

                if related:
                    enriched['enrichment']['related_samples'] = related

                    # Correlate threat scores
                    related_scores = [r.get('threat_score', 0) for r in related]
                    avg_related_score = sum(related_scores) / len(related_scores) if related_scores else 0

                    enriched['enrichment']['additional_context']['related_avg_threat_score'] = avg_related_score

                    # If our score differs significantly from related, note it
                    our_score = analysis.get('threat_score', 0)
                    if abs(our_score - avg_related_score) > 20:
                        enriched['enrichment']['confidence_adjustments'].append({
                            'reason': 'score_deviation_from_related',
                            'adjustment': -0.1 if our_score < avg_related_score else 0.05,
                            'note': f"Score differs from related samples avg ({avg_related_score:.1f})"
                        })

                # Check for family consistency
                detected_family = analysis.get('malware_family')
                if detected_family and pre_context and pre_context.get('known_family'):
                    known_family = pre_context['known_family']
                    if detected_family.lower() == known_family.lower():
                        enriched['enrichment']['confidence_adjustments'].append({
                            'reason': 'family_confirmed',
                            'adjustment': 0.1,
                            'note': f"Family '{detected_family}' matches prior knowledge"
                        })
                    else:
                        enriched['enrichment']['confidence_adjustments'].append({
                            'reason': 'family_mismatch',
                            'adjustment': -0.15,
                            'note': f"Family '{detected_family}' differs from known '{known_family}'"
                        })

            except Exception as e:
                logger.warning(f"Post-enrichment from knowledge graph failed: {e}")

        # Calculate final confidence score
        base_confidence = self._calculate_base_confidence(analysis_result)
        adjustments = enriched['enrichment'].get('confidence_adjustments', [])
        total_adjustment = sum(a.get('adjustment', 0) for a in adjustments)

        final_confidence = max(0.0, min(1.0, base_confidence + total_adjustment))
        enriched['enrichment']['confidence'] = {
            'base': base_confidence,
            'adjustments': total_adjustment,
            'final': final_confidence,
        }

        # Add threat intelligence context
        enriched['enrichment']['threat_context'] = self._build_threat_context(
            analysis_result, enriched.get('pre_analysis_context')
        )

        self.stats['post_enrichments'] += 1
        logger.debug(f"Post-enrichment completed for {sample_hash[:16]}")

        return enriched

    def enrich_iocs(
        self,
        iocs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich IOCs with additional context from knowledge graph.

        Args:
            iocs: List of IOCs

        Returns:
            Enriched IOC list
        """
        if not self.knowledge_graph:
            return iocs

        enriched_iocs = []
        for ioc in iocs:
            enriched = ioc.copy()
            ioc_value = ioc.get('value', '')

            # Find samples containing this IOC
            samples = self.knowledge_graph.get_ioc_samples(ioc_value)
            if samples:
                enriched['known_samples'] = len(samples)
                enriched['max_threat_score'] = max(
                    s.get('threat_score', 0) for s in samples
                )

                # Boost confidence if IOC seen in multiple samples
                if len(samples) > 1:
                    enriched['confidence'] = min(
                        1.0, ioc.get('confidence', 0.5) + 0.1 * len(samples)
                    )

            enriched_iocs.append(enriched)

        return enriched_iocs

    def enrich_techniques(
        self,
        techniques: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich ATT&CK techniques with prevalence data.

        Args:
            techniques: List of techniques

        Returns:
            Enriched technique list
        """
        if not self.knowledge_graph:
            return techniques

        enriched_techniques = []
        for technique in techniques:
            enriched = technique.copy()
            technique_id = technique.get('technique_id', '')

            # Get samples using this technique
            samples = self.knowledge_graph.get_technique_samples(technique_id, limit=100)
            if samples:
                enriched['prevalence'] = len(samples)
                enriched['avg_threat_score'] = sum(
                    s.get('threat_score', 0) for s in samples
                ) / len(samples)

            enriched_techniques.append(enriched)

        return enriched_techniques

    def _calculate_base_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate base confidence from analysis quality indicators."""
        confidence = 0.5  # Start at neutral

        analysis = analysis_result.get('analysis', {})

        # Higher threat score = more confident in detection
        threat_score = analysis.get('threat_score', 0)
        if threat_score >= 80:
            confidence += 0.2
        elif threat_score >= 60:
            confidence += 0.1
        elif threat_score <= 20:
            confidence -= 0.1

        # Having a family identified increases confidence
        if analysis.get('malware_family'):
            confidence += 0.1

        # Having techniques mapped increases confidence
        techniques = analysis_result.get('techniques', [])
        if len(techniques) >= 5:
            confidence += 0.1
        elif len(techniques) >= 2:
            confidence += 0.05

        # Having IOCs increases confidence
        iocs = analysis_result.get('iocs', [])
        if len(iocs) >= 10:
            confidence += 0.1
        elif len(iocs) >= 3:
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def _build_threat_context(
        self,
        analysis_result: Dict[str, Any],
        pre_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build threat intelligence context."""
        context = {
            'is_known_threat': False,
            'threat_type': 'unknown',
            'severity_assessment': 'unknown',
            'recommendations': [],
        }

        analysis = analysis_result.get('analysis', {})
        threat_score = analysis.get('threat_score', 0)
        malware_family = analysis.get('malware_family')
        threat_actor = analysis.get('threat_actor')

        # Determine if this is a known threat
        if pre_context and pre_context.get('prior_analysis'):
            context['is_known_threat'] = True

        # Determine threat type
        if malware_family:
            # Could enhance with family->type mapping
            context['threat_type'] = 'malware'
            if 'ransomware' in malware_family.lower():
                context['threat_type'] = 'ransomware'
            elif 'rat' in malware_family.lower() or 'trojan' in malware_family.lower():
                context['threat_type'] = 'rat'
            elif 'stealer' in malware_family.lower():
                context['threat_type'] = 'infostealer'

        # Severity assessment
        if threat_score >= 80:
            context['severity_assessment'] = 'critical'
            context['recommendations'].append('Immediate incident response required')
            context['recommendations'].append('Isolate affected systems')
        elif threat_score >= 60:
            context['severity_assessment'] = 'high'
            context['recommendations'].append('Prioritize for analysis')
            context['recommendations'].append('Check for lateral movement')
        elif threat_score >= 40:
            context['severity_assessment'] = 'medium'
            context['recommendations'].append('Schedule for detailed analysis')
        else:
            context['severity_assessment'] = 'low'
            context['recommendations'].append('Monitor for additional indicators')

        # Add actor-specific recommendations
        if threat_actor:
            context['recommendations'].append(
                f'Review {threat_actor} TTPs and update detections'
            )

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        return self.stats.copy()

