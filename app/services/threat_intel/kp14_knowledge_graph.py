"""
KP14 Knowledge Graph Builder

Builds and maintains a knowledge graph from KP14 analysis results,
enabling relationship discovery between samples, threat actors,
malware families, techniques, and IOCs.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Knowledge graph node types"""
    SAMPLE = "sample"
    MALWARE_FAMILY = "malware_family"
    THREAT_ACTOR = "threat_actor"
    CAMPAIGN = "campaign"
    TECHNIQUE = "technique"
    IOC = "ioc"
    C2_SERVER = "c2_server"


class EdgeType(str, Enum):
    """Knowledge graph edge types"""
    INSTANCE_OF = "instance_of"        # Sample -> MalwareFamily
    ATTRIBUTED_TO = "attributed_to"    # Sample -> ThreatActor
    PART_OF = "part_of"                # Sample -> Campaign
    USES = "uses"                      # Sample -> Technique
    CONTAINS = "contains"              # Sample -> IOC
    CONNECTS_TO = "connects_to"        # Sample -> C2Server
    RELATED_TO = "related_to"          # Generic relationship
    VARIANT_OF = "variant_of"          # Sample -> Sample
    TARGETS = "targets"                # ThreatActor -> (industry/region)
    DEVELOPS = "develops"              # ThreatActor -> MalwareFamily


@dataclass
class GraphNode:
    """Knowledge graph node"""
    node_id: str
    node_type: NodeType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "kp14"


@dataclass
class GraphEdge:
    """Knowledge graph edge"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 0.5
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class KP14KnowledgeGraph:
    """
    Knowledge graph for KP14 threat intelligence.

    Maintains nodes and edges representing relationships between
    malware samples, threat actors, techniques, IOCs, etc.
    """

    def __init__(self):
        """Initialize knowledge graph."""
        # Node storage: node_id -> GraphNode
        self._nodes: Dict[str, GraphNode] = {}

        # Edge storage: (source_id, target_id, edge_type) -> GraphEdge
        self._edges: Dict[Tuple[str, str, EdgeType], GraphEdge] = {}

        # Index: node_type -> set of node_ids
        self._type_index: Dict[NodeType, Set[str]] = defaultdict(set)

        # Index: adjacency list for fast traversal
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

        # Statistics
        self.stats = {
            'nodes_added': 0,
            'edges_added': 0,
            'analyses_processed': 0,
        }

        logger.info("KP14 Knowledge Graph initialized")

    def add_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """
        Add KP14 analysis result to knowledge graph.

        Args:
            analysis_result: Normalized KP14 analysis result

        Returns:
            Sample node ID
        """
        sample = analysis_result.get('sample', {})
        analysis = analysis_result.get('analysis', {})

        sample_hash = sample.get('hash_sha256', '')
        if not sample_hash:
            logger.warning("Cannot add analysis without sample hash")
            return ""

        # Create sample node
        sample_node_id = f"sample:{sample_hash[:16]}"
        self._add_node(GraphNode(
            node_id=sample_node_id,
            node_type=NodeType.SAMPLE,
            name=f"Sample {sample_hash[:8]}",
            attributes={
                'hash_sha256': sample_hash,
                'hash_md5': sample.get('hash_md5'),
                'hash_sha1': sample.get('hash_sha1'),
                'file_type': sample.get('file_type'),
                'file_size': sample.get('file_size'),
                'threat_score': analysis.get('threat_score', 0),
            },
            confidence=0.95,
            source='kp14'
        ))

        # Add malware family relationship
        malware_family = analysis.get('malware_family')
        if malware_family:
            family_node_id = self._add_malware_family(malware_family)
            self._add_edge(GraphEdge(
                source_id=sample_node_id,
                target_id=family_node_id,
                edge_type=EdgeType.INSTANCE_OF,
                confidence=analysis.get('family_confidence', 0.7)
            ))

        # Add threat actor relationship
        threat_actor = analysis.get('threat_actor')
        if threat_actor:
            actor_node_id = self._add_threat_actor(threat_actor)
            self._add_edge(GraphEdge(
                source_id=sample_node_id,
                target_id=actor_node_id,
                edge_type=EdgeType.ATTRIBUTED_TO,
                confidence=analysis.get('attribution_confidence', 0.6)
            ))

        # Add campaign relationship
        campaign = analysis.get('campaign')
        if campaign:
            campaign_node_id = self._add_campaign(campaign)
            self._add_edge(GraphEdge(
                source_id=sample_node_id,
                target_id=campaign_node_id,
                edge_type=EdgeType.PART_OF,
                confidence=0.7
            ))

        # Add technique relationships
        for technique in analysis_result.get('techniques', []):
            technique_node_id = self._add_technique(technique)
            self._add_edge(GraphEdge(
                source_id=sample_node_id,
                target_id=technique_node_id,
                edge_type=EdgeType.USES,
                confidence=technique.get('confidence', 0.5)
            ))

        # Add IOC relationships
        for ioc in analysis_result.get('iocs', []):
            ioc_node_id = self._add_ioc(ioc)
            self._add_edge(GraphEdge(
                source_id=sample_node_id,
                target_id=ioc_node_id,
                edge_type=EdgeType.CONTAINS,
                confidence=ioc.get('confidence', 0.5)
            ))

        # Add C2 relationships
        for c2 in analysis_result.get('c2_endpoints', []):
            c2_node_id = self._add_c2_server(c2)
            self._add_edge(GraphEdge(
                source_id=sample_node_id,
                target_id=c2_node_id,
                edge_type=EdgeType.CONNECTS_TO,
                confidence=c2.get('confidence', 0.5)
            ))

        self.stats['analyses_processed'] += 1
        logger.debug(f"Added analysis to knowledge graph: {sample_node_id}")

        return sample_node_id

    def find_related_samples(
        self,
        sample_hash: str,
        max_depth: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find samples related to given sample.

        Args:
            sample_hash: Sample hash
            max_depth: Max traversal depth
            limit: Max results

        Returns:
            List of related samples with relationship info
        """
        sample_node_id = f"sample:{sample_hash[:16]}"
        if sample_node_id not in self._nodes:
            return []

        related = []
        visited = {sample_node_id}
        queue = [(sample_node_id, 0, [])]  # (node_id, depth, path)

        while queue and len(related) < limit:
            current_id, depth, path = queue.pop(0)

            if depth >= max_depth:
                continue

            # Get neighbors
            for neighbor_id in self._adjacency.get(current_id, set()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                neighbor = self._nodes.get(neighbor_id)
                if not neighbor:
                    continue

                new_path = path + [neighbor_id]

                # If neighbor is a sample, add to results
                if neighbor.node_type == NodeType.SAMPLE and neighbor_id != sample_node_id:
                    related.append({
                        'node_id': neighbor_id,
                        'sample_hash': neighbor.attributes.get('hash_sha256'),
                        'relationship_path': new_path,
                        'depth': depth + 1,
                        'threat_score': neighbor.attributes.get('threat_score', 0),
                    })
                else:
                    # Continue traversal through non-sample nodes
                    queue.append((neighbor_id, depth + 1, new_path))

            # Also check reverse adjacency
            for neighbor_id in self._reverse_adjacency.get(current_id, set()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                neighbor = self._nodes.get(neighbor_id)
                if not neighbor:
                    continue

                new_path = path + [neighbor_id]

                if neighbor.node_type == NodeType.SAMPLE and neighbor_id != sample_node_id:
                    related.append({
                        'node_id': neighbor_id,
                        'sample_hash': neighbor.attributes.get('hash_sha256'),
                        'relationship_path': new_path,
                        'depth': depth + 1,
                        'threat_score': neighbor.attributes.get('threat_score', 0),
                    })
                else:
                    queue.append((neighbor_id, depth + 1, new_path))

        # Sort by depth then threat score
        related.sort(key=lambda x: (x['depth'], -x['threat_score']))

        return related[:limit]

    def get_threat_actor_samples(
        self,
        threat_actor: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all samples attributed to a threat actor."""
        actor_id = f"actor:{threat_actor.lower().replace(' ', '_')}"
        if actor_id not in self._nodes:
            return []

        samples = []
        for sample_id in self._reverse_adjacency.get(actor_id, set()):
            sample = self._nodes.get(sample_id)
            if sample and sample.node_type == NodeType.SAMPLE:
                samples.append({
                    'node_id': sample_id,
                    'sample_hash': sample.attributes.get('hash_sha256'),
                    'threat_score': sample.attributes.get('threat_score', 0),
                    'first_seen': sample.first_seen.isoformat(),
                })

        samples.sort(key=lambda x: x.get('threat_score', 0), reverse=True)
        return samples[:limit]

    def get_malware_family_samples(
        self,
        malware_family: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all samples of a malware family."""
        family_id = f"family:{malware_family.lower().replace(' ', '_')}"
        if family_id not in self._nodes:
            return []

        samples = []
        for sample_id in self._reverse_adjacency.get(family_id, set()):
            sample = self._nodes.get(sample_id)
            if sample and sample.node_type == NodeType.SAMPLE:
                samples.append({
                    'node_id': sample_id,
                    'sample_hash': sample.attributes.get('hash_sha256'),
                    'threat_score': sample.attributes.get('threat_score', 0),
                    'first_seen': sample.first_seen.isoformat(),
                })

        samples.sort(key=lambda x: x.get('threat_score', 0), reverse=True)
        return samples[:limit]

    def get_technique_samples(
        self,
        technique_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get samples using a specific ATT&CK technique."""
        tech_node_id = f"technique:{technique_id}"
        if tech_node_id not in self._nodes:
            return []

        samples = []
        for sample_id in self._reverse_adjacency.get(tech_node_id, set()):
            sample = self._nodes.get(sample_id)
            if sample and sample.node_type == NodeType.SAMPLE:
                samples.append({
                    'node_id': sample_id,
                    'sample_hash': sample.attributes.get('hash_sha256'),
                    'threat_score': sample.attributes.get('threat_score', 0),
                })

        return samples[:limit]

    def get_ioc_samples(self, ioc_value: str) -> List[Dict[str, Any]]:
        """Get samples containing a specific IOC."""
        # Need to search by IOC value since node_id includes type
        ioc_nodes = [
            nid for nid, node in self._nodes.items()
            if node.node_type == NodeType.IOC and
            node.attributes.get('value') == ioc_value
        ]

        samples = []
        for ioc_id in ioc_nodes:
            for sample_id in self._reverse_adjacency.get(ioc_id, set()):
                sample = self._nodes.get(sample_id)
                if sample and sample.node_type == NodeType.SAMPLE:
                    samples.append({
                        'node_id': sample_id,
                        'sample_hash': sample.attributes.get('hash_sha256'),
                        'threat_score': sample.attributes.get('threat_score', 0),
                    })

        return samples

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        type_counts = {t.value: len(ids) for t, ids in self._type_index.items()}

        return {
            'total_nodes': len(self._nodes),
            'total_edges': len(self._edges),
            'analyses_processed': self.stats['analyses_processed'],
            'node_counts': type_counts,
        }

    def _add_node(self, node: GraphNode) -> str:
        """Add or update a node."""
        if node.node_id in self._nodes:
            # Update existing node
            existing = self._nodes[node.node_id]
            existing.last_seen = datetime.now(timezone.utc)
            existing.attributes.update(node.attributes)
            if node.confidence > existing.confidence:
                existing.confidence = node.confidence
        else:
            self._nodes[node.node_id] = node
            self._type_index[node.node_type].add(node.node_id)
            self.stats['nodes_added'] += 1

        return node.node_id

    def _add_edge(self, edge: GraphEdge) -> None:
        """Add or update an edge."""
        key = (edge.source_id, edge.target_id, edge.edge_type)

        if key in self._edges:
            # Update existing edge
            existing = self._edges[key]
            existing.weight += 1
            if edge.confidence > existing.confidence:
                existing.confidence = edge.confidence
        else:
            self._edges[key] = edge
            self._adjacency[edge.source_id].add(edge.target_id)
            self._reverse_adjacency[edge.target_id].add(edge.source_id)
            self.stats['edges_added'] += 1

    def _add_malware_family(self, family_name: str) -> str:
        """Add malware family node."""
        node_id = f"family:{family_name.lower().replace(' ', '_')}"
        self._add_node(GraphNode(
            node_id=node_id,
            node_type=NodeType.MALWARE_FAMILY,
            name=family_name,
            confidence=0.8
        ))
        return node_id

    def _add_threat_actor(self, actor_name: str) -> str:
        """Add threat actor node."""
        node_id = f"actor:{actor_name.lower().replace(' ', '_')}"
        self._add_node(GraphNode(
            node_id=node_id,
            node_type=NodeType.THREAT_ACTOR,
            name=actor_name,
            confidence=0.7
        ))
        return node_id

    def _add_campaign(self, campaign_name: str) -> str:
        """Add campaign node."""
        node_id = f"campaign:{campaign_name.lower().replace(' ', '_')}"
        self._add_node(GraphNode(
            node_id=node_id,
            node_type=NodeType.CAMPAIGN,
            name=campaign_name,
            confidence=0.7
        ))
        return node_id

    def _add_technique(self, technique: Dict[str, Any]) -> str:
        """Add ATT&CK technique node."""
        technique_id = technique.get('technique_id', 'unknown')
        node_id = f"technique:{technique_id}"
        self._add_node(GraphNode(
            node_id=node_id,
            node_type=NodeType.TECHNIQUE,
            name=technique.get('name', technique_id),
            attributes={
                'technique_id': technique_id,
                'tactic': technique.get('tactic'),
            },
            confidence=technique.get('confidence', 0.5)
        ))
        return node_id

    def _add_ioc(self, ioc: Dict[str, Any]) -> str:
        """Add IOC node."""
        ioc_type = ioc.get('type', 'unknown')
        ioc_value = ioc.get('value', '')
        node_id = f"ioc:{ioc_type}:{hash(ioc_value) & 0xFFFFFFFF:08x}"
        self._add_node(GraphNode(
            node_id=node_id,
            node_type=NodeType.IOC,
            name=f"{ioc_type}: {ioc_value[:32]}",
            attributes={
                'type': ioc_type,
                'value': ioc_value,
            },
            confidence=ioc.get('confidence', 0.5)
        ))
        return node_id

    def _add_c2_server(self, c2: Dict[str, Any]) -> str:
        """Add C2 server node."""
        c2_value = c2.get('value', '')
        node_id = f"c2:{hash(c2_value) & 0xFFFFFFFF:08x}"
        self._add_node(GraphNode(
            node_id=node_id,
            node_type=NodeType.C2_SERVER,
            name=c2_value[:64],
            attributes={
                'value': c2_value,
                'type': c2.get('type'),
                'protocol': c2.get('protocol'),
                'port': c2.get('port'),
            },
            confidence=c2.get('confidence', 0.5)
        ))
        return node_id

