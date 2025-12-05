#!/usr/bin/env python3
"""
Oracle Query System for DSMIL Brain

Hub-originated distributed query system:
- All queries originate from hub
- Distributed to nodes for local correlation
- Results aggregated and synthesized by hub
"""

import hashlib
import threading
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Set, Callable
from datetime import datetime, timezone
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of oracle queries"""
    ENTITY_SEARCH = auto()      # "Who is most likely to do X?"
    PROBABILITY = auto()        # "What is the probability of Y?"
    KNOWLEDGE_GAP = auto()      # "What don't we know that we should?"
    CAUSATION = auto()          # "What caused Z?"
    PREDICTION = auto()         # "What will X cause?"
    FINGERPRINT_MATCH = auto()  # "Who matches this fingerprint?"
    RELATIONSHIP = auto()       # "What's the relationship between A and B?"
    AGGREGATION = auto()        # General aggregation query


@dataclass
class DistributedQuery:
    """A query to be distributed to nodes"""
    query_id: str
    query_type: QueryType

    # Query content
    natural_language: str
    structured_form: Dict[str, Any] = field(default_factory=dict)

    # Routing
    target_nodes: List[str] = field(default_factory=list)  # Empty = all nodes
    required_capabilities: Set[str] = field(default_factory=set)

    # Timing
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: float = 30.0


@dataclass
class NodeResponse:
    """Response from a single node"""
    node_id: str
    query_id: str

    # Content
    data: Any = None
    confidence: float = 0.0

    # Metadata
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DistributedResponse:
    """Synthesized response from all nodes"""
    response_id: str
    query_id: str

    # Node responses
    node_responses: List[NodeResponse] = field(default_factory=list)
    nodes_responded: int = 0
    nodes_failed: int = 0

    # Synthesized answer
    answer: Any = None
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)

    # Timing
    total_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OracleQuerySystem:
    """
    Oracle Query System

    All intelligence queries originate from hub, are distributed to nodes,
    and synthesized by the hub.

    Usage:
        oracle = OracleQuerySystem()

        # Register node handlers
        oracle.register_node("node-1", handler=node1_handler)

        # Query
        response = oracle.query("Who is most likely to attack target X?")
    """

    # Supported query patterns
    QUERY_TYPES = {
        "who": QueryType.ENTITY_SEARCH,
        "probability": QueryType.PROBABILITY,
        "what don't we know": QueryType.KNOWLEDGE_GAP,
        "what caused": QueryType.CAUSATION,
        "what will": QueryType.PREDICTION,
        "matches": QueryType.FINGERPRINT_MATCH,
        "relationship": QueryType.RELATIONSHIP,
    }

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

        self._nodes: Dict[str, Dict] = {}  # node_id -> {handler, capabilities}
        self._queries: Dict[str, DistributedQuery] = {}
        self._responses: Dict[str, DistributedResponse] = {}

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()

        logger.info("OracleQuerySystem initialized")

    def register_node(self, node_id: str,
                     handler: Callable[[DistributedQuery], NodeResponse],
                     capabilities: Optional[Set[str]] = None):
        """Register a node with its query handler"""
        with self._lock:
            self._nodes[node_id] = {
                "handler": handler,
                "capabilities": capabilities or set(),
            }
            logger.info(f"Registered node {node_id}")

    def unregister_node(self, node_id: str):
        """Unregister a node"""
        with self._lock:
            self._nodes.pop(node_id, None)

    def parse_query(self, natural_language: str) -> DistributedQuery:
        """Parse natural language query into structured form"""
        nl_lower = natural_language.lower()

        # Determine query type
        query_type = QueryType.AGGREGATION
        for pattern, qtype in self.QUERY_TYPES.items():
            if pattern in nl_lower:
                query_type = qtype
                break

        # Extract structured form (simplified)
        structured = {
            "raw": natural_language,
            "keywords": [w for w in natural_language.split() if len(w) > 3],
        }

        query = DistributedQuery(
            query_id=hashlib.sha256(f"{natural_language}:{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            query_type=query_type,
            natural_language=natural_language,
            structured_form=structured,
        )

        return query

    def route_query(self, query: DistributedQuery) -> List[str]:
        """Determine which nodes to query"""
        with self._lock:
            if query.target_nodes:
                # Specific nodes requested
                return [n for n in query.target_nodes if n in self._nodes]

            if query.required_capabilities:
                # Filter by capability
                return [
                    nid for nid, info in self._nodes.items()
                    if query.required_capabilities.issubset(info.get("capabilities", set()))
                ]

            # Default: all nodes
            return list(self._nodes.keys())

    def broadcast_query(self, query: DistributedQuery,
                       target_nodes: List[str]) -> List[NodeResponse]:
        """Broadcast query to target nodes and collect responses"""
        responses = []

        def query_node(node_id: str) -> NodeResponse:
            try:
                handler = self._nodes[node_id]["handler"]
                return handler(query)
            except Exception as e:
                return NodeResponse(
                    node_id=node_id,
                    query_id=query.query_id,
                    error=str(e),
                )

        # Submit queries in parallel
        futures = {
            self._executor.submit(query_node, nid): nid
            for nid in target_nodes
        }

        # Collect results with timeout
        for future in as_completed(futures, timeout=query.timeout_seconds):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                node_id = futures[future]
                responses.append(NodeResponse(
                    node_id=node_id,
                    query_id=query.query_id,
                    error=str(e),
                ))

        return responses

    def synthesize_responses(self, query: DistributedQuery,
                            node_responses: List[NodeResponse]) -> DistributedResponse:
        """Synthesize node responses into final answer"""
        start_time = datetime.now(timezone.utc)

        # Separate successful and failed responses
        successful = [r for r in node_responses if r.error is None]
        failed = [r for r in node_responses if r.error is not None]

        # Aggregate data
        all_data = [r.data for r in successful if r.data is not None]
        avg_confidence = sum(r.confidence for r in successful) / len(successful) if successful else 0

        # Synthesize based on query type
        answer = self._synthesize_by_type(query.query_type, all_data)

        # Find contradictions
        contradictions = self._find_contradictions(all_data)

        response = DistributedResponse(
            response_id=hashlib.sha256(f"resp:{query.query_id}".encode()).hexdigest()[:16],
            query_id=query.query_id,
            node_responses=node_responses,
            nodes_responded=len(successful),
            nodes_failed=len(failed),
            answer=answer,
            confidence=avg_confidence,
            supporting_evidence=[str(d) for d in all_data[:5]],
            contradictions=contradictions,
            total_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        )

        return response

    def _synthesize_by_type(self, query_type: QueryType, data: List) -> Any:
        """Synthesize data based on query type"""
        if not data:
            return None

        if query_type == QueryType.ENTITY_SEARCH:
            # Return most mentioned entity
            entities = {}
            for d in data:
                if isinstance(d, dict) and "entity" in d:
                    e = d["entity"]
                    entities[e] = entities.get(e, 0) + 1
            if entities:
                return max(entities.items(), key=lambda x: x[1])[0]

        elif query_type == QueryType.PROBABILITY:
            # Average probabilities
            probs = [d.get("probability", 0.5) for d in data if isinstance(d, dict)]
            return sum(probs) / len(probs) if probs else 0.5

        elif query_type == QueryType.RELATIONSHIP:
            # Merge relationship data
            relationships = []
            for d in data:
                if isinstance(d, dict) and "relationships" in d:
                    relationships.extend(d["relationships"])
            return relationships

        # Default: return all data
        return data

    def _find_contradictions(self, data: List) -> List[str]:
        """Find contradictions in data"""
        contradictions = []
        # Would implement actual contradiction detection
        return contradictions

    def query(self, natural_language: str,
             target_nodes: Optional[List[str]] = None) -> DistributedResponse:
        """
        Execute a distributed query

        This is the main entry point for the oracle system.
        """
        with self._lock:
            # 1. Parse query
            query = self.parse_query(natural_language)
            if target_nodes:
                query.target_nodes = target_nodes
            self._queries[query.query_id] = query

            # 2. Route query
            targets = self.route_query(query)

            if not targets:
                return DistributedResponse(
                    response_id=hashlib.sha256(f"empty:{query.query_id}".encode()).hexdigest()[:16],
                    query_id=query.query_id,
                    answer="No nodes available to process query",
                )

            # 3. Broadcast to nodes
            node_responses = self.broadcast_query(query, targets)

            # 4. Synthesize
            response = self.synthesize_responses(query, node_responses)
            self._responses[response.response_id] = response

            return response

    def get_stats(self) -> Dict:
        """Get system statistics"""
        with self._lock:
            return {
                "registered_nodes": len(self._nodes),
                "queries_processed": len(self._queries),
                "responses_generated": len(self._responses),
            }


# Example node handler
def example_node_handler(query: DistributedQuery) -> NodeResponse:
    """Example node handler for testing"""
    import time
    import random

    # Simulate processing
    time.sleep(random.uniform(0.01, 0.1))

    return NodeResponse(
        node_id="example-node",
        query_id=query.query_id,
        data={
            "entity": "APT29",
            "probability": random.uniform(0.3, 0.9),
            "relationships": [{"a": "entity1", "b": "entity2", "type": "communicates"}],
        },
        confidence=random.uniform(0.5, 0.9),
        processing_time_ms=random.uniform(10, 100),
    )


if __name__ == "__main__":
    print("Oracle Query System Self-Test")
    print("=" * 50)

    oracle = OracleQuerySystem()

    print("\n[1] Register Nodes")
    for i in range(3):
        def handler(q, node_id=f"node-{i}"):
            import time
            import random
            time.sleep(0.05)
            return NodeResponse(
                node_id=node_id,
                query_id=q.query_id,
                data={"entity": f"entity-{random.randint(1,3)}", "probability": random.uniform(0.4, 0.8)},
                confidence=random.uniform(0.6, 0.9),
            )
        oracle.register_node(f"node-{i}", handler, {"intel", "search"})
    print("    Registered 3 nodes")

    print("\n[2] Query: Entity Search")
    response = oracle.query("Who is most likely to attack our systems?")
    print(f"    Query ID: {response.query_id}")
    print(f"    Nodes responded: {response.nodes_responded}")
    print(f"    Answer: {response.answer}")
    print(f"    Confidence: {response.confidence:.2f}")
    print(f"    Time: {response.total_time_ms:.1f}ms")

    print("\n[3] Query: Probability")
    response = oracle.query("What is the probability of APT29 attacking?")
    print(f"    Answer: {response.answer}")

    print("\n[4] Query: Relationship")
    response = oracle.query("What is the relationship between APT29 and target?")
    print(f"    Answer type: {type(response.answer)}")

    print("\n[5] Query Types Supported")
    for pattern, qtype in OracleQuerySystem.QUERY_TYPES.items():
        print(f"    '{pattern}...' -> {qtype.name}")

    print("\n[6] Statistics")
    stats = oracle.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Oracle Query System test complete")

