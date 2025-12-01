"""
Knowledge Graph Service
Phase 3: Intelligence Layer - Build and query knowledge graphs from memories
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import uuid

logger = structlog.get_logger()

class KnowledgeGraphService:
    """
    Service for building and querying knowledge graphs from memory content.
    Creates nodes for entities and edges for relationships.
    """
    
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
        self.graph = defaultdict(list)  # In-memory graph: node_id -> [(edge_type, target_id)]
        self.nodes = {}  # node_id -> node_data
        self.edges = []  # list of edges
    
    async def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: Dict[str, Any]
    ) -> str:
        """
        Add a node to the knowledge graph.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (e.g., "PERSON", "TECHNOLOGY", "CONCEPT")
            properties: Additional properties for the node
        
        Returns:
            node_id
        """
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "properties": properties,
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.debug("Node added to knowledge graph", node_id=node_id, node_type=node_type)
        return node_id
    
    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an edge between two nodes in the knowledge graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship (e.g., "MENTIONS", "RELATED_TO", "PART_OF")
            properties: Additional edge properties
        """
        edge = {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
            "properties": properties or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.edges.append(edge)
        self.graph[source_id].append((edge_type, target_id))
        
        logger.debug("Edge added to knowledge graph", 
                    source=source_id, target=target_id, type=edge_type)
    
    async def build_graph_from_memory(
        self,
        memory_id: uuid.UUID,
        enrichment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build knowledge graph nodes and edges from memory enrichment data.
        
        Args:
            memory_id: The memory ID
            enrichment_data: NLP enrichment data (entities, relationships, etc.)
        
        Returns:
            Graph statistics
        """
        logger.info("Building knowledge graph from memory", memory_id=str(memory_id))
        
        memory_node_id = f"memory:{memory_id}"
        await self.add_node(
            memory_node_id,
            "MEMORY",
            {"memory_id": str(memory_id)}
        )
        
        nodes_created = 1
        edges_created = 0
        
        # Add entity nodes and connect to memory
        entities = enrichment_data.get("entities", [])
        for entity in entities:
            entity_id = self._generate_entity_id(entity["text"], entity["label"])
            
            # Add or update entity node
            await self.add_node(
                entity_id,
                entity["label"],
                {
                    "text": entity["text"],
                    "occurrences": 1
                }
            )
            nodes_created += 1
            
            # Connect entity to memory
            await self.add_edge(
                memory_node_id,
                entity_id,
                "MENTIONS",
                {"position": entity.get("start", 0)}
            )
            edges_created += 1
        
        # Add relationship edges
        relationships = enrichment_data.get("relationships", [])
        for rel in relationships:
            subj_id = self._generate_entity_id(rel["subject"], "ENTITY")
            obj_id = self._generate_entity_id(rel["object"], "ENTITY")
            
            # Ensure nodes exist
            if subj_id not in self.nodes:
                await self.add_node(subj_id, "ENTITY", {"text": rel["subject"]})
                nodes_created += 1
            
            if obj_id not in self.nodes:
                await self.add_node(obj_id, "ENTITY", {"text": rel["object"]})
                nodes_created += 1
            
            # Add relationship edge
            await self.add_edge(
                subj_id,
                obj_id,
                rel["predicate"].upper().replace(" ", "_"),
                {"from_memory": str(memory_id)}
            )
            edges_created += 1
        
        # Add keyword nodes
        keywords = enrichment_data.get("keywords", [])
        for kw in keywords[:5]:  # Top 5 keywords
            kw_id = self._generate_entity_id(kw["term"], "KEYWORD")
            
            if kw_id not in self.nodes:
                await self.add_node(
                    kw_id,
                    "KEYWORD",
                    {"text": kw["term"], "importance": kw["score"]}
                )
                nodes_created += 1
            
            await self.add_edge(
                memory_node_id,
                kw_id,
                "HAS_KEYWORD",
                {"score": kw["score"]}
            )
            edges_created += 1
        
        stats = {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges)
        }
        
        logger.info("Knowledge graph built", **stats)
        return stats
    
    def _generate_entity_id(self, text: str, entity_type: str) -> str:
        """Generate a consistent ID for an entity"""
        # Normalize text
        normalized = text.lower().strip()
        return f"{entity_type}:{normalized.replace(' ', '_')}"
    
    async def find_related_nodes(
        self,
        node_id: str,
        max_depth: int = 2,
        edge_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find nodes related to a given node through graph traversal.
        
        Args:
            node_id: Starting node
            max_depth: Maximum depth to traverse
            edge_types: Filter by specific edge types
        
        Returns:
            List of related nodes with their distances
        """
        visited = set()
        queue = [(node_id, 0)]  # (node_id, depth)
        related = []
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if current_id != node_id and current_id in self.nodes:
                related.append({
                    "node": self.nodes[current_id],
                    "distance": depth
                })
            
            # Get neighbors
            for edge_type, neighbor_id in self.graph.get(current_id, []):
                if edge_types is None or edge_type in edge_types:
                    queue.append((neighbor_id, depth + 1))
        
        logger.debug("Related nodes found", 
                    start_node=node_id, count=len(related), max_depth=max_depth)
        
        return related
    
    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.
        
        Returns:
            List of node IDs in the path, or None if no path exists
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        visited = set()
        queue = [(source_id, [source_id])]
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == target_id:
                logger.debug("Path found", length=len(path))
                return path
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            for _, neighbor_id in self.graph.get(current_id, []):
                if neighbor_id not in visited:
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        logger.debug("No path found", source=source_id, target=target_id)
        return None
    
    async def get_node_neighborhood(
        self,
        node_id: str,
        radius: int = 1
    ) -> Dict[str, Any]:
        """
        Get the local neighborhood around a node.
        
        Returns:
            Dictionary with nodes and edges in the neighborhood
        """
        if node_id not in self.nodes:
            return {"nodes": [], "edges": []}
        
        related = await self.find_related_nodes(node_id, max_depth=radius)
        related_ids = {node_id} | {r["node"]["id"] for r in related}
        
        # Get edges within neighborhood
        neighborhood_edges = [
            edge for edge in self.edges
            if edge["source"] in related_ids and edge["target"] in related_ids
        ]
        
        return {
            "center": self.nodes[node_id],
            "nodes": [self.nodes[node_id]] + [r["node"] for r in related],
            "edges": neighborhood_edges,
            "radius": radius
        }
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)
        
        for node in self.nodes.values():
            node_types[node["type"]] += 1
        
        for edge in self.edges:
            edge_types[edge["type"]] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "avg_degree": len(self.edges) * 2 / len(self.nodes) if self.nodes else 0
        }
    
    async def export_graph(self, format: str = "dict") -> Any:
        """
        Export the knowledge graph in various formats.
        
        Args:
            format: "dict", "networkx", "cytoscape", etc.
        
        Returns:
            Graph in requested format
        """
        if format == "dict":
            return {
                "nodes": list(self.nodes.values()),
                "edges": self.edges
            }
        elif format == "cytoscape":
            # Format for Cytoscape.js visualization
            return {
                "elements": {
                    "nodes": [
                        {"data": node}
                        for node in self.nodes.values()
                    ],
                    "edges": [
                        {
                            "data": {
                                "id": f"{edge['source']}-{edge['target']}",
                                "source": edge["source"],
                                "target": edge["target"],
                                "type": edge["type"],
                                **edge.get("properties", {})
                            }
                        }
                        for edge in self.edges
                    ]
                }
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global instance
knowledge_graph = KnowledgeGraphService()
