import pytest
from uuid import uuid4
from app.services.enrichment.knowledge_graph import KnowledgeGraphService

@pytest.mark.asyncio
class TestKnowledgeGraph:
    """Test knowledge graph service"""
    
    async def test_add_node(self):
        """Test adding a node to the graph"""
        kg = KnowledgeGraphService()
        
        node_id = await kg.add_node(
            "test_node_1",
            "PERSON",
            {"name": "John Doe"}
        )
        
        assert node_id == "test_node_1"
        assert "test_node_1" in kg.nodes
        assert kg.nodes["test_node_1"]["type"] == "PERSON"
    
    async def test_add_edge(self):
        """Test adding an edge between nodes"""
        kg = KnowledgeGraphService()
        
        await kg.add_node("node_a", "PERSON", {"name": "Alice"})
        await kg.add_node("node_b", "TECHNOLOGY", {"name": "Python"})
        
        await kg.add_edge("node_a", "node_b", "KNOWS", {"level": "expert"})
        
        assert len(kg.edges) == 1
        assert kg.graph["node_a"][0] == ("KNOWS", "node_b")
    
    async def test_build_graph_from_memory(self):
        """Test building graph from memory enrichment data"""
        kg = KnowledgeGraphService()
        
        memory_id = uuid4()
        enrichment = {
            "entities": [
                {"text": "Python", "label": "TECHNOLOGY", "start": 0},
                {"text": "FastAPI", "label": "TECHNOLOGY", "start": 10}
            ],
            "keywords": [
                {"term": "programming", "score": 0.8},
                {"term": "web", "score": 0.6}
            ],
            "relationships": [
                {"subject": "Python", "predicate": "used_for", "object": "web development"}
            ]
        }
        
        stats = await kg.build_graph_from_memory(memory_id, enrichment)
        
        assert stats["nodes_created"] > 0
        assert stats["edges_created"] > 0
    
    async def test_find_related_nodes(self):
        """Test finding related nodes"""
        kg = KnowledgeGraphService()
        
        await kg.add_node("a", "PERSON", {})
        await kg.add_node("b", "TECHNOLOGY", {})
        await kg.add_node("c", "CONCEPT", {})
        
        await kg.add_edge("a", "b", "KNOWS")
        await kg.add_edge("b", "c", "RELATED_TO")
        
        related = await kg.find_related_nodes("a", max_depth=2)
        
        assert len(related) >= 1
    
    async def test_find_shortest_path(self):
        """Test finding shortest path between nodes"""
        kg = KnowledgeGraphService()
        
        await kg.add_node("a", "TYPE", {})
        await kg.add_node("b", "TYPE", {})
        await kg.add_node("c", "TYPE", {})
        
        await kg.add_edge("a", "b", "EDGE")
        await kg.add_edge("b", "c", "EDGE")
        
        path = await kg.find_shortest_path("a", "c")
        
        assert path == ["a", "b", "c"]
    
    async def test_get_graph_stats(self):
        """Test getting graph statistics"""
        kg = KnowledgeGraphService()
        
        await kg.add_node("a", "PERSON", {})
        await kg.add_node("b", "TECHNOLOGY", {})
        await kg.add_edge("a", "b", "KNOWS")
        
        stats = await kg.get_graph_stats()
        
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert "node_types" in stats
        assert "edge_types" in stats
    
    async def test_export_graph(self):
        """Test exporting graph"""
        kg = KnowledgeGraphService()
        
        await kg.add_node("a", "TYPE", {"label": "A"})
        await kg.add_edge("a", "a", "SELF")
        
        exported = await kg.export_graph(format="dict")
        
        assert "nodes" in exported
        assert "edges" in exported
        assert len(exported["nodes"]) == 1
