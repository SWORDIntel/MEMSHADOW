import pytest
from app.services.enrichment.multimodal_embeddings import (
    MultiModalEmbeddingService,
    ContentType
)

@pytest.mark.asyncio
class TestMultiModalEmbeddings:
    """Test multi-modal embedding service"""
    
    async def test_text_embedding(self):
        """Test text content embedding"""
        service = MultiModalEmbeddingService()
        
        result = await service.generate_embedding(
            "This is test text",
            ContentType.TEXT
        )
        
        assert "embedding" in result
        assert "content_type" in result
        assert result["content_type"] == ContentType.TEXT
        assert isinstance(result["embedding"], list)
    
    async def test_code_embedding(self):
        """Test code content embedding"""
        service = MultiModalEmbeddingService()
        
        code = "def hello():\n    print('Hello, world!')"
        result = await service.generate_embedding(
            code,
            ContentType.CODE,
            metadata={"language": "python"}
        )
        
        assert result["content_type"] == ContentType.CODE
        assert result["language"] == "python"
    
    async def test_image_embedding(self):
        """Test image content embedding"""
        service = MultiModalEmbeddingService()
        
        # Mock image data
        image_data = b"fake_image_data"
        result = await service.generate_embedding(
            image_data,
            ContentType.IMAGE
        )
        
        assert result["content_type"] == ContentType.IMAGE
        assert result["method"] == "CLIP"
    
    async def test_similarity_computation(self):
        """Test embedding similarity computation"""
        service = MultiModalEmbeddingService()
        
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]
        
        # Identical embeddings
        sim_same = await service.compute_similarity(emb1, emb2, method="cosine")
        assert sim_same > 0.99
        
        # Different embeddings
        sim_diff = await service.compute_similarity(emb1, emb3, method="cosine")
        assert sim_diff < 0.1
    
    async def test_batch_embeddings(self):
        """Test batch embedding generation"""
        service = MultiModalEmbeddingService()
        
        contents = [
            ("Text 1", ContentType.TEXT),
            ("Text 2", ContentType.TEXT),
        ]
        
        results = await service.generate_batch_embeddings(contents)
        
        assert len(results) == 2
        assert all("embedding" in r for r in results)
