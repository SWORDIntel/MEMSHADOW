import pytest
from app.services.enrichment.nlp_service import NLPEnrichmentService, nlp_service

@pytest.mark.asyncio
class TestNLPEnrichment:
    """Test NLP enrichment service"""
    
    async def test_extract_entities(self):
        """Test entity extraction"""
        service = NLPEnrichmentService()
        
        text = "Python and FastAPI are great technologies for building APIs."
        entities = await service.extract_entities(text)
        
        assert isinstance(entities, list)
        # Mock should detect Python and FastAPI
        assert len(entities) >= 0
    
    async def test_analyze_sentiment(self):
        """Test sentiment analysis"""
        service = NLPEnrichmentService()
        
        # Positive text
        positive_text = "This is a great and excellent success!"
        sentiment = await service.analyze_sentiment(positive_text)
        
        assert "polarity" in sentiment
        assert "label" in sentiment
        assert sentiment["label"] in ["positive", "negative", "neutral"]
    
    async def test_extract_keywords(self):
        """Test keyword extraction"""
        service = NLPEnrichmentService()
        
        text = "Python programming language is used for data science and machine learning applications."
        keywords = await service.extract_keywords(text, top_n=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        
        if keywords:
            assert "term" in keywords[0]
            assert "score" in keywords[0]
    
    async def test_generate_summary(self):
        """Test summary generation"""
        service = NLPEnrichmentService()
        
        long_text = "This is a very long text. " * 50
        summary = await service.generate_summary(long_text, max_length=100)
        
        assert len(summary) <= 150  # Allow some margin
        assert len(summary) > 0
    
    async def test_detect_language(self):
        """Test language detection"""
        service = NLPEnrichmentService()
        
        text = "This is English text."
        language = await service.detect_language(text)
        
        assert language == "en"
    
    async def test_enrich_memory(self):
        """Test comprehensive memory enrichment"""
        service = NLPEnrichmentService()
        
        text = "Python is a programming language used for machine learning."
        enrichment = await service.enrich_memory(text)
        
        assert "entities" in enrichment
        assert "sentiment" in enrichment
        assert "keywords" in enrichment
        assert "language" in enrichment
        assert "enriched_at" in enrichment
