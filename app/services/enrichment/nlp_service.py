"""
NLP Enrichment Service
Phase 3: Intelligence Layer - Natural Language Processing for memory enrichment
"""

from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime
import hashlib
from app.core.config import settings, MemoryOperationMode

logger = structlog.get_logger()

class NLPEnrichmentService:
    """
    NLP service for memory enrichment using spaCy.
    Provides entity extraction, sentiment analysis, summarization, and more.
    """
    
    _nlp_model = None
    _sentiment_model = None
    
    @classmethod
    def _initialize_models(cls):
        """Initialize NLP models lazily"""
        if cls._nlp_model is None:
            try:
                # In production, these would be actual spaCy models
                # For now, we'll use a mock structure that can be replaced
                logger.info("Initializing NLP models")
                
                # Mock structure - in production:
                # import spacy
                # cls._nlp_model = spacy.load("en_core_web_lg")
                cls._nlp_model = MockNLPModel()
                cls._sentiment_model = MockSentimentModel()
                
                logger.info("NLP models initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize NLP models", error=str(e))
                raise
    
    def __init__(self):
        self._initialize_models()
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Returns entities in format:
        [{"text": "Python", "label": "TECHNOLOGY", "start": 0, "end": 6}, ...]
        """
        try:
            # In production with spaCy:
            # doc = self._nlp_model(text)
            # entities = [
            #     {
            #         "text": ent.text,
            #         "label": ent.label_,
            #         "start": ent.start_char,
            #         "end": ent.end_char
            #     }
            #     for ent in doc.ents
            # ]
            
            entities = self._nlp_model.extract_entities(text)
            
            logger.debug("Entities extracted", count=len(entities))
            return entities
            
        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return []
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Returns: {"polarity": 0.5, "subjectivity": 0.3, "label": "positive"}
        """
        try:
            sentiment = self._sentiment_model.analyze(text)
            
            logger.debug("Sentiment analyzed", sentiment=sentiment["label"])
            return sentiment
            
        except Exception as e:
            logger.error("Sentiment analysis failed", error=str(e))
            return {"polarity": 0.0, "subjectivity": 0.0, "label": "neutral"}
    
    async def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key terms from text using TF-IDF or other methods.
        
        Returns: [{"term": "python", "score": 0.85}, ...]
        """
        try:
            # In production, use actual keyword extraction
            # Could use TF-IDF, TextRank, RAKE, etc.
            keywords = self._nlp_model.extract_keywords(text, top_n)
            
            logger.debug("Keywords extracted", count=len(keywords))
            return keywords
            
        except Exception as e:
            logger.error("Keyword extraction failed", error=str(e))
            return []
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate extractive or abstractive summary of text.
        
        For long memories, creates a concise summary.
        """
        try:
            if len(text) <= max_length:
                return text
            
            # In production, use summarization models
            # Could use BART, T5, or extractive methods
            summary = self._nlp_model.summarize(text, max_length)
            
            logger.debug("Summary generated", original_len=len(text), summary_len=len(summary))
            return summary
            
        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            return text[:max_length] + "..."
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        try:
            # In production, use langdetect or spaCy's language detection
            language = self._nlp_model.detect_language(text)
            return language
        except Exception as e:
            logger.error("Language detection failed", error=str(e))
            return "en"  # Default to English
    
    async def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract subject-predicate-object relationships from text.
        
        Returns: [{"subject": "Python", "predicate": "is", "object": "programming language"}, ...]
        """
        try:
            # In production, use dependency parsing
            # doc = self._nlp_model(text)
            # Extract relationships using dependency trees
            
            relationships = self._nlp_model.extract_relationships(text)
            
            logger.debug("Relationships extracted", count=len(relationships))
            return relationships
            
        except Exception as e:
            logger.error("Relationship extraction failed", error=str(e))
            return []
    
    async def enrich_memory(self, content: str) -> Dict[str, Any]:
        """
        Comprehensive enrichment of memory content.

        Returns all NLP enrichments in one go.
        Processing level depends on MEMORY_OPERATION_MODE:
        - LIGHTWEIGHT: Skip all enrichment (returns minimal data)
        - ONLINE: Basic enrichment (entities, keywords, language only)
        - LOCAL: Full enrichment (all features)
        """
        mode = settings.MEMORY_OPERATION_MODE
        logger.info("Starting memory enrichment", mode=mode.value)

        # LIGHTWEIGHT mode: Skip enrichment entirely
        if mode == MemoryOperationMode.LIGHTWEIGHT:
            return {
                "enriched_at": datetime.utcnow().isoformat(),
                "mode": "lightweight",
                "skipped": True
            }

        # ONLINE mode: Basic enrichment only
        if mode == MemoryOperationMode.ONLINE:
            enrichment = {
                "entities": await self.extract_entities(content),
                "keywords": await self.extract_keywords(content, top_n=5),  # Reduced count
                "language": await self.detect_language(content),
                "enriched_at": datetime.utcnow().isoformat(),
                "mode": "online"
            }

            logger.info("Memory enrichment completed (ONLINE mode)",
                       entities_count=len(enrichment["entities"]))
            return enrichment

        # LOCAL mode: Full enrichment
        enrichment = {
            "entities": await self.extract_entities(content),
            "sentiment": await self.analyze_sentiment(content),
            "keywords": await self.extract_keywords(content),
            "language": await self.detect_language(content),
            "relationships": await self.extract_relationships(content),
            "enriched_at": datetime.utcnow().isoformat(),
            "mode": "local"
        }

        # Generate summary if content is long
        if len(content) > 500:
            enrichment["summary"] = await self.generate_summary(content)

        logger.info("Memory enrichment completed (LOCAL mode)",
                   entities_count=len(enrichment["entities"]),
                   keywords_count=len(enrichment["keywords"]))

        return enrichment


# Mock models for structure - to be replaced with actual implementations

class MockNLPModel:
    """Mock NLP model for testing structure"""
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Mock entity extraction"""
        # Simple regex-based mock for common patterns
        entities = []
        
        # Mock detection of common entity types
        if "python" in text.lower():
            entities.append({"text": "Python", "label": "TECHNOLOGY", "start": 0, "end": 6})
        if "fastapi" in text.lower():
            entities.append({"text": "FastAPI", "label": "TECHNOLOGY", "start": 0, "end": 7})
        
        return entities
    
    def extract_keywords(self, text: str, top_n: int) -> List[Dict[str, Any]]:
        """Mock keyword extraction"""
        # Split by whitespace and return most common words
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [
            {"term": word, "score": freq / len(words)}
            for word, freq in sorted_words[:top_n]
        ]
        return keywords
    
    def summarize(self, text: str, max_length: int) -> str:
        """Mock summarization - just truncate"""
        sentences = text.split('. ')
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + ". "
            else:
                break
        return summary.strip()
    
    def detect_language(self, text: str) -> str:
        """Mock language detection"""
        return "en"
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Mock relationship extraction"""
        return []


class MockSentimentModel:
    """Mock sentiment model for testing structure"""
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Mock sentiment analysis"""
        # Simple heuristic based on word presence
        positive_words = ["good", "great", "excellent", "happy", "success"]
        negative_words = ["bad", "terrible", "awful", "sad", "failure"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            label = "positive"
            polarity = 0.5
        elif neg_count > pos_count:
            label = "negative"
            polarity = -0.5
        else:
            label = "neutral"
            polarity = 0.0
        
        return {
            "polarity": polarity,
            "subjectivity": 0.5,
            "label": label
        }


# Global instance
nlp_service = NLPEnrichmentService()
