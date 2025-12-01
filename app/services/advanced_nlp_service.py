"""
Advanced NLP Service for MEMSHADOW MCP Integration

Provides sophisticated natural language querying capabilities:
- Query expansion and semantic enrichment
- Multi-query fusion
- Contextual reranking
- Intent classification
- Entity extraction
"""

from typing import List, Dict, Any, Tuple
import re
from sentence_transformers import SentenceTransformer, util
import torch
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class AdvancedNLPService:
    """
    Advanced NLP service for semantic search and query understanding
    """

    def __init__(self):
        self.model = None
        self.device = None
        self._initialize_model()

        # Query expansion templates
        self.expansion_templates = {
            'what_is': [
                'definition of {}',
                'explain {}',
                'describe {}',
                '{} meaning'
            ],
            'how_to': [
                'steps to {}',
                'guide for {}',
                'tutorial on {}',
                'instructions for {}'
            ],
            'when': [
                'timing of {}',
                'schedule for {}',
                '{} occurrence'
            ],
            'why': [
                'reason for {}',
                'purpose of {}',
                '{} explanation',
                'rationale behind {}'
            ]
        }

    def _initialize_model(self):
        """Initialize the advanced NLP model"""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Use a more advanced model for 2048-dim embeddings
            # Note: The actual model should support 2048 dimensions
            # For demonstration, using a model that can be extended
            self.model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=self.device
            )

            logger.info(
                "Advanced NLP model initialized",
                model=settings.EMBEDDING_MODEL,
                device=str(self.device),
                dimension=settings.EMBEDDING_DIMENSION
            )

        except Exception as e:
            logger.error("Failed to initialize NLP model", error=str(e))
            raise

    async def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple semantically related queries

        Args:
            query: Original query

        Returns:
            List of expanded queries including the original
        """
        if not settings.NLP_QUERY_EXPANSION:
            return [query]

        expanded = [query]

        # Detect query intent
        intent = self._detect_intent(query)

        if intent and intent in self.expansion_templates:
            # Extract the main subject
            subject = self._extract_subject(query)

            if subject:
                # Generate expansions
                for template in self.expansion_templates[intent]:
                    expanded.append(template.format(subject))

        logger.debug(f"Query expanded from 1 to {len(expanded)} variants", original=query)

        return expanded[:5]  # Limit to top 5 expansions

    def _detect_intent(self, query: str) -> str:
        """
        Detect the intent of a query

        Args:
            query: Query text

        Returns:
            Intent type or None
        """
        query_lower = query.lower()

        if query_lower.startswith(('what is', 'what are', 'define')):
            return 'what_is'
        elif query_lower.startswith(('how to', 'how do', 'how can')):
            return 'how_to'
        elif query_lower.startswith(('when', 'what time')):
            return 'when'
        elif query_lower.startswith(('why', 'what caused', 'reason')):
            return 'why'

        return None

    def _extract_subject(self, query: str) -> str:
        """
        Extract the main subject from a query

        Args:
            query: Query text

        Returns:
            Extracted subject
        """
        # Remove common question words
        subject = re.sub(
            r'^(what is|what are|how to|how do|when|why|define|explain)\s+',
            '',
            query.lower()
        )

        # Remove question marks and extra spaces
        subject = re.sub(r'[?.]', '', subject).strip()

        return subject

    async def multi_query_fusion(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Fuse results from multiple queries using reciprocal rank fusion

        Args:
            queries: List of queries
            top_k: Number of top results to return

        Returns:
            List of (result_id, fused_score) tuples
        """
        # Generate embeddings for all queries
        embeddings = self.model.encode(queries, convert_to_tensor=True)

        # In a real implementation, this would:
        # 1. Execute each query against the vector DB
        # 2. Apply reciprocal rank fusion
        # 3. Return fused results

        # For now, return the mean embedding as a single query
        mean_embedding = torch.mean(embeddings, dim=0)

        logger.debug(f"Fused {len(queries)} queries into unified representation")

        return mean_embedding.cpu().tolist()

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'ips': [],
            'urls': [],
            'emails': [],
            'hashes': [],
            'cves': [],
            'domains': []
        }

        # IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        entities['ips'] = re.findall(ip_pattern, text)

        # URLs
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        entities['urls'] = re.findall(url_pattern, text)

        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)

        # MD5, SHA1, SHA256 hashes
        hash_pattern = r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b'
        entities['hashes'] = re.findall(hash_pattern, text)

        # CVE identifiers
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        entities['cves'] = re.findall(cve_pattern, text, re.IGNORECASE)

        # Domain names
        domain_pattern = r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]\b'
        entities['domains'] = re.findall(domain_pattern, text.lower())

        logger.debug("Entity extraction completed", entity_counts={k: len(v) for k, v in entities.items()})

        return entities

    async def classify_security_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify the security-related intent of a query

        Args:
            query: Query text

        Returns:
            Classification result with intent and confidence
        """
        query_lower = query.lower()

        intents = {
            'vulnerability_search': [
                'vulnerability', 'exploit', 'cve', 'weakness', 'bug'
            ],
            'threat_intelligence': [
                'threat', 'malware', 'actor', 'campaign', 'attack'
            ],
            'incident_analysis': [
                'incident', 'breach', 'compromise', 'intrusion'
            ],
            'configuration_info': [
                'configure', 'setup', 'install', 'deploy'
            ],
            'compliance': [
                'compliance', 'regulation', 'policy', 'audit'
            ],
            'reconnaissance': [
                'scan', 'enumerate', 'discover', 'map'
            ]
        }

        detected_intent = 'general_query'
        max_matches = 0

        for intent, keywords in intents.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_intent = intent

        confidence = min(max_matches * 0.3, 1.0)

        return {
            'intent': detected_intent,
            'confidence': confidence,
            'entities': await self.extract_entities(query)
        }

    async def semantic_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on semantic similarity to query

        Args:
            query: Original query
            results: List of result dictionaries with 'content' field
            top_k: Number of top results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Generate embeddings for all results
        result_texts = [r.get('content', '') for r in results]
        result_embeddings = self.model.encode(result_texts, convert_to_tensor=True)

        # Calculate cosine similarities
        similarities = util.cos_sim(query_embedding, result_embeddings)[0]

        # Sort by similarity
        sorted_indices = torch.argsort(similarities, descending=True)

        # Rerank results
        reranked = []
        for idx in sorted_indices[:top_k]:
            result = results[idx.item()].copy()
            result['rerank_score'] = similarities[idx].item()
            reranked.append(result)

        logger.debug(f"Reranked {len(results)} results, returning top {len(reranked)}")

        return reranked


# Global instance
advanced_nlp_service = AdvancedNLPService()
