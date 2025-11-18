# MEMSHADOW Phase 3: Intelligence Layer Implementation Guide

## Executive Summary

Phase 3 (Weeks 17-24) focuses on advanced AI-driven memory processing, transforming MEMSHADOW from a simple memory store into an intelligent system capable of understanding, predicting, and connecting information.

---

## 1. NLP Enrichment Pipeline

### 1.1 Core Architecture

```python
# app/services/enrichment/nlp_service.py
class NLPEnrichmentService:
    async def enrich_memory(self, content: str) -> Dict[str, Any]:
        """Comprehensive NLP enrichment"""
        return {
            "entities": await self.extract_entities(content),
            "sentiment": await self.analyze_sentiment(content),
            "keywords": await self.extract_keywords(content),
            "language": await self.detect_language(content),
            "relationships": await self.extract_relationships(content),
            "summary": await self.generate_summary(content)
        }
```

### 1.2 Entity Extraction

Extracts named entities using spaCy:
- **PERSON**: People and characters
- **TECHNOLOGY**: Programming languages, frameworks, tools
- **ORGANIZATION**: Companies, institutions
- **LOCATION**: Places and locations
- **DATE**: Temporal references
- **CUSTOM**: Domain-specific entities

### 1.3 Sentiment Analysis

Multi-dimensional sentiment scoring:
- **Polarity**: -1.0 (negative) to +1.0 (positive)
- **Subjectivity**: 0.0 (objective) to 1.0 (subjective)
- **Label**: positive, negative, neutral

### 1.4 Keyword Extraction

TF-IDF and TextRank based extraction:
- Identifies key terms and concepts
- Provides importance scores
- Filters common stopwords

---

## 2. Knowledge Graph Construction

### 2.1 Graph Structure

```python
# app/services/enrichment/knowledge_graph.py
class KnowledgeGraphService:
    async def build_graph_from_memory(
        self,
        memory_id: UUID,
        enrichment_data: Dict
    ):
        # Create nodes for entities
        # Create edges for relationships
        # Connect to memory node
        pass
```

### 2.2 Node Types

- **MEMORY**: Memory entries
- **ENTITY**: Extracted entities
- **KEYWORD**: Important terms
- **PERSON**: People and characters
- **TECHNOLOGY**: Tools and frameworks
- **CONCEPT**: Abstract concepts

### 2.3 Edge Types

- **MENTIONS**: Memory mentions entity
- **RELATED_TO**: Generic relationship
- **HAS_KEYWORD**: Memory has keyword
- **KNOWS**: Knowledge relationship
- **PART_OF**: Hierarchical relationship
- **USES**: Usage relationship

### 2.4 Graph Queries

```python
# Find related nodes
related = await kg.find_related_nodes(node_id, max_depth=2)

# Find shortest path
path = await kg.find_shortest_path(source_id, target_id)

# Get neighborhood
neighborhood = await kg.get_node_neighborhood(node_id, radius=1)

# Export for visualization
graph_data = await kg.export_graph(format="cytoscape")
```

---

## 3. Multi-Modal Embeddings

### 3.1 Supported Modalities

```python
class ContentType(Enum):
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
```

### 3.2 Embedding Strategies

**Text**: Sentence Transformers (all-mpnet-base-v2)
- 768-dimensional embeddings
- Optimized for semantic similarity

**Code**: Code-aware embeddings
- Language-specific preprocessing
- AST-aware representations
- Support for CodeBERT, GraphCodeBERT

**Images**: CLIP multimodal embeddings
- 512-dimensional joint text-image space
- Enables cross-modal search

**Documents**: Structure-aware embedding
- Section-based embedding
- Hierarchical aggregation

### 3.3 Cross-Modal Retrieval

```python
# Find images related to text query
await multimodal.find_cross_modal_matches(
    query_embedding=text_embedding,
    candidate_embeddings=image_embeddings,
    top_k=10
)
```

---

## 4. Local LLM Integration

### 4.1 Supported Models

- **Phi-3-mini**: 3.8B parameter model, efficient
- **Gemma-2B**: Google's efficient model
- **Llama-3-8B**: Meta's latest (quantized)
- **Mistral-7B**: High-quality 7B model

### 4.2 Enrichment Tasks

```python
# app/services/enrichment/local_llm.py
class LocalLLMService:
    async def enrich_with_context(self, text: str):
        return {
            "summary": await self.generate_summary(text),
            "insights": await self.extract_insights(text),
            "questions": await self.generate_questions(text),
            "classifications": await self.classify_content(text, categories)
        }
```

### 4.3 Privacy Benefits

- No data leaves local environment
- Sensitive content never sent to cloud
- Full control over model and inference
- GDPR and compliance friendly

---

## 5. Predictive Retrieval

### 5.1 Prediction Strategies

**Sequence-based**:
- Learns common access patterns
- "Users who viewed X often view Y next"
- Markov chain-style prediction

**Temporal-based**:
- Time-of-day patterns
- Day-of-week patterns
- Seasonal trends

**Context-based**:
- Query similarity matching
- Topic clustering
- Project context awareness

### 5.2 Implementation

```python
# Record access for learning
await predictive.record_access(
    user_id=user.id,
    memory_id=memory.id,
    context={"query": query, "project": current_project}
)

# Get predictions
predictions = await predictive.predict_next_memories(
    user_id=user.id,
    current_context={"query": "async python"},
    top_k=5
)

# Preload cache
preload_ids = await predictive.preload_cache(user_id)
```

### 5.3 Cache Preloading

- Predicts likely next accesses
- Preloads into Redis cache
- Reduces retrieval latency
- Improves user experience

---

## 6. Database Schema

### 6.1 Knowledge Graph Tables

```sql
CREATE TABLE kg_nodes (
    id UUID PRIMARY KEY,
    node_id VARCHAR(255) UNIQUE NOT NULL,
    node_type VARCHAR(50) NOT NULL,
    label VARCHAR(255) NOT NULL,
    properties JSONB DEFAULT '{}',
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE kg_edges (
    id UUID PRIMARY KEY,
    source_node_id VARCHAR(255) NOT NULL,
    target_node_id VARCHAR(255) NOT NULL,
    edge_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    properties JSONB DEFAULT '{}',
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_kg_node_type_user ON kg_nodes(node_type, user_id);
CREATE INDEX idx_kg_edge_source_target ON kg_edges(source_node_id, target_node_id);
```

### 6.2 Memory Enrichment Table

```sql
CREATE TABLE memory_enrichments (
    id UUID PRIMARY KEY,
    memory_id UUID REFERENCES memories(id),
    entities JSONB DEFAULT '[]',
    keywords JSONB DEFAULT '[]',
    sentiment JSONB DEFAULT '{}',
    language VARCHAR(10) DEFAULT 'en',
    summary TEXT,
    relationships JSONB DEFAULT '[]',
    llm_summary TEXT,
    insights JSONB DEFAULT '[]',
    questions JSONB DEFAULT '[]',
    enriched_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.3 Access Patterns Table

```sql
CREATE TABLE access_patterns (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    memory_id UUID REFERENCES memories(id),
    query TEXT,
    context JSONB DEFAULT '{}',
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    hour_of_day INTEGER,
    day_of_week INTEGER,
    session_id UUID
);

CREATE INDEX idx_access_user_time ON access_patterns(user_id, accessed_at);
CREATE INDEX idx_access_temporal ON access_patterns(user_id, hour_of_day, day_of_week);
```

---

## 7. Integration with Memory Service

### 7.1 Enrichment Pipeline

```python
# app/workers/enrichment_tasks.py
@celery_app.task
async def enrich_memory_task(memory_id: str, content: str):
    # NLP enrichment
    nlp_data = await nlp_service.enrich_memory(content)
    
    # LLM enrichment (if enabled)
    llm_data = await local_llm.enrich_with_context(content)
    
    # Build knowledge graph
    await kg_service.build_graph_from_memory(memory_id, nlp_data)
    
    # Store enrichment
    await store_enrichment(memory_id, nlp_data, llm_data)
```

### 7.2 Retrieval Enhancement

```python
# app/services/memory_service.py
async def search_memories(
    self,
    user_id: UUID,
    query: str
):
    # Record access for prediction
    await predictive.record_access(user_id, query)
    
    # Get predictions
    predicted_ids = await predictive.predict_next_memories(user_id)
    
    # Semantic search
    results = await self._semantic_search(query)
    
    # Knowledge graph enhancement
    kg_related = await kg.find_related_nodes(results)
    
    return results + kg_related
```

---

## 8. Performance Considerations

### 8.1 Async Processing

All enrichment runs asynchronously:
- Background Celery workers
- Non-blocking memory ingestion
- Progressive enhancement

### 8.2 Caching Strategy

```python
# Redis caching for predictions
await redis.setex(
    f"predictions:{user_id}",
    3600,  # 1 hour TTL
    json.dumps(predictions)
)
```

### 8.3 Resource Management

**CPU Intensive**:
- NLP processing
- LLM inference
- Graph computations

**Solution**:
- Dedicated worker pools
- Priority queues
- Resource limits

---

## 9. Testing

### 9.1 Unit Tests

```python
# tests/unit/test_nlp_enrichment.py
async def test_enrich_memory():
    enrichment = await nlp_service.enrich_memory(text)
    assert "entities" in enrichment
    assert "keywords" in enrichment
```

### 9.2 Integration Tests

```python
# tests/integration/test_enrichment_pipeline.py
async def test_full_enrichment_pipeline():
    memory = await create_memory(content)
    enrichment = await enrich_memory_task(memory.id)
    kg_data = await kg_service.get_graph_stats()
    assert kg_data["total_nodes"] > 0
```

---

## 10. Production Deployment

### 10.1 Model Requirements

**NLP Models**:
- spaCy: en_core_web_lg (560MB)
- Sentiment: fine-tuned BERT (440MB)

**LLM Models**:
- Phi-3-mini-4k-instruct (4-bit): ~2.3GB
- Context window: 4K tokens

### 10.2 Hardware Recommendations

**Minimum**:
- 16GB RAM
- 4 CPU cores
- 20GB disk (for models)

**Recommended**:
- 32GB RAM
- 8 CPU cores (or 4 cores + GPU)
- SSD storage
- Optional: NPU for accelerated inference

### 10.3 Configuration

```yaml
# config/enrichment.yaml
nlp:
  enabled: true
  model: "en_core_web_lg"
  batch_size: 32

llm:
  enabled: false  # Enable for privacy-sensitive deployments
  model: "phi-3-mini"
  quantization: "4bit"
  max_tokens: 200

knowledge_graph:
  enabled: true
  max_depth: 3
  export_format: "cytoscape"

predictive:
  enabled: true
  prediction_threshold: 0.3
  cache_preload: true
```

---

## Summary

Phase 3 transforms MEMSHADOW into an intelligent system:

1. **NLP Enrichment**: Deep understanding of content
2. **Knowledge Graphs**: Connected information networks
3. **Multi-Modal**: Support for diverse content types
4. **Local LLM**: Privacy-preserving intelligence
5. **Predictive**: Anticipates user needs

These capabilities enable MEMSHADOW to not just store memories, but to understand, connect, and intelligently surface information when needed.
