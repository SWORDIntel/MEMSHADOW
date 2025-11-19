# MEMSHADOW Intelligence Analysis Guide

## 2048-Dimensional Vector Intelligence System

### Overview

MEMSHADOW now features a comprehensive intelligence analysis platform with:
- **2048-dimensional vector embeddings** for superior semantic understanding
- **Fuzzy matching** combining string and semantic similarity
- **VantaBlackWidow-inspired** security analysis capabilities
- **Cross-system integration** for multi-source intelligence correlation

---

## Core Capabilities

### 1. Advanced NLP & Querying

**Query Expansion:**
```python
from app.services.advanced_nlp_service import advanced_nlp_service

# Automatically expands "what is SQL injection" to:
# - "definition of SQL injection"
# - "explain SQL injection"
# - "describe SQL injection"
# - "SQL injection meaning"
queries = await advanced_nlp_service.expand_query("what is SQL injection")
```

**Intent Classification:**
```python
# Classifies security-related queries
intent = await advanced_nlp_service.classify_security_intent(
    "Find vulnerabilities in authentication system"
)
# Returns: {
#   "intent": "vulnerability_search",
#   "confidence": 0.9,
#   "entities": {...}
# }
```

### 2. Fuzzy Matching (Hybrid Scoring)

**String + Semantic Similarity:**
```python
from app.services.fuzzy_vector_intel import fuzzy_vector_intel

matches = await fuzzy_vector_intel.fuzzy_match_text(
    query="CVE-2024-1234",
    candidates=[
        "CVE-2024-1234 allows remote code execution",
        "Critical vulnerability CVE-2024-1235",
        "Security advisory for CVE-2024-1234"
    ],
    threshold=0.7
)

# Returns scored matches:
# {
#   "candidate": "...",
#   "string_similarity": 0.85,
#   "semantic_similarity": 0.92,
#   "hybrid_score": 0.89  # (0.4*string + 0.6*semantic)
# }
```

### 3. IoC (Indicator of Compromise) Extraction

**Supported IoC Types:**
- IPv4/IPv6 addresses
- Domain names
- URLs
- Email addresses
- File hashes (MD5, SHA1, SHA256)
- CVE identifiers
- Cryptocurrency addresses (Bitcoin, Ethereum)
- Registry keys
- Windows/Unix file paths
- Mutexes
- ASNs

**Usage:**
```python
from app.services.vanta_blackwidow.ioc_identifier import ioc_identifier

text = """
The malware connected to 192.168.1.100 and downloaded
payload from evil.com. Hash: a1b2c3d4e5f6...
CVE-2024-1234 was exploited.
"""

iocs = ioc_identifier.extract_iocs(text)

# Returns IoC objects with:
# - type: 'ipv4', 'domain', 'md5', 'cve', etc.
# - value: The actual indicator
# - threat_level: 'critical', 'high', 'medium', 'low', 'unknown'
# - confidence: 0.0-1.0
# - context: Surrounding text
# - metadata: Additional information
```

**Threat Intelligence:**
```python
report = ioc_identifier.generate_report(iocs)

# Returns:
# {
#   "total_iocs": 15,
#   "by_type": {"ipv4": 3, "domain": 2, ...},
#   "by_threat_level": {"critical": 1, "high": 3, ...},
#   "high_confidence_critical": [...]
# }
```

### 4. Payload Fuzzing (InjectX-Style)

**120+ Attack Payloads:**
- SQL Injection (21 payloads)
- XSS (20 payloads)
- Command Injection (20 payloads)
- SSRF (15 payloads)
- Template Injection (16 payloads)
- XXE (4 payloads)
- LFI/Path Traversal (24 payloads)

**Usage:**
```python
from app.services.vanta_blackwidow.payload_fuzzer import payload_fuzzer

# Get all SQL injection payloads
sql_payloads = payload_fuzzer.get_payloads('sql_injection')

# Fuzz a parameter
test_cases = payload_fuzzer.fuzz_parameter(
    param_name="username",
    param_value="admin",
    attack_types=['sql_injection', 'xss']
)

# Test for vulnerabilities
result = payload_fuzzer.detect_vulnerability(
    response_text="<h1>Welcome admin' OR '1'='1</h1>",
    attack_type='sql_injection',
    payload="admin' OR '1'='1"
)

# Returns:
# {
#   "vulnerable": True,
#   "confidence": 0.9,
#   "evidence": ["Payload reflected in response"]
# }
```

### 5. Comprehensive Intelligence Analysis

**End-to-End Pipeline:**
```python
from app.services.fuzzy_vector_intel import fuzzy_vector_intel

report = await fuzzy_vector_intel.comprehensive_intelligence_analysis(
    text="""
    Suspicious activity detected from 10.0.0.5.
    Connection to C2 server at malicious.example.com.
    Ransomware hash: a1b2c3d4e5f6...
    """
)

# Returns comprehensive report:
# {
#   "extracted_iocs": {...},
#   "vector_representation": [...],  # 2048-dim vector
#   "cross_system_matches": {...},
#   "ioc_correlations": {...},
#   "threat_assessment": {
#     "threat_level": "CRITICAL",
#     "confidence": 0.9
#   }
# }
```

---

## MCP Integration

### Advanced Search

**Endpoint:** `POST /api/v1/mcp/memory/search/advanced`

**Features:**
- Query expansion
- IoC extraction
- Intent classification
- Semantic reranking
- 2048-dim vector search

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/mcp/memory/search/advanced \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find SQL injection vulnerabilities",
    "use_fuzzy": true,
    "expand_query": true,
    "extract_iocs": true,
    "limit": 10
  }'
```

**Response:**
```json
{
  "tool": "advanced_search",
  "status": "success",
  "query": {
    "original": "Find SQL injection vulnerabilities",
    "expanded": [
      "Find SQL injection vulnerabilities",
      "SQL injection vulnerability detection",
      "Identify SQL injection weaknesses"
    ],
    "intent": {
      "intent": "vulnerability_search",
      "confidence": 0.9
    }
  },
  "iocs_found": [],
  "num_results": 5,
  "results": [
    {
      "id": "...",
      "content": "...",
      "rerank_score": 0.95
    }
  ],
  "metadata": {
    "vector_dim": 2048,
    "fuzzy_matching": true,
    "reranked": true
  }
}
```

### Intelligence Analysis

**Endpoint:** `POST /api/v1/mcp/memory/intelligence/analyze`

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/mcp/memory/intelligence/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Malware beacon to 192.168.1.100:4444. CVE-2024-1234 exploited."
  }'
```

**Response:**
```json
{
  "tool": "intelligence_analysis",
  "status": "success",
  "analysis": {
    "extracted_iocs": {
      "total": 3,
      "by_type": {"ipv4": 1, "cve": 1},
      "high_threat": ["192.168.1.100", "CVE-2024-1234"]
    },
    "vector_representation": [...],  // 2048-dim
    "threat_assessment": {
      "threat_level": "HIGH",
      "confidence": 0.8
    }
  }
}
```

### Fuzzy Matching

**Endpoint:** `POST /api/v1/mcp/memory/fuzzy/match`

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/mcp/memory/fuzzy/match \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SQL injection attack",
    "candidates": [
      "SQL injection vulnerability in login",
      "XSS in search parameter",
      "Database injection flaw"
    ],
    "threshold": 0.7
  }'
```

---

## Cross-System Integration

### Integrating External Vector Systems

```python
from app.services.fuzzy_vector_intel import fuzzy_vector_intel
import numpy as np

# Your external system's 2048-dim vectors
external_vectors = np.array([...])  # shape: (n_samples, 2048)
metadata = [{"id": "ext1", "source": "ThreatDB"}, ...]

# Integrate
await fuzzy_vector_intel.integrate_external_vectors(
    system_name="ThreatIntelDB",
    vectors=external_vectors,
    metadata=metadata
)

# Now search across systems
results = await fuzzy_vector_intel.cross_system_search(
    query="Find ransomware indicators",
    systems=["ThreatIntelDB", "VulnDB"],
    top_k_per_system=5
)
```

---

## Use Cases

### 1. Threat Hunting

```python
# Search for indicators across all sources
analysis = await fuzzy_vector_intel.comprehensive_intelligence_analysis(
    text=log_data
)

# Extract all high-threat IoCs
threats = [
    ioc for ioc in analysis['extracted_iocs']
    if ioc['threat_level'] in ['high', 'critical']
]

# Find correlations
correlations = await fuzzy_vector_intel.correlate_iocs(threats)
```

### 2. Vulnerability Research

```python
# Fuzzy search for similar vulnerabilities
matches = await fuzzy_vector_intel.fuzzy_match_text(
    query="Remote code execution in authentication",
    candidates=vulnerability_database,
    threshold=0.75
)

# Test with payloads
test_cases = payload_fuzzer.fuzz_parameter(
    param_name="username",
    param_value="admin",
    attack_types=['sql_injection', 'command_injection']
)
```

### 3. Incident Response

```python
# Analyze incident data
report = await fuzzy_vector_intel.comprehensive_intelligence_analysis(
    text=incident_log
)

# Cluster related incidents
vectors = await fuzzy_vector_intel.vectorize_intelligence(incident_logs)
clusters = await fuzzy_vector_intel.cluster_intelligence(
    vectors,
    metadata=incident_metadata
)
```

---

## Performance Notes

- **Vector Dimension:** 2048 (vs 768 previously)
- **Fuzzy Matching:** Hybrid scoring (40% string + 60% semantic)
- **IoC Extraction:** Regex-based (sub-second for most documents)
- **Payload Library:** 120+ pre-loaded vectors
- **Cross-System:** Supports unlimited external systems
- **Clustering:** DBSCAN with cosine similarity

---

## Security Considerations

1. **IoC Privacy:** IoCs are stored with context - ensure compliance with data retention policies
2. **Payload Testing:** Only use against authorized systems (Arena environment)
3. **Vector Storage:** 2048-dim vectors are larger - monitor storage
4. **API Authentication:** All MCP endpoints require valid authentication
5. **Rate Limiting:** Recommended for fuzzy matching and analysis endpoints

---

## Next Steps

1. **Deploy Arena** for SWARM testing
2. **Integrate external threat feeds** using cross-system integration
3. **Create custom missions** for automated testing
4. **Build MCP server** to expose tools to AI assistants
5. **Tune similarity thresholds** based on your use cases

---

## Support

For issues or questions:
- GitHub: https://github.com/SWORDIntel/MEMSHADOW
- Documentation: See README.md and SWARM_README.md
