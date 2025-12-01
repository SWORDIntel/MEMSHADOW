"""
KP14 Integration API Endpoints

REST API for receiving and querying KP14 malware analysis data.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime
import structlog

from app.schemas.kp14 import (
    KP14AnalysisCreate,
    KP14AnalysisResponse,
    KP14AnalysisSearch,
    IOCBatch,
    TechniqueBatch,
    MeshThreatIntel,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/kp14", tags=["KP14 Integration"])

# In-memory storage (replace with proper database in production)
_analyses: Dict[str, Dict[str, Any]] = {}
_knowledge_graph = None
_enrichment_service = None


def get_knowledge_graph():
    """Get or create knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        try:
            from app.services.threat_intel.kp14_knowledge_graph import KP14KnowledgeGraph
            _knowledge_graph = KP14KnowledgeGraph()
        except Exception as e:
            logger.warning(f"Failed to initialize knowledge graph: {e}")
    return _knowledge_graph


def get_enrichment_service():
    """Get or create enrichment service instance."""
    global _enrichment_service
    if _enrichment_service is None:
        try:
            from app.services.threat_intel.kp14_enrichment import KP14EnrichmentService
            _enrichment_service = KP14EnrichmentService(
                knowledge_graph=get_knowledge_graph()
            )
        except Exception as e:
            logger.warning(f"Failed to initialize enrichment service: {e}")
    return _enrichment_service


@router.post("/analyses", response_model=Dict[str, Any])
async def create_analysis(
    analysis: KP14AnalysisCreate,
    background_tasks: BackgroundTasks,
    enrich: bool = Query(True, description="Apply enrichment")
):
    """
    Store a KP14 analysis result.

    This endpoint receives analysis data from KP14 (typically via mesh sync)
    and stores it in MEMSHADOW's threat intelligence store.
    """
    try:
        # Convert to dict
        analysis_data = analysis.model_dump()
        analysis_id = analysis.analysis_id

        # Add to knowledge graph in background
        kg = get_knowledge_graph()
        if kg:
            background_tasks.add_task(kg.add_analysis, analysis_data)

        # Apply enrichment if enabled
        if enrich:
            enrichment = get_enrichment_service()
            if enrichment:
                analysis_data = enrichment.post_enrich(analysis_data)

        # Store analysis
        _analyses[analysis_id] = {
            'id': analysis_id,
            'analysis_id': analysis_id,
            **analysis_data,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
        }

        logger.info("KP14 analysis stored", analysis_id=analysis_id)

        return {
            'status': 'success',
            'analysis_id': analysis_id,
            'enriched': enrich,
        }

    except Exception as e:
        logger.error("Failed to store analysis", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{analysis_id}", response_model=Dict[str, Any])
async def get_analysis(analysis_id: str):
    """Get a specific KP14 analysis by ID."""
    if analysis_id not in _analyses:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return _analyses[analysis_id]


@router.post("/analyses/search", response_model=List[Dict[str, Any]])
async def search_analyses(search: KP14AnalysisSearch):
    """
    Search KP14 analyses.

    Supports filtering by hash, malware family, threat actor,
    threat score range, techniques, and IOCs.
    """
    results = []

    for analysis_id, analysis in _analyses.items():
        # Apply filters
        if search.hash:
            sample_hash = analysis.get('sample', {}).get('hash_sha256', '')
            if search.hash.lower() not in sample_hash.lower():
                continue

        if search.malware_family:
            family = analysis.get('analysis', {}).get('malware_family', '')
            if search.malware_family.lower() not in (family or '').lower():
                continue

        if search.threat_actor:
            actor = analysis.get('analysis', {}).get('threat_actor', '')
            if search.threat_actor.lower() not in (actor or '').lower():
                continue

        if search.min_threat_score is not None:
            score = analysis.get('analysis', {}).get('threat_score', 0)
            if score < search.min_threat_score:
                continue

        if search.max_threat_score is not None:
            score = analysis.get('analysis', {}).get('threat_score', 0)
            if score > search.max_threat_score:
                continue

        if search.technique_ids:
            techniques = analysis.get('techniques', [])
            tech_ids = {t.get('technique_id', '') for t in techniques}
            if not any(tid in tech_ids for tid in search.technique_ids):
                continue

        if search.ioc_value:
            iocs = analysis.get('iocs', [])
            ioc_values = {i.get('value', '') for i in iocs}
            if not any(search.ioc_value.lower() in v.lower() for v in ioc_values):
                continue

        results.append(analysis)

    # Apply pagination
    start = search.offset
    end = start + search.limit

    return results[start:end]


@router.get("/analyses/by-hash/{sample_hash}", response_model=Optional[Dict[str, Any]])
async def get_analysis_by_hash(sample_hash: str):
    """Get analysis by sample hash."""
    for analysis in _analyses.values():
        if analysis.get('sample', {}).get('hash_sha256', '').startswith(sample_hash):
            return analysis

    raise HTTPException(status_code=404, detail="Analysis not found for hash")


@router.post("/iocs/batch", response_model=Dict[str, Any])
async def store_ioc_batch(batch: IOCBatch, background_tasks: BackgroundTasks):
    """
    Store a batch of IOCs from KP14.

    Used for bulk IOC ingestion from mesh broadcasts.
    """
    try:
        # Enrich IOCs
        enrichment = get_enrichment_service()
        enriched_iocs = batch.iocs
        if enrichment:
            enriched_iocs = enrichment.enrich_iocs([i.model_dump() for i in batch.iocs])

        logger.info("IOC batch stored", count=len(batch.iocs))

        return {
            'status': 'success',
            'iocs_stored': len(batch.iocs),
            'campaign_id': batch.campaign_id,
        }

    except Exception as e:
        logger.error("Failed to store IOC batch", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/techniques/batch", response_model=Dict[str, Any])
async def store_technique_batch(batch: TechniqueBatch, background_tasks: BackgroundTasks):
    """
    Store a batch of ATT&CK technique mappings from KP14.
    """
    try:
        # Enrich techniques
        enrichment = get_enrichment_service()
        enriched_techniques = batch.techniques
        if enrichment:
            enriched_techniques = enrichment.enrich_techniques(
                [t.model_dump() for t in batch.techniques]
            )

        logger.info("Technique batch stored",
                   count=len(batch.techniques),
                   sample_hash=batch.sample_hash[:16])

        return {
            'status': 'success',
            'techniques_stored': len(batch.techniques),
            'sample_hash': batch.sample_hash,
        }

    except Exception as e:
        logger.error("Failed to store technique batch", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/related/{sample_hash}", response_model=List[Dict[str, Any]])
async def get_related_samples(
    sample_hash: str,
    limit: int = Query(10, ge=1, le=50)
):
    """
    Find samples related to the given hash.

    Uses knowledge graph to find samples that share:
    - Same malware family
    - Same threat actor
    - Same C2 infrastructure
    - Same IOCs
    - Similar techniques
    """
    kg = get_knowledge_graph()
    if not kg:
        return []

    return kg.find_related_samples(sample_hash, max_depth=2, limit=limit)


@router.get("/family/{family_name}/samples", response_model=List[Dict[str, Any]])
async def get_family_samples(
    family_name: str,
    limit: int = Query(50, ge=1, le=100)
):
    """Get all samples of a malware family."""
    kg = get_knowledge_graph()
    if not kg:
        return []

    return kg.get_malware_family_samples(family_name, limit=limit)


@router.get("/actor/{actor_name}/samples", response_model=List[Dict[str, Any]])
async def get_actor_samples(
    actor_name: str,
    limit: int = Query(50, ge=1, le=100)
):
    """Get all samples attributed to a threat actor."""
    kg = get_knowledge_graph()
    if not kg:
        return []

    return kg.get_threat_actor_samples(actor_name, limit=limit)


@router.get("/technique/{technique_id}/samples", response_model=List[Dict[str, Any]])
async def get_technique_samples(
    technique_id: str,
    limit: int = Query(50, ge=1, le=100)
):
    """Get samples using a specific ATT&CK technique."""
    kg = get_knowledge_graph()
    if not kg:
        return []

    return kg.get_technique_samples(technique_id, limit=limit)


@router.get("/ioc/{ioc_value}/samples", response_model=List[Dict[str, Any]])
async def get_ioc_samples(ioc_value: str):
    """Get samples containing a specific IOC."""
    kg = get_knowledge_graph()
    if not kg:
        return []

    return kg.get_ioc_samples(ioc_value)


@router.get("/pre-enrich/{sample_hash}", response_model=Dict[str, Any])
async def pre_enrich_sample(sample_hash: str):
    """
    Get pre-analysis enrichment for a sample.

    Call this before analyzing a sample to get prior intelligence
    that can inform the analysis.
    """
    enrichment = get_enrichment_service()
    if not enrichment:
        return {'sample_hash': sample_hash, 'enrichment': None}

    context = enrichment.pre_enrich(sample_hash)
    return context


@router.post("/mesh/threat-intel", response_model=Dict[str, Any])
async def receive_mesh_threat_intel(
    intel: MeshThreatIntel,
    background_tasks: BackgroundTasks
):
    """
    Receive threat intelligence from mesh network.

    This endpoint handles incoming messages from KP14 nodes
    via the dsmil-mesh network.
    """
    try:
        if intel.type == "kp14_analysis":
            # Convert to analysis and store
            analysis_data = {
                'analysis_id': intel.analysis_id,
                'sample': intel.sample.model_dump() if intel.sample else {},
                'analysis': intel.analysis.model_dump() if intel.analysis else {},
                'iocs': [i.model_dump() for i in intel.iocs],
                'techniques': [t.model_dump() for t in intel.techniques],
                'c2_endpoints': [c.model_dump() for c in intel.c2_endpoints],
                'source_node': intel.source_node,
            }

            # Add to knowledge graph
            kg = get_knowledge_graph()
            if kg:
                background_tasks.add_task(kg.add_analysis, analysis_data)

            # Store
            if intel.analysis_id:
                _analyses[intel.analysis_id] = {
                    **analysis_data,
                    'created_at': datetime.utcnow().isoformat(),
                }

            logger.info("Mesh threat intel received",
                       type=intel.type,
                       source=intel.source_node)

        return {
            'status': 'success',
            'type': intel.type,
            'source_node': intel.source_node,
        }

    except Exception as e:
        logger.error("Failed to process mesh threat intel", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=Dict[str, Any])
async def get_kp14_stats():
    """Get KP14 integration statistics."""
    kg = get_knowledge_graph()
    enrichment = get_enrichment_service()

    stats = {
        'total_analyses': len(_analyses),
        'knowledge_graph': kg.get_stats() if kg else None,
        'enrichment': enrichment.get_stats() if enrichment else None,
    }

    return stats

