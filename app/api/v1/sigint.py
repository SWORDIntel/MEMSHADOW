"""
MEMSHADOW Phase-0 SIGINT/GEOINT API Endpoints

REST API for ingesting and querying intelligence observations.
"""

import logging
from typing import List, Optional
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, Float
from sqlalchemy.orm import selectinload

from app.core.security import get_current_user
from app.db.postgres import get_db
from app.models.user import User
from app.models.sigint_geoint import (
    Observation,
    Device,
    Identity,
    Service,
    AreaOfInterest
)
from app.schemas.sigint_geoint import (
    ObservationCreate,
    ObservationResponse,
    ObservationQuery,
    ObservationUpdate,
    DeviceCreate,
    DeviceResponse,
    IdentityCreate,
    IdentityResponse,
    AOICreate,
    AOIResponse,
    GeoIPEnrichmentRequest,
    GeoIPEnrichmentResponse,
    AOIMembershipUpdate,
    TimelineQuery
)
from app.services.sigint.geo_enricher import get_geo_enricher


logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Observation Endpoints
# ============================================================================

@router.post("/observations", response_model=ObservationResponse, status_code=status.HTTP_201_CREATED)
async def create_observation(
    observation: ObservationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new intelligence observation.

    This endpoint is called by sigint-net-agent and other collectors.
    Observations are automatically enriched with GeoIP and AOI data.
    """
    try:
        # Enrich with GeoIP and AOI data
        geo_enricher = get_geo_enricher()
        obs_dict = observation.model_dump()

        if geo_enricher.is_available():
            obs_dict = await geo_enricher.enrich(obs_dict)
        else:
            logger.warning("Geo enricher not available - observation will not be enriched")

        # Generate node_id if not provided
        if not obs_dict.get("node_id"):
            ts = obs_dict["timestamp"].isoformat() if isinstance(obs_dict["timestamp"], datetime) else obs_dict["timestamp"]
            host = obs_dict["host"]
            channel = obs_dict.get("labels", {}).get("channel", "UNKNOWN")
            identifier = obs_dict.get("payload", {}).get("src_ip", str(uuid4())[:8])
            obs_dict["node_id"] = f"obs:{ts}:{host}:{channel}:{identifier}"

        # Create observation model
        db_obs = Observation(
            user_id=current_user.id,
            node_id=obs_dict["node_id"],
            modality=obs_dict["modality"],
            source=obs_dict["source"],
            timestamp=obs_dict["timestamp"] if isinstance(obs_dict["timestamp"], datetime) else datetime.fromisoformat(obs_dict["timestamp"]),
            host=obs_dict["host"],
            sensor_id=obs_dict["sensor_id"],
            labels=obs_dict.get("labels", {}),
            subjects=obs_dict.get("subjects", []),
            location=obs_dict.get("location"),
            payload=obs_dict.get("payload", {}),
            signals=obs_dict.get("signals", {}),
            embedding=obs_dict.get("embedding"),
            aoi_memberships=obs_dict.get("aoi_memberships", [])
        )

        db.add(db_obs)
        await db.commit()
        await db.refresh(db_obs)

        logger.info(f"Created observation: {db_obs.node_id}")
        return db_obs

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create observation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create observation: {str(e)}"
        )


@router.get("/observations", response_model=List[ObservationResponse])
async def query_observations(
    modality: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    host: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    aoi_ids: Optional[str] = Query(None, description="Comma-separated AOI node IDs"),
    min_risk_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Query intelligence observations with filters.

    Supports filtering by modality, channel, time range, AOI, and more.
    """
    try:
        # Build query
        query = select(Observation).where(Observation.user_id == current_user.id)

        if modality:
            query = query.where(Observation.modality == modality)

        if channel:
            query = query.where(Observation.labels["channel"].astext == channel)

        if source:
            query = query.where(Observation.source == source)

        if host:
            query = query.where(Observation.host == host)

        if start_time:
            query = query.where(Observation.timestamp >= start_time)

        if end_time:
            query = query.where(Observation.timestamp <= end_time)

        if aoi_ids:
            aoi_list = [aoi_id.strip() for aoi_id in aoi_ids.split(",")]
            # Check if any of the AOI IDs are in the aoi_memberships array
            query = query.where(
                or_(*[Observation.aoi_memberships.any(aoi_id) for aoi_id in aoi_list])
            )

        if min_risk_score is not None:
            # Filter by risk_score in signals JSONB field
            query = query.where(
                func.cast(Observation.signals["risk_score"], Float) >= min_risk_score
            )

        # Apply pagination
        query = query.order_by(Observation.timestamp.desc())
        query = query.offset(offset).limit(limit)

        result = await db.execute(query)
        observations = result.scalars().all()

        logger.info(f"Queried {len(observations)} observations")
        return observations

    except Exception as e:
        logger.error(f"Failed to query observations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query observations: {str(e)}"
        )


@router.get("/observations/{observation_id}", response_model=ObservationResponse)
async def get_observation(
    observation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific observation by ID."""
    query = select(Observation).where(
        and_(
            Observation.id == observation_id,
            Observation.user_id == current_user.id
        )
    )

    result = await db.execute(query)
    obs = result.scalar_one_or_none()

    if not obs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Observation not found"
        )

    return obs


@router.get("/observations/by-node/{node_id}", response_model=ObservationResponse)
async def get_observation_by_node_id(
    node_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific observation by node_id."""
    query = select(Observation).where(
        and_(
            Observation.node_id == node_id,
            Observation.user_id == current_user.id
        )
    )

    result = await db.execute(query)
    obs = result.scalar_one_or_none()

    if not obs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Observation not found"
        )

    return obs


@router.get("/observations/stats/summary")
async def get_observation_stats(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get summary statistics for observations.

    Returns counts by modality, channel, and time period.
    """
    try:
        query = select(Observation).where(Observation.user_id == current_user.id)

        if start_time:
            query = query.where(Observation.timestamp >= start_time)

        if end_time:
            query = query.where(Observation.timestamp <= end_time)

        result = await db.execute(query)
        observations = result.scalars().all()

        # Compute statistics
        total_count = len(observations)
        modality_counts = {}
        channel_counts = {}

        for obs in observations:
            # Count by modality
            modality_counts[obs.modality] = modality_counts.get(obs.modality, 0) + 1

            # Count by channel
            channel = obs.labels.get("channel", "unknown")
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

        return {
            "total_observations": total_count,
            "by_modality": modality_counts,
            "by_channel": channel_counts,
            "time_range": {
                "start": start_time,
                "end": end_time
            }
        }

    except Exception as e:
        logger.error(f"Failed to get observation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


# ============================================================================
# Device Endpoints
# ============================================================================

@router.post("/devices", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def create_device(
    device: DeviceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new device node."""
    try:
        # Generate node_id if not provided
        node_id = device.node_id or f"device:{device.device_type.lower()}:{device.identifier}"

        db_device = Device(
            user_id=current_user.id,
            node_id=node_id,
            device_type=device.device_type,
            identifier=device.identifier,
            labels=device.labels,
            first_seen=device.first_seen,
            last_seen=device.last_seen,
            risk_score=device.risk_score
        )

        db.add(db_device)
        await db.commit()
        await db.refresh(db_device)

        logger.info(f"Created device: {db_device.node_id}")
        return db_device

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create device: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create device: {str(e)}"
        )


@router.get("/devices", response_model=List[DeviceResponse])
async def list_devices(
    device_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all device nodes."""
    query = select(Device).where(Device.user_id == current_user.id)

    if device_type:
        query = query.where(Device.device_type == device_type)

    query = query.order_by(Device.last_seen.desc())
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    devices = result.scalars().all()

    return devices


# ============================================================================
# AOI Endpoints
# ============================================================================

@router.post("/aois", response_model=AOIResponse, status_code=status.HTTP_201_CREATED)
async def create_aoi(
    aoi: AOICreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new Area of Interest."""
    try:
        # Generate node_id if not provided
        node_id = aoi.node_id or f"aoi:{aoi.name.lower().replace(' ', '-')}"

        db_aoi = AreaOfInterest(
            user_id=current_user.id,
            node_id=node_id,
            name=aoi.name,
            category=aoi.category,
            description=aoi.description,
            labels=aoi.labels,
            geometry=aoi.geometry.model_dump()
        )

        db.add(db_aoi)
        await db.commit()
        await db.refresh(db_aoi)

        # Load into geo enricher cache
        geo_enricher = get_geo_enricher()
        if geo_enricher.is_available():
            geo_enricher.aoi_matcher.load_aoi(
                node_id,
                aoi.geometry,
                {"name": aoi.name, "category": aoi.category, "labels": aoi.labels}
            )

        logger.info(f"Created AOI: {db_aoi.node_id}")
        return db_aoi

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create AOI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create AOI: {str(e)}"
        )


@router.get("/aois", response_model=List[AOIResponse])
async def list_aois(
    category: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all Areas of Interest."""
    query = select(AreaOfInterest).where(AreaOfInterest.user_id == current_user.id)

    if category:
        query = query.where(AreaOfInterest.category == category)

    query = query.order_by(AreaOfInterest.name)
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    aois = result.scalars().all()

    return aois


@router.get("/aois/{aoi_id}/observations", response_model=List[ObservationResponse])
async def get_aoi_observations(
    aoi_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all observations within a specific AOI."""
    # First verify AOI exists
    aoi_query = select(AreaOfInterest).where(
        and_(
            AreaOfInterest.node_id == aoi_id,
            AreaOfInterest.user_id == current_user.id
        )
    )
    result = await db.execute(aoi_query)
    aoi = result.scalar_one_or_none()

    if not aoi:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AOI not found"
        )

    # Query observations in this AOI
    query = select(Observation).where(
        and_(
            Observation.user_id == current_user.id,
            Observation.aoi_memberships.any(aoi_id)
        )
    )

    if start_time:
        query = query.where(Observation.timestamp >= start_time)

    if end_time:
        query = query.where(Observation.timestamp <= end_time)

    query = query.order_by(Observation.timestamp.desc())
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    observations = result.scalars().all()

    return observations


# ============================================================================
# Enrichment Endpoints
# ============================================================================

@router.post("/enrich/geoip", response_model=GeoIPEnrichmentResponse)
async def enrich_ip_address(
    request: GeoIPEnrichmentRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Enrich an IP address with GeoIP data.

    Useful for manual enrichment or testing.
    """
    geo_enricher = get_geo_enricher()

    if not geo_enricher.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GeoIP enrichment service not available"
        )

    try:
        location = await geo_enricher.geoip_enricher.enrich_ip(request.ip_address)

        return GeoIPEnrichmentResponse(
            ip_address=request.ip_address,
            location=location,
            success=location is not None,
            error=None if location else "No location data found"
        )

    except Exception as e:
        logger.error(f"Failed to enrich IP: {e}")
        return GeoIPEnrichmentResponse(
            ip_address=request.ip_address,
            location=None,
            success=False,
            error=str(e)
        )


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint for SIGINT/GEOINT subsystem."""
    geo_enricher = get_geo_enricher()

    return {
        "status": "healthy",
        "geoip_available": geo_enricher.geoip_enricher.is_available(),
        "aoi_matching_available": geo_enricher.aoi_matcher.is_available(),
        "aoi_count": len(geo_enricher.aoi_matcher.aoi_cache)
    }
