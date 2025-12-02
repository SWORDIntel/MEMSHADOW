"""
MEMSHADOW Phase-0 GeoIP Enricher Service

Enriches observations with geographic data using GeoIP databases
and matches them against Areas of Interest using point-in-polygon checks.

Dependencies:
    pip install geoip2 shapely
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import ipaddress

try:
    import geoip2.database
    import geoip2.errors
    GEOIP2_AVAILABLE = True
except ImportError:
    GEOIP2_AVAILABLE = False
    logging.warning("geoip2 not installed. GeoIP enrichment will be disabled.")

try:
    from shapely.geometry import Point, shape
    from shapely.prepared import prep
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logging.warning("shapely not installed. AOI matching will be disabled.")

from app.core.config import settings
from app.schemas.sigint_geoint import (
    GeographicLocation,
    LocationRole,
    GeoJSONGeometry
)


logger = logging.getLogger(__name__)


class GeoIPEnricher:
    """
    Enriches IP addresses with geographic data using MaxMind GeoLite2 databases.

    Downloads:
        GeoLite2-City: https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
        GeoLite2-ASN: https://dev.maxmind.com/geoip/geolite2-free-geolocation-data

    Place the .mmdb files in /var/lib/memshadow/geoip/ or configure via env vars.
    """

    def __init__(
        self,
        city_db_path: Optional[str] = None,
        asn_db_path: Optional[str] = None
    ):
        """
        Initialize GeoIP enricher with database paths.

        Args:
            city_db_path: Path to GeoLite2-City.mmdb
            asn_db_path: Path to GeoLite2-ASN.mmdb
        """
        self.city_db_path = city_db_path or getattr(
            settings, "GEOIP_CITY_DB_PATH", "/var/lib/memshadow/geoip/GeoLite2-City.mmdb"
        )
        self.asn_db_path = asn_db_path or getattr(
            settings, "GEOIP_ASN_DB_PATH", "/var/lib/memshadow/geoip/GeoLite2-ASN.mmdb"
        )

        self.city_reader: Optional[Any] = None
        self.asn_reader: Optional[Any] = None

        if GEOIP2_AVAILABLE:
            self._load_databases()
        else:
            logger.warning("GeoIP2 library not available. Install with: pip install geoip2")

    def _load_databases(self):
        """Load GeoIP databases if they exist."""
        if Path(self.city_db_path).exists():
            try:
                self.city_reader = geoip2.database.Reader(self.city_db_path)
                logger.info(f"Loaded GeoIP City database: {self.city_db_path}")
            except Exception as e:
                logger.error(f"Failed to load GeoIP City database: {e}")
        else:
            logger.warning(f"GeoIP City database not found: {self.city_db_path}")

        if Path(self.asn_db_path).exists():
            try:
                self.asn_reader = geoip2.database.Reader(self.asn_db_path)
                logger.info(f"Loaded GeoIP ASN database: {self.asn_db_path}")
            except Exception as e:
                logger.error(f"Failed to load GeoIP ASN database: {e}")
        else:
            logger.warning(f"GeoIP ASN database not found: {self.asn_db_path}")

    def is_available(self) -> bool:
        """Check if GeoIP enrichment is available."""
        return GEOIP2_AVAILABLE and (self.city_reader is not None or self.asn_reader is not None)

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private (RFC1918, etc.)"""
        try:
            addr = ipaddress.ip_address(ip)
            return addr.is_private or addr.is_loopback or addr.is_link_local
        except ValueError:
            return False

    async def enrich_ip(
        self,
        ip_address: str,
        role: LocationRole = LocationRole.REMOTE_GEOIP
    ) -> Optional[GeographicLocation]:
        """
        Enrich an IP address with GeoIP data.

        Args:
            ip_address: IP address to enrich
            role: Location role (default: REMOTE_GEOIP)

        Returns:
            GeographicLocation object or None if enrichment fails
        """
        if not self.is_available():
            logger.debug("GeoIP enrichment not available")
            return None

        # Skip private IPs
        if self._is_private_ip(ip_address):
            logger.debug(f"Skipping private IP: {ip_address}")
            return None

        try:
            # Get city/location data
            city_data = None
            if self.city_reader:
                try:
                    city_data = self.city_reader.city(ip_address)
                except geoip2.errors.AddressNotFoundError:
                    logger.debug(f"No city data for IP: {ip_address}")

            # Get ASN data
            asn_data = None
            if self.asn_reader:
                try:
                    asn_data = self.asn_reader.asn(ip_address)
                except geoip2.errors.AddressNotFoundError:
                    logger.debug(f"No ASN data for IP: {ip_address}")

            # Build GeographicLocation
            if city_data and city_data.location.latitude and city_data.location.longitude:
                return GeographicLocation(
                    lat=city_data.location.latitude,
                    lon=city_data.location.longitude,
                    accuracy_m=city_data.location.accuracy_radius * 1000 if city_data.location.accuracy_radius else 50000,
                    crs="EPSG:4326",
                    role=role,
                    city=city_data.city.name if city_data.city.name else None,
                    country=city_data.country.iso_code if city_data.country.iso_code else None,
                    asn=asn_data.autonomous_system_number if asn_data else None,
                    as_org=asn_data.autonomous_system_organization if asn_data else None
                )
            elif asn_data:
                # Have ASN but no location - return with ASN only
                logger.debug(f"No location data for IP {ip_address}, but have ASN")
                return None

            return None

        except Exception as e:
            logger.error(f"Error enriching IP {ip_address}: {e}")
            return None

    async def enrich_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich an observation with GeoIP data based on source/destination IPs.

        Args:
            observation: Observation dict (matching ObservationBase schema)

        Returns:
            Updated observation dict with location field populated
        """
        if not self.is_available():
            return observation

        payload = observation.get("payload", {})

        # Try to extract remote IP from various payload fields
        remote_ip = payload.get("src_ip") or payload.get("remote_ip") or payload.get("client_ip")

        if remote_ip:
            location = await self.enrich_ip(remote_ip, role=LocationRole.REMOTE_GEOIP)
            if location:
                observation["location"] = location.model_dump()
                logger.debug(f"Enriched observation with location: {location.city}, {location.country}")

        return observation

    def close(self):
        """Close GeoIP database readers."""
        if self.city_reader:
            self.city_reader.close()
        if self.asn_reader:
            self.asn_reader.close()


class AOIMatcher:
    """
    Matches observations against Areas of Interest using point-in-polygon checks.

    Uses Shapely for geometric operations.
    """

    def __init__(self):
        """Initialize AOI matcher."""
        if not SHAPELY_AVAILABLE:
            logger.warning("Shapely not available. AOI matching will be disabled.")

        # Cache of prepared geometries for fast point-in-polygon checks
        # Format: {aoi_node_id: (prepared_geometry, aoi_metadata)}
        self.aoi_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}

    def is_available(self) -> bool:
        """Check if AOI matching is available."""
        return SHAPELY_AVAILABLE

    def load_aoi(self, aoi_node_id: str, geometry: GeoJSONGeometry, metadata: Optional[Dict[str, Any]] = None):
        """
        Load an AOI into the matcher's cache.

        Args:
            aoi_node_id: Node ID of the AOI (e.g., "aoi:beirut")
            geometry: GeoJSON geometry
            metadata: Optional AOI metadata (name, category, etc.)
        """
        if not SHAPELY_AVAILABLE:
            return

        try:
            # Convert GeoJSON to Shapely geometry
            geom_dict = {
                "type": geometry.type,
                "coordinates": geometry.coordinates
            }
            shapely_geom = shape(geom_dict)

            # Prepare geometry for fast point-in-polygon checks
            prepared_geom = prep(shapely_geom)

            self.aoi_cache[aoi_node_id] = (prepared_geom, metadata or {})
            logger.info(f"Loaded AOI: {aoi_node_id} ({metadata.get('name', 'unnamed')})")

        except Exception as e:
            logger.error(f"Failed to load AOI {aoi_node_id}: {e}")

    def load_aois_from_db(self, aois: List[Dict[str, Any]]):
        """
        Load multiple AOIs from database results.

        Args:
            aois: List of AOI dicts with node_id, geometry, name, etc.
        """
        for aoi in aois:
            try:
                geom = GeoJSONGeometry(**aoi["geometry"])
                metadata = {
                    "name": aoi.get("name"),
                    "category": aoi.get("category"),
                    "labels": aoi.get("labels", {})
                }
                self.load_aoi(aoi["node_id"], geom, metadata)
            except Exception as e:
                logger.error(f"Failed to load AOI {aoi.get('node_id')}: {e}")

    def match_point(self, lat: float, lon: float) -> List[str]:
        """
        Match a geographic point against all loaded AOIs.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            List of AOI node IDs that contain this point
        """
        if not SHAPELY_AVAILABLE:
            return []

        point = Point(lon, lat)  # Shapely uses (x, y) = (lon, lat)
        matches = []

        for aoi_node_id, (prepared_geom, metadata) in self.aoi_cache.items():
            try:
                if prepared_geom.contains(point):
                    matches.append(aoi_node_id)
                    logger.debug(f"Point ({lat}, {lon}) matched AOI: {metadata.get('name', aoi_node_id)}")
            except Exception as e:
                logger.error(f"Error checking point against AOI {aoi_node_id}: {e}")

        return matches

    async def enrich_observation_with_aois(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich an observation with AOI memberships based on its location.

        Args:
            observation: Observation dict with location field

        Returns:
            Updated observation with aoi_memberships populated
        """
        if not SHAPELY_AVAILABLE:
            return observation

        location = observation.get("location")
        if not location:
            return observation

        lat = location.get("lat")
        lon = location.get("lon")
        if lat is None or lon is None:
            return observation

        # Match against AOIs
        aoi_matches = self.match_point(lat, lon)

        # Update observation
        if aoi_matches:
            observation["aoi_memberships"] = aoi_matches
            logger.debug(f"Observation matched {len(aoi_matches)} AOIs")

        return observation


class GeoEnricherService:
    """
    Combined service for GeoIP enrichment and AOI matching.

    This is the main service used by the observation ingestion pipeline.
    """

    def __init__(
        self,
        city_db_path: Optional[str] = None,
        asn_db_path: Optional[str] = None
    ):
        """Initialize the geo enricher service."""
        self.geoip_enricher = GeoIPEnricher(city_db_path, asn_db_path)
        self.aoi_matcher = AOIMatcher()
        logger.info("GeoEnricherService initialized")

    def is_available(self) -> bool:
        """Check if any enrichment is available."""
        return self.geoip_enricher.is_available() or self.aoi_matcher.is_available()

    async def enrich(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fully enrich an observation with GeoIP and AOI data.

        Args:
            observation: Observation dict

        Returns:
            Enriched observation dict
        """
        # Step 1: GeoIP enrichment
        observation = await self.geoip_enricher.enrich_observation(observation)

        # Step 2: AOI matching
        observation = await self.aoi_matcher.enrich_observation_with_aois(observation)

        return observation

    async def load_aois(self, aois: List[Dict[str, Any]]):
        """Load AOIs into the matcher cache."""
        self.aoi_matcher.load_aois_from_db(aois)

    def close(self):
        """Clean up resources."""
        self.geoip_enricher.close()


# Singleton instance
_geo_enricher_service: Optional[GeoEnricherService] = None


def get_geo_enricher() -> GeoEnricherService:
    """Get or create the singleton GeoEnricher service."""
    global _geo_enricher_service
    if _geo_enricher_service is None:
        _geo_enricher_service = GeoEnricherService()
    return _geo_enricher_service
