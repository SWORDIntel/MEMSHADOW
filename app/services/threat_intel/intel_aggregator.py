"""
Threat Intelligence Aggregation System
Integrates multiple threat intelligence feeds
Classification: UNCLASSIFIED

Supported Sources:
- MISP (Malware Information Sharing Platform)
- OpenCTI (Open Cyber Threat Intelligence)
- AlienVault OTX
- AbuseIPDB
- VirusTotal
- Shodan
- URLhaus
- ThreatFox
"""

import asyncio
import aiohttp
import structlog
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

logger = structlog.get_logger()


class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IOCType(Enum):
    """Types of Indicators of Compromise"""
    IP_ADDRESS = "ipv4-addr"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH_MD5 = "md5"
    FILE_HASH_SHA1 = "sha1"
    FILE_HASH_SHA256 = "sha256"
    EMAIL = "email"
    CRYPTOCURRENCY = "cryptocurrency"
    CVE = "cve"
    YARA_RULE = "yara"


@dataclass
class ThreatIndicator:
    """Threat intelligence indicator"""
    ioc_type: IOCType
    value: str
    threat_level: ThreatLevel
    source: str
    first_seen: datetime
    last_seen: datetime
    confidence: int  # 0-100
    tags: List[str]
    description: str
    related_campaigns: List[str]
    kill_chain_phases: List[str]
    ttps: List[str]  # MITRE ATT&CK TTPs
    raw_data: Dict


class ThreatIntelAggregator:
    """Aggregates threat intelligence from multiple sources"""

    def __init__(self, config: Dict):
        self.config = config
        self.indicators: Dict[str, ThreatIndicator] = {}
        self.session: Optional[aiohttp.ClientSession] = None

        # API keys
        self.misp_url = config.get('misp_url')
        self.misp_key = config.get('misp_key')
        self.opencti_url = config.get('opencti_url')
        self.opencti_key = config.get('opencti_key')
        self.otx_key = config.get('otx_key')
        self.vt_key = config.get('vt_key')
        self.shodan_key = config.get('shodan_key')
        self.abuseipdb_key = config.get('abuseipdb_key')

        # Caching
        self.cache_ttl = timedelta(hours=6)
        self.last_update: Dict[str, datetime] = {}

    async def initialize(self):
        """Initialize HTTP session and connections"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'MEMSHADOW-ThreatIntel/2.1'}
        )
        logger.info("Threat intelligence aggregator initialized")

    async def close(self):
        """Close connections"""
        if self.session:
            await self.session.close()

    # ========================================================================
    # MISP Integration
    # ========================================================================

    async def fetch_misp_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from MISP"""
        if not self.misp_url or not self.misp_key:
            logger.warning("MISP not configured")
            return []

        try:
            url = f"{self.misp_url}/attributes/restSearch"
            headers = {
                'Authorization': self.misp_key,
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }

            # Search for recent indicators (last 7 days)
            payload = {
                'returnFormat': 'json',
                'last': '7d',
                'to_ids': True,  # Only indicators marked for detection
                'published': True,
                'enforceWarninglist': True,
            }

            async with self.session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    indicators = self._parse_misp_response(data)
                    logger.info("MISP indicators fetched", count=len(indicators))
                    return indicators
                else:
                    logger.error("MISP fetch failed", status=resp.status)
                    return []

        except Exception as e:
            logger.error("MISP integration error", error=str(e))
            return []

    def _parse_misp_response(self, data: Dict) -> List[ThreatIndicator]:
        """Parse MISP API response"""
        indicators = []

        for attr in data.get('response', {}).get('Attribute', []):
            try:
                ioc_type = self._map_misp_type(attr.get('type'))
                if not ioc_type:
                    continue

                threat_level = self._map_misp_threat_level(
                    attr.get('Event', {}).get('threat_level_id', 3)
                )

                indicator = ThreatIndicator(
                    ioc_type=ioc_type,
                    value=attr.get('value'),
                    threat_level=threat_level,
                    source='MISP',
                    first_seen=datetime.fromtimestamp(int(attr.get('timestamp', 0))),
                    last_seen=datetime.utcnow(),
                    confidence=int(attr.get('distribution', 0)) * 20,  # Rough mapping
                    tags=[tag.get('name', '') for tag in attr.get('Tag', [])],
                    description=attr.get('comment', ''),
                    related_campaigns=[],
                    kill_chain_phases=[],
                    ttps=[],
                    raw_data=attr
                )
                indicators.append(indicator)

            except Exception as e:
                logger.error("MISP indicator parse error", error=str(e))
                continue

        return indicators

    def _map_misp_type(self, misp_type: str) -> Optional[IOCType]:
        """Map MISP attribute type to IOC type"""
        mapping = {
            'ip-dst': IOCType.IP_ADDRESS,
            'ip-src': IOCType.IP_ADDRESS,
            'domain': IOCType.DOMAIN,
            'hostname': IOCType.DOMAIN,
            'url': IOCType.URL,
            'md5': IOCType.FILE_HASH_MD5,
            'sha1': IOCType.FILE_HASH_SHA1,
            'sha256': IOCType.FILE_HASH_SHA256,
            'email-src': IOCType.EMAIL,
            'email-dst': IOCType.EMAIL,
            'btc': IOCType.CRYPTOCURRENCY,
        }
        return mapping.get(misp_type)

    def _map_misp_threat_level(self, level_id: int) -> ThreatLevel:
        """Map MISP threat level to our enum"""
        mapping = {
            1: ThreatLevel.CRITICAL,
            2: ThreatLevel.HIGH,
            3: ThreatLevel.MEDIUM,
            4: ThreatLevel.LOW,
        }
        return mapping.get(level_id, ThreatLevel.MEDIUM)

    # ========================================================================
    # OpenCTI Integration
    # ========================================================================

    async def fetch_opencti_indicators(self) -> List[ThreatIndicator]:
        """Fetch indicators from OpenCTI"""
        if not self.opencti_url or not self.opencti_key:
            logger.warning("OpenCTI not configured")
            return []

        try:
            url = f"{self.opencti_url}/graphql"
            headers = {
                'Authorization': f'Bearer {self.opencti_key}',
                'Content-Type': 'application/json'
            }

            # GraphQL query for indicators
            query = """
            query GetIndicators {
              indicators(first: 1000, orderBy: created_at, orderMode: desc) {
                edges {
                  node {
                    id
                    pattern
                    pattern_type
                    name
                    description
                    valid_from
                    valid_until
                    x_opencti_score
                    labels {
                      edges {
                        node {
                          value
                        }
                      }
                    }
                    killChainPhases {
                      edges {
                        node {
                          kill_chain_name
                          phase_name
                        }
                      }
                    }
                  }
                }
              }
            }
            """

            payload = {'query': query}

            async with self.session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    indicators = self._parse_opencti_response(data)
                    logger.info("OpenCTI indicators fetched", count=len(indicators))
                    return indicators
                else:
                    logger.error("OpenCTI fetch failed", status=resp.status)
                    return []

        except Exception as e:
            logger.error("OpenCTI integration error", error=str(e))
            return []

    def _parse_opencti_response(self, data: Dict) -> List[ThreatIndicator]:
        """Parse OpenCTI GraphQL response"""
        indicators = []

        for edge in data.get('data', {}).get('indicators', {}).get('edges', []):
            try:
                node = edge.get('node', {})

                # Parse STIX pattern
                pattern = node.get('pattern', '')
                ioc_value, ioc_type = self._parse_stix_pattern(pattern)

                if not ioc_type or not ioc_value:
                    continue

                # Map OpenCTI score to threat level
                score = node.get('x_opencti_score', 50)
                threat_level = self._score_to_threat_level(score)

                # Extract tags
                tags = [
                    edge['node']['value']
                    for edge in node.get('labels', {}).get('edges', [])
                ]

                # Extract kill chain phases
                kill_chain = [
                    f"{edge['node']['kill_chain_name']}:{edge['node']['phase_name']}"
                    for edge in node.get('killChainPhases', {}).get('edges', [])
                ]

                indicator = ThreatIndicator(
                    ioc_type=ioc_type,
                    value=ioc_value,
                    threat_level=threat_level,
                    source='OpenCTI',
                    first_seen=datetime.fromisoformat(node.get('valid_from', datetime.utcnow().isoformat())),
                    last_seen=datetime.utcnow(),
                    confidence=score,
                    tags=tags,
                    description=node.get('description', ''),
                    related_campaigns=[],
                    kill_chain_phases=kill_chain,
                    ttps=[],
                    raw_data=node
                )
                indicators.append(indicator)

            except Exception as e:
                logger.error("OpenCTI indicator parse error", error=str(e))
                continue

        return indicators

    def _parse_stix_pattern(self, pattern: str) -> tuple[Optional[str], Optional[IOCType]]:
        """Parse STIX 2.x pattern to extract IOC value and type"""
        # Example: "[ipv4-addr:value = '192.0.2.0']"
        try:
            if 'ipv4-addr:value' in pattern:
                value = pattern.split("'")[1]
                return value, IOCType.IP_ADDRESS
            elif 'domain-name:value' in pattern:
                value = pattern.split("'")[1]
                return value, IOCType.DOMAIN
            elif 'url:value' in pattern:
                value = pattern.split("'")[1]
                return value, IOCType.URL
            elif 'file:hashes.MD5' in pattern:
                value = pattern.split("'")[1]
                return value, IOCType.FILE_HASH_MD5
            elif 'file:hashes.SHA-256' in pattern:
                value = pattern.split("'")[1]
                return value, IOCType.FILE_HASH_SHA256
        except:
            pass

        return None, None

    def _score_to_threat_level(self, score: int) -> ThreatLevel:
        """Convert OpenCTI score (0-100) to threat level"""
        if score >= 80:
            return ThreatLevel.CRITICAL
        elif score >= 60:
            return ThreatLevel.HIGH
        elif score >= 40:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    # ========================================================================
    # AbuseIPDB Integration
    # ========================================================================

    async def check_ip_reputation(self, ip: str) -> Optional[ThreatIndicator]:
        """Check IP reputation via AbuseIPDB"""
        if not self.abuseipdb_key:
            return None

        try:
            url = 'https://api.abuseipdb.com/api/v2/check'
            headers = {
                'Accept': 'application/json',
                'Key': self.abuseipdb_key
            }
            params = {
                'ipAddress': ip,
                'maxAgeInDays': 90,
                'verbose': True
            }

            async with self.session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_abuseipdb_response(data)
                else:
                    logger.error("AbuseIPDB check failed", status=resp.status, ip=ip)
                    return None

        except Exception as e:
            logger.error("AbuseIPDB error", error=str(e), ip=ip)
            return None

    def _parse_abuseipdb_response(self, data: Dict) -> Optional[ThreatIndicator]:
        """Parse AbuseIPDB response"""
        try:
            abuse_data = data.get('data', {})
            abuse_score = abuse_data.get('abuseConfidenceScore', 0)

            if abuse_score < 25:  # Low abuse score, not significant
                return None

            # Map abuse score to threat level
            if abuse_score >= 75:
                threat_level = ThreatLevel.CRITICAL
            elif abuse_score >= 50:
                threat_level = ThreatLevel.HIGH
            elif abuse_score >= 25:
                threat_level = ThreatLevel.MEDIUM
            else:
                threat_level = ThreatLevel.LOW

            indicator = ThreatIndicator(
                ioc_type=IOCType.IP_ADDRESS,
                value=abuse_data.get('ipAddress'),
                threat_level=threat_level,
                source='AbuseIPDB',
                first_seen=datetime.utcnow() - timedelta(days=90),
                last_seen=datetime.utcnow(),
                confidence=abuse_score,
                tags=['abuse', 'scanning'] if abuse_data.get('isWhitelisted') is False else [],
                description=f"Abuse score: {abuse_score}. Reports: {abuse_data.get('totalReports', 0)}",
                related_campaigns=[],
                kill_chain_phases=[],
                ttps=[],
                raw_data=abuse_data
            )
            return indicator

        except Exception as e:
            logger.error("AbuseIPDB parse error", error=str(e))
            return None

    # ========================================================================
    # Aggregation and Correlation
    # ========================================================================

    async def update_all_feeds(self):
        """Update all threat intelligence feeds"""
        logger.info("Updating threat intelligence feeds...")

        tasks = []

        if self.misp_url:
            tasks.append(self.fetch_misp_indicators())

        if self.opencti_url:
            tasks.append(self.fetch_opencti_indicators())

        # Fetch all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate indicators
        all_indicators = []
        for result in results:
            if isinstance(result, list):
                all_indicators.extend(result)
            elif isinstance(result, Exception):
                logger.error("Feed update error", error=str(result))

        # Deduplicate and store
        self._store_indicators(all_indicators)

        logger.info("Threat intelligence updated",
                   total_indicators=len(self.indicators))

    def _store_indicators(self, indicators: List[ThreatIndicator]):
        """Store indicators with deduplication"""
        for indicator in indicators:
            # Create unique key
            key = f"{indicator.ioc_type.value}:{indicator.value}"

            # If exists, merge (keep highest confidence)
            if key in self.indicators:
                existing = self.indicators[key]
                if indicator.confidence > existing.confidence:
                    self.indicators[key] = indicator
                # Merge tags
                existing.tags = list(set(existing.tags + indicator.tags))
            else:
                self.indicators[key] = indicator

    def check_ioc(self, ioc_type: IOCType, value: str) -> Optional[ThreatIndicator]:
        """Check if IOC is known threat"""
        key = f"{ioc_type.value}:{value}"
        return self.indicators.get(key)

    def get_indicators_by_level(self, level: ThreatLevel) -> List[ThreatIndicator]:
        """Get all indicators of specific threat level"""
        return [
            ind for ind in self.indicators.values()
            if ind.threat_level == level
        ]

    def export_for_blocking(self, min_confidence: int = 70) -> Dict[str, List[str]]:
        """Export high-confidence indicators for automated blocking"""
        blocking_lists = {
            'ips': [],
            'domains': [],
            'urls': [],
            'hashes': []
        }

        for indicator in self.indicators.values():
            if indicator.confidence < min_confidence:
                continue

            if indicator.ioc_type == IOCType.IP_ADDRESS:
                blocking_lists['ips'].append(indicator.value)
            elif indicator.ioc_type == IOCType.DOMAIN:
                blocking_lists['domains'].append(indicator.value)
            elif indicator.ioc_type == IOCType.URL:
                blocking_lists['urls'].append(indicator.value)
            elif indicator.ioc_type in [IOCType.FILE_HASH_MD5, IOCType.FILE_HASH_SHA256]:
                blocking_lists['hashes'].append(indicator.value)

        return blocking_lists
