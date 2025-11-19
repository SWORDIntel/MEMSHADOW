"""
IoC (Indicator of Compromise) Identifier

Extracts and classifies security-relevant indicators from text and network data:
- IP addresses (with reputation scoring)
- Domains and URLs
- File hashes (MD5, SHA1, SHA256)
- Email addresses
- CVE identifiers
- Cryptocurrency addresses
- YARA rule matching
"""

import re
import hashlib
from typing import Dict, List, Any, Set
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger()


@dataclass
class IoC:
    """Indicator of Compromise data class"""
    type: str
    value: str
    context: str = ""
    confidence: float = 1.0
    threat_level: str = "unknown"  # low, medium, high, critical
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class IoC_Identifier:
    """
    Advanced IoC identification and classification system
    """

    # Regex patterns for various IoC types
    PATTERNS = {
        'ipv4': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        'ipv6': r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
        'domain': r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b',
        'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'md5': r'\b[a-fA-F0-9]{32}\b',
        'sha1': r'\b[a-fA-F0-9]{40}\b',
        'sha256': r'\b[a-fA-F0-9]{64}\b',
        'cve': r'CVE-\d{4}-\d{4,7}',
        'bitcoin': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
        'ethereum': r'\b0x[a-fA-F0-9]{40}\b',
        'registry_key': r'HKEY_(?:LOCAL_MACHINE|CURRENT_USER|CLASSES_ROOT|USERS|CURRENT_CONFIG)\\[^\s]+',
        'file_path_win': r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
        'file_path_unix': r'/(?:[^/\0]+/)*[^/\0]+',
        'mutex': r'(?:Global|Local)\\[A-Za-z0-9_\-]+',
        'user_agent': r'User-Agent: .+',
        'asn': r'AS\d{1,10}\b'
    }

    # Known malicious patterns (simplified - in production, use threat intel feeds)
    MALICIOUS_DOMAINS = {
        'evil.com', 'badguy.net', 'malware.xyz'
    }

    # Private IP ranges (RFC 1918)
    PRIVATE_IP_RANGES = [
        ('10.0.0.0', '10.255.255.255'),
        ('172.16.0.0', '172.31.255.255'),
        ('192.168.0.0', '192.168.255.255'),
        ('127.0.0.0', '127.255.255.255')
    ]

    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE if name != 'file_path_unix' else 0)
            for name, pattern in self.PATTERNS.items()
        }

        logger.info("IoC Identifier initialized")

    def extract_iocs(self, text: str) -> List[IoC]:
        """
        Extract all IoCs from text

        Args:
            text: Input text to analyze

        Returns:
            List of IoC objects
        """
        iocs = []

        for ioc_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)

            for match in matches:
                # Extract context (surrounding text)
                match_pos = text.find(match)
                context_start = max(0, match_pos - 50)
                context_end = min(len(text), match_pos + len(match) + 50)
                context = text[context_start:context_end]

                ioc = IoC(
                    type=ioc_type,
                    value=match,
                    context=context
                )

                # Classify and enrich
                self._classify_ioc(ioc)

                iocs.append(ioc)

        # Deduplicate
        seen = set()
        unique_iocs = []
        for ioc in iocs:
            key = (ioc.type, ioc.value)
            if key not in seen:
                seen.add(key)
                unique_iocs.append(ioc)

        logger.info(f"Extracted {len(unique_iocs)} unique IoCs from text", types=self._count_by_type(unique_iocs))

        return unique_iocs

    def _classify_ioc(self, ioc: IoC):
        """
        Classify and enrich an IoC with threat intelligence

        Args:
            ioc: IoC object to classify (modified in place)
        """
        if ioc.type in ['ipv4', 'ipv6']:
            self._classify_ip(ioc)
        elif ioc.type == 'domain':
            self._classify_domain(ioc)
        elif ioc.type in ['md5', 'sha1', 'sha256']:
            self._classify_hash(ioc)
        elif ioc.type == 'cve':
            self._classify_cve(ioc)
        elif ioc.type in ['bitcoin', 'ethereum']:
            ioc.threat_level = 'medium'  # Crypto addresses in security context are suspicious
        elif ioc.type in ['registry_key', 'mutex']:
            ioc.threat_level = 'medium'  # Persistence mechanisms

    def _classify_ip(self, ioc: IoC):
        """Classify IP address"""
        ip = ioc.value

        # Check if private
        if self._is_private_ip(ip):
            ioc.metadata['scope'] = 'private'
            ioc.threat_level = 'low'
        else:
            ioc.metadata['scope'] = 'public'
            ioc.threat_level = 'medium'

            # In production, query threat intel APIs (VirusTotal, AbuseIPDB, etc.)
            # For now, simple heuristic
            if any(keyword in ioc.context.lower() for keyword in ['malware', 'attack', 'exploit', 'c2', 'command']):
                ioc.threat_level = 'high'
                ioc.confidence = 0.7

    def _classify_domain(self, ioc: IoC):
        """Classify domain"""
        domain = ioc.value.lower()

        # Check against known malicious domains
        if domain in self.MALICIOUS_DOMAINS:
            ioc.threat_level = 'critical'
            ioc.confidence = 1.0
            ioc.metadata['status'] = 'known_malicious'
        # Check for suspicious TLDs
        elif any(domain.endswith(tld) for tld in ['.xyz', '.tk', '.ml', '.ga', '.cf']):
            ioc.threat_level = 'medium'
            ioc.confidence = 0.5
            ioc.metadata['note'] = 'Suspicious TLD'
        # Check for DGA-like patterns (long random strings)
        elif len(domain.split('.')[0]) > 20:
            ioc.threat_level = 'medium'
            ioc.confidence = 0.6
            ioc.metadata['note'] = 'Possible DGA domain'
        else:
            ioc.threat_level = 'low'

    def _classify_hash(self, ioc: IoC):
        """Classify file hash"""
        # In production, query VirusTotal, MalwareBazaar, etc.
        # For now, mark as unknown
        ioc.threat_level = 'unknown'
        ioc.metadata['hash_type'] = ioc.type.upper()

        # Check context for malware indicators
        if any(keyword in ioc.context.lower() for keyword in ['malware', 'virus', 'trojan', 'ransomware']):
            ioc.threat_level = 'high'
            ioc.confidence = 0.7

    def _classify_cve(self, ioc: IoC):
        """Classify CVE identifier"""
        # Extract CVE year
        cve_parts = ioc.value.split('-')
        if len(cve_parts) >= 2:
            year = cve_parts[1]
            ioc.metadata['year'] = year

        # In production, query NVD or CVE databases for CVSS score
        # For now, mark as medium by default
        ioc.threat_level = 'medium'
        ioc.metadata['note'] = 'Query NVD for details'

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private range"""
        try:
            octets = [int(x) for x in ip.split('.')]
            ip_int = (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]

            for start_str, end_str in self.PRIVATE_IP_RANGES:
                start_octets = [int(x) for x in start_str.split('.')]
                end_octets = [int(x) for x in end_str.split('.')]

                start_int = (start_octets[0] << 24) + (start_octets[1] << 16) + (start_octets[2] << 8) + start_octets[3]
                end_int = (end_octets[0] << 24) + (end_octets[1] << 16) + (end_octets[2] << 8) + end_octets[3]

                if start_int <= ip_int <= end_int:
                    return True

            return False

        except:
            return False

    def _count_by_type(self, iocs: List[IoC]) -> Dict[str, int]:
        """Count IoCs by type"""
        counts = {}
        for ioc in iocs:
            counts[ioc.type] = counts.get(ioc.type, 0) + 1
        return counts

    def generate_report(self, iocs: List[IoC]) -> Dict[str, Any]:
        """
        Generate a comprehensive IoC report

        Args:
            iocs: List of IoCs

        Returns:
            Report dictionary
        """
        report = {
            'total_iocs': len(iocs),
            'by_type': self._count_by_type(iocs),
            'by_threat_level': {
                'critical': len([i for i in iocs if i.threat_level == 'critical']),
                'high': len([i for i in iocs if i.threat_level == 'high']),
                'medium': len([i for i in iocs if i.threat_level == 'medium']),
                'low': len([i for i in iocs if i.threat_level == 'low']),
                'unknown': len([i for i in iocs if i.threat_level == 'unknown'])
            },
            'high_confidence_critical': [
                asdict(ioc) for ioc in iocs
                if ioc.threat_level == 'critical' and ioc.confidence >= 0.7
            ],
            'all_iocs': [asdict(ioc) for ioc in iocs]
        }

        return report


# Global instance
ioc_identifier = IoC_Identifier()
