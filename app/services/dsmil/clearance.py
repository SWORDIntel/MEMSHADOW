"""
DSMILSYSTEM Clearance Token System

Handles clearance token validation and Rules of Engagement (ROE) enforcement:
- Token validation
- Upward-only flow enforcement
- ROE rule evaluation
- Access denial logging
"""
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import structlog

logger = structlog.get_logger()


class ClearanceLevel(str, Enum):
    """Clearance levels"""
    UNCLASSIFIED = "UNCLASSIFIED"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"
    TOP_SECRET = "TOP_SECRET"


class AccessDecision(str, Enum):
    """Access decision result"""
    ALLOWED = "ALLOWED"
    DENIED_CLEARANCE = "DENIED_CLEARANCE"
    DENIED_UPWARD_FLOW = "DENIED_UPWARD_FLOW"
    DENIED_ROE = "DENIED_ROE"


class ClearanceValidator:
    """
    Validates clearance tokens and enforces access rules.
    
    Rules:
    1. Clearance levels: UNCLASSIFIED < CONFIDENTIAL < SECRET < TOP_SECRET
    2. Upward-only flows: Lower layers cannot read higher-layer memory
    3. ROE rules: Additional restrictions based on Rules of Engagement
    """
    
    CLEARANCE_LEVELS = {
        ClearanceLevel.UNCLASSIFIED: 0,
        ClearanceLevel.CONFIDENTIAL: 1,
        ClearanceLevel.SECRET: 2,
        ClearanceLevel.TOP_SECRET: 3,
    }
    
    def __init__(self):
        self.roe_rules: Dict[str, Dict[str, Any]] = {}
    
    def parse_clearance_token(self, token: str) -> Tuple[ClearanceLevel, Optional[Dict[str, Any]]]:
        """
        Parse clearance token.
        
        Format: "LEVEL[:metadata]"
        Example: "SECRET:compartment=SCI"
        
        Returns:
            (clearance_level, metadata_dict)
        """
        parts = token.split(":", 1)
        level_str = parts[0].upper()
        
        try:
            level = ClearanceLevel(level_str)
        except ValueError:
            logger.warning("Invalid clearance level, defaulting to UNCLASSIFIED", token=token)
            level = ClearanceLevel.UNCLASSIFIED
        
        metadata = {}
        if len(parts) > 1:
            # Parse metadata (key=value pairs)
            for pair in parts[1].split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    metadata[key.strip()] = value.strip()
        
        return level, metadata
    
    def check_clearance_level(
        self,
        required_level: ClearanceLevel,
        token_level: ClearanceLevel
    ) -> bool:
        """
        Check if token has sufficient clearance level.
        
        Args:
            required_level: Required clearance level
            token_level: Token clearance level
            
        Returns:
            True if token has sufficient clearance
        """
        required_rank = self.CLEARANCE_LEVELS.get(required_level, 0)
        token_rank = self.CLEARANCE_LEVELS.get(token_level, 0)
        
        return token_rank >= required_rank
    
    def check_upward_only_flow(
        self,
        source_layer: int,
        target_layer: int,
        operation: str = "read"
    ) -> Tuple[bool, Optional[str]]:
        """
        Enforce upward-only flow rule.
        
        Lower layers (higher numbers) cannot read from higher layers (lower numbers).
        Higher layers can read from lower layers if clearance is sufficient.
        
        Args:
            source_layer: Layer of the memory being accessed
            target_layer: Layer of the requester
            operation: Operation type ("read", "write", "delete")
            
        Returns:
            (allowed, reason_if_denied)
        """
        # Upward-only: target_layer >= source_layer for reads
        # Writes can go anywhere (subject to clearance)
        if operation == "read":
            if target_layer < source_layer:
                return False, f"Upward-only flow violation: Layer {target_layer} cannot read from Layer {source_layer}"
            return True, None
        else:
            # Writes/deletes allowed in both directions
            return True, None
    
    def check_roe_rules(
        self,
        token: str,
        layer_id: int,
        device_id: int,
        operation: str,
        memory_roe: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check Rules of Engagement (ROE) rules.
        
        Args:
            token: Clearance token
            layer_id: Layer ID
            device_id: Device ID
            operation: Operation type
            memory_roe: ROE metadata from memory
            
        Returns:
            (allowed, reason_if_denied)
        """
        # Check memory-specific ROE rules
        if memory_roe:
            # Example: Check if device is allowed
            allowed_devices = memory_roe.get("allowed_devices", [])
            if allowed_devices and device_id not in allowed_devices:
                return False, f"Device {device_id} not in allowed devices list"
            
            # Example: Check time-based restrictions
            time_restrictions = memory_roe.get("time_restrictions", {})
            if time_restrictions:
                # Implement time-based checks if needed
                pass
        
        # Check token-specific ROE rules
        level, metadata = self.parse_clearance_token(token)
        
        # Example: Check compartment access
        required_compartment = memory_roe.get("compartment") if memory_roe else None
        if required_compartment:
            token_compartment = metadata.get("compartment")
            if token_compartment != required_compartment:
                return False, f"Compartment mismatch: required {required_compartment}, token has {token_compartment}"
        
        return True, None
    
    def validate_access(
        self,
        token: str,
        source_layer: int,
        target_layer: int,
        source_clearance: str,
        source_roe: Optional[Dict[str, Any]],
        operation: str = "read"
    ) -> Tuple[AccessDecision, Optional[str]]:
        """
        Validate access request.
        
        Args:
            token: Requester's clearance token
            source_layer: Layer of the memory being accessed
            target_layer: Layer of the requester
            source_clearance: Clearance level of the memory
            source_roe: ROE metadata of the memory
            operation: Operation type
            
        Returns:
            (decision, reason_if_denied)
        """
        # Parse tokens
        token_level, token_metadata = self.parse_clearance_token(token)
        source_level, _ = self.parse_clearance_token(source_clearance)
        
        # Check clearance level
        if not self.check_clearance_level(source_level, token_level):
            return (
                AccessDecision.DENIED_CLEARANCE,
                f"Insufficient clearance: required {source_level.value}, token has {token_level.value}"
            )
        
        # Check upward-only flow
        allowed, reason = self.check_upward_only_flow(source_layer, target_layer, operation)
        if not allowed:
            return AccessDecision.DENIED_UPWARD_FLOW, reason
        
        # Check ROE rules
        allowed, reason = self.check_roe_rules(
            token,
            source_layer,
            0,  # Device ID not available in this context
            operation,
            source_roe
        )
        if not allowed:
            return AccessDecision.DENIED_ROE, reason
        
        return AccessDecision.ALLOWED, None


# Global validator instance
clearance_validator = ClearanceValidator()
