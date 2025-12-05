"""
DSMILSYSTEM Event Bus Integration

Handles event emission and consumption using Redis Streams:
- Per-layer streams (layer:{layer_id}:in, layer:{layer_id}:out)
- Per-device streams (device:{device_id}:events)
- Event schemas with correlation IDs
- Stream consumers for processing
"""
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

from app.db.redis import redis_client

logger = structlog.get_logger()


class EventBus:
    """
    Event bus for DSMILSYSTEM memory operations.
    
    Uses Redis Streams for event routing:
    - layer:{layer_id}:in - Incoming events for a layer
    - layer:{layer_id}:out - Outgoing events from a layer
    - device:{device_id}:events - Device-specific events
    """
    
    def __init__(self):
        self.redis = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = await redis_client.get_client()
        logger.info("Event bus initialized")
    
    async def emit_memory_event(
        self,
        event_type: str,
        layer_id: int,
        device_id: int,
        memory_id: str,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Emit a memory event to appropriate streams.
        
        Args:
            event_type: Type of event (store, search, delete, etc.)
            layer_id: Layer ID (2-9)
            device_id: Device ID (0-103)
            memory_id: Memory ID
            correlation_id: Correlation ID for event tracking
            metadata: Additional event metadata
            
        Returns:
            Event ID
        """
        if not self.redis:
            await self.initialize()
        
        correlation_id = correlation_id or str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        event_data = {
            "event_type": event_type,
            "memory_id": memory_id,
            "layer_id": layer_id,
            "device_id": device_id,
            "correlation_id": correlation_id,
            "timestamp": timestamp,
            "metadata": json.dumps(metadata or {})
        }
        
        # Emit to layer streams
        layer_in_stream = f"layer:{layer_id}:in"
        layer_out_stream = f"layer:{layer_id}:out"
        
        # Emit to device stream
        device_stream = f"device:{device_id}:events"
        
        event_id = None
        
        try:
            # Add to layer IN stream
            event_id = await self.redis.xadd(
                layer_in_stream,
                event_data,
                maxlen=10000  # Keep last 10k events
            )
            
            # Add to layer OUT stream
            await self.redis.xadd(
                layer_out_stream,
                event_data,
                maxlen=10000
            )
            
            # Add to device stream
            await self.redis.xadd(
                device_stream,
                event_data,
                maxlen=10000
            )
            
            logger.info(
                "Memory event emitted",
                event_type=event_type,
                layer_id=layer_id,
                device_id=device_id,
                memory_id=memory_id,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to emit memory event",
                error=str(e),
                event_type=event_type,
                layer_id=layer_id,
                device_id=device_id
            )
            raise
        
        return event_id
    
    async def read_layer_events(
        self,
        layer_id: int,
        stream: str = "in",
        count: int = 10,
        last_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Read events from a layer stream.
        
        Args:
            layer_id: Layer ID
            stream: Stream type ("in" or "out")
            count: Number of events to read
            last_id: Last event ID (for pagination)
            
        Returns:
            List of events
        """
        if not self.redis:
            await self.initialize()
        
        stream_name = f"layer:{layer_id}:{stream}"
        
        if last_id:
            events = await self.redis.xread({stream_name: last_id}, count=count)
        else:
            events = await self.redis.xread({stream_name: "0"}, count=count)
        
        result = []
        for stream, messages in events:
            for msg_id, data in messages:
                event = {
                    "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                    "data": {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in data.items()
                    }
                }
                # Parse metadata JSON
                if "metadata" in event["data"]:
                    try:
                        event["data"]["metadata"] = json.loads(event["data"]["metadata"])
                    except:
                        pass
                result.append(event)
        
        return result
    
    async def read_device_events(
        self,
        device_id: int,
        count: int = 10,
        last_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Read events from a device stream.
        
        Args:
            device_id: Device ID
            count: Number of events to read
            last_id: Last event ID (for pagination)
            
        Returns:
            List of events
        """
        if not self.redis:
            await self.initialize()
        
        stream_name = f"device:{device_id}:events"
        
        if last_id:
            events = await self.redis.xread({stream_name: last_id}, count=count)
        else:
            events = await self.redis.xread({stream_name: "0"}, count=count)
        
        result = []
        for stream, messages in events:
            for msg_id, data in messages:
                event = {
                    "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                    "data": {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in data.items()
                    }
                }
                # Parse metadata JSON
                if "metadata" in event["data"]:
                    try:
                        event["data"]["metadata"] = json.loads(event["data"]["metadata"])
                    except:
                        pass
                result.append(event)
        
        return result


# Global event bus instance
event_bus = EventBus()
