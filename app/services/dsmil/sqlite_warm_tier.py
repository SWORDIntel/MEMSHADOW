"""
DSMILSYSTEM SQLite Warm Tier

Manages warm-tier memory storage using tmpfs-mounted SQLite databases:
- Per-layer SQLite instances
- Fast local access for frequently accessed memories
- Automatic promotion/demotion logic
- Sync mechanism with PostgreSQL cold tier
"""
import sqlite3
import json
import os
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
from uuid import UUID
import structlog

logger = structlog.get_logger()


class SQLiteWarmTier:
    """
    SQLite-based warm tier for frequently accessed memories.
    
    Uses tmpfs-mounted SQLite databases for fast local access.
    Each layer has its own SQLite database.
    """
    
    def __init__(self, base_path: str = "/tmp/memshadow_warm"):
        """
        Initialize warm tier.
        
        Args:
            base_path: Base path for SQLite databases (should be tmpfs-mounted)
        """
        self.base_path = base_path
        self.connections: Dict[int, sqlite3.Connection] = {}
        os.makedirs(base_path, exist_ok=True)
        self._initialize_schemas()
    
    def _get_db_path(self, layer_id: int) -> str:
        """Get SQLite database path for a layer"""
        return os.path.join(self.base_path, f"layer_{layer_id}.db")
    
    def _get_connection(self, layer_id: int) -> sqlite3.Connection:
        """Get or create SQLite connection for a layer"""
        if layer_id not in self.connections:
            db_path = self._get_db_path(layer_id)
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self.connections[layer_id] = conn
        return self.connections[layer_id]
    
    def _initialize_schemas(self):
        """Initialize SQLite schemas for all layers"""
        for layer_id in range(2, 10):  # Layers 2-9
            conn = self._get_connection(layer_id)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    device_id INTEGER NOT NULL,
                    clearance_token TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding BLOB,
                    tags TEXT NOT NULL,
                    extra_data TEXT NOT NULL,
                    roe_metadata TEXT NOT NULL,
                    correlation_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed_at TEXT
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_device_id ON memories(device_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_clearance_token ON memories(clearance_token)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at ON memories(accessed_at)
            """)
            
            conn.commit()
        
        logger.info("SQLite warm tier schemas initialized", layers=list(range(2, 10)))
    
    async def store(
        self,
        layer_id: int,
        memory_id: str,
        device_id: int,
        clearance_token: str,
        user_id: str,
        content: str,
        content_hash: str,
        embedding: Optional[bytes],
        tags: List[str],
        extra_data: Dict[str, Any],
        roe_metadata: Dict[str, Any],
        correlation_id: Optional[str]
    ) -> bool:
        """
        Store memory in warm tier.
        
        Args:
            layer_id: Layer ID (2-9)
            memory_id: Memory ID
            device_id: Device ID (0-103)
            clearance_token: Clearance token
            user_id: User ID
            content: Memory content
            content_hash: Content hash
            embedding: Embedding vector (bytes)
            tags: List of tags
            extra_data: Extra metadata
            roe_metadata: ROE metadata
            correlation_id: Correlation ID
            
        Returns:
            True if stored successfully
        """
        try:
            conn = self._get_connection(layer_id)
            cursor = conn.cursor()
            
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO memories (
                    id, device_id, clearance_token, user_id, content, content_hash,
                    embedding, tags, extra_data, roe_metadata, correlation_id,
                    created_at, updated_at, accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                memory_id,
                device_id,
                clearance_token,
                user_id,
                content,
                content_hash,
                embedding,
                json.dumps(tags),
                json.dumps(extra_data),
                json.dumps(roe_metadata),
                correlation_id,
                now,
                now,
                now
            ))
            
            conn.commit()
            
            logger.debug(
                "Memory stored in warm tier",
                layer_id=layer_id,
                memory_id=memory_id,
                device_id=device_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to store memory in warm tier",
                error=str(e),
                layer_id=layer_id,
                memory_id=memory_id
            )
            return False
    
    async def retrieve(
        self,
        layer_id: int,
        memory_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve memory from warm tier.
        
        Args:
            layer_id: Layer ID
            memory_id: Memory ID
            
        Returns:
            Memory dict or None if not found
        """
        try:
            conn = self._get_connection(layer_id)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM memories WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Update access tracking
            now = datetime.utcnow().isoformat()
            cursor.execute("""
                UPDATE memories
                SET accessed_at = ?, access_count = access_count + 1, last_accessed_at = ?
                WHERE id = ?
            """, (now, now, memory_id))
            conn.commit()
            
            return {
                "id": row["id"],
                "layer_id": layer_id,
                "device_id": row["device_id"],
                "clearance_token": row["clearance_token"],
                "user_id": row["user_id"],
                "content": row["content"],
                "content_hash": row["content_hash"],
                "embedding": row["embedding"],
                "tags": json.loads(row["tags"]),
                "extra_data": json.loads(row["extra_data"]),
                "roe_metadata": json.loads(row["roe_metadata"]),
                "correlation_id": row["correlation_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "accessed_at": row["accessed_at"],
                "access_count": row["access_count"]
            }
            
        except Exception as e:
            logger.error(
                "Failed to retrieve memory from warm tier",
                error=str(e),
                layer_id=layer_id,
                memory_id=memory_id
            )
            return None
    
    async def search(
        self,
        layer_id: int,
        device_id: Optional[int] = None,
        clearance_token: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories in warm tier.
        
        Args:
            layer_id: Layer ID
            device_id: Filter by device ID (optional)
            clearance_token: Filter by clearance token (optional)
            limit: Maximum results
            
        Returns:
            List of memory dicts
        """
        try:
            conn = self._get_connection(layer_id)
            cursor = conn.cursor()
            
            query = "SELECT * FROM memories WHERE 1=1"
            params = []
            
            if device_id is not None:
                query += " AND device_id = ?"
                params.append(device_id)
            
            if clearance_token:
                query += " AND clearance_token = ?"
                params.append(clearance_token)
            
            query += " ORDER BY accessed_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "layer_id": layer_id,
                    "device_id": row["device_id"],
                    "clearance_token": row["clearance_token"],
                    "user_id": row["user_id"],
                    "content": row["content"],
                    "content_hash": row["content_hash"],
                    "embedding": row["embedding"],
                    "tags": json.loads(row["tags"]),
                    "extra_data": json.loads(row["extra_data"]),
                    "roe_metadata": json.loads(row["roe_metadata"]),
                    "correlation_id": row["correlation_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "accessed_at": row["accessed_at"],
                    "access_count": row["access_count"]
                })
            
            return results
            
        except Exception as e:
            logger.error(
                "Failed to search warm tier",
                error=str(e),
                layer_id=layer_id
            )
            return []
    
    async def delete(
        self,
        layer_id: int,
        memory_id: str
    ) -> bool:
        """
        Delete memory from warm tier.
        
        Args:
            layer_id: Layer ID
            memory_id: Memory ID
            
        Returns:
            True if deleted successfully
        """
        try:
            conn = self._get_connection(layer_id)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            
            logger.debug(
                "Memory deleted from warm tier",
                layer_id=layer_id,
                memory_id=memory_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete memory from warm tier",
                error=str(e),
                layer_id=layer_id,
                memory_id=memory_id
            )
            return False
    
    async def promote_from_hot(
        self,
        layer_id: int,
        memory_data: Dict[str, Any]
    ) -> bool:
        """
        Promote memory from hot tier (Redis) to warm tier.
        
        Args:
            layer_id: Layer ID
            memory_data: Memory data from hot tier
            
        Returns:
            True if promoted successfully
        """
        return await self.store(
            layer_id=layer_id,
            memory_id=memory_data["id"],
            device_id=memory_data["device_id"],
            clearance_token=memory_data["clearance_token"],
            user_id=memory_data["user_id"],
            content=memory_data["content"],
            content_hash=memory_data["content_hash"],
            embedding=memory_data.get("embedding"),
            tags=memory_data.get("tags", []),
            extra_data=memory_data.get("extra_data", {}),
            roe_metadata=memory_data.get("roe_metadata", {}),
            correlation_id=memory_data.get("correlation_id")
        )
    
    async def demote_to_cold(
        self,
        layer_id: int,
        memory_id: str
    ) -> Dict[str, Any]:
        """
        Demote memory from warm tier to cold tier (PostgreSQL).
        Returns memory data for storage in PostgreSQL.
        
        Args:
            layer_id: Layer ID
            memory_id: Memory ID
            
        Returns:
            Memory data dict
        """
        memory = await self.retrieve(layer_id, memory_id)
        if memory:
            await self.delete(layer_id, memory_id)
        return memory or {}
    
    def close(self):
        """Close all database connections"""
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()


# Global warm tier instance
warm_tier = SQLiteWarmTier()
