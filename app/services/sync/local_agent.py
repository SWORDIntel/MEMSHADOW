"""
Local Sync Agent
Phase 4: Distributed Architecture - Local-to-cloud synchronization agent
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import structlog
from enum import Enum

logger = structlog.get_logger()

class SyncStatus(str, Enum):
    """Sync operation status"""
    IDLE = "idle"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    ERROR = "error"
    COMPLETED = "completed"

class SyncDirection(str, Enum):
    """Direction of sync"""
    UPLOAD = "upload"
    DOWNLOAD = "download"
    BIDIRECTIONAL = "bidirectional"

class LocalSyncAgent:
    """
    Local sync agent for hybrid local-cloud architecture.
    
    Features:
    - Differential sync (only changed data)
    - Conflict resolution
    - Offline-first operation
    - Bandwidth optimization
    - L1/L2 local caching
    """
    
    def __init__(
        self,
        local_cache_path: str = "/var/cache/memshadow",
        sync_interval: int = 300,  # 5 minutes
        cloud_endpoint: str = "https://api.memshadow.cloud"
    ):
        self.local_cache_path = Path(local_cache_path)
        self.sync_interval = sync_interval
        self.cloud_endpoint = cloud_endpoint
        
        # Create cache directories
        self.l1_cache = self.local_cache_path / "l1"  # RAM-like (hot data)
        self.l2_cache = self.local_cache_path / "l2"  # SSD (warm data)
        
        self.l1_cache.mkdir(parents=True, exist_ok=True)
        self.l2_cache.mkdir(parents=True, exist_ok=True)
        
        # Sync state
        self.status = SyncStatus.IDLE
        self.last_sync = None
        self.pending_changes = []
        self.conflict_queue = []
        
        logger.info("Local sync agent initialized",
                   l1_path=str(self.l1_cache),
                   l2_path=str(self.l2_cache))
    
    async def start_sync_loop(self):
        """Main sync loop - runs periodically"""
        logger.info("Starting sync loop", interval=self.sync_interval)
        
        while True:
            try:
                await self.sync()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error("Sync loop error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry
    
    async def sync(
        self,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Perform synchronization with cloud.
        
        Args:
            direction: Direction of sync
            force: Force full sync (ignore differential)
        
        Returns:
            Sync statistics
        """
        if self.status == SyncStatus.SYNCING and not force:
            logger.warning("Sync already in progress")
            return {"status": "skipped", "reason": "sync_in_progress"}
        
        self.status = SyncStatus.SYNCING
        start_time = datetime.utcnow()
        
        logger.info("Starting sync", direction=direction, force=force)
        
        try:
            stats = {
                "direction": direction,
                "uploaded": 0,
                "downloaded": 0,
                "conflicts": 0,
                "errors": 0,
                "start_time": start_time.isoformat()
            }
            
            # Get changes since last sync
            if not force and self.last_sync:
                changes = await self._get_differential_changes()
            else:
                changes = await self._get_all_changes()
            
            # Upload changes
            if direction in [SyncDirection.UPLOAD, SyncDirection.BIDIRECTIONAL]:
                upload_result = await self._upload_changes(changes["local"])
                stats["uploaded"] = upload_result["count"]
                stats["conflicts"] += upload_result["conflicts"]
            
            # Download changes
            if direction in [SyncDirection.DOWNLOAD, SyncDirection.BIDIRECTIONAL]:
                download_result = await self._download_changes()
                stats["downloaded"] = download_result["count"]
                stats["conflicts"] += download_result["conflicts"]
            
            # Resolve conflicts
            if stats["conflicts"] > 0:
                await self._resolve_conflicts()
            
            self.last_sync = datetime.utcnow()
            self.status = SyncStatus.COMPLETED
            
            end_time = datetime.utcnow()
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            stats["end_time"] = end_time.isoformat()
            
            logger.info("Sync completed", **stats)
            return stats
            
        except Exception as e:
            self.status = SyncStatus.ERROR
            logger.error("Sync failed", error=str(e), exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def _get_differential_changes(self) -> Dict[str, List[Dict]]:
        """Get only changed items since last sync"""
        logger.debug("Computing differential changes")
        
        # Load sync manifest (tracks what's been synced)
        manifest = await self._load_sync_manifest()
        
        local_changes = []
        
        # Check L1 cache for changes
        for item_file in self.l1_cache.glob("*.json"):
            item_data = json.loads(item_file.read_text())
            item_hash = self._compute_hash(item_data)
            
            # Check if changed
            manifest_hash = manifest.get(str(item_file.name), {}).get("hash")
            if item_hash != manifest_hash:
                local_changes.append({
                    "id": item_data["id"],
                    "type": "memory",
                    "data": item_data,
                    "hash": item_hash,
                    "modified_at": item_data.get("updated_at", datetime.utcnow().isoformat())
                })
        
        return {"local": local_changes}
    
    async def _get_all_changes(self) -> Dict[str, List[Dict]]:
        """Get all items (full sync)"""
        logger.debug("Getting all changes for full sync")
        
        all_changes = []
        
        for item_file in self.l1_cache.glob("*.json"):
            item_data = json.loads(item_file.read_text())
            all_changes.append({
                "id": item_data["id"],
                "type": "memory",
                "data": item_data,
                "hash": self._compute_hash(item_data)
            })
        
        return {"local": all_changes}
    
    async def _upload_changes(self, changes: List[Dict]) -> Dict[str, Any]:
        """Upload local changes to cloud"""
        logger.info("Uploading changes to cloud", count=len(changes))
        
        uploaded = 0
        conflicts = 0
        
        for change in changes:
            try:
                # In production, would make actual API call
                # response = await self._api_call("POST", "/sync/upload", data=change)
                
                # For now, simulate upload
                logger.debug("Uploading item", item_id=change["id"])
                uploaded += 1
                
                # Update manifest
                await self._update_manifest_entry(change["id"], change["hash"])
                
            except Exception as e:
                logger.error("Upload failed for item", item_id=change["id"], error=str(e))
                conflicts += 1
        
        return {"count": uploaded, "conflicts": conflicts}
    
    async def _download_changes(self) -> Dict[str, Any]:
        """Download cloud changes to local"""
        logger.info("Downloading changes from cloud")
        
        # In production:
        # response = await self._api_call("GET", "/sync/changes", 
        #                                 params={"since": self.last_sync})
        # changes = response["changes"]
        
        # Mock for now
        changes = []
        
        downloaded = 0
        conflicts = 0
        
        for change in changes:
            try:
                # Check for conflicts
                local_file = self.l1_cache / f"{change['id']}.json"
                if local_file.exists():
                    local_data = json.loads(local_file.read_text())
                    if local_data.get("updated_at") > change["data"].get("updated_at"):
                        # Local is newer - conflict
                        self.conflict_queue.append({
                            "local": local_data,
                            "remote": change["data"],
                            "type": "update_conflict"
                        })
                        conflicts += 1
                        continue
                
                # Write to L1 cache
                local_file.write_text(json.dumps(change["data"], indent=2))
                downloaded += 1
                
            except Exception as e:
                logger.error("Download failed for item", item_id=change["id"], error=str(e))
        
        return {"count": downloaded, "conflicts": conflicts}
    
    async def _resolve_conflicts(self):
        """Resolve sync conflicts"""
        logger.info("Resolving conflicts", count=len(self.conflict_queue))
        
        for conflict in self.conflict_queue:
            # Default strategy: last-write-wins
            local_time = datetime.fromisoformat(conflict["local"].get("updated_at", "2000-01-01"))
            remote_time = datetime.fromisoformat(conflict["remote"].get("updated_at", "2000-01-01"))
            
            if remote_time > local_time:
                # Remote wins - overwrite local
                item_id = conflict["remote"]["id"]
                local_file = self.l1_cache / f"{item_id}.json"
                local_file.write_text(json.dumps(conflict["remote"], indent=2))
                logger.info("Conflict resolved: remote wins", item_id=item_id)
            else:
                # Local wins - upload to cloud
                logger.info("Conflict resolved: local wins", item_id=conflict["local"]["id"])
                await self._upload_changes([{
                    "id": conflict["local"]["id"],
                    "data": conflict["local"],
                    "hash": self._compute_hash(conflict["local"])
                }])
        
        self.conflict_queue.clear()
    
    def _compute_hash(self, data: Dict) -> str:
        """Compute hash of data for change detection"""
        # Remove timestamp fields that don't affect content
        hashable = {k: v for k, v in data.items() if k not in ["updated_at", "accessed_at"]}
        serialized = json.dumps(hashable, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    async def _load_sync_manifest(self) -> Dict[str, Any]:
        """Load sync manifest (tracks what's been synced)"""
        manifest_file = self.local_cache_path / "sync_manifest.json"
        
        if manifest_file.exists():
            return json.loads(manifest_file.read_text())
        
        return {}
    
    async def _update_manifest_entry(self, item_id: str, item_hash: str):
        """Update manifest for an item"""
        manifest = await self._load_sync_manifest()
        manifest[item_id] = {
            "hash": item_hash,
            "synced_at": datetime.utcnow().isoformat()
        }
        
        manifest_file = self.local_cache_path / "sync_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        l1_files = list(self.l1_cache.glob("*.json"))
        l2_files = list(self.l2_cache.glob("*.json"))
        
        l1_size = sum(f.stat().st_size for f in l1_files)
        l2_size = sum(f.stat().st_size for f in l2_files)
        
        return {
            "l1": {
                "count": len(l1_files),
                "size_bytes": l1_size,
                "size_mb": l1_size / 1024 / 1024
            },
            "l2": {
                "count": len(l2_files),
                "size_bytes": l2_size,
                "size_mb": l2_size / 1024 / 1024
            },
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "status": self.status,
            "pending_conflicts": len(self.conflict_queue)
        }
    
    async def cache_item(self, item_id: str, data: Dict, tier: str = "l1"):
        """Cache an item in L1 or L2"""
        cache_dir = self.l1_cache if tier == "l1" else self.l2_cache
        
        item_file = cache_dir / f"{item_id}.json"
        item_file.write_text(json.dumps(data, indent=2))
        
        logger.debug("Item cached", item_id=item_id, tier=tier)
    
    async def get_cached_item(self, item_id: str) -> Optional[Dict]:
        """Retrieve cached item (check L1 then L2)"""
        # Check L1 first (hot)
        l1_file = self.l1_cache / f"{item_id}.json"
        if l1_file.exists():
            logger.debug("Cache hit L1", item_id=item_id)
            return json.loads(l1_file.read_text())
        
        # Check L2 (warm)
        l2_file = self.l2_cache / f"{item_id}.json"
        if l2_file.exists():
            logger.debug("Cache hit L2", item_id=item_id)
            data = json.loads(l2_file.read_text())
            
            # Promote to L1
            await self.cache_item(item_id, data, tier="l1")
            
            return data
        
        logger.debug("Cache miss", item_id=item_id)
        return None


# Import for async
import asyncio

# Global instance
local_sync_agent = LocalSyncAgent()
