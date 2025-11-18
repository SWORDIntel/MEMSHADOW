"""
Predictive Retrieval Service
Phase 3: Intelligence Layer - Predict and preload relevant memories
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

class PredictiveRetrievalService:
    """
    Service for predicting user's next information needs.
    Uses usage patterns, temporal patterns, and context to preload relevant memories.
    """
    
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
        self.access_history = []  # (user_id, memory_id, timestamp, context)
        self.user_patterns = defaultdict(lambda: {
            "common_sequences": [],
            "time_patterns": {},
            "topic_clusters": []
        })
    
    async def record_access(
        self,
        user_id: str,
        memory_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record memory access for pattern learning.
        
        Args:
            user_id: User ID
            memory_id: Accessed memory ID
            context: Optional context (query, time, location, etc.)
        """
        access_record = {
            "user_id": user_id,
            "memory_id": memory_id,
            "timestamp": datetime.utcnow(),
            "context": context or {}
        }
        
        self.access_history.append(access_record)
        
        # Update user patterns
        await self._update_patterns(user_id, memory_id, context)
        
        logger.debug("Access recorded", user_id=user_id, memory_id=memory_id)
    
    async def predict_next_memories(
        self,
        user_id: str,
        current_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict which memories the user is likely to need next.
        
        Args:
            user_id: User ID
            current_context: Current context (recent queries, time, etc.)
            top_k: Number of predictions to return
        
        Returns:
            List of predicted memory IDs with confidence scores
        """
        logger.info("Predicting next memories", user_id=user_id)
        
        predictions = []
        
        # Strategy 1: Sequence-based prediction
        sequence_preds = await self._predict_from_sequences(user_id, top_k)
        predictions.extend(sequence_preds)
        
        # Strategy 2: Temporal pattern prediction
        temporal_preds = await self._predict_from_temporal_patterns(user_id, top_k)
        predictions.extend(temporal_preds)
        
        # Strategy 3: Context-based prediction
        if current_context:
            context_preds = await self._predict_from_context(user_id, current_context, top_k)
            predictions.extend(context_preds)
        
        # Aggregate and rank predictions
        prediction_scores = defaultdict(float)
        for pred in predictions:
            prediction_scores[pred["memory_id"]] += pred["confidence"]
        
        # Sort by score and return top k
        ranked = sorted(
            [
                {"memory_id": mid, "confidence": score}
                for mid, score in prediction_scores.items()
            ],
            key=lambda x: x["confidence"],
            reverse=True
        )
        
        return ranked[:top_k]
    
    async def _update_patterns(
        self,
        user_id: str,
        memory_id: str,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Update user access patterns"""
        patterns = self.user_patterns[user_id]
        
        # Update sequence patterns (what follows what)
        recent_accesses = [
            a for a in self.access_history
            if a["user_id"] == user_id
            and (datetime.utcnow() - a["timestamp"]).total_seconds() < 3600
        ]
        
        if len(recent_accesses) >= 2:
            # Record sequences
            for i in range(len(recent_accesses) - 1):
                sequence = (recent_accesses[i]["memory_id"], recent_accesses[i+1]["memory_id"])
                patterns["common_sequences"].append(sequence)
        
        # Update temporal patterns (time of day, day of week)
        now = datetime.utcnow()
        hour = now.hour
        day_of_week = now.weekday()
        
        time_key = f"{day_of_week}_{hour}"
        if time_key not in patterns["time_patterns"]:
            patterns["time_patterns"][time_key] = []
        patterns["time_patterns"][time_key].append(memory_id)
    
    async def _predict_from_sequences(
        self,
        user_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Predict based on common access sequences"""
        patterns = self.user_patterns[user_id]
        
        if not self.access_history:
            return []
        
        # Get most recent memory accessed by user
        user_accesses = [a for a in self.access_history if a["user_id"] == user_id]
        if not user_accesses:
            return []
        
        last_memory = user_accesses[-1]["memory_id"]
        
        # Find what typically follows this memory
        sequences = patterns["common_sequences"]
        following_memories = Counter([
            seq[1] for seq in sequences if seq[0] == last_memory
        ])
        
        predictions = []
        for memory_id, count in following_memories.most_common(top_k):
            confidence = count / len(sequences) if sequences else 0
            predictions.append({
                "memory_id": memory_id,
                "confidence": confidence,
                "method": "sequence"
            })
        
        return predictions
    
    async def _predict_from_temporal_patterns(
        self,
        user_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Predict based on time-of-day and day-of-week patterns"""
        patterns = self.user_patterns[user_id]
        
        now = datetime.utcnow()
        hour = now.hour
        day_of_week = now.weekday()
        time_key = f"{day_of_week}_{hour}"
        
        temporal_memories = patterns["time_patterns"].get(time_key, [])
        if not temporal_memories:
            return []
        
        # Count frequency
        memory_counts = Counter(temporal_memories)
        total = len(temporal_memories)
        
        predictions = []
        for memory_id, count in memory_counts.most_common(top_k):
            predictions.append({
                "memory_id": memory_id,
                "confidence": count / total,
                "method": "temporal"
            })
        
        return predictions
    
    async def _predict_from_context(
        self,
        user_id: str,
        context: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Predict based on current context (query, topic, etc.)"""
        # Context-based prediction would use:
        # - Recent query terms
        # - Current topic/project
        # - Location/environment
        # - Connected memories from knowledge graph
        
        query = context.get("query", "")
        if not query:
            return []
        
        # Simple approach: find memories accessed with similar queries
        similar_accesses = []
        for access in self.access_history:
            if access["user_id"] == user_id:
                past_query = access["context"].get("query", "")
                if past_query and self._query_similarity(query, past_query) > 0.5:
                    similar_accesses.append(access["memory_id"])
        
        if not similar_accesses:
            return []
        
        memory_counts = Counter(similar_accesses)
        total = len(similar_accesses)
        
        predictions = []
        for memory_id, count in memory_counts.most_common(top_k):
            predictions.append({
                "memory_id": memory_id,
                "confidence": count / total,
                "method": "context"
            })
        
        return predictions
    
    def _query_similarity(self, query1: str, query2: str) -> float:
        """Simple query similarity based on word overlap"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    async def get_usage_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        user_accesses = [a for a in self.access_history if a["user_id"] == user_id]
        
        if not user_accesses:
            return {
                "total_accesses": 0,
                "unique_memories": 0,
                "patterns": {}
            }
        
        memory_ids = [a["memory_id"] for a in user_accesses]
        memory_counts = Counter(memory_ids)
        
        # Time distribution
        hour_distribution = Counter([a["timestamp"].hour for a in user_accesses])
        
        return {
            "total_accesses": len(user_accesses),
            "unique_memories": len(set(memory_ids)),
            "most_accessed": memory_counts.most_common(5),
            "hour_distribution": dict(hour_distribution),
            "patterns": {
                "sequences": len(self.user_patterns[user_id]["common_sequences"]),
                "time_patterns": len(self.user_patterns[user_id]["time_patterns"])
            }
        }
    
    async def preload_cache(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Preload memories into cache based on predictions.
        
        Returns list of memory IDs that should be cached.
        """
        predictions = await self.predict_next_memories(user_id, context, top_k=10)
        
        # Filter by confidence threshold
        high_confidence = [
            p["memory_id"]
            for p in predictions
            if p["confidence"] > 0.3
        ]
        
        logger.info("Preload recommendations generated", 
                   user_id=user_id, count=len(high_confidence))
        
        return high_confidence


# Global instance
predictive_retrieval = PredictiveRetrievalService()
