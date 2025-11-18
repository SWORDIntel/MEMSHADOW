import pytest
from uuid import uuid4
from app.services.enrichment.predictive_retrieval import PredictiveRetrievalService

@pytest.mark.asyncio
class TestPredictiveRetrieval:
    """Test predictive retrieval service"""
    
    async def test_record_access(self):
        """Test recording memory access"""
        service = PredictiveRetrievalService()
        
        user_id = str(uuid4())
        memory_id = str(uuid4())
        
        await service.record_access(user_id, memory_id, {"query": "test"})
        
        assert len(service.access_history) == 1
        assert service.access_history[0]["user_id"] == user_id
    
    async def test_predict_from_sequences(self):
        """Test sequence-based prediction"""
        service = PredictiveRetrievalService()
        
        user_id = str(uuid4())
        mem1 = str(uuid4())
        mem2 = str(uuid4())
        
        # Record a sequence
        await service.record_access(user_id, mem1)
        await service.record_access(user_id, mem2)
        await service.record_access(user_id, mem1)
        await service.record_access(user_id, mem2)
        
        # Predict next
        predictions = await service.predict_next_memories(user_id, top_k=5)
        
        assert isinstance(predictions, list)
    
    async def test_usage_statistics(self):
        """Test usage statistics"""
        service = PredictiveRetrievalService()
        
        user_id = str(uuid4())
        mem1 = str(uuid4())
        
        await service.record_access(user_id, mem1)
        await service.record_access(user_id, mem1)
        
        stats = await service.get_usage_statistics(user_id)
        
        assert stats["total_accesses"] == 2
        assert stats["unique_memories"] == 1
    
    async def test_preload_cache(self):
        """Test cache preloading recommendations"""
        service = PredictiveRetrievalService()
        
        user_id = str(uuid4())
        
        # Record some accesses
        for _ in range(5):
            await service.record_access(user_id, str(uuid4()))
        
        preload_list = await service.preload_cache(user_id)
        
        assert isinstance(preload_list, list)
