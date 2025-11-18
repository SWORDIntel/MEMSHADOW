"""
Tests for NPU Accelerator Service
Phase 4: Distributed Architecture
"""

import pytest
from app.services.distributed.npu_accelerator import (
    NPUAccelerator,
    AcceleratorType,
    QuantizationType
)


@pytest.mark.asyncio
async def test_npu_accelerator_initialization():
    """Test NPU accelerator initialization"""
    accelerator = NPUAccelerator()

    assert accelerator is not None
    assert accelerator.active_accelerator in [
        AcceleratorType.NPU,
        AcceleratorType.GPU,
        AcceleratorType.CPU
    ]
    assert len(accelerator.available_hardware) > 0
    assert AcceleratorType.CPU in accelerator.available_hardware


@pytest.mark.asyncio
async def test_hardware_detection():
    """Test hardware detection"""
    accelerator = NPUAccelerator()

    # CPU should always be available
    assert AcceleratorType.CPU in accelerator.available_hardware

    # Check detection logic
    available = accelerator.available_hardware
    assert isinstance(available, list)
    assert all(isinstance(hw, AcceleratorType) for hw in available)


@pytest.mark.asyncio
async def test_load_embedding_model():
    """Test loading embedding model"""
    accelerator = NPUAccelerator()

    model_info = await accelerator.load_model(
        model_name="sentence-transformers",
        model_type="embedding",
        quantization=QuantizationType.INT8
    )

    assert model_info["name"] == "sentence-transformers"
    assert model_info["type"] == "embedding"
    assert model_info["quantization"] == QuantizationType.INT8
    assert model_info["accelerator"] == accelerator.active_accelerator
    assert model_info["embedding_dim"] == 384


@pytest.mark.asyncio
async def test_generate_embeddings():
    """Test embedding generation"""
    accelerator = NPUAccelerator()

    texts = [
        "This is a test memory",
        "Another test memory",
        "Memory about neural networks"
    ]

    result = await accelerator.generate_embeddings(
        texts=texts,
        model="sentence-transformers"
    )

    assert "embeddings" in result
    assert len(result["embeddings"]) == len(texts)
    assert result["model"] == "sentence-transformers"
    assert result["accelerator"] == accelerator.active_accelerator
    assert "inference_time_ms" in result

    # Check embedding dimensions
    for embedding in result["embeddings"]:
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # sentence-transformers dimension


@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing of embeddings"""
    accelerator = NPUAccelerator()

    # Large batch
    texts = [f"Memory {i}" for i in range(100)]

    result = await accelerator.generate_embeddings(
        texts=texts,
        model="sentence-transformers",
        batch_size=32
    )

    assert len(result["embeddings"]) == 100
    assert result["inference_time_ms"] > 0


@pytest.mark.asyncio
async def test_quantization_levels():
    """Test different quantization levels"""
    accelerator = NPUAccelerator()

    quantization_types = [
        QuantizationType.NONE,
        QuantizationType.FLOAT16,
        QuantizationType.INT8,
        QuantizationType.INT4
    ]

    for quant_type in quantization_types:
        model_info = await accelerator.load_model(
            model_name="sentence-transformers",
            model_type="embedding",
            quantization=quant_type
        )

        assert model_info["quantization"] == quant_type

        # INT4 should be smallest, NONE should be largest
        if quant_type == QuantizationType.INT4:
            assert model_info["model_size_mb"] < 30
        elif quant_type == QuantizationType.NONE:
            assert model_info["model_size_mb"] > 50


@pytest.mark.asyncio
async def test_performance_stats():
    """Test performance statistics tracking"""
    accelerator = NPUAccelerator()

    # Generate some embeddings
    texts = ["Test memory 1", "Test memory 2"]
    await accelerator.generate_embeddings(texts, model="sentence-transformers")

    stats = await accelerator.get_performance_stats()

    assert stats["total_inferences"] >= 2
    assert stats["total_time_ms"] > 0
    assert stats["avg_inference_time_ms"] > 0
    assert "accelerator_usage" in stats
    assert stats["active_accelerator"] in [
        AcceleratorType.NPU,
        AcceleratorType.GPU,
        AcceleratorType.CPU
    ]


@pytest.mark.asyncio
async def test_device_optimization_recommendations():
    """Test device optimization recommendations"""
    accelerator = NPUAccelerator()

    recommendations = await accelerator.optimize_for_device()

    assert "current_accelerator" in recommendations
    assert "recommendations" in recommendations
    assert isinstance(recommendations["recommendations"], list)

    # Should have recommendations based on active accelerator
    if accelerator.active_accelerator == AcceleratorType.CPU:
        assert any(
            "quantization" in str(rec)
            for rec in recommendations["recommendations"]
        )


@pytest.mark.asyncio
async def test_run_inference_classification():
    """Test classification inference"""
    accelerator = NPUAccelerator()

    result = await accelerator.run_inference(
        inputs="This is a positive statement",
        model="sentiment-classifier",
        task="classification"
    )

    assert "predictions" in result
    assert "confidence" in result
    assert result["model"] == "sentiment-classifier"


@pytest.mark.asyncio
async def test_run_inference_generation():
    """Test text generation inference"""
    accelerator = NPUAccelerator()

    result = await accelerator.run_inference(
        inputs="What is machine learning?",
        model="phi-3-mini",
        task="generation",
        max_tokens=50,
        temperature=0.7
    )

    assert "generated_text" in result
    assert "tokens_generated" in result
    assert result["model"] == "phi-3-mini"
