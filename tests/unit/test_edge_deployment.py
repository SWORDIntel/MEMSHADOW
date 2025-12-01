"""
Tests for Edge Deployment Service
Phase 4: Distributed Architecture
"""

import pytest
from app.services.distributed.edge_deployment import (
    EdgeDeploymentService,
    EdgeProfile,
    ComputeMode
)


@pytest.mark.asyncio
async def test_edge_service_initialization():
    """Test edge deployment service initialization"""
    service = EdgeDeploymentService()

    assert service is not None
    assert service.deployment_path.exists()
    assert service.cache_path.exists()
    assert len(service.profile_configs) == 4


@pytest.mark.asyncio
async def test_profile_configurations():
    """Test predefined profile configurations"""
    service = EdgeDeploymentService()

    # Minimal profile
    minimal = service.profile_configs[EdgeProfile.MINIMAL]
    assert minimal.profile == EdgeProfile.MINIMAL
    assert minimal.compute_mode == ComputeMode.INFERENCE_ONLY
    assert minimal.max_memory_mb == 256
    assert minimal.enable_local_embeddings is False
    assert minimal.enable_local_llm is False

    # Performance profile
    perf = service.profile_configs[EdgeProfile.PERFORMANCE]
    assert perf.profile == EdgeProfile.PERFORMANCE
    assert perf.compute_mode == ComputeMode.FULL_LOCAL
    assert perf.max_memory_mb == 8192
    assert perf.enable_local_embeddings is True
    assert perf.enable_local_llm is True


@pytest.mark.asyncio
async def test_detect_device_capabilities():
    """Test device capability detection"""
    service = EdgeDeploymentService()

    capabilities = await service.detect_device_capabilities()

    assert "ram_mb" in capabilities
    assert "cpu_cores" in capabilities
    assert "storage_mb" in capabilities
    assert "has_gpu" in capabilities
    assert "has_npu" in capabilities
    assert "network_type" in capabilities
    assert "detected_at" in capabilities


@pytest.mark.asyncio
async def test_configure_for_minimal_device():
    """Test configuration for minimal device"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=256,
        storage_mb=512,
        has_npu=False,
        has_gpu=False
    )

    assert config.profile == EdgeProfile.MINIMAL
    assert config.compute_mode == ComputeMode.INFERENCE_ONLY
    assert config.enable_local_embeddings is False
    assert config.enable_local_llm is False


@pytest.mark.asyncio
async def test_configure_for_light_device():
    """Test configuration for light device"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=1024,
        storage_mb=8192,
        has_npu=False
    )

    assert config.profile == EdgeProfile.LIGHT
    assert config.max_memory_mb == 1024
    assert config.enable_local_embeddings is True


@pytest.mark.asyncio
async def test_configure_for_standard_device():
    """Test configuration for standard device"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=3000,
        storage_mb=32768,
        has_npu=False
    )

    assert config.profile == EdgeProfile.STANDARD
    assert config.enable_local_llm is True
    assert config.enable_knowledge_graph is True


@pytest.mark.asyncio
async def test_configure_for_performance_device():
    """Test configuration for performance device"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=8192,
        storage_mb=102400,
        has_npu=True,
        has_gpu=True
    )

    assert config.profile == EdgeProfile.PERFORMANCE
    assert config.compute_mode == ComputeMode.FULL_LOCAL
    assert config.enable_local_embeddings is True
    assert config.enable_local_llm is True


@pytest.mark.asyncio
async def test_configure_for_battery_powered():
    """Test configuration adjustment for battery-powered devices"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=2048,
        storage_mb=16384,
        battery_powered=True
    )

    # Should have more conservative settings
    base_interval = service.profile_configs[EdgeProfile.STANDARD].sync_interval_seconds
    assert config.sync_interval_seconds >= base_interval


@pytest.mark.asyncio
async def test_configure_with_hardware_acceleration():
    """Test configuration with NPU/GPU"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=2048,
        storage_mb=16384,
        has_npu=True
    )

    # Should enable local processing with hardware acceleration
    assert config.enable_local_embeddings is True


@pytest.mark.asyncio
async def test_configure_with_limited_storage():
    """Test configuration with limited storage"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=2048,
        storage_mb=512  # Very limited
    )

    # Should have reduced cache and disabled features
    assert config.max_cache_size_mb <= 100
    assert config.enable_knowledge_graph is False


@pytest.mark.asyncio
async def test_deploy():
    """Test deployment process"""
    service = EdgeDeploymentService()

    config = await service.configure_for_device(
        ram_mb=2048,
        storage_mb=16384
    )

    result = await service.deploy(config)

    assert result["status"] == "success"
    assert result["profile"] == config.profile
    assert "duration_seconds" in result
    assert "steps" in result
    assert len(result["steps"]) == 6  # 6 deployment steps


@pytest.mark.asyncio
async def test_deploy_minimal_profile():
    """Test deployment with minimal profile"""
    service = EdgeDeploymentService()

    config = service.profile_configs[EdgeProfile.MINIMAL]
    result = await service.deploy(config)

    assert result["status"] == "success"
    assert result["profile"] == EdgeProfile.MINIMAL

    # Check deployment steps
    steps = result["steps"]
    assert any(s["step"] == "create_directory_structure" for s in steps)
    assert any(s["step"] == "deploy_core_services" for s in steps)
    assert any(s["step"] == "configure_sync" for s in steps)


@pytest.mark.asyncio
async def test_deploy_with_optional_services():
    """Test deployment with optional services enabled"""
    service = EdgeDeploymentService()

    config = service.profile_configs[EdgeProfile.STANDARD]
    result = await service.deploy(config)

    assert result["status"] == "success"

    # Find optional services step
    optional_step = next(
        s for s in result["steps"]
        if s["step"] == "deploy_optional_services"
    )

    services = optional_step["services_deployed"]

    # Standard profile should have multiple services
    assert "embedding_service" in services
    assert "llm_service" in services
    assert "knowledge_graph_service" in services


@pytest.mark.asyncio
async def test_get_deployment_status():
    """Test getting deployment status"""
    service = EdgeDeploymentService()

    status = await service.get_deployment_status()

    assert "deployed" in status
    assert isinstance(status["deployed"], bool)


@pytest.mark.asyncio
async def test_get_resource_usage():
    """Test getting resource usage"""
    service = EdgeDeploymentService()

    usage = await service.get_resource_usage()

    assert "memory_mb" in usage
    assert "memory_percent" in usage
    assert "cpu_percent" in usage
    assert "cache_size_mb" in usage
    assert "measured_at" in usage


@pytest.mark.asyncio
async def test_update_deployment():
    """Test updating existing deployment"""
    service = EdgeDeploymentService()

    # Create initial deployment
    config1 = service.profile_configs[EdgeProfile.LIGHT]
    await service.deploy(config1)

    # Update to higher profile
    config2 = service.profile_configs[EdgeProfile.STANDARD]
    result = await service.update_deployment(config2)

    assert result["status"] == "success"
    assert result["profile"] == EdgeProfile.STANDARD


@pytest.mark.asyncio
async def test_quantization_configuration():
    """Test quantization level configuration"""
    service = EdgeDeploymentService()

    # Minimal should use INT4
    minimal = service.profile_configs[EdgeProfile.MINIMAL]
    assert minimal.quantization_level == "int4"

    # Performance can use no quantization
    perf = service.profile_configs[EdgeProfile.PERFORMANCE]
    assert perf.quantization_level == "none"

    # Light/Standard should use INT8
    light = service.profile_configs[EdgeProfile.LIGHT]
    assert light.quantization_level == "int8"
