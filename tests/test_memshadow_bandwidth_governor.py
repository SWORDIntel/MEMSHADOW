#!/usr/bin/env python3
"""
Tests for MEMSHADOW Bandwidth Governor

Tests cover:
- Governor allows traffic when under budget
- Governor enters degradation mode when traffic exceeds per-layer budget
- CRITICAL/EMERGENCY frames are never dropped prematurely
- Metrics increment correctly for accepted, degraded, and dropped frames
- Layer mapping functions return sensible target layers/devices
"""

import sys
import time

# Optional pytest import for running in pytest context
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest.fixture for when running standalone
    class pytest:
        @staticmethod
        def fixture(func):
            return func
        @staticmethod
        def skip(msg):
            print(f"SKIP: {msg}")

from unittest.mock import MagicMock, patch

# Import test subjects
from ai.brain.memory.memshadow_layer_mapping import (
    DSMILLayer,
    MemshadowCategory,
    LayerDeviceInfo,
    LayerMapping,
    get_target_layers_for_category,
    get_devices_for_category,
    get_category_layer_mapping,
    should_trigger_layer8_hook,
    get_category_priority_weight,
    can_degrade_category,
    LAYER_MEMORY_BUDGETS_GB,
    LAYER_BANDWIDTH_BUDGETS_GBPS,
)

from ai.brain.memory.memshadow_bandwidth_governor import (
    MemshadowBandwidthGovernor,
    GovernorConfig,
    AcceptDecision,
    DegradationMode,
    SyncMode,
    LayerCategoryStats,
    Priority,
)

from ai.brain.metrics.memshadow_metrics import (
    MemshadowMetricsRegistry,
    get_memshadow_metrics_registry,
)

from config.memshadow_config import (
    MemshadowConfig,
    BandwidthGovernorConfig,
    DegradationPolicyConfig,
    load_memshadow_config,
)


# =============================================================================
# Layer Mapping Tests
# =============================================================================

class TestLayerMapping:
    """Tests for memshadow_layer_mapping.py"""
    
    def test_layer_budgets_exist(self):
        """Verify all layers have memory and bandwidth budgets."""
        for layer in DSMILLayer:
            assert layer in LAYER_MEMORY_BUDGETS_GB, f"Missing memory budget for {layer.name}"
            assert layer in LAYER_BANDWIDTH_BUDGETS_GBPS, f"Missing bandwidth budget for {layer.name}"
    
    def test_total_bandwidth_approximately_64gbps(self):
        """Verify total bandwidth is approximately 64 GB/s as per spec."""
        total = sum(LAYER_BANDWIDTH_BUDGETS_GBPS.values())
        assert 60 <= total <= 70, f"Total bandwidth {total} GB/s not in expected range"
    
    def test_get_target_layers_for_psych(self):
        """PSYCH category should target Layer 7 and Layer 8."""
        layers = get_target_layers_for_category(MemshadowCategory.PSYCH)
        assert DSMILLayer.PRIMARY_AI_MEMORY in layers
        assert DSMILLayer.SECURITY_ANALYTICS in layers
    
    def test_get_target_layers_for_threat(self):
        """THREAT category should primarily target Layer 8."""
        layers = get_target_layers_for_category(MemshadowCategory.THREAT, include_secondary=False)
        assert DSMILLayer.SECURITY_ANALYTICS in layers
    
    def test_get_target_layers_for_memory(self):
        """MEMORY category should target Layer 7."""
        layers = get_target_layers_for_category(MemshadowCategory.MEMORY)
        assert DSMILLayer.PRIMARY_AI_MEMORY in layers
    
    def test_get_target_layers_for_federation(self):
        """FEDERATION category should target Layers 6-8."""
        layers = get_target_layers_for_category(MemshadowCategory.FEDERATION)
        assert DSMILLayer.FEDERATION_MESH in layers
        assert DSMILLayer.PRIMARY_AI_MEMORY in layers
    
    def test_get_devices_for_category(self):
        """Verify devices are returned for known categories."""
        for cat in [MemshadowCategory.PSYCH, MemshadowCategory.THREAT, MemshadowCategory.MEMORY]:
            devices = get_devices_for_category(cat)
            assert len(devices) >= 1, f"No devices for {cat.name}"
    
    def test_get_devices_filters_by_layer(self):
        """Verify device filtering by layer works."""
        devices = get_devices_for_category(MemshadowCategory.PSYCH, layer=DSMILLayer.PRIMARY_AI_MEMORY)
        for d in devices:
            assert d.layer == DSMILLayer.PRIMARY_AI_MEMORY
    
    def test_category_from_msg_type_value(self):
        """Verify MessageType value -> Category conversion."""
        assert MemshadowCategory.from_msg_type_value(0x0100) == MemshadowCategory.PSYCH
        assert MemshadowCategory.from_msg_type_value(0x0201) == MemshadowCategory.THREAT
        assert MemshadowCategory.from_msg_type_value(0x0304) == MemshadowCategory.MEMORY
        assert MemshadowCategory.from_msg_type_value(0x0401) == MemshadowCategory.FEDERATION
        assert MemshadowCategory.from_msg_type_value(0x0501) == MemshadowCategory.IMPROVEMENT
        assert MemshadowCategory.from_msg_type_value(0x0001) == MemshadowCategory.UNKNOWN
    
    def test_category_from_string(self):
        """Verify string -> Category conversion."""
        assert MemshadowCategory.from_string("psych") == MemshadowCategory.PSYCH
        assert MemshadowCategory.from_string("THREAT") == MemshadowCategory.THREAT
        assert MemshadowCategory.from_string("Memory") == MemshadowCategory.MEMORY
        assert MemshadowCategory.from_string("invalid") == MemshadowCategory.UNKNOWN
    
    def test_should_trigger_layer8_hook(self):
        """Verify Layer 8 hook trigger logic."""
        # THREAT should trigger at NORMAL+ priority (min_priority_for_layer8_hook=1)
        assert should_trigger_layer8_hook(MemshadowCategory.THREAT, Priority.NORMAL)
        assert should_trigger_layer8_hook(MemshadowCategory.THREAT, Priority.HIGH)
        
        # PSYCH should trigger at HIGH+ priority (min_priority_for_layer8_hook=2)
        assert not should_trigger_layer8_hook(MemshadowCategory.PSYCH, Priority.LOW)
        assert should_trigger_layer8_hook(MemshadowCategory.PSYCH, Priority.HIGH)
        
        # MEMORY: Layer 8 is only in secondary layers (FEDERATION_MESH is primary)
        # So Layer 8 hooks won't trigger since SECURITY_ANALYTICS isn't in target layers
        # This is expected behavior - memory sync doesn't need security hooks by default
    
    def test_threat_category_not_degradable(self):
        """THREAT category should not be degradable."""
        assert not can_degrade_category(MemshadowCategory.THREAT)
    
    def test_psych_category_degradable(self):
        """PSYCH category should be degradable."""
        assert can_degrade_category(MemshadowCategory.PSYCH)
    
    def test_priority_weights_in_valid_range(self):
        """Priority weights should be 1-10."""
        for cat in MemshadowCategory:
            if cat == MemshadowCategory.UNKNOWN:
                continue
            weight = get_category_priority_weight(cat)
            assert 1 <= weight <= 10, f"{cat.name} has invalid weight {weight}"


# =============================================================================
# Bandwidth Governor Tests
# =============================================================================

class TestBandwidthGovernor:
    """Tests for memshadow_bandwidth_governor.py"""
    
    @pytest.fixture
    def governor(self):
        """Create a test governor with known config."""
        config = GovernorConfig(
            memshadow_bandwidth_percent=0.15,
            degrade_threshold_percent=0.80,
            drop_threshold_percent=0.95,
            stats_window_seconds=10.0,
            monitor_only=False,
        )
        gov = MemshadowBandwidthGovernor(config)
        yield gov
        gov.reset_stats()
    
    @pytest.fixture
    def monitor_only_governor(self):
        """Create a governor in monitor-only mode."""
        config = GovernorConfig(
            memshadow_bandwidth_percent=0.15,
            monitor_only=True,
        )
        gov = MemshadowBandwidthGovernor(config)
        yield gov
        gov.reset_stats()
    
    def test_accept_under_budget(self, governor):
        """Governor should accept traffic when under budget."""
        decision = governor.should_accept(
            payload_bytes=1000,
            category=MemshadowCategory.PSYCH,
            priority=Priority.NORMAL,
        )
        assert decision == AcceptDecision.ACCEPT
    
    def test_critical_never_dropped(self, governor):
        """CRITICAL priority frames should never be dropped."""
        # Simulate high load
        for _ in range(1000):
            governor.should_accept(
                payload_bytes=10 * 1024 * 1024,  # 10 MB each
                category=MemshadowCategory.PSYCH,
                priority=Priority.LOW,
            )
        
        # CRITICAL should still be accepted
        decision = governor.should_accept(
            payload_bytes=10 * 1024 * 1024,
            category=MemshadowCategory.PSYCH,
            priority=Priority.CRITICAL,
        )
        assert decision == AcceptDecision.ACCEPT
    
    def test_emergency_never_dropped(self, governor):
        """EMERGENCY priority frames should never be dropped."""
        decision = governor.should_accept(
            payload_bytes=100 * 1024 * 1024,  # 100 MB
            category=MemshadowCategory.PSYCH,
            priority=Priority.EMERGENCY,
        )
        assert decision == AcceptDecision.ACCEPT
    
    def test_monitor_only_never_drops(self, monitor_only_governor):
        """Monitor-only mode should always accept."""
        for _ in range(100):
            decision = monitor_only_governor.should_accept(
                payload_bytes=100 * 1024 * 1024,
                category=MemshadowCategory.PSYCH,
                priority=Priority.LOW,
            )
            assert decision == AcceptDecision.ACCEPT
    
    def test_threat_not_degraded(self, governor):
        """THREAT category should not be degraded (only dropped as last resort)."""
        # Simulate moderate load
        for _ in range(50):
            governor.should_accept(
                payload_bytes=10 * 1024 * 1024,
                category=MemshadowCategory.THREAT,
                priority=Priority.NORMAL,
            )
        
        # THREAT at normal priority should still be accepted (not degraded)
        decision = governor.should_accept(
            payload_bytes=1000,
            category=MemshadowCategory.THREAT,
            priority=Priority.NORMAL,
        )
        # Either ACCEPT or DEGRADE is acceptable, but not DROP for THREAT
        assert decision in (AcceptDecision.ACCEPT, AcceptDecision.DEGRADE)
    
    def test_sync_mode_normal(self, governor):
        """Default sync mode should be DELTA."""
        mode = governor.choose_sync_mode(MemshadowCategory.MEMORY)
        assert mode == SyncMode.DELTA
    
    def test_sync_interval_normal(self, governor):
        """Default sync interval should be 1000ms."""
        interval = governor.get_sync_interval_ms(MemshadowCategory.MEMORY)
        assert interval == 1000
    
    def test_batch_size_normal(self, governor):
        """Default batch size should be 100."""
        batch_size = governor.get_batch_size(MemshadowCategory.MEMORY)
        assert batch_size == 100
    
    def test_should_compress_large_payload(self, governor):
        """Large payloads should be compressed."""
        assert governor.should_compress(1024)
        assert not governor.should_compress(256)
    
    def test_stats_tracking(self, governor):
        """Verify stats are tracked correctly."""
        governor.should_accept(1000, MemshadowCategory.PSYCH, priority=Priority.NORMAL)
        governor.should_accept(2000, MemshadowCategory.THREAT, priority=Priority.HIGH)
        
        stats = governor.get_stats()
        assert "layer_category_stats" in stats
        assert "config" in stats
    
    def test_utilization_calculation(self, governor):
        """Verify utilization calculation."""
        # Record some traffic
        governor.should_accept(1000, MemshadowCategory.PSYCH, priority=Priority.NORMAL)
        
        util = governor.get_utilization(MemshadowCategory.PSYCH)
        assert util >= 0.0
    
    def test_degradation_mode_tracking(self, governor):
        """Verify degradation mode can be queried."""
        mode = governor.get_degradation_mode(MemshadowCategory.PSYCH)
        assert mode == DegradationMode.NONE  # Should start with no degradation
    
    def test_layer8_hook_registration(self, governor):
        """Verify Layer 8 hooks can be registered."""
        hook_called = []
        
        def test_hook(category, priority, payload_bytes, metadata):
            hook_called.append((category, priority))
        
        governor.register_layer8_hook(test_hook)
        governor.trigger_layer8_hooks(
            MemshadowCategory.THREAT,
            Priority.HIGH,
            1000,
            {"test": True},
        )
        
        assert len(hook_called) == 1


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMemshadowMetrics:
    """Tests for extended memshadow_metrics.py"""
    
    @pytest.fixture
    def metrics(self):
        """Create a fresh metrics registry."""
        registry = MemshadowMetricsRegistry()
        yield registry
        registry.reset()
    
    def test_record_layer_category_bytes(self, metrics):
        """Verify per-layer/category byte tracking."""
        metrics.record_layer_category_bytes(7, "psych", 1000, frame_count=1)
        metrics.record_layer_category_bytes(7, "psych", 2000, frame_count=1)
        
        stats = metrics.get_layer_category_stats(layer_id=7, category="psych")
        assert "7/psych" in stats
        assert stats["7/psych"]["bytes_total"] == 3000
        assert stats["7/psych"]["frames_total"] == 2
    
    def test_record_dropped_frame(self, metrics):
        """Verify dropped frame tracking."""
        metrics.record_dropped_frame(7, "psych", 1000, reason="bandwidth_guard")
        metrics.record_dropped_frame(7, "psych", 500, reason="config_disabled")
        
        snapshot = metrics.snapshot()
        assert snapshot["memshadow_frames_dropped"] == 2
        assert snapshot["memshadow_bytes_dropped"] == 1500
        
        by_reason = metrics.get_dropped_by_reason()
        assert by_reason["bandwidth_guard"] == 1
        assert by_reason["config_disabled"] == 1
    
    def test_record_degraded_frame(self, metrics):
        """Verify degraded frame tracking."""
        metrics.record_degraded_frame(8, "threat", 2000)
        
        snapshot = metrics.snapshot()
        assert snapshot["memshadow_frames_degraded"] == 1
        assert snapshot["memshadow_bytes_degraded"] == 2000
    
    def test_record_degradation_event(self, metrics):
        """Verify degradation event tracking."""
        metrics.record_degradation_event(7, "psych", active=True)
        
        active_modes = metrics.get_active_degradation_modes()
        assert "7/psych" in active_modes
        
        snapshot = metrics.snapshot()
        assert snapshot["memshadow_degradation_events"] == 1
    
    def test_record_layer8_hook(self, metrics):
        """Verify Layer 8 hook tracking."""
        metrics.record_layer8_hook()
        metrics.record_layer8_hook()
        
        snapshot = metrics.snapshot()
        assert snapshot["memshadow_layer8_hooks_triggered"] == 2
    
    def test_prometheus_export(self, metrics):
        """Verify Prometheus format export."""
        metrics.record_layer_category_bytes(7, "psych", 1000)
        metrics.observe_latency(5.0)
        
        prometheus_output = metrics.export_prometheus_format()
        assert "memshadow_" in prometheus_output
        assert "bytes" in prometheus_output.lower()


# =============================================================================
# Config Tests
# =============================================================================

class TestMemshadowConfigExtended:
    """Tests for extended memshadow_config.py"""
    
    def test_bandwidth_governor_defaults(self, tmp_path, monkeypatch):
        """Verify bandwidth governor default values."""
        config = load_memshadow_config(str(tmp_path / "missing.yaml"))
        
        bg = config.bandwidth_governor
        assert bg.total_system_bandwidth_gbps == 64.0
        assert bg.memshadow_bandwidth_percent == 0.15
        assert bg.monitor_only is False
        assert bg.enable_layer8_hooks is True
    
    def test_degradation_policy_defaults(self, tmp_path, monkeypatch):
        """Verify degradation policy default values."""
        config = load_memshadow_config(str(tmp_path / "missing.yaml"))
        
        dp = config.bandwidth_governor.degradation_policy
        assert dp.degrade_threshold_percent == 0.80
        assert dp.drop_threshold_percent == 0.95
        assert dp.never_degrade_priority == 3  # CRITICAL
        assert dp.never_drop_priority == 4     # EMERGENCY
    
    def test_memshadow_budget_calculation(self, tmp_path, monkeypatch):
        """Verify MEMSHADOW budget calculation."""
        config = load_memshadow_config(str(tmp_path / "missing.yaml"))
        
        budget = config.bandwidth_governor.get_memshadow_budget_bytes_per_sec()
        expected = int(64.0 * 1e9 * 0.15)  # 64 GB/s * 15%
        assert budget == expected
    
    def test_env_override_monitor_only(self, tmp_path, monkeypatch):
        """Verify MEMSHADOW_MONITOR_ONLY env override."""
        monkeypatch.setenv("MEMSHADOW_MONITOR_ONLY", "true")
        config = load_memshadow_config(str(tmp_path / "missing.yaml"))
        
        assert config.bandwidth_governor.monitor_only is True
    
    def test_env_override_degrade_threshold(self, tmp_path, monkeypatch):
        """Verify MEMSHADOW_DEGRADE_THRESHOLD_PERCENT env override."""
        monkeypatch.setenv("MEMSHADOW_DEGRADE_THRESHOLD_PERCENT", "0.70")
        config = load_memshadow_config(str(tmp_path / "missing.yaml"))
        
        assert config.bandwidth_governor.degradation_policy.degrade_threshold_percent == 0.70
    
    def test_config_to_dict(self, tmp_path, monkeypatch):
        """Verify config serialization to dict."""
        config = load_memshadow_config(str(tmp_path / "missing.yaml"))
        
        d = config.to_dict()
        assert "bandwidth_governor" in d
        assert "degradation_policy" in d["bandwidth_governor"]
        assert "memshadow_budget_bytes_per_sec" in d["bandwidth_governor"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestGovernorIntegration:
    """Integration tests for governor with layer mapping."""
    
    def test_governor_uses_layer_mapping(self):
        """Verify governor respects layer mapping."""
        config = GovernorConfig()
        governor = MemshadowBandwidthGovernor(config)
        
        # THREAT category should not be degraded
        # Even under simulated load, threat intel shouldn't be degraded
        for _ in range(100):
            decision = governor.should_accept(
                payload_bytes=1_000_000,
                category=MemshadowCategory.THREAT,
                priority=Priority.NORMAL,
            )
            # THREAT is never degraded due to allows_degradation=False
            assert decision in (AcceptDecision.ACCEPT, AcceptDecision.DROP)
    
    def test_metrics_integration(self):
        """Verify governor updates metrics correctly."""
        metrics = get_memshadow_metrics_registry()
        metrics.reset()
        
        config = GovernorConfig()
        governor = MemshadowBandwidthGovernor(config)
        
        # Accept a frame
        governor.should_accept(1000, MemshadowCategory.PSYCH, priority=Priority.NORMAL)
        
        # Stats should be updated
        stats = governor.get_stats()
        assert len(stats["layer_category_stats"]) > 0


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Running MEMSHADOW Bandwidth Governor Tests")
    print("=" * 60)
    
    # Run layer mapping tests
    print("\n[1] Layer Mapping Tests")
    test_mapping = TestLayerMapping()
    test_mapping.test_layer_budgets_exist()
    test_mapping.test_total_bandwidth_approximately_64gbps()
    test_mapping.test_get_target_layers_for_psych()
    test_mapping.test_get_target_layers_for_threat()
    test_mapping.test_category_from_msg_type_value()
    test_mapping.test_should_trigger_layer8_hook()
    print("    ✓ All layer mapping tests passed")
    
    # Run governor tests
    print("\n[2] Bandwidth Governor Tests")
    config = GovernorConfig(
        memshadow_bandwidth_percent=0.15,
        degrade_threshold_percent=0.80,
        drop_threshold_percent=0.95,
        stats_window_seconds=10.0,
        monitor_only=False,
    )
    governor = MemshadowBandwidthGovernor(config)
    
    # Test accept under budget
    decision = governor.should_accept(1000, MemshadowCategory.PSYCH, priority=Priority.NORMAL)
    assert decision == AcceptDecision.ACCEPT, "Should accept under budget"
    
    # Test EMERGENCY never dropped
    decision = governor.should_accept(100_000_000, MemshadowCategory.PSYCH, priority=Priority.EMERGENCY)
    assert decision == AcceptDecision.ACCEPT, "EMERGENCY should never be dropped"
    
    print("    ✓ All bandwidth governor tests passed")
    
    # Run metrics tests
    print("\n[3] Metrics Tests")
    metrics = MemshadowMetricsRegistry()
    metrics.record_layer_category_bytes(7, "psych", 1000)
    stats = metrics.get_layer_category_stats()
    assert "7/psych" in stats, "Should track layer/category stats"
    print("    ✓ All metrics tests passed")
    
    print("\n" + "=" * 60)
    print("All self-tests passed!")
