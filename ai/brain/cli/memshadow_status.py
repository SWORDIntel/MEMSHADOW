#!/usr/bin/env python3
"""
MEMSHADOW Status CLI

Displays current MEMSHADOW configuration, per-layer/per-category throughput,
and active degradation modes without requiring a running web server.

Usage:
    python3 -m ai.brain.cli.memshadow_status
    python3 -m ai.brain.cli.memshadow_status --json
    python3 -m ai.brain.cli.memshadow_status --watch --interval 5
    python3 -m ai.brain.cli.memshadow_status --category psych
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent paths for imports
_this_dir = Path(__file__).parent
_brain_dir = _this_dir.parent
_ai_dir = _brain_dir.parent
_root_dir = _ai_dir.parent

for p in [_root_dir, _ai_dir, _brain_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _safe_import():
    """Safely import required modules with fallbacks."""
    imports = {
        "config_available": False,
        "metrics_available": False,
        "layer_mapping_available": False,
        "governor_available": False,
    }
    
    try:
        from config.memshadow_config import get_memshadow_config
        imports["config"] = get_memshadow_config
        imports["config_available"] = True
    except ImportError as e:
        imports["config_error"] = str(e)
    
    try:
        from ai.brain.metrics.memshadow_metrics import get_memshadow_metrics_registry
        imports["metrics"] = get_memshadow_metrics_registry
        imports["metrics_available"] = True
    except ImportError as e:
        imports["metrics_error"] = str(e)
    
    try:
        from ai.brain.memory.memshadow_layer_mapping import (
            DSMILLayer,
            MemshadowCategory,
            LAYER_MEMORY_BUDGETS_GB,
            LAYER_BANDWIDTH_BUDGETS_GBPS,
            get_target_layers_for_category,
            get_category_layer_mapping,
        )
        imports["layer_mapping"] = {
            "DSMILLayer": DSMILLayer,
            "MemshadowCategory": MemshadowCategory,
            "LAYER_MEMORY_BUDGETS_GB": LAYER_MEMORY_BUDGETS_GB,
            "LAYER_BANDWIDTH_BUDGETS_GBPS": LAYER_BANDWIDTH_BUDGETS_GBPS,
            "get_target_layers_for_category": get_target_layers_for_category,
            "get_category_layer_mapping": get_category_layer_mapping,
        }
        imports["layer_mapping_available"] = True
    except ImportError as e:
        imports["layer_mapping_error"] = str(e)
    
    try:
        from ai.brain.memory.memshadow_bandwidth_governor import (
            get_bandwidth_governor,
            DegradationMode,
            AcceptDecision,
        )
        imports["governor"] = {
            "get_bandwidth_governor": get_bandwidth_governor,
            "DegradationMode": DegradationMode,
            "AcceptDecision": AcceptDecision,
        }
        imports["governor_available"] = True
    except ImportError as e:
        imports["governor_error"] = str(e)
    
    return imports


def format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1e12:
        return f"{b / 1e12:.2f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    if b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{b} B"


def format_rate(bps: float) -> str:
    """Format bytes per second as human-readable rate."""
    if bps >= 1e9:
        return f"{bps / 1e9:.2f} GB/s"
    if bps >= 1e6:
        return f"{bps / 1e6:.2f} MB/s"
    if bps >= 1e3:
        return f"{bps / 1e3:.2f} KB/s"
    return f"{bps:.0f} B/s"


def get_status_data(imports: Dict[str, Any], category_filter: Optional[str] = None) -> Dict[str, Any]:
    """Gather all status data."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "modules_available": {
            "config": imports["config_available"],
            "metrics": imports["metrics_available"],
            "layer_mapping": imports["layer_mapping_available"],
            "governor": imports["governor_available"],
        },
    }
    
    # Configuration
    if imports["config_available"]:
        config = imports["config"]()
        status["config"] = config.to_dict()
    else:
        status["config"] = {"error": imports.get("config_error", "Not available")}
    
    # Layer budgets
    if imports["layer_mapping_available"]:
        lm = imports["layer_mapping"]
        status["layer_budgets"] = {
            layer.name: {
                "memory_gb": lm["LAYER_MEMORY_BUDGETS_GB"].get(layer, 0),
                "bandwidth_gbps": lm["LAYER_BANDWIDTH_BUDGETS_GBPS"].get(layer, 0),
            }
            for layer in lm["DSMILLayer"]
        }
        
        # Category mappings
        status["category_mappings"] = {}
        for cat in lm["MemshadowCategory"]:
            if cat.name == "UNKNOWN":
                continue
            if category_filter and cat.name.lower() != category_filter.lower():
                continue
            mapping = lm["get_category_layer_mapping"](cat)
            status["category_mappings"][cat.name] = {
                "primary_layers": [l.name for l in mapping.primary_layers],
                "secondary_layers": [l.name for l in mapping.secondary_layers],
                "priority_weight": mapping.priority_weight,
                "allows_degradation": mapping.allows_degradation,
            }
    else:
        status["layer_budgets"] = {"error": imports.get("layer_mapping_error", "Not available")}
    
    # Metrics
    if imports["metrics_available"]:
        registry = imports["metrics"]()
        snapshot = registry.snapshot()
        
        # Filter layer_category_stats if category filter provided
        if category_filter and "layer_category_stats" in snapshot:
            snapshot["layer_category_stats"] = {
                k: v for k, v in snapshot["layer_category_stats"].items()
                if v.get("category", "").lower() == category_filter.lower()
            }
        
        status["metrics"] = snapshot
    else:
        status["metrics"] = {"error": imports.get("metrics_error", "Not available")}
    
    # Governor status
    if imports["governor_available"]:
        try:
            governor = imports["governor"]["get_bandwidth_governor"]()
            gov_stats = governor.get_stats(category=category_filter)
            status["governor"] = gov_stats
        except Exception as e:
            status["governor"] = {"error": str(e)}
    else:
        status["governor"] = {"error": imports.get("governor_error", "Not available")}
    
    return status


def print_text_status(status: Dict[str, Any]) -> None:
    """Print status in human-readable text format."""
    print("=" * 70)
    print("MEMSHADOW STATUS")
    print(f"Timestamp: {status['timestamp']}")
    print("=" * 70)
    
    # Module availability
    print("\n[Modules]")
    for module, available in status["modules_available"].items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {module}")
    
    # Configuration
    print("\n[Configuration]")
    if "error" in status.get("config", {}):
        print(f"  Error: {status['config']['error']}")
    else:
        config = status["config"]
        print(f"  Sync Interval: {config.get('background_sync_interval_seconds', '?')}s")
        print(f"  Max Batch Items: {config.get('max_batch_items', '?')}")
        print(f"  Compression Threshold: {format_bytes(config.get('compression_threshold_bytes', 0))}")
        print(f"  P2P for Critical: {'Yes' if config.get('enable_p2p_for_critical') else 'No'}")
        
        # Ingest toggles
        print("  Ingest Categories:")
        for key in ["psych", "threat", "memory", "federation", "improvement"]:
            enabled = config.get(f"enable_{key}_ingest", True)
            icon = "✓" if enabled else "✗"
            print(f"    {icon} {key.upper()}")
        
        # Bandwidth governor config
        bg = config.get("bandwidth_governor", {})
        if bg:
            print(f"\n  [Bandwidth Governor]")
            print(f"    System Bandwidth: {bg.get('total_system_bandwidth_gbps', '?')} GB/s")
            print(f"    MEMSHADOW Share: {bg.get('memshadow_bandwidth_percent', 0) * 100:.1f}%")
            print(f"    Budget: {format_rate(bg.get('memshadow_budget_bytes_per_sec', 0))}")
            print(f"    Monitor Only: {'Yes' if bg.get('monitor_only') else 'No'}")
            print(f"    Layer 8 Hooks: {'Enabled' if bg.get('enable_layer8_hooks') else 'Disabled'}")
            
            dp = bg.get("degradation_policy", {})
            if dp:
                print(f"    Degrade Threshold: {dp.get('degrade_threshold_percent', 0) * 100:.0f}%")
                print(f"    Drop Threshold: {dp.get('drop_threshold_percent', 0) * 100:.0f}%")
    
    # Layer budgets
    print("\n[Layer Budgets]")
    if "error" in status.get("layer_budgets", {}):
        print(f"  Error: {status['layer_budgets']['error']}")
    else:
        for layer_name, budget in sorted(status["layer_budgets"].items()):
            print(f"  Layer {layer_name}:")
            print(f"    Memory: {budget.get('memory_gb', 0):.1f} GB")
            print(f"    Bandwidth: {budget.get('bandwidth_gbps', 0):.1f} GB/s")
    
    # Category mappings
    print("\n[Category → Layer Mappings]")
    if "error" not in status.get("category_mappings", {"error": True}):
        for cat_name, mapping in status.get("category_mappings", {}).items():
            print(f"  {cat_name}:")
            print(f"    Primary: {', '.join(mapping.get('primary_layers', []))}")
            if mapping.get("secondary_layers"):
                print(f"    Secondary: {', '.join(mapping.get('secondary_layers', []))}")
            print(f"    Weight: {mapping.get('priority_weight', '?')}/10")
            print(f"    Degradable: {'Yes' if mapping.get('allows_degradation') else 'No'}")
    
    # Throughput metrics
    print("\n[Throughput Metrics]")
    if "error" in status.get("metrics", {}):
        print(f"  Error: {status['metrics']['error']}")
    else:
        metrics = status["metrics"]
        
        # Global counters
        print("  Global:")
        print(f"    Batches Sent: {metrics.get('memshadow_batches_sent', 0):,}")
        print(f"    Batches Received: {metrics.get('memshadow_batches_received', 0):,}")
        print(f"    Frames Accepted: {metrics.get('memshadow_frames_accepted', 0):,}")
        print(f"    Frames Dropped: {metrics.get('memshadow_frames_dropped', 0):,}")
        print(f"    Frames Degraded: {metrics.get('memshadow_frames_degraded', 0):,}")
        print(f"    Bytes Accepted: {format_bytes(metrics.get('memshadow_bytes_accepted', 0))}")
        print(f"    Bytes Dropped: {format_bytes(metrics.get('memshadow_bytes_dropped', 0))}")
        
        # Latency
        if "memshadow_sync_latency_ms_avg" in metrics:
            print(f"    Latency Avg: {metrics['memshadow_sync_latency_ms_avg']:.2f} ms")
            print(f"    Latency P95: {metrics.get('memshadow_sync_latency_ms_p95', 0):.2f} ms")
        
        # Per-category
        print("\n  Per-Category:")
        for key in ["psych", "threat", "memory", "federation", "improvement"]:
            count = metrics.get(f"memshadow_{key}_messages", 0)
            if count > 0:
                print(f"    {key.upper()}: {count:,} messages")
        
        # Layer/category breakdown
        lc_stats = metrics.get("layer_category_stats", {})
        if lc_stats:
            print("\n  Per-Layer/Category:")
            for key, stats in sorted(lc_stats.items()):
                print(f"    {key}:")
                print(f"      Bytes: {format_bytes(stats.get('bytes_total', 0))}")
                print(f"      Frames: {stats.get('frames_total', 0):,}")
                if stats.get("degradation_active"):
                    print("      ⚠ DEGRADATION ACTIVE")
        
        # Dropped by reason
        dropped = metrics.get("dropped_by_reason", {})
        if dropped:
            print("\n  Drops by Reason:")
            for reason, count in dropped.items():
                print(f"    {reason}: {count:,}")
    
    # Governor status
    print("\n[Bandwidth Governor Status]")
    if "error" in status.get("governor", {}):
        print(f"  Error: {status['governor']['error']}")
    else:
        gov = status["governor"]
        
        # Config summary
        gov_config = gov.get("config", {})
        print(f"  Monitor Only: {'Yes' if gov_config.get('monitor_only') else 'No'}")
        print(f"  Degrade at: {gov_config.get('degrade_threshold_percent', 0) * 100:.0f}%")
        print(f"  Drop at: {gov_config.get('drop_threshold_percent', 0) * 100:.0f}%")
        
        # Per-layer stats
        lc_stats = gov.get("layer_category_stats", [])
        if lc_stats:
            print("\n  Layer/Category Stats:")
            for stat in lc_stats:
                layer = stat.get("layer", "?")
                cat = stat.get("category", "?")
                bps = stat.get("bytes_per_second", 0)
                util = stat.get("utilization_percent", 0)
                print(f"    {layer}/{cat}:")
                print(f"      Rate: {format_rate(bps)}")
                print(f"      Utilization: {util:.1f}%")
        
        # Active degradation modes
        degradation = gov.get("degradation_modes", {})
        if degradation:
            print("\n  ⚠ Active Degradation Modes:")
            for key, mode in degradation.items():
                print(f"    {key}: {mode}")
        else:
            print("\n  ✓ No degradation modes active")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="MEMSHADOW Status CLI - Display configuration and throughput metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m ai.brain.cli.memshadow_status           # Text output
  python3 -m ai.brain.cli.memshadow_status --json    # JSON output
  python3 -m ai.brain.cli.memshadow_status --watch   # Continuous monitoring
  python3 -m ai.brain.cli.memshadow_status -c psych  # Filter by category
        """,
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuously watch and refresh status",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=5,
        help="Refresh interval in seconds for watch mode (default: 5)",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="Filter to specific category (psych, threat, memory, federation, improvement)",
    )
    parser.add_argument(
        "--prometheus",
        action="store_true",
        help="Output metrics in Prometheus format",
    )
    
    args = parser.parse_args()
    
    # Import modules
    imports = _safe_import()
    
    if args.prometheus:
        if imports["metrics_available"]:
            registry = imports["metrics"]()
            print(registry.export_prometheus_format())
        else:
            print("# ERROR: Metrics module not available")
            sys.exit(1)
        return
    
    def show_status():
        status = get_status_data(imports, category_filter=args.category)
        
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            if args.watch:
                # Clear screen for watch mode
                print("\033[2J\033[H", end="")
            print_text_status(status)
    
    if args.watch:
        try:
            while True:
                show_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        show_status()


if __name__ == "__main__":
    main()
