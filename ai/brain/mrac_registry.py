#!/usr/bin/env python3
"""
MRAC registry for remote app control nodes.

Stores a simple JSON index at /mnt/miltop/var/mrac_nodes.json with:
{
  "app_id_hex": {
      "name": str,
      "capabilities": dict,
      "last_seen": iso8601,
      "status": str,
      "telemetry": dict
  }
}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

MRAC_PATH = Path("/mnt/miltop/var/mrac_nodes.json")
MRAC_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load() -> Dict[str, Any]:
    if MRAC_PATH.exists():
        try:
            return json.loads(MRAC_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save(data: Dict[str, Any]) -> None:
    MRAC_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_register(app_id: str, name: str, capabilities: Dict[str, Any]) -> None:
    data = _load()
    node = data.get(app_id, {})
    node.update(
        {
            "name": name,
            "capabilities": capabilities,
            "last_seen": _now(),
            "status": "registered",
        }
    )
    data[app_id] = node
    _save(data)


def update_heartbeat(app_id: str, telemetry: Dict[str, Any]) -> None:
    data = _load()
    node = data.get(app_id, {})
    node.setdefault("telemetry", {}).update(telemetry)
    node["last_seen"] = _now()
    node.setdefault("status", "online")
    data[app_id] = node
    _save(data)


def update_command_ack(app_id: str, command_id: str, status: str) -> None:
    data = _load()
    node = data.get(app_id, {})
    node["last_seen"] = _now()
    node.setdefault("last_command", {})["id"] = command_id
    node["last_command"]["status"] = status
    data[app_id] = node
    _save(data)


def list_nodes() -> Dict[str, Any]:
    return _load()
