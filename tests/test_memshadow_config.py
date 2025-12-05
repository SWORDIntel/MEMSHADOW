from config.memshadow_config import load_memshadow_config


def test_memshadow_config_defaults(tmp_path, monkeypatch):
    cfg = load_memshadow_config(str(tmp_path / "missing.yaml"))
    assert cfg.background_sync_interval_seconds == 30
    assert cfg.max_batch_items == 100
    assert cfg.compression_threshold_bytes == 1024
    assert cfg.enable_p2p_for_critical is True
    assert cfg.enable_shrink_ingest is True
    assert cfg.enable_psych_ingest is True
    assert cfg.enable_threat_ingest is True
    assert cfg.enable_memory_ingest is True
    assert cfg.enable_federation_ingest is True
    assert cfg.enable_improvement_ingest is True


def test_memshadow_config_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMSHADOW_MAX_BATCH_ITEMS", "42")
    monkeypatch.setenv("MEMSHADOW_ENABLE_P2P_FOR_CRITICAL", "false")
    monkeypatch.setenv("MEMSHADOW_ENABLE_THREAT_INGEST", "0")
    cfg = load_memshadow_config(str(tmp_path / "missing.yaml"))
    assert cfg.max_batch_items == 42
    assert cfg.enable_p2p_for_critical is False
    assert cfg.enable_threat_ingest is False
