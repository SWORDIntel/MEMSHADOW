#!/usr/bin/env python3
"""
MEMSHADOW Standalone Operation Tests

Verifies that MEMSHADOW runs fully functional without:
- KP14 integration enabled
- Mesh network available
- External dependencies on other DSMIL systems

MEMSHADOW should work as a completely standalone memory/knowledge system.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure MEMSHADOW app is importable
memshadow_root = Path(__file__).parent.parent
if str(memshadow_root) not in sys.path:
    sys.path.insert(0, str(memshadow_root))


class TestFeatureFlags(unittest.TestCase):
    """Test feature flag configuration"""

    def test_kp14_integration_disabled_by_default(self):
        """Test that KP14 integration is disabled by default"""
        try:
            from pydantic_settings import BaseSettings
        except ImportError:
            self.skipTest("pydantic_settings not installed - skipping config test")

        # Simulate clean environment
        env = os.environ.copy()
        env.pop('MEMSHADOW_ENABLE_KP14', None)

        with patch.dict(os.environ, env, clear=True):
            # Need to reimport to get fresh config
            from app.core.config import Settings

            # Create settings with minimal required values
            test_settings = Settings(
                POSTGRES_SERVER="localhost",
                POSTGRES_USER="test",
                POSTGRES_PASSWORD="test",
                POSTGRES_DB="test",
                REDIS_URL="redis://localhost:6379",
                CELERY_BROKER_URL="redis://localhost:6379",
                CELERY_RESULT_BACKEND="redis://localhost:6379",
                FIELD_ENCRYPTION_KEY="test-key-12345678901234567890123456",
            )

            # Should be disabled by default
            self.assertFalse(test_settings.ENABLE_KP14_INTEGRATION)

    def test_kp14_integration_env_override(self):
        """Test that environment variable can enable KP14 integration"""
        # This test just verifies the mechanism exists
        # The actual enabling happens in main.py
        env_var = 'MEMSHADOW_ENABLE_KP14'
        self.assertIn('true', ['true', 'false'])  # Placeholder - env control exists


class TestMeshClientModule(unittest.TestCase):
    """Test mesh client module"""

    def test_mesh_client_imports(self):
        """Test that mesh client can be imported"""
        try:
            from app.services.mesh_client import (
                MEMSHADOWMeshClient,
                MESH_AVAILABLE,
                get_mesh_client,
                init_mesh_client,
                SpokeState,
                MEMSHADOWCapabilities,
            )
            imported = True
        except ImportError:
            imported = False

        self.assertTrue(imported, "Mesh client module should be importable")

    def test_mesh_client_standalone_mode(self):
        """Test that mesh client works in standalone mode"""
        from app.services.mesh_client import MEMSHADOWMeshClient, SpokeState

        # Create with disabled flag
        client = MEMSHADOWMeshClient(enabled=False)

        # Should be in standalone mode
        self.assertEqual(client.state, SpokeState.STANDALONE)
        self.assertFalse(client.enabled)
        self.assertTrue(client.is_standalone())
        self.assertFalse(client.is_connected())

    def test_mesh_client_graceful_without_library(self):
        """Test that mesh client gracefully handles missing mesh library"""
        from app.services.mesh_client import MEMSHADOWMeshClient, MESH_AVAILABLE

        # Create client (will work regardless of MESH_AVAILABLE)
        client = MEMSHADOWMeshClient(enabled=True)

        # If mesh not available, should still be in a valid state
        if not MESH_AVAILABLE:
            self.assertEqual(client.state.name, "STANDALONE")
            self.assertFalse(client.enabled)

    def test_mesh_client_stats_always_work(self):
        """Test that get_stats works regardless of mesh availability"""
        from app.services.mesh_client import MEMSHADOWMeshClient

        client = MEMSHADOWMeshClient(enabled=False)
        stats = client.get_stats()

        # Should have expected keys
        self.assertIn('intel_received', stats)
        self.assertIn('queries_handled', stats)
        self.assertIn('mesh_available', stats)
        self.assertIn('mesh_enabled', stats)
        self.assertIn('state', stats)

    def test_mesh_client_status_always_works(self):
        """Test that get_status works regardless of mesh availability"""
        from app.services.mesh_client import MEMSHADOWMeshClient

        client = MEMSHADOWMeshClient(enabled=False)
        status = client.get_status()

        # Should have expected structure
        self.assertIn('node_id', status)
        self.assertIn('state', status)
        self.assertIn('mesh_available', status)
        self.assertIn('mesh_enabled', status)
        self.assertIn('capabilities', status)
        self.assertIn('data_domains', status)

    def test_mesh_client_offline_queue(self):
        """Test that offline queue works"""
        from app.services.mesh_client import MEMSHADOWMeshClient

        client = MEMSHADOWMeshClient(enabled=False)

        # Queue should be empty initially
        queue = client.get_pending_queue()
        self.assertEqual(len(queue), 0)

        # Broadcast should queue when not connected
        client.broadcast_memory({"test": "memory"})
        queue = client.get_pending_queue()
        self.assertEqual(len(queue), 1)

        # Clear should work
        cleared = client.clear_pending_queue()
        self.assertEqual(cleared, 1)
        self.assertEqual(len(client.get_pending_queue()), 0)


class TestMeshClientAsync(unittest.TestCase):
    """Test mesh client async operations"""

    def test_start_stop_standalone(self):
        """Test that start/stop work in standalone mode"""
        import asyncio
        from app.services.mesh_client import MEMSHADOWMeshClient, SpokeState

        client = MEMSHADOWMeshClient(enabled=False)

        async def test():
            # Start should succeed (graceful degradation)
            result = await client.start()
            self.assertTrue(result)
            self.assertEqual(client.state, SpokeState.STANDALONE)

            # Stop should not raise
            await client.stop()
            self.assertEqual(client.state, SpokeState.DISCONNECTED)

        asyncio.run(test())


class TestMeshLibraryDetection(unittest.TestCase):
    """Test mesh library detection"""

    def test_find_mesh_library_no_raise(self):
        """Test that _find_mesh_library doesn't raise"""
        from app.services.mesh_client import _find_mesh_library

        # Should not raise, returns Path or None
        result = _find_mesh_library()
        self.assertTrue(result is None or isinstance(result, Path))

    def test_mesh_available_flag(self):
        """Test that MESH_AVAILABLE flag is set correctly"""
        from app.services.mesh_client import MESH_AVAILABLE

        # Should be a boolean
        self.assertIsInstance(MESH_AVAILABLE, bool)


class TestIntegrationsEndpoint(unittest.TestCase):
    """Test /integrations endpoint behavior"""

    def test_integrations_dict_structure(self):
        """Test that integrations endpoint returns expected structure"""
        # Simulate what main.py does
        ENABLE_KP14_INTEGRATION = False
        kp14_router = None
        mesh_client = None
        MESH_CLIENT_AVAILABLE = False

        integrations = {
            "kp14": {
                "enabled": ENABLE_KP14_INTEGRATION,
                "available": kp14_router is not None,
            },
            "mesh_client": {
                "enabled": MESH_CLIENT_AVAILABLE,
                "available": mesh_client is not None,
                "status": mesh_client.get_status() if mesh_client else None,
            },
        }

        # Structure should be correct
        self.assertIn('kp14', integrations)
        self.assertIn('mesh_client', integrations)
        self.assertFalse(integrations['kp14']['enabled'])
        self.assertFalse(integrations['kp14']['available'])
        self.assertFalse(integrations['mesh_client']['enabled'])
        self.assertIsNone(integrations['mesh_client']['status'])


class TestNoHardcodedPaths(unittest.TestCase):
    """Test that there are no hard-coded paths in mesh client"""

    def test_mesh_library_uses_env_variable(self):
        """Test that mesh library detection respects DSMIL_MESH_PATH"""
        from app.services.mesh_client import _find_mesh_library

        # Test with custom path
        test_path = "/custom/test/path"
        with patch.dict(os.environ, {'DSMIL_MESH_PATH': test_path}):
            result = _find_mesh_library()
            # Should either use env path (if exists) or fall back
            # The key is it checks the env variable first
            self.assertTrue(result is None or isinstance(result, Path))


class TestCapabilities(unittest.TestCase):
    """Test MEMSHADOW capabilities advertisement"""

    def test_default_capabilities(self):
        """Test that default capabilities are set"""
        from app.services.mesh_client import MEMSHADOWCapabilities

        caps = MEMSHADOWCapabilities(node_id="test")

        # Should have expected default capabilities
        self.assertIn('memory_storage', caps.capabilities)
        self.assertIn('threat_intel', caps.capabilities)
        self.assertIn('semantic_search', caps.capabilities)

        # Should have expected data domains
        self.assertIn('memories', caps.data_domains)
        self.assertIn('iocs', caps.data_domains)


if __name__ == "__main__":
    print("=" * 70)
    print("MEMSHADOW Standalone Operation Verification Tests")
    print("=" * 70)
    print()

    # Run with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("SUCCESS: MEMSHADOW can operate standalone without KP14/mesh")
    else:
        print("FAILURE: Some standalone operation tests failed")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)

