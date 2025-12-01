"""
Integration Test for New Memlayer-Inspired Features
Tests operation modes, task reminders, and SDK functionality
"""

import sys
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_section(title):
    """Print a section header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")


def print_success(message):
    """Print success message"""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message):
    """Print error message"""
    print(f"{RED}✗ {message}{RESET}")


def print_warning(message):
    """Print warning message"""
    print(f"{YELLOW}⚠ {message}{RESET}")


def print_info(message):
    """Print info message"""
    print(f"  {message}")


async def test_operation_modes():
    """Test operation mode configuration"""
    print_section("Test 1: Operation Modes")

    try:
        from app.core.config import settings, MemoryOperationMode

        # Check enum exists
        assert hasattr(MemoryOperationMode, 'LOCAL')
        assert hasattr(MemoryOperationMode, 'ONLINE')
        assert hasattr(MemoryOperationMode, 'LIGHTWEIGHT')
        print_success("MemoryOperationMode enum defined")

        # Check config has the setting
        assert hasattr(settings, 'MEMORY_OPERATION_MODE')
        print_success(f"Current mode: {settings.MEMORY_OPERATION_MODE.value}")

        return True

    except Exception as e:
        print_error(f"Operation mode test failed: {e}")
        return False


async def test_task_reminder_model():
    """Test task reminder model"""
    print_section("Test 2: Task Reminder Model")

    try:
        from app.models.task_reminder import TaskReminder, ReminderStatus, ReminderPriority

        # Check enums
        assert hasattr(ReminderStatus, 'PENDING')
        assert hasattr(ReminderPriority, 'HIGH')
        print_success("ReminderStatus and ReminderPriority enums defined")

        # Check model attributes
        test_reminder = TaskReminder(
            user_id=uuid4(),
            title="Test Reminder",
            reminder_date=datetime.now(),
            status=ReminderStatus.PENDING,
            priority=ReminderPriority.MEDIUM
        )

        assert hasattr(test_reminder, 'is_overdue')
        assert hasattr(test_reminder, 'should_remind')
        print_success("TaskReminder model has computed properties")

        return True

    except Exception as e:
        print_error(f"Task reminder model test failed: {e}")
        return False


async def test_task_reminder_service():
    """Test task reminder service"""
    print_section("Test 3: Task Reminder Service")

    try:
        from app.services.task_reminder_service import TaskReminderService

        # Check service methods exist
        assert hasattr(TaskReminderService, 'create_reminder')
        assert hasattr(TaskReminderService, 'list_reminders')
        assert hasattr(TaskReminderService, 'get_pending_reminders')
        assert hasattr(TaskReminderService, 'mark_as_completed')
        print_success("TaskReminderService has all required methods")

        return True

    except Exception as e:
        print_error(f"Task reminder service test failed: {e}")
        return False


async def test_api_endpoints():
    """Test API endpoint registration"""
    print_section("Test 4: API Endpoints")

    try:
        from app.main import app

        # Get all routes
        routes = [route.path for route in app.routes]

        # Check for reminder endpoints
        reminder_endpoints = [
            '/api/v1/reminders/',
            '/api/v1/reminders/{reminder_id}',
        ]

        found_endpoints = []
        for endpoint in reminder_endpoints:
            if any(endpoint in route for route in routes):
                found_endpoints.append(endpoint)

        if found_endpoints:
            print_success(f"Found {len(found_endpoints)} reminder endpoints")
            for endpoint in found_endpoints:
                print_info(f"  - {endpoint}")
        else:
            print_warning("No reminder endpoints found in routes")

        return len(found_endpoints) > 0

    except Exception as e:
        print_error(f"API endpoint test failed: {e}")
        return False


async def test_celery_tasks():
    """Test Celery task registration"""
    print_section("Test 5: Celery Tasks")

    try:
        from app.workers.celery_app import celery_app

        # Check beat schedule
        beat_schedule = celery_app.conf.beat_schedule

        assert 'check-reminders' in beat_schedule
        print_success("check-reminders task registered in Celery Beat")
        print_info(f"  Schedule: Every {beat_schedule['check-reminders']['schedule']} seconds")

        # Check task exists
        from app.workers import tasks
        assert hasattr(tasks, 'check_and_send_reminders')
        print_success("check_and_send_reminders task defined")

        return True

    except Exception as e:
        print_error(f"Celery task test failed: {e}")
        return False


async def test_sdk_structure():
    """Test SDK package structure"""
    print_section("Test 6: SDK Package Structure")

    try:
        import os

        sdk_path = 'sdk/memshadow_sdk'

        required_files = [
            'sdk/memshadow_sdk/__init__.py',
            'sdk/memshadow_sdk/client.py',
            'sdk/memshadow_sdk/wrappers/__init__.py',
            'sdk/memshadow_sdk/wrappers/openai.py',
            'sdk/memshadow_sdk/wrappers/anthropic.py',
            'sdk/setup.py',
            'sdk/README.md',
        ]

        all_exist = True
        for file_path in required_files:
            if os.path.exists(file_path):
                print_success(f"{file_path} exists")
            else:
                print_error(f"{file_path} not found")
                all_exist = False

        return all_exist

    except Exception as e:
        print_error(f"SDK structure test failed: {e}")
        return False


async def test_chromadb_batch_operations():
    """Test ChromaDB batch operations"""
    print_section("Test 7: ChromaDB Batch Operations")

    try:
        from app.db.chromadb import ChromaDBClient

        # Check method exists
        client = ChromaDBClient()
        assert hasattr(client, 'add_embeddings_batch')
        print_success("add_embeddings_batch method exists")

        # Check method signature
        import inspect
        sig = inspect.signature(client.add_embeddings_batch)
        params = list(sig.parameters.keys())

        assert 'memory_ids' in params
        assert 'embeddings' in params
        assert 'metadatas' in params
        print_success("Batch method has correct signature")

        return True

    except Exception as e:
        print_error(f"ChromaDB batch operations test failed: {e}")
        return False


async def test_memory_service_modes():
    """Test MemoryService mode awareness"""
    print_section("Test 8: MemoryService Mode Awareness")

    try:
        from app.services.memory_service import MemoryService
        from app.core.config import MemoryOperationMode
        import inspect

        # Check that MemoryOperationMode is imported in memory_service
        source = inspect.getsource(MemoryService)

        if 'MemoryOperationMode' in source:
            print_success("MemoryService imports MemoryOperationMode")
        else:
            print_warning("MemoryOperationMode may not be used in MemoryService")

        # Check methods exist
        assert hasattr(MemoryService, 'generate_and_store_embedding')
        assert hasattr(MemoryService, 'search_memories')
        print_success("MemoryService has required methods")

        return True

    except Exception as e:
        print_error(f"MemoryService mode test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}MEMSHADOW Integration Tests{RESET}")
    print(f"{BLUE}Memlayer-Inspired Features{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    tests = [
        ("Operation Modes", test_operation_modes),
        ("Task Reminder Model", test_task_reminder_model),
        ("Task Reminder Service", test_task_reminder_service),
        ("API Endpoints", test_api_endpoints),
        ("Celery Tasks", test_celery_tasks),
        ("SDK Structure", test_sdk_structure),
        ("ChromaDB Batch Ops", test_chromadb_batch_operations),
        ("MemoryService Modes", test_memory_service_modes),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")

    print(f"\n{BLUE}{'='*60}{RESET}")
    if passed == total:
        print(f"{GREEN}All tests passed! ({passed}/{total}){RESET}")
        print(f"{GREEN}✓ New features are ready to use{RESET}")
    else:
        print(f"{YELLOW}Some tests failed ({passed}/{total}){RESET}")
        print(f"{YELLOW}⚠ Check the errors above{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
