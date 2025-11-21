"""
MEMSHADOW SDK - Basic Usage Example
Demonstrates basic client operations for memory ingestion and retrieval
"""

from memshadow_sdk import MemshadowClient
from datetime import datetime, timedelta


def main():
    # Initialize the client
    print("Initializing MEMSHADOW client...")
    client = MemshadowClient(
        api_url="http://localhost:8000/api/v1",
        api_key="your-api-key-here",  # Replace with your API key
        user_id="demo_user"
    )

    # Example 1: Ingest memories
    print("\n=== Ingesting Memories ===")
    memories_to_store = [
        "I am a Python developer with 5 years of experience",
        "My favorite framework is FastAPI",
        "I'm currently working on a memory persistence project",
        "The project uses PostgreSQL, Redis, and ChromaDB"
    ]

    for content in memories_to_store:
        try:
            memory = client.ingest(content)
            print(f"✓ Stored: {content[:50]}... (ID: {memory['id']})")
        except Exception as e:
            print(f"✗ Failed to store: {e}")

    # Example 2: Retrieve memories
    print("\n=== Retrieving Memories ===")
    queries = [
        "What is my job?",
        "What technologies am I using?",
        "Tell me about frameworks"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            results = client.retrieve(query, limit=3)
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['content']}")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error: {e}")

    # Example 3: Create task reminders
    print("\n=== Creating Task Reminders ===")

    # Create a reminder for 1 hour from now
    reminder_date = datetime.now() + timedelta(hours=1)
    due_date = datetime.now() + timedelta(days=1)

    try:
        reminder = client.create_reminder(
            title="Review code changes",
            reminder_date=reminder_date,
            due_date=due_date,
            description="Review the new authentication feature implementation",
            priority="high"
        )
        print(f"✓ Created reminder: {reminder['title']} (ID: {reminder['id']})")
    except Exception as e:
        print(f"✗ Failed to create reminder: {e}")

    # Example 4: List reminders
    print("\n=== Listing Pending Reminders ===")
    try:
        reminders = client.list_reminders(status="pending", limit=10)
        if reminders:
            for reminder in reminders:
                print(f"  • {reminder['title']} (Priority: {reminder['priority']})")
                print(f"    Due: {reminder.get('due_date', 'No due date')}")
        else:
            print("  No pending reminders")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
