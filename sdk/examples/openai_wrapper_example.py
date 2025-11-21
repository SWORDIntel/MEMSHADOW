"""
MEMSHADOW SDK - OpenAI Wrapper Example
Demonstrates automatic memory persistence with OpenAI ChatGPT
"""

from memshadow_sdk import OpenAI


def main():
    print("=== MEMSHADOW + OpenAI Integration Demo ===\n")

    # Initialize the OpenAI wrapper with MEMSHADOW
    client = OpenAI(
        api_key="your-openai-api-key",           # Your OpenAI API key
        memshadow_url="http://localhost:8000/api/v1",
        memshadow_api_key="your-memshadow-key",  # Your MEMSHADOW API key
        model="gpt-4",
        user_id="demo_user",
        auto_inject_context=True,                # Automatically inject relevant memories
        context_limit=5                          # Inject up to 5 relevant memories
    )

    # Session 1: Store some information
    print("Session 1: Introducing myself")
    print("-" * 50)

    conversation_1 = [
        "Hi! My name is Alice and I work as a senior software engineer at TechCorp.",
        "I specialize in backend development with Python and FastAPI.",
        "My favorite database is PostgreSQL, and I love using Redis for caching."
    ]

    for message in conversation_1:
        print(f"\nUser: {message}")
        response = client.chat([{"role": "user", "content": message}])
        assistant_reply = response.choices[0].message.content
        print(f"Assistant: {assistant_reply}")

    print("\n" + "="*50)
    print("\nMemories stored! Now let's test retrieval...\n")
    print("="*50)

    # Session 2: Ask questions that require memory
    print("\nSession 2: Testing memory recall")
    print("-" * 50)

    questions = [
        "What's my name?",
        "Where do I work?",
        "What's my job role?",
        "What technologies do I use?"
    ]

    for question in questions:
        print(f"\nUser: {question}")
        response = client.chat([{"role": "user", "content": question}])
        assistant_reply = response.choices[0].message.content
        print(f"Assistant: {assistant_reply}")

    # Session 3: Multi-turn conversation with memory
    print("\n" + "="*50)
    print("\nSession 3: Natural conversation with memory")
    print("-" * 50)

    multi_turn = [
        {"role": "user", "content": "Can you help me with a database question?"},
        {"role": "user", "content": "What would you recommend for my use case at work?"}
    ]

    for msg in multi_turn:
        print(f"\n{msg['role'].capitalize()}: {msg['content']}")
        response = client.chat([msg])
        assistant_reply = response.choices[0].message.content
        print(f"Assistant: {assistant_reply}")

    # Manually retrieve memories
    print("\n" + "="*50)
    print("\nManual Memory Retrieval")
    print("-" * 50)

    print("\nSearching for 'database' memories:")
    memories = client.retrieve_memories("database", limit=3)
    for i, mem in enumerate(memories, 1):
        print(f"{i}. {mem['content']}")

    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("✓ Automatic conversation storage")
    print("✓ Semantic memory retrieval")
    print("✓ Context injection for relevant responses")
    print("✓ Cross-session memory persistence")


if __name__ == "__main__":
    main()
