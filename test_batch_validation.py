"""
Unit test for ChromaDB batch operations validation fix
Tests the corrected length validation logic
"""

from unittest.mock import Mock


def test_batch_validation_all_same_length():
    """Test that validation passes when all lists have same length"""
    from app.db.chromadb import ChromaDBClient
    from app.core.config import settings

    client = ChromaDBClient()
    client.collection = Mock()
    client.collection.add = Mock()

    # Mock settings
    settings.EMBEDDING_DIMENSION = 3

    memory_ids = ["id1", "id2", "id3"]
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    metadatas = [{"a": 1}, {"b": 2}, {"c": 3}]

    # Should not raise
    import asyncio
    asyncio.run(client.add_embeddings_batch(memory_ids, embeddings, metadatas))

    # Verify collection.add was called
    client.collection.add.assert_called_once()
    print("✓ Test passed: All same length (3, 3, 3)")


def test_batch_validation_two_ids_one_embedding():
    """Test that validation fails when one list is shorter (2 ids, 1 embedding, 2 metadata)"""
    from app.db.chromadb import ChromaDBClient
    from app.core.config import settings

    client = ChromaDBClient()
    client.collection = Mock()

    settings.EMBEDDING_DIMENSION = 3

    memory_ids = ["id1", "id2"]  # 2 items
    embeddings = [[1.0, 2.0, 3.0]]  # 1 item - MISMATCH!
    metadatas = [{"a": 1}, {"b": 2}]  # 2 items

    # Should raise ValueError
    import asyncio
    try:
        asyncio.run(client.add_embeddings_batch(memory_ids, embeddings, metadatas))
        print("✗ Test FAILED: Should have raised ValueError for (2, 1, 2)")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Length mismatch" in str(e)
        assert "memory_ids=2" in str(e)
        assert "embeddings=1" in str(e)
        assert "metadatas=2" in str(e)
        print("✓ Test passed: Caught mismatch (2, 1, 2)")


def test_batch_validation_all_different_lengths():
    """Test that validation fails when all three have different lengths"""
    from app.db.chromadb import ChromaDBClient
    from app.core.config import settings

    client = ChromaDBClient()
    client.collection = Mock()

    settings.EMBEDDING_DIMENSION = 3

    memory_ids = ["id1"]  # 1 item
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2 items
    metadatas = [{"a": 1}, {"b": 2}, {"c": 3}]  # 3 items

    # Should raise ValueError
    import asyncio
    try:
        asyncio.run(client.add_embeddings_batch(memory_ids, embeddings, metadatas))
        print("✗ Test FAILED: Should have raised ValueError for (1, 2, 3)")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Length mismatch" in str(e)
        print("✓ Test passed: Caught mismatch (1, 2, 3)")


def test_batch_validation_two_same_one_different():
    """Test that validation fails when two lists match but one differs"""
    from app.db.chromadb import ChromaDBClient
    from app.core.config import settings

    client = ChromaDBClient()
    client.collection = Mock()

    settings.EMBEDDING_DIMENSION = 3

    memory_ids = ["id1", "id2"]  # 2 items
    embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2 items (same as ids)
    metadatas = [{"a": 1}]  # 1 item - MISMATCH!

    # Should raise ValueError
    import asyncio
    try:
        asyncio.run(client.add_embeddings_batch(memory_ids, embeddings, metadatas))
        print("✗ Test FAILED: Should have raised ValueError for (2, 2, 1)")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Length mismatch" in str(e)
        assert "metadatas=1" in str(e)
        print("✓ Test passed: Caught mismatch (2, 2, 1)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ChromaDB Batch Validation Tests")
    print("Testing the corrected length validation logic")
    print("="*60 + "\n")

    try:
        test_batch_validation_all_same_length()
        test_batch_validation_two_ids_one_embedding()
        test_batch_validation_all_different_lengths()
        test_batch_validation_two_same_one_different()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("The validation fix correctly catches all mismatch cases!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
