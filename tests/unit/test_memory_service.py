import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from uuid import uuid4
from datetime import datetime

from app.services.memory_service import MemoryService
from app.models.memory import Memory

@pytest.fixture
def mock_db_session():
    """Fixture for a mocked async database session."""
    session = AsyncMock()
    # The 'add' method on an AsyncSession is synchronous, so we mock it as such.
    session.add = MagicMock()
    return session

@pytest.fixture
def memory_service_fixture(mock_db_session):
    """Fixture for MemoryService with its dependencies mocked."""
    with patch("app.services.memory_service.EmbeddingService") as mock_embedding_service_cls, \
         patch("app.services.memory_service.chroma_client", new_callable=AsyncMock) as mock_chroma_client, \
         patch("app.services.memory_service.redis_client", new_callable=AsyncMock) as mock_redis_client:

        mock_embedding_service = mock_embedding_service_cls.return_value
        mock_embedding_service.generate_embedding = AsyncMock()

        # Configure async methods on the mocks
        mock_chroma_client.add_embedding = AsyncMock()
        mock_chroma_client.search_similar = AsyncMock()
        mock_redis_client.cache_set = AsyncMock()

        service = MemoryService(mock_db_session)
        service.embedding_service = mock_embedding_service

        yield service, mock_chroma_client, mock_redis_client

@pytest.mark.asyncio
async def test_create_memory_success(memory_service_fixture, mock_db_session: AsyncMock):
    # Arrange
    memory_service, _, _ = memory_service_fixture
    user_id = uuid4()
    content = "This is a new memory."
    extra_data = {"source": "test"}

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_result

    # Act
    memory = await memory_service.create_memory(user_id, content, extra_data)

    # Assert
    assert memory is not None
    assert memory.user_id == user_id
    assert memory.content == content
    assert memory.extra_data == extra_data
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()

@pytest.mark.asyncio
async def test_create_memory_duplicate_error(memory_service_fixture, mock_db_session: AsyncMock):
    # Arrange
    memory_service, _, _ = memory_service_fixture
    user_id = uuid4()
    content = "This is a duplicate memory."
    extra_data = {"source": "test"}

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = Memory(id=uuid4(), user_id=user_id, content=content, content_hash="somehash")
    mock_db_session.execute.return_value = mock_result

    # Act & Assert
    with pytest.raises(ValueError, match="Duplicate memory content"):
        await memory_service.create_memory(user_id, content, extra_data)

    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()

@pytest.mark.asyncio
async def test_generate_and_store_embedding(memory_service_fixture, mock_db_session: AsyncMock):
    # Arrange
    memory_service, mock_chroma_client, mock_redis_client = memory_service_fixture
    memory_id = uuid4()
    content = "some content to embed"
    embedding_vector = [0.1, 0.2, 0.3]
    mock_memory = Memory(
        id=memory_id,
        user_id=uuid4(),
        content=content,
        content_hash="hash",
        created_at=datetime.utcnow(),
        extra_data={}
    )

    memory_service.embedding_service.generate_embedding.return_value = embedding_vector
    mock_db_session.get.return_value = mock_memory

    # Act
    await memory_service.generate_and_store_embedding(memory_id, content)

    # Assert
    memory_service.embedding_service.generate_embedding.assert_called_once_with(content)
    assert mock_memory.embedding == embedding_vector
    mock_chroma_client.add_embedding.assert_called_once_with(
        memory_id=str(memory_id),
        embedding=embedding_vector,
        metadata=ANY
    )
    mock_redis_client.cache_set.assert_called_once()
    mock_db_session.commit.assert_called_once()

@pytest.mark.asyncio
async def test_search_memories(memory_service_fixture, mock_db_session: AsyncMock):
    # Arrange
    memory_service, mock_chroma_client, mock_redis_client = memory_service_fixture
    user_id = uuid4()
    query = "search query"
    query_embedding = [0.4, 0.5, 0.6]
    memory_id_1 = uuid4()
    memory_id_2 = uuid4()

    memory_service.embedding_service.generate_embedding.return_value = query_embedding
    mock_chroma_client.search_similar.return_value = {
        'ids': [[str(memory_id_1), str(memory_id_2)]],
        'distances': [[0.1, 0.2]]
    }

    mock_memories = [
        Memory(id=memory_id_1, user_id=user_id, content="content 1", extra_data={}),
        Memory(id=memory_id_2, user_id=user_id, content="content 2", extra_data={})
    ]
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_memories
    mock_db_session.execute.return_value = mock_result

    # Act
    results = await memory_service.search_memories(user_id, query)

    # Assert
    assert len(results) == 2
    memory_service.embedding_service.generate_embedding.assert_called_once_with(query)
    mock_chroma_client.search_similar.assert_called_once_with(
        query_embedding=query_embedding,
        n_results=ANY,
        where={'user_id': str(user_id)}
    )
    mock_db_session.execute.assert_called_once()
    mock_db_session.commit.assert_called_once()
    assert results[0].extra_data['access_count'] == 1