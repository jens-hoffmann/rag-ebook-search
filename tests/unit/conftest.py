"""Pytest fixtures and configuration for unit tests.

Unit tests use mocked database and external services - no running services required.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from rag_ebook_search.database import get_db
from rag_ebook_search.services import get_container
from rag_ebook_search.main import app
from rag_ebook_search.ports.document_loader import DocumentLoaderPort
from rag_ebook_search.ports.embedding import EmbeddingPort
from rag_ebook_search.ports.llm import LLMPort
from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.ports.vector_store import VectorStorePort

# Module-level storage for books, shared across all test requests
_stored_books: dict[str, dict[str, Any]] = {}


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Provide a mock vector store."""
    mock = MagicMock(spec=VectorStorePort)
    mock.add_documents = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.delete_by_book_id = AsyncMock()
    return mock


@pytest.fixture
def mock_embedding() -> MagicMock:
    """Provide a mock embedding service."""
    mock = MagicMock(spec=EmbeddingPort)
    mock.embed_documents.return_value = [[0.1] * 10]
    mock.embed_query = AsyncMock(return_value=[0.1] * 10)
    return mock


@pytest.fixture
def mock_llm() -> MagicMock:
    """Provide a mock LLM."""
    mock = MagicMock(spec=LLMPort)
    mock.generate = AsyncMock(return_value="Mock LLM response")
    mock.generate_with_context = AsyncMock(return_value="Mock LLM response with context")
    return mock


@pytest.fixture
def mock_rag_chain() -> MagicMock:
    """Provide a mock RAG chain."""
    mock = MagicMock(spec=RAGChainPort)
    mock.ask = AsyncMock()
    return mock


@pytest.fixture
def mock_document_loader() -> MagicMock:
    """Provide a mock document loader."""
    mock = MagicMock(spec=DocumentLoaderPort)
    mock.extract_text = MagicMock(return_value="Mock extracted text")
    mock.split_text = MagicMock(return_value=[])
    return mock


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a mocked async database session that simulates real database operations."""
    # Create a mock session with proper async behavior
    mock_session = AsyncMock(spec=AsyncSession)

    # Clear storage at the start of each test
    _stored_books.clear()

    def mock_add(obj: Any) -> None:
        """Mock db.add() to store object in memory."""
        # Generate and set id if not already set
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = str(uuid.uuid4())
        # Set timestamps if not already set
        if not hasattr(obj, "created_at") or obj.created_at is None:
            obj.created_at = datetime.now(timezone.utc)
        if not hasattr(obj, "updated_at") or obj.updated_at is None:
            obj.updated_at = datetime.now(timezone.utc)
        _stored_books[obj.id] = {
            "id": obj.id,
            "title": getattr(obj, "title", ""),
            "author": getattr(obj, "author", None),
            "filename": getattr(obj, "filename", ""),
            "file_type": getattr(obj, "file_type", ""),
            "description": getattr(obj, "description", None),
            "created_at": obj.created_at,
            "updated_at": obj.updated_at,
        }

    async def mock_commit() -> None:
        """Mock db.commit() as no-op."""
        pass

    async def mock_rollback() -> None:
        """Mock db.rollback() as no-op."""
        pass

    async def mock_refresh(obj: Any) -> None:
        """Mock db.refresh() to update object with stored data."""
        if hasattr(obj, "id") and obj.id in _stored_books:
            for key, value in _stored_books[obj.id].items():
                setattr(obj, key, value)

    async def mock_execute(statement: Any) -> MagicMock:
        """Mock db.execute() to return query results."""
        mock_result = MagicMock()

        # Helper to create mock book from stored data
        def create_mock_book(book_data: dict[str, Any]) -> MagicMock:
            mock_book = MagicMock()
            mock_book.id = book_data["id"]
            mock_book.title = book_data["title"]
            mock_book.author = book_data["author"]
            mock_book.filename = book_data["filename"]
            mock_book.file_type = book_data["file_type"]
            mock_book.description = book_data["description"]
            mock_book.created_at = book_data["created_at"]
            mock_book.updated_at = book_data["updated_at"]
            return mock_book

        # Handle select statements for Book
        if hasattr(statement, "_where_criteria") and statement._where_criteria:
            # Single book lookup by ID (has where criteria)
            book_id = None
            for crit in statement._where_criteria:
                if hasattr(crit, "right") and hasattr(crit.right, "value"):
                    book_id = crit.right.value
                    break

            if book_id and book_id in _stored_books:
                book_data = _stored_books[book_id]
                mock_result.scalar_one_or_none.return_value = create_mock_book(book_data)
            else:
                mock_result.scalar_one_or_none.return_value = None
        else:
            # List all books (handles offset/limit)
            books_list = [create_mock_book(book_data) for book_data in _stored_books.values()]

            # Check if statement has offset/limit attributes
            skip = getattr(statement, "_offset", 0) or 0
            limit = getattr(statement, "_limit", None)

            if skip > 0:
                books_list = books_list[skip:]
            if limit is not None:
                books_list = books_list[:limit]

            mock_result.scalars.return_value.all.return_value = books_list
            mock_result.scalar_one_or_none.return_value = MagicMock()
            mock_result.scalar_one_or_none.return_value.count = len(_stored_books)

        return mock_result

    async def mock_delete(obj: Any) -> None:
        """Mock db.delete() to remove object from storage."""
        if hasattr(obj, "id") and obj.id in _stored_books:
            del _stored_books[obj.id]

    mock_session.add = mock_add
    mock_session.commit = mock_commit
    mock_session.rollback = mock_rollback
    mock_session.refresh = mock_refresh
    mock_session.execute = mock_execute
    mock_session.delete = mock_delete

    yield mock_session


@pytest.fixture
def client(
    db_session: AsyncSession,
    mock_vector_store: MagicMock,
    mock_rag_chain: MagicMock,
    mock_document_loader: MagicMock,
) -> Generator[TestClient, None, None]:
    """Provide a TestClient with mocked database dependency and lifespan."""

    async def _get_test_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    # Mock the lifespan context manager to prevent database connection on startup
    @asynccontextmanager
    async def mock_lifespan(app):
        yield

    # Create a mock container
    mock_container = MagicMock()
    mock_container.vector_store = mock_vector_store
    mock_container.rag_chain = mock_rag_chain
    mock_container.document_loader = mock_document_loader
    mock_container.embedding = MagicMock()
    mock_container.llm = MagicMock()

    def _get_mock_container() -> MagicMock:
        return mock_container

    # Override dependencies
    app.dependency_overrides[get_db] = _get_test_db
    app.dependency_overrides[get_container] = _get_mock_container
    app.router.lifespan_context = mock_lifespan

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
