"""Integration test fixtures using testcontainers + PostgreSQL with VectorChord."""

import os
import socket
import subprocess
import time
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

# Set environment variables BEFORE importing app to ensure correct database URL
# These must be set before any rag_ebook_search imports
os.environ["LMSTUDIO_BASE_URL"] = "http://localhost:1234/v1"
os.environ["LMSTUDIO_API_KEY"] = "test-key"
os.environ["CHUNK_SIZE"] = "500"
os.environ["CHUNK_OVERLAP"] = "50"
# Skip database init in app lifespan - we handle it in db_engine fixture
os.environ["SKIP_DB_INIT"] = "true"
# Placeholder DATABASE_URL - will be overridden by postgres_container fixture
os.environ["DATABASE_URL"] = "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from rag_ebook_search.database import get_db
from rag_ebook_search.services import get_container, get_document_loader, get_rag_chain, get_vector_store
from rag_ebook_search.main import app
from rag_ebook_search.models import Base
from rag_ebook_search.ports.document_loader import DocumentLoaderPort
from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.ports.vector_store import VectorStorePort

POSTGRES_IMAGE = "docker.io/tensorchord/vchord-postgres:pg16-v0.2.2"


def _get_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_postgres(container_id: str, timeout: int = 60) -> None:
    """Wait for PostgreSQL to be ready using pg_isready inside the container."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["podman", "exec", container_id, "pg_isready", "-U", "testuser", "-d", "testdb"],
            capture_output=True,
        )
        if result.returncode == 0:
            return
        time.sleep(0.5)
    raise TimeoutError(f"PostgreSQL did not become ready within {timeout}s")


@pytest.fixture(scope="session")
def postgres_container() -> Generator[dict, None, None]:
    """Spin up a PostgreSQL container with VectorChord for integration tests."""
    host_port = _get_free_port()

    result = subprocess.run(
        [
            "podman",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{host_port}:5432",
            "-e",
            "POSTGRES_USER=testuser",
            "-e",
            "POSTGRES_PASSWORD=testpass",
            "-e",
            "POSTGRES_DB=testdb",
            POSTGRES_IMAGE,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    container_id = result.stdout.strip()

    _wait_for_postgres(container_id)

    yield {"host": "127.0.0.1", "port": host_port, "id": container_id}

    subprocess.run(["podman", "stop", "-t", "5", container_id], capture_output=True)


@pytest.fixture(scope="session")
def db_engine(postgres_container: dict) -> Generator:
    """Create async SQLAlchemy engine connected to the testcontainers postgres."""
    host = postgres_container["host"]
    port = postgres_container["port"]

    database_url = f"postgresql+asyncpg://testuser:testpass@{host}:{port}/testdb"
    # Update the DATABASE_URL for any code that reads it later
    os.environ["DATABASE_URL"] = database_url

    engine = create_async_engine(database_url, future=True, poolclass=NullPool)

    async def _init() -> None:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

    import asyncio

    asyncio.run(_init())

    yield engine

    async def _drop() -> None:
        async with engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                await conn.execute(table.delete())
        await engine.dispose()

    asyncio.run(_drop())


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Provide a mock vector store for integration tests."""
    mock = MagicMock(spec=VectorStorePort)
    mock.add_documents = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.delete_by_book_id = AsyncMock()
    return mock


@pytest.fixture
def mock_document_loader() -> MagicMock:
    """Provide a mock document loader for integration tests."""
    mock = MagicMock(spec=DocumentLoaderPort)
    mock.extract_text = MagicMock(return_value="Mock extracted text")
    mock.split_text = MagicMock(return_value=[])
    return mock


@pytest.fixture
def mock_rag_chain() -> MagicMock:
    """Provide a mock RAG chain for integration tests."""
    mock = MagicMock(spec=RAGChainPort)
    mock.ask = AsyncMock()
    return mock


@pytest.fixture
def integration_client(
    db_engine: object,
    mock_vector_store: MagicMock,
    mock_document_loader: MagicMock,
    mock_rag_chain: MagicMock,
) -> Generator[TestClient, None, None]:
    """Provide a TestClient with real database and mocked external services."""

    async def _get_db() -> AsyncGenerator[AsyncSession, None]:
        SessionLocal = async_sessionmaker(
            db_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        async with SessionLocal() as session:
            yield session

    # Create a mock container
    mock_container = MagicMock()
    mock_container.vector_store = mock_vector_store
    mock_container.document_loader = mock_document_loader
    mock_container.rag_chain = mock_rag_chain
    mock_container.embedding = MagicMock()
    mock_container.llm = MagicMock()

    def _get_mock_container() -> MagicMock:
        return mock_container

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_container] = _get_mock_container
    # Override individual service dependencies directly
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_document_loader] = lambda: mock_document_loader
    app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain

    # Clean tables before each test
    import asyncio

    async def _clean() -> None:
        async with db_engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                await conn.execute(table.delete())

    asyncio.run(_clean())

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
