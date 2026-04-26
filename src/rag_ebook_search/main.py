"""FastAPI application entry point."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from rag_ebook_search.config import settings
from rag_ebook_search.database import engine
from rag_ebook_search.services import get_container
from rag_ebook_search.services.container import DependencyContainer
from rag_ebook_search.services.fastapi_deps import get_container as get_container_dep
from rag_ebook_search.models import Base
from rag_ebook_search.routers.books import router as books_router
from rag_ebook_search.routers.rag import router as rag_router
from rag_ebook_search.routers.search import router as search_router

# Global dependency container
_container: DependencyContainer | None = None


def get_container() -> DependencyContainer:
    """Get the global dependency container.

    Returns:
        DependencyContainer instance.
    """
    global _container
    if _container is None:
        _container = DependencyContainer(settings)
    return _container


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: create tables on startup."""
    # Initialize dependency container
    container = get_container()
    app.state.container = container

    # Override the get_container dependency
    app.dependency_overrides[get_container_dep] = get_container

    # Skip database initialization if SKIP_DB_INIT is set (for testing)
    if os.environ.get("SKIP_DB_INIT") != "true":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Ensure VectorChord extension is enabled
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    yield
    await engine.dispose()


app = FastAPI(
    title="RAG Ebook Search",
    description="Search and chat with your PDF and EPUB ebooks using RAG.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(books_router)
app.include_router(search_router)
app.include_router(rag_router)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
