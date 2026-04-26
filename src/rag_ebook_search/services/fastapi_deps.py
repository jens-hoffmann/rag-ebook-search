"""FastAPI dependency providers for dependency injection."""

from fastapi import Depends

from rag_ebook_search.services.container import DependencyContainer
from rag_ebook_search.ports.document_loader import DocumentLoaderPort
from rag_ebook_search.ports.embedding import EmbeddingPort
from rag_ebook_search.ports.llm import LLMPort
from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.ports.vector_store import VectorStorePort


def get_container() -> "DependencyContainer":
    """Get the global dependency container.

    This function is set up in main.py to return the application's
    singleton container instance.

    Returns:
        DependencyContainer instance.
    """
    # This will be overridden in main.py
    raise RuntimeError(
        "Dependency container not initialized. "
        "Ensure app.state.container is set in main.py"
    )


def get_vector_store(container: "DependencyContainer" = Depends(get_container)) -> VectorStorePort:
    """FastAPI dependency for vector store.

    Args:
        container: Dependency container (injected by FastAPI).

    Returns:
        VectorStorePort implementation.
    """
    return container.vector_store


def get_embedding(container: "DependencyContainer" = Depends(get_container)) -> EmbeddingPort:
    """FastAPI dependency for embedding service.

    Args:
        container: Dependency container (injected by FastAPI).

    Returns:
        EmbeddingPort implementation.
    """
    return container.embedding


def get_llm(container: "DependencyContainer" = Depends(get_container)) -> LLMPort:
    """FastAPI dependency for LLM service.

    Args:
        container: Dependency container (injected by FastAPI).

    Returns:
        LLMPort implementation.
    """
    return container.llm


def get_rag_chain(container: "DependencyContainer" = Depends(get_container)) -> RAGChainPort:
    """FastAPI dependency for RAG chain.

    Args:
        container: Dependency container (injected by FastAPI).

    Returns:
        RAGChainPort implementation.
    """
    return container.rag_chain


def get_document_loader(
    container: "DependencyContainer" = Depends(get_container),
) -> DocumentLoaderPort:
    """FastAPI dependency for document loader.

    Args:
        container: Dependency container (injected by FastAPI).

    Returns:
        DocumentLoaderPort implementation.
    """
    return container.document_loader
