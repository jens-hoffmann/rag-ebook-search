"""Dependency injection container and providers."""

from rag_ebook_search.services.container import DependencyContainer
from rag_ebook_search.services.fastapi_deps import (
    get_container,
    get_document_loader,
    get_embedding,
    get_llm,
    get_rag_chain,
    get_vector_store,
)

__all__ = [
    "DependencyContainer",
    "get_container",
    "get_vector_store",
    "get_embedding",
    "get_llm",
    "get_rag_chain",
    "get_document_loader",
]
