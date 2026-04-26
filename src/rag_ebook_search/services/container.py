"""Dependency injection container for the RAG application."""

from functools import cached_property

from rag_ebook_search.adapters.document_loader import LangChainDocumentLoaderAdapter
from rag_ebook_search.adapters.embedding import LangChainEmbeddingAdapter
from rag_ebook_search.adapters.llm import LangChainLLMAdapter
from rag_ebook_search.adapters.rag_chain import LangChainRAGChainAdapter
from rag_ebook_search.adapters.vector_store import LangChainVectorStoreAdapter
from rag_ebook_search.config import Settings
from rag_ebook_search.ports.document_loader import DocumentLoaderPort
from rag_ebook_search.ports.embedding import EmbeddingPort
from rag_ebook_search.ports.llm import LLMPort
from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.ports.vector_store import VectorStorePort


class DependencyContainer:
    """DI container for managing port implementations.

    Provides lazy initialization of dependencies and allows overriding
    implementations for testing purposes.

    Example:
        container = DependencyContainer(config)
        vector_store = container.vector_store
        rag_chain = container.rag_chain
    """

    def __init__(self, config: Settings):
        """Initialize the dependency container.

        Args:
            config: Application settings.
        """
        self._config = config
        self._embedding: EmbeddingPort | None = None
        self._vector_store: VectorStorePort | None = None
        self._llm: LLMPort | None = None
        self._rag_chain: RAGChainPort | None = None
        self._document_loader: DocumentLoaderPort | None = None

    @property
    def config(self) -> Settings:
        """Get the application settings."""
        return self._config

    @property
    def embedding(self) -> EmbeddingPort:
        """Get the embedding port implementation.

        Returns:
            EmbeddingPort implementation.
        """
        if self._embedding is None:
            self._embedding = LangChainEmbeddingAdapter(self._config)
        return self._embedding

    @property
    def vector_store(self) -> VectorStorePort:
        """Get the vector store port implementation.

        Returns:
            VectorStorePort implementation.
        """
        if self._vector_store is None:
            self._vector_store = LangChainVectorStoreAdapter(
                config=self._config,
                embedding=self.embedding,
            )
        return self._vector_store

    @property
    def llm(self) -> LLMPort:
        """Get the LLM port implementation.

        Returns:
            LLMPort implementation.
        """
        if self._llm is None:
            self._llm = LangChainLLMAdapter(self._config)
        return self._llm

    @property
    def rag_chain(self) -> RAGChainPort:
        """Get the RAG chain port implementation.

        Returns:
            RAGChainPort implementation.
        """
        if self._rag_chain is None:
            self._rag_chain = LangChainRAGChainAdapter(
                config=self._config,
                vector_store=self.vector_store,
                llm=self.llm,
            )
        return self._rag_chain

    @property
    def document_loader(self) -> DocumentLoaderPort:
        """Get the document loader port implementation.

        Returns:
            DocumentLoaderPort implementation.
        """
        if self._document_loader is None:
            self._document_loader = LangChainDocumentLoaderAdapter(self._config)
        return self._document_loader

    def register_embedding(self, impl: EmbeddingPort) -> None:
        """Register a custom embedding implementation.

        Args:
            impl: Custom EmbeddingPort implementation.
        """
        self._embedding = impl

    def register_vector_store(self, impl: VectorStorePort) -> None:
        """Register a custom vector store implementation.

        Args:
            impl: Custom VectorStorePort implementation.
        """
        self._vector_store = impl

    def register_llm(self, impl: LLMPort) -> None:
        """Register a custom LLM implementation.

        Args:
            impl: Custom LLMPort implementation.
        """
        self._llm = impl

    def register_rag_chain(self, impl: RAGChainPort) -> None:
        """Register a custom RAG chain implementation.

        Args:
            impl: Custom RAGChainPort implementation.
        """
        self._rag_chain = impl

    def register_document_loader(self, impl: DocumentLoaderPort) -> None:
        """Register a custom document loader implementation.

        Args:
            impl: Custom DocumentLoaderPort implementation.
        """
        self._document_loader = impl

    def reset(self) -> None:
        """Reset all cached implementations.

        Useful for testing to ensure clean state between tests.
        """
        self._embedding = None
        self._vector_store = None
        self._llm = None
        self._rag_chain = None
        self._document_loader = None
