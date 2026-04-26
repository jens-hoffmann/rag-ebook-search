"""Adapters for external services and frameworks.

Adapters implement the port interfaces using concrete external libraries
like LangChain, PostgreSQL, LMStudio, etc.
"""

from rag_ebook_search.adapters.document_loader import LangChainDocumentLoaderAdapter
from rag_ebook_search.adapters.embedding import LangChainEmbeddingAdapter
from rag_ebook_search.adapters.llm import LangChainLLMAdapter
from rag_ebook_search.adapters.rag_chain import LangChainRAGChainAdapter
from rag_ebook_search.adapters.vector_store import LangChainVectorStoreAdapter

__all__ = [
    "LangChainVectorStoreAdapter",
    "LangChainEmbeddingAdapter",
    "LangChainLLMAdapter",
    "LangChainRAGChainAdapter",
    "LangChainDocumentLoaderAdapter",
]
