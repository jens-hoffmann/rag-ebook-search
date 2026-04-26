"""Port interfaces for the RAG ebook search application.

Ports define the abstract interfaces that external adapters must implement.
This allows the application core to remain independent of external frameworks.
"""

from rag_ebook_search.ports.document_loader import DocumentLoaderPort
from rag_ebook_search.ports.embedding import EmbeddingPort
from rag_ebook_search.ports.llm import LLMPort
from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.ports.vector_store import VectorStorePort

__all__ = [
    "VectorStorePort",
    "EmbeddingPort",
    "LLMPort",
    "RAGChainPort",
    "DocumentLoaderPort",
]
