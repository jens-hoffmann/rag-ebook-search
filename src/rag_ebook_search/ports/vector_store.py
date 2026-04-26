"""Vector store port interface."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class VectorStorePort(ABC):
    """Port for vector similarity operations.

    This abstract base class defines the interface that all vector store
    implementations must follow. It allows the application to swap vector
    store providers without changing business logic.
    """

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add document chunks to the vector store.

        Args:
            documents: List of Document chunks to embed and store.
        """
        pass

    @abstractmethod
    async def search(
        self, query: str, book_id: str | None = None, top_k: int = 5
    ) -> List[Document]:
        """Search for similar documents in the vector store.

        Args:
            query: Search query text.
            book_id: Optional book ID to filter results.
            top_k: Number of results to return.

        Returns:
            List of matching Document chunks.
        """
        pass

    @abstractmethod
    async def delete_by_book_id(self, book_id: str) -> None:
        """Delete all chunks belonging to a specific book.

        Args:
            book_id: ID of the book whose chunks should be removed.
        """
        pass
