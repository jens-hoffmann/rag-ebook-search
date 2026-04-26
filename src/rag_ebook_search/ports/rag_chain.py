"""RAG chain port interface."""

from abc import ABC, abstractmethod

from rag_ebook_search.schemas import RAGResponse


class RAGChainPort(ABC):
    """Port for RAG pipeline operations.

    This abstract base class defines the interface that all RAG chain
    implementations must follow. It allows the application to swap
    RAG implementations without changing business logic.
    """

    @abstractmethod
    async def ask(
        self, question: str, book_id: str | None = None, top_k: int = 5
    ) -> RAGResponse:
        """Execute the RAG pipeline: retrieve and generate answer.

        Args:
            question: User question.
            book_id: Optional book ID to scope retrieval.
            top_k: Number of chunks to retrieve.

        Returns:
            RAGResponse with answer and source excerpts.
        """
        pass
