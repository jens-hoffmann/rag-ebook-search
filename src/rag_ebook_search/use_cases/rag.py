"""RAG use case for question answering."""

from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.schemas import RAGResponse


class RAGUseCase:
    """Use case for RAG-based question answering.

    Orchestrates the RAG flow by using the RAGChainPort.
    """

    def __init__(self, rag_chain: RAGChainPort):
        """Initialize the RAG use case.

        Args:
            rag_chain: RAG chain port implementation.
        """
        self._rag_chain = rag_chain

    async def ask(
        self, question: str, book_id: str | None = None, top_k: int = 5
    ) -> RAGResponse:
        """Ask a question and get an answer based on ebook content.

        Args:
            question: User question.
            book_id: Optional book ID to scope retrieval.
            top_k: Number of chunks to retrieve.

        Returns:
            RAGResponse with answer and source excerpts.
        """
        return await self._rag_chain.ask(
            question=question,
            book_id=book_id,
            top_k=top_k,
        )
