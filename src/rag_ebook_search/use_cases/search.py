"""Search use case for vector similarity search."""

from typing import List

from langchain_core.documents import Document

from rag_ebook_search.ports.vector_store import VectorStorePort
from rag_ebook_search.schemas import SearchResponse, SearchResult


class SearchUseCase:
    """Use case for searching similar documents.

    Orchestrates the search flow by using the VectorStorePort.
    """

    def __init__(self, vector_store: VectorStorePort):
        """Initialize the search use case.

        Args:
            vector_store: Vector store port implementation.
        """
        self._vector_store = vector_store

    async def search(
        self, query: str, book_id: str | None = None, limit: int = 5
    ) -> SearchResponse:
        """Search for documents similar to the query.

        Args:
            query: Search query text.
            book_id: Optional book ID to filter results.
            limit: Maximum number of results to return.

        Returns:
            SearchResponse with matching results.
        """
        documents = await self._vector_store.search(
            query, book_id=book_id, top_k=limit
        )

        results = self._documents_to_results(documents)
        return SearchResponse(query=query, results=results)

    def _documents_to_results(self, documents: List[Document]) -> List[SearchResult]:
        """Convert Document objects to SearchResult schemas.

        Args:
            documents: List of retrieved documents.

        Returns:
            List of search results.
        """
        results: List[SearchResult] = []
        for doc in documents:
            meta = doc.metadata or {}
            results.append(
                SearchResult(
                    book_id=meta.get("book_id", ""),
                    book_title=meta.get("book_title", "Unknown"),
                    chunk_text=doc.page_content,
                    score=meta.get("score", 0.0),
                )
            )
        return results
