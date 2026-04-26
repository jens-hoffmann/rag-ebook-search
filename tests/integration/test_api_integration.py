"""Integration tests for API endpoints with real database."""

import io
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag_ebook_search.schemas import RAGResponse, SearchResult


class TestAPIIntegration:
    """Integration tests for search and RAG endpoints."""

    def test_search_endpoint(
        self,
        integration_client: TestClient,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test search endpoint with real database."""
        mock_vector_store.search.return_value = [
            Document(
                page_content="Test chunk content",
                metadata={"book_id": "123", "book_title": "Test Book", "score": 0.95},
            )
        ]

        response = integration_client.get("/search/?q=hello&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "hello"
        assert len(data["results"]) == 1
        assert data["results"][0]["book_title"] == "Test Book"

    def test_rag_endpoint(
        self,
        integration_client: TestClient,
        mock_rag_chain: MagicMock,
    ) -> None:
        """Test RAG endpoint with real database."""
        mock_rag_chain.ask.return_value = RAGResponse(
            question="What is AI?",
            answer="AI stands for Artificial Intelligence.",
            sources=[
                SearchResult(
                    book_id="123",
                    book_title="Test Book",
                    chunk_text="AI is artificial intelligence.",
                    score=0.95,
                )
            ],
        )

        response = integration_client.post(
            "/rag/",
            json={"question": "What is AI?", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is AI?"
        assert "Artificial Intelligence" in data["answer"]
        assert len(data["sources"]) == 1

    def test_rag_with_book_filter(
        self,
        integration_client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
        mock_rag_chain: MagicMock,
    ) -> None:
        """Test RAG scoped to a specific book."""
        mock_document_loader.extract_text.return_value = "Test content."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        create_resp = integration_client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            data={"title": "RAG Integration Book"},
        )
        assert create_resp.status_code == 201, create_resp.text
        book_id = create_resp.json()["id"]

        mock_rag_chain.ask.return_value = RAGResponse(
            question="Test?",
            answer="Test answer.",
            sources=[],
        )

        response = integration_client.post(
            "/rag/",
            json={"question": "Test?", "book_id": book_id, "top_k": 3},
        )
        assert response.status_code == 200
        mock_rag_chain.ask.assert_called_once_with(
            question="Test?",
            book_id=book_id,
            top_k=3,
        )

    def test_search_invalid_book_id(self, integration_client: TestClient) -> None:
        """Test search with non-existent book_id."""
        response = integration_client.get("/search/?q=hello&book_id=nonexistent")
        assert response.status_code == 404

    def test_rag_invalid_book_id(self, integration_client: TestClient) -> None:
        """Test RAG with non-existent book_id."""
        response = integration_client.post(
            "/rag/",
            json={"question": "Test?", "book_id": "nonexistent", "top_k": 5},
        )
        assert response.status_code == 404
