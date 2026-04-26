"""Tests for search endpoints."""

import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document


class TestSearchRouter:
    """Tests for the search router."""

    def test_search(
        self, client: TestClient, mock_vector_store: MagicMock
    ) -> None:
        """Test vector search endpoint."""
        mock_vector_store.search.return_value = [
            Document(
                page_content="Test chunk content",
                metadata={"book_id": "123", "book_title": "Test Book", "score": 0.95},
            )
        ]

        response = client.get("/search/?q=hello&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "hello"
        assert len(data["results"]) == 1
        assert data["results"][0]["book_title"] == "Test Book"

    def test_search_with_book_filter(
        self,
        client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test search with book_id filter."""
        mock_document_loader.extract_text.return_value = "Test content."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        create_resp = client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            data={"title": "Filter Test"},
        )
        assert create_resp.status_code == 201, create_resp.text
        book_id = create_resp.json()["id"]

        mock_vector_store.search.return_value = []

        response = client.get(f"/search/?q=hello&book_id={book_id}")
        assert response.status_code == 200
        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["book_id"] == book_id

    def test_search_invalid_book_id(self, client: TestClient) -> None:
        """Test search with non-existent book_id."""
        response = client.get("/search/?q=hello&book_id=nonexistent")
        assert response.status_code == 404
