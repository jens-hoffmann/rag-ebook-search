"""Tests for book management endpoints."""

import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag_ebook_search.schemas import RAGResponse, SearchResult


class TestBooksRouter:
    """Tests for the books router."""

    def test_upload_pdf(
        self,
        client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test uploading a PDF file."""
        mock_document_loader.extract_text.return_value = "Test PDF content."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        response = client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")},
            data={"title": "Test PDF", "author": "Test Author"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["title"] == "Test PDF"
        assert data["author"] == "Test Author"
        assert data["file_type"] == "pdf"

    def test_upload_unsupported_file(self, client: TestClient) -> None:
        """Test uploading an unsupported file type."""
        response = client.post(
            "/books/",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    def test_list_books(
        self,
        client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test listing books."""
        mock_document_loader.extract_text.return_value = "Test PDF content."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            data={"title": "List Test"},
        )

        response = client.get("/books/")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["books"]) >= 1

    def test_get_book(
        self,
        client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test getting a single book."""
        mock_document_loader.extract_text.return_value = "Test PDF content."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        create_resp = client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            data={"title": "Get Test"},
        )
        assert create_resp.status_code == 201, create_resp.text
        book_id = create_resp.json()["id"]

        response = client.get(f"/books/{book_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == book_id
        assert data["title"] == "Get Test"

    def test_get_book_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent book."""
        response = client.get("/books/non-existent-id")
        assert response.status_code == 404

    def test_delete_book(
        self,
        client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test deleting a book."""
        mock_document_loader.extract_text.return_value = "Test PDF content."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None
        mock_vector_store.delete_by_book_id.return_value = None

        create_resp = client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            data={"title": "Delete Test"},
        )
        assert create_resp.status_code == 201, create_resp.text
        book_id = create_resp.json()["id"]

        response = client.delete(f"/books/{book_id}")
        assert response.status_code == 204

        get_resp = client.get(f"/books/{book_id}")
        assert get_resp.status_code == 404

    def test_health_check(self, client: TestClient) -> None:
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
