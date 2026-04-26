"""Integration tests for book management with real database."""

import io
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document


class TestBooksIntegration:
    """Integration tests for books router with real PostgreSQL."""

    def test_health_check(self, integration_client: TestClient) -> None:
        """Test the health check endpoint."""
        response = integration_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_upload_and_list_pdf(
        self,
        integration_client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test uploading a PDF and listing books."""
        mock_document_loader.extract_text.return_value = "This is test PDF content for integration testing."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk 1", metadata={}),
            Document(page_content="Chunk 2", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        response = integration_client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake-pdf"), "application/pdf")},
            data={"title": "Integration Test PDF", "author": "Test Author"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["title"] == "Integration Test PDF"
        assert data["author"] == "Test Author"
        assert data["file_type"] == "pdf"

        # List books
        list_resp = integration_client.get("/books/")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["total"] >= 1
        assert any(b["title"] == "Integration Test PDF" for b in list_data["books"])

    def test_upload_unsupported_file(self, integration_client: TestClient) -> None:
        """Test uploading an unsupported file type."""
        response = integration_client.post(
            "/books/",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400

    def test_get_book(
        self,
        integration_client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test getting a single book."""
        mock_document_loader.extract_text.return_value = "Test content for get book."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        create_resp = integration_client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            data={"title": "Get Integration Test"},
        )
        assert create_resp.status_code == 201, create_resp.text
        book_id = create_resp.json()["id"]

        response = integration_client.get(f"/books/{book_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == book_id
        assert data["title"] == "Get Integration Test"

    def test_get_book_not_found(self, integration_client: TestClient) -> None:
        """Test getting a non-existent book."""
        response = integration_client.get("/books/non-existent-id")
        assert response.status_code == 404

    def test_delete_book(
        self,
        integration_client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test deleting a book."""
        mock_document_loader.extract_text.return_value = "Test content for delete book."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None
        mock_vector_store.delete_by_book_id.return_value = None

        create_resp = integration_client.post(
            "/books/",
            files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            data={"title": "Delete Integration Test"},
        )
        assert create_resp.status_code == 201, create_resp.text
        book_id = create_resp.json()["id"]

        response = integration_client.delete(f"/books/{book_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_resp = integration_client.get(f"/books/{book_id}")
        assert get_resp.status_code == 404

    def test_upload_epub(
        self,
        integration_client: TestClient,
        mock_document_loader: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test uploading an EPUB file."""
        mock_document_loader.extract_text.return_value = "Test EPUB content."
        mock_document_loader.split_text.return_value = [
            Document(page_content="Chunk", metadata={}),
        ]
        mock_vector_store.add_documents.return_value = None

        response = integration_client.post(
            "/books/",
            files={"file": ("test.epub", io.BytesIO(b"fake-epub"), "application/epub+zip")},
            data={"title": "Integration Test EPUB"},
        )
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["title"] == "Integration Test EPUB"
        assert data["file_type"] == "epub"
