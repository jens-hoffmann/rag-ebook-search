"""Upload use case for book processing."""

from fastapi import HTTPException, status
from langchain_core.documents import Document

from rag_ebook_search.exceptions import DocumentProcessingError, VectorStoreError
from rag_ebook_search.logging_config import get_logger
from rag_ebook_search.ports.document_loader import DocumentLoaderPort
from rag_ebook_search.ports.vector_store import VectorStorePort

logger = get_logger(__name__)


class UploadUseCase:
    """Use case for uploading and processing books.

    Orchestrates the book upload flow including text extraction,
    chunking, and vector embedding.
    """

    def __init__(
        self,
        document_loader: DocumentLoaderPort,
        vector_store: VectorStorePort,
    ):
        """Initialize the upload use case.

        Args:
            document_loader: Document loader port implementation.
            vector_store: Vector store port implementation.
        """
        self._document_loader = document_loader
        self._vector_store = vector_store

    async def upload_book(
        self,
        content: bytes,
        file_type: str,
        book_id: str,
        book_title: str,
    ) -> None:
        """Process and upload a book to the vector store.

        Args:
            content: Raw file content.
            file_type: File type (pdf or epub).
            book_id: Unique book identifier.
            book_title: Book title for metadata.

        Raises:
            HTTPException: If text extraction or embedding fails.
        """
        # Extract text from file
        try:
            text = self._document_loader.extract_text(content, file_type)
            logger.info(f"Extracted text from book '{book_title}' ({len(text)} characters)")
        except Exception as exc:
            logger.error(f"Failed to extract text from book '{book_title}': {exc!s}")
            raise DocumentProcessingError(
                message="Failed to extract text",
                details=str(exc),
            ) from exc

        if not text.strip():
            logger.warning(f"No text extracted from book '{book_title}'")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the file",
            )

        # Split text into chunks
        chunks = self._document_loader.split_text(text)
        logger.info(f"Split book '{book_title}' into {len(chunks)} chunks")

        # Add metadata to chunks
        self._add_metadata_to_chunks(chunks, book_id, book_title)

        # Add to vector store
        try:
            await self._vector_store.add_documents(chunks)
            logger.info(f"Stored {len(chunks)} embeddings for book '{book_title}'")
        except Exception as exc:
            logger.error(f"Failed to store embeddings for book '{book_title}': {exc!s}")
            raise VectorStoreError(
                message="Failed to process embeddings",
                details=str(exc),
            ) from exc

    def _add_metadata_to_chunks(
        self, chunks: list[Document], book_id: str, book_title: str
    ) -> None:
        """Add book metadata to document chunks.

        Args:
            chunks: List of document chunks.
            book_id: Unique book identifier.
            book_title: Book title.
        """
        for chunk in chunks:
            chunk.metadata["book_id"] = book_id
            chunk.metadata["book_title"] = book_title

    async def delete_book(self, book_id: str) -> None:
        """Delete all vector embeddings for a book.

        Args:
            book_id: Unique book identifier.
        """
        await self._vector_store.delete_by_book_id(book_id)
