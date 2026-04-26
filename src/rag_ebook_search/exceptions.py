"""Custom exceptions for the RAG Ebook Search application."""


class RAGApplicationError(Exception):
    """Base exception for all RAG application errors."""

    def __init__(self, message: str, details: str | None = None):
        """Initialize the exception.

        Args:
            message: User-friendly error message.
            details: Additional technical details (optional).
        """
        self.message = message
        self.details = details
        super().__init__(self.message)


class DocumentProcessingError(RAGApplicationError):
    """Raised when document text extraction or processing fails."""

    def __init__(self, message: str = "Failed to process document", details: str | None = None):
        super().__init__(message, details)


class EmbeddingError(RAGApplicationError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str = "Failed to generate embeddings", details: str | None = None):
        super().__init__(message, details)


class VectorStoreError(RAGApplicationError):
    """Raised when vector store operations fail."""

    def __init__(self, message: str = "Failed to process embeddings", details: str | None = None):
        super().__init__(message, details)


class DatabaseError(RAGApplicationError):
    """Raised when database operations fail."""

    def __init__(self, message: str = "Database operation failed", details: str | None = None):
        super().__init__(message, details)


class LLMError(RAGApplicationError):
    """Raised when LLM operations fail."""

    def __init__(self, message: str = "LLM operation failed", details: str | None = None):
        super().__init__(message, details)


class BookNotFoundError(RAGApplicationError):
    """Raised when a requested book is not found."""

    def __init__(self, book_id: str):
        super().__init__(f"Book not found: {book_id}", details=f"Book ID: {book_id}")


class InvalidFileTypeError(RAGApplicationError):
    """Raised when an unsupported file type is uploaded."""

    def __init__(self, file_type: str):
        super().__init__(
            f"Unsupported file type: {file_type}",
            details=f"Supported types: PDF, EPUB"
        )
