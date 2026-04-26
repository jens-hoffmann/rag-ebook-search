"""Document loader port interface."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class DocumentLoaderPort(ABC):
    """Port for document loading and text extraction.

    This abstract base class defines the interface that all document
    loader implementations must follow. It allows the application to
    swap document loaders without changing business logic.
    """

    @abstractmethod
    def extract_text(self, content: bytes, file_type: str) -> str:
        """Extract text from file content.

        Args:
            content: Raw file content as bytes.
            file_type: File type (e.g., "pdf", "epub").

        Returns:
            Extracted plain text.

        Raises:
            ValueError: If file type is not supported.
        """
        pass

    @abstractmethod
    def split_text(self, text: str) -> List[Document]:
        """Split text into chunks with metadata.

        Args:
            text: Full document text to split.

        Returns:
            List of Document chunks.
        """
        pass
