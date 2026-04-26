"""LangChain document loader adapter."""

import io
from typing import List

import ebooklib  # type: ignore[import-untyped]
from bs4 import BeautifulSoup
from ebooklib import epub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_ebook_search.config import Settings
from rag_ebook_search.ports.document_loader import DocumentLoaderPort


class LangChainDocumentLoaderAdapter(DocumentLoaderPort):
    """LangChain implementation of the DocumentLoaderPort.

    Supports PDF and EPUB file formats.
    """

    def __init__(self, config: Settings):
        """Initialize the document loader adapter.

        Args:
            config: Application settings.
        """
        self._config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_pdf_text(self, file_bytes: bytes) -> str:
        """Extract text from PDF bytes.

        Args:
            file_bytes: Raw PDF content.

        Returns:
            Extracted plain text.
        """
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)

    def _extract_epub_text(self, file_bytes: bytes) -> str:
        """Extract text from EPUB bytes.

        Args:
            file_bytes: Raw EPUB content.

        Returns:
            Extracted plain text.
        """
        book = epub.read_book(io.BytesIO(file_bytes))
        text_parts: List[str] = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)

    def extract_text(self, content: bytes, file_type: str) -> str:
        """Extract text from PDF or EPUB bytes.

        Args:
            content: Raw file content.
            file_type: Either "pdf" or "epub".

        Returns:
            Extracted plain text.

        Raises:
            ValueError: If file_type is not supported.
        """
        if file_type.lower() == "pdf":
            return self._extract_pdf_text(content)
        if file_type.lower() == "epub":
            return self._extract_epub_text(content)
        raise ValueError(f"Unsupported file type: {file_type}")

    def split_text(self, text: str) -> List[Document]:
        """Split text into chunks using RecursiveCharacterTextSplitter.

        Args:
            text: Full document text.

        Returns:
            List of Document chunks with metadata.
        """
        return self._splitter.create_documents([text])
