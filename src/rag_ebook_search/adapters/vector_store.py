"""LangChain vector store adapter."""

from typing import List

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import text

from rag_ebook_search.config import Settings
from rag_ebook_search.database import engine
from rag_ebook_search.ports.embedding import EmbeddingPort
from rag_ebook_search.ports.vector_store import VectorStorePort


def _get_sync_connection_string(database_url: str) -> str:
    """Convert asyncpg connection string to psycopg2 for PGVector.

    Args:
        database_url: Asyncpg connection URL.

    Returns:
        Psycopg2-compatible connection URL.
    """
    return database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


class LangChainVectorStoreAdapter(VectorStorePort):
    """LangChain implementation of the VectorStorePort.

    Uses PGVector with PostgreSQL for vector similarity search.
    """

    def __init__(
        self,
        config: Settings,
        embedding: EmbeddingPort,
        collection_name: str = "ebook_chunks",
    ):
        """Initialize the vector store adapter.

        Args:
            config: Application settings.
            embedding: Embedding port implementation.
            collection_name: Name of the collection/table.
        """
        self._config = config
        self._embedding = embedding
        self._collection_name = collection_name
        self._store: PGVector | None = None

    def _get_store(self) -> PGVector:
        """Get or create the PGVector store instance.

        Returns:
            Configured PGVector instance.
        """
        if self._store is None:
            connection_string = _get_sync_connection_string(self._config.database_url)
            # Create a wrapper that uses our embedding port
            self._store = PGVector(
                connection=connection_string,
                embeddings=self._embedding,  # type: ignore[arg-type]
                collection_name=self._collection_name,
                use_jsonb=True,
            )
        return self._store

    async def add_documents(self, documents: List[Document]) -> None:
        """Add document chunks to the vector store.

        Args:
            documents: List of Document chunks to embed and store.
        """
        store = self._get_store()
        store.add_documents(documents)

    async def search(
        self, query: str, book_id: str | None = None, top_k: int = 5
    ) -> List[Document]:
        """Search for similar documents in the vector store.

        Args:
            query: Search query text.
            book_id: Optional book ID to filter results.
            top_k: Number of results to return.

        Returns:
            List of matching Document chunks.
        """
        store = self._get_store()
        filter_dict: dict | None = None
        if book_id:
            filter_dict = {"book_id": book_id}

        return store.similarity_search(query, k=top_k, filter=filter_dict)

    async def delete_by_book_id(self, book_id: str) -> None:
        """Delete all chunks belonging to a specific book.

        Args:
            book_id: ID of the book whose chunks should be removed.
        """
        async with engine.connect() as conn:
            await conn.execute(
                text(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'book_id' = :book_id
                    """
                ),
                {"book_id": book_id},
            )
            await conn.commit()
