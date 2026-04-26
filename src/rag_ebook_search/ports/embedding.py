"""Embedding port interface."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingPort(ABC):
    """Port for text embedding operations.

    This abstract base class defines the interface that all embedding
    implementations must follow. It allows the application to swap
    embedding providers without changing business logic.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        pass
