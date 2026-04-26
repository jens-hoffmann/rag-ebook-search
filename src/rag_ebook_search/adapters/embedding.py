"""LangChain embedding adapter."""

from typing import List

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from rag_ebook_search.config import Settings
from rag_ebook_search.ports.embedding import EmbeddingPort


class LangChainEmbeddingAdapter(EmbeddingPort):
    """LangChain implementation of the EmbeddingPort.

    Uses OpenAI-compatible embeddings via LMStudio or other providers.
    """

    def __init__(self, config: Settings):
        """Initialize the embedding adapter.

        Args:
            config: Application settings.
        """
        self._config = config
        self._embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            base_url=config.lmstudio_base_url,
            api_key=SecretStr(config.lmstudio_api_key),
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._embeddings.embed_documents(texts)

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return await self._embeddings.aembed_query(text)
