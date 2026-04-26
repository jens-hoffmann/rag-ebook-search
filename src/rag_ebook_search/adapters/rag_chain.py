"""LangChain RAG chain adapter."""

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_ebook_search.config import Settings
from rag_ebook_search.ports.llm import LLMPort
from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.ports.vector_store import VectorStorePort
from rag_ebook_search.schemas import RAGResponse, SearchResult

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided ebook excerpts.
Use only the information from the provided context to answer the question.
If the context does not contain enough information to answer the question, say so clearly.
Always cite the source excerpts you used in your answer.
"""

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)


class LangChainRAGChainAdapter(RAGChainPort):
    """LangChain implementation of the RAGChainPort.

    Combines vector search with LLM generation for RAG queries.
    """

    def __init__(
        self,
        config: Settings,
        vector_store: VectorStorePort,
        llm: LLMPort,
    ):
        """Initialize the RAG chain adapter.

        Args:
            config: Application settings.
            vector_store: Vector store port implementation.
            llm: LLM port implementation.
        """
        self._config = config
        self._vector_store = vector_store
        self._llm = llm
        self._prompt = _PROMPT_TEMPLATE

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a single context string.

        Args:
            documents: List of retrieved documents.

        Returns:
            Formatted context string.
        """
        parts: List[str] = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"[Excerpt {i}]\n{doc.page_content}\n")
        return "\n".join(parts)

    def _documents_to_sources(self, documents: List[Document]) -> List[SearchResult]:
        """Convert Document objects to SearchResult schemas.

        Args:
            documents: List of retrieved documents.

        Returns:
            List of search results.
        """
        sources: List[SearchResult] = []
        for doc in documents:
            meta = doc.metadata or {}
            sources.append(
                SearchResult(
                    book_id=meta.get("book_id", ""),
                    book_title=meta.get("book_title", "Unknown"),
                    chunk_text=doc.page_content,
                    score=meta.get("score", 0.0),
                )
            )
        return sources

    async def ask(
        self, question: str, book_id: str | None = None, top_k: int = 5
    ) -> RAGResponse:
        """Execute the RAG pipeline: retrieve and generate answer.

        Args:
            question: User question.
            book_id: Optional book ID to scope retrieval.
            top_k: Number of chunks to retrieve.

        Returns:
            RAGResponse with answer and source excerpts.
        """
        # Retrieve relevant chunks
        documents = await self._vector_store.search(
            question, book_id=book_id, top_k=top_k
        )

        # Format context
        context = self._format_context(documents)

        # Generate answer using LLM
        answer = await self._llm.generate_with_context(context, question)

        # Convert documents to sources
        sources = self._documents_to_sources(documents)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
        )
