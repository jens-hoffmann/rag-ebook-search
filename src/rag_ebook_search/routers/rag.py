"""RAG Q&A router."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_ebook_search.database import get_db
from rag_ebook_search.services import get_rag_chain
from rag_ebook_search.models import Book
from rag_ebook_search.ports.rag_chain import RAGChainPort
from rag_ebook_search.schemas import RAGRequest, RAGResponse
from rag_ebook_search.use_cases.rag import RAGUseCase

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/", response_model=RAGResponse)
async def rag_query(
    request: RAGRequest,
    db: AsyncSession = Depends(get_db),
    rag_chain: RAGChainPort = Depends(get_rag_chain),
) -> RAGResponse:
    """Ask a question and get an answer based on ebook content."""
    # Validate book_id if provided
    if request.book_id:
        result = await db.execute(select(Book).where(Book.id == request.book_id))
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Book not found",
            )

    # Use RAG use case
    rag_use_case = RAGUseCase(rag_chain=rag_chain)
    return await rag_use_case.ask(
        question=request.question,
        book_id=request.book_id,
        top_k=request.top_k,
    )
