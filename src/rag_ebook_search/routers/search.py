"""Search router for vector similarity search."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_ebook_search.database import get_db
from rag_ebook_search.services import get_vector_store
from rag_ebook_search.models import Book
from rag_ebook_search.ports.vector_store import VectorStorePort
from rag_ebook_search.schemas import SearchResponse
from rag_ebook_search.use_cases.search import SearchUseCase

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    book_id: str | None = Query(None, description="Filter by book ID"),
    limit: int = Query(5, ge=1, le=20, description="Number of results"),
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> SearchResponse:
    """Search for text similar to the query across all ebook chunks."""
    # Validate book_id if provided
    if book_id:
        result = await db.execute(select(Book).where(Book.id == book_id))
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Book not found",
            )

    # Use search use case
    search_use_case = SearchUseCase(vector_store=vector_store)
    return await search_use_case.search(query=q, book_id=book_id, limit=limit)
