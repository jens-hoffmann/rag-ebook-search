"""Pydantic schemas for request and response validation."""

from datetime import datetime

from pydantic import BaseModel, Field


class BookBase(BaseModel):
    """Base book schema."""

    title: str = Field(..., min_length=1, max_length=500)
    author: str | None = Field(None, max_length=500)
    description: str | None = Field(None)


class BookCreate(BookBase):
    """Schema for creating a book."""

    pass


class BookResponse(BookBase):
    """Schema for book response."""

    id: str
    filename: str
    file_type: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BookListResponse(BaseModel):
    """Schema for listing books."""

    books: list[BookResponse]
    total: int


class SearchResult(BaseModel):
    """Schema for a single search result."""

    book_id: str
    book_title: str
    chunk_text: str
    score: float


class SearchResponse(BaseModel):
    """Schema for search response."""

    query: str
    results: list[SearchResult]


class RAGRequest(BaseModel):
    """Schema for RAG query request."""

    question: str = Field(..., min_length=1)
    book_id: str | None = Field(None)
    top_k: int = Field(5, ge=1, le=20)


class RAGResponse(BaseModel):
    """Schema for RAG query response."""

    question: str
    answer: str
    sources: list[SearchResult]
