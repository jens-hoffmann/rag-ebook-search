"""Book management router."""

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_ebook_search.database import get_db
from rag_ebook_search.exceptions import DocumentProcessingError, VectorStoreError
from rag_ebook_search.logging_config import get_logger
from rag_ebook_search.services import get_document_loader, get_vector_store
from rag_ebook_search.models import Book
from rag_ebook_search.ports.document_loader import DocumentLoaderPort
from rag_ebook_search.ports.vector_store import VectorStorePort
from rag_ebook_search.schemas import BookListResponse, BookResponse
from rag_ebook_search.use_cases.upload import UploadUseCase

logger = get_logger(__name__)

router = APIRouter(prefix="/books", tags=["books"])


def _detect_file_type(filename: str) -> str:
    """Detect file type from filename extension."""
    if filename.lower().endswith(".pdf"):
        return "pdf"
    if filename.lower().endswith(".epub"):
        return "epub"
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Only PDF and EPUB files are supported",
    )


@router.post("/", response_model=BookResponse, status_code=status.HTTP_201_CREATED)
async def upload_book(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    author: str | None = Form(None),
    description: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    document_loader: DocumentLoaderPort = Depends(get_document_loader),
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> Book:
    """Upload a new PDF or EPUB book, parse it, chunk it, and store embeddings."""
    file_type = _detect_file_type(file.filename or "")

    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    # Create book record
    book = Book(
        title=title or (file.filename or "Untitled"),
        author=author,
        filename=file.filename or "untitled",
        file_type=file_type,
        description=description,
    )
    db.add(book)
    await db.commit()
    await db.refresh(book)

    # Use upload use case to process the book
    upload_use_case = UploadUseCase(
        document_loader=document_loader,
        vector_store=vector_store,
    )

    try:
        await upload_use_case.upload_book(
            content=content,
            file_type=file_type,
            book_id=book.id,
            book_title=book.title,
        )
        logger.info(f"Successfully uploaded book: {book.title} (ID: {book.id})")
    except DocumentProcessingError as exc:
        logger.error(f"Document processing failed for book '{book.title}': {exc.message}")
        await db.delete(book)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        ) from exc
    except VectorStoreError as exc:
        logger.error(f"Vector store error for book '{book.title}': {exc.message}")
        await db.delete(book)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except HTTPException:
        # Rollback: delete book record if processing fails
        await db.delete(book)
        await db.commit()
        raise

    return book


@router.get("/", response_model=BookListResponse)
async def list_books(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """List all uploaded books."""
    result = await db.execute(select(Book).offset(skip).limit(limit))
    books: List[Book] = list(result.scalars().all())
    total_result = await db.execute(select(Book))
    total = len(total_result.scalars().all())
    return {"books": books, "total": total}


@router.get("/{book_id}", response_model=BookResponse)
async def get_book(book_id: str, db: AsyncSession = Depends(get_db)) -> Book:
    """Get a single book by ID."""
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Book not found",
        )
    return book


@router.delete("/{book_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(
    book_id: str,
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> None:
    """Delete a book and all its vector embeddings."""
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Book not found",
        )

    # Delete vector chunks using the port
    await vector_store.delete_by_book_id(book_id)

    # Delete book record
    await db.delete(book)
    await db.commit()
