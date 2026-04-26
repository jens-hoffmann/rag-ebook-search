# 📚 RAG Ebook Search

A RAG (Retrieval-Augmented Generation) application that makes your PDF and EPUB ebooks searchable and chat-able using local LLMs via LMStudio.

## Features

- **Upload** PDF and EPUB ebooks
- **Semantic search** across all your books using vector embeddings
- **RAG Chat** — ask questions and get AI-powered answers with cited sources
- **Streamlit UI** for easy interaction
- **FastAPI backend** with async database support
- **VectorChord** extension on PostgreSQL for vector storage
- **LMStudio** integration for both embeddings and LLM inference
- **Ports & Adapters architecture** for clean separation of concerns
- **Centralized logging and error handling** with custom exceptions

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│   Postgres  │
│     UI      │     │   Backend   │     │ +VectorChord│
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │   LMStudio  │
                    │ (Embeddings │
                    │   + LLM)    │
                    └─────────────┘
```

### Ports & Adapters Structure

```
src/rag_ebook_search/
├── main.py              # FastAPI app entry point
├── streamlit_app.py     # Streamlit UI
├── config.py            # Settings (from env)
├── database.py          # Async SQLAlchemy engine
├── models.py            # SQLAlchemy ORM models
├── schemas.py           # Pydantic request/response schemas
├── exceptions.py        # Custom exception hierarchy
├── logging_config.py   # Centralized logging setup
├── routers/            # FastAPI route handlers
│   ├── books.py        # Book CRUD endpoints
│   ├── search.py       # Vector search endpoint
│   └── rag.py          # RAG Q&A endpoint
├── ports/              # Interface definitions (hexagon core)
│   ├── document_loader.py
│   ├── embedding.py
│   ├── llm.py
│   ├── vector_store.py
│   └── rag_chain.py
├── adapters/           # Implementation of ports
│   ├── document_loader.py  # PDF/EPUB parsing (LangChain)
│   ├── embedding.py         # LMStudio embeddings
│   ├── llm.py              # LMStudio LLM
│   ├── vector_store.py      # PGVector store
│   └── rag_chain.py         # RAG pipeline
├── services/            # Dependency injection & FastAPI deps
│   ├── container.py     # DI container
│   └── fastapi_deps.py  # FastAPI dependency providers
└── use_cases/          # Application business logic
    ├── upload.py        # Book upload orchestration
    ├── search.py        # Search orchestration
    └── rag.py           # RAG orchestration
```

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for package management
- [Podman](https://podman.io/) (or Docker) for containers
- PostgreSQL with [VectorChord](https://github.com/tensorchord/VectorChord) extension
- [LMStudio](https://lmstudio.ai/) running locally with embedding and LLM models loaded

## Setup

### 1. Clone and install dependencies

```bash
uv pip install -e ".[dev]"
```

### 2. Configure environment

Copy the example env file and adjust values:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `DATABASE_URL` — your PostgreSQL connection string
- `LMSTUDIO_BASE_URL` — LMStudio local API (default: `http://localhost:1234/v1`)
- `EMBEDDING_MODEL` and `LLM_MODEL` — model names loaded in LMStudio

### 3. Run the backend

```bash
uvicorn rag_ebook_search.main:app --reload
```

Or with Python module path:

```bash
PYTHONPATH=src uvicorn rag_ebook_search.main:app --reload
```

### 4. Run the Streamlit frontend

In a new terminal:

```bash
PYTHONPATH=src API_BASE_URL=http://localhost:8000 streamlit run src/rag_ebook_search/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/books/` | Upload PDF/EPUB |
| GET | `/books/` | List all books |
| GET | `/books/{id}` | Get book details |
| DELETE | `/books/{id}` | Delete book + embeddings |
| GET | `/search/?q=...` | Semantic search |
| POST | `/rag/` | RAG Q&A |
| GET | `/health` | Health check |

## Testing with tox

Run all test environments:

```bash
tox
```

Run specific environments:

```bash
tox -e test        # unit tests
tox -e integration # integration tests with PostgreSQL
tox -e lint        # ruff linting
tox -e type        # mypy type checking
```

## Container Deployment

Build and run with Podman Compose:

```bash
podman-compose up --build
```

Or with podman directly:

```bash
podman build -t rag-ebook-search -f Containerfile .
podman run -p 8000:8000 --env-file .env rag-ebook-search
```

## LMStudio Setup

1. Open LMStudio
2. Download and load an **embedding model** (e.g. `nomic-embed-text-v1.5`)
3. Download and load a **chat LLM** (e.g. `Llama-3.2-3B-Instruct`)
4. Start the local server in LMStudio (default port 1234)

## Error Handling

The application uses a custom exception hierarchy for structured error handling:

- `RAGApplicationError` — base exception
- `DocumentProcessingError` — PDF/EPUB parsing failures
- `VectorStoreError` — embedding/vector storage failures
- `DatabaseError` — database operation failures
- `LLMError` — LLM inference failures
- `BookNotFoundError` — requested book not found
- `InvalidFileTypeError` — unsupported file type

Logs are centralized via `logging_config.py` with configurable log levels via `LOG_LEVEL` environment variable.

## License

MIT
