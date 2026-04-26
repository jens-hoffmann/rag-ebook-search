"""Streamlit frontend for the RAG Ebook Search application."""

import os

import httpx
import streamlit as st

from rag_ebook_search.logging_config import setup_logging

# Initialize logging
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Ebook Search",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Ebook Search")
st.markdown("Upload your PDF or EPUB ebooks, search them, and ask questions with AI.")


def _api_client() -> httpx.Client:
    """Return a configured HTTP client."""
    return httpx.Client(base_url=API_BASE_URL, timeout=120.0)


def _get_error_message(status_code: int, response_text: str) -> str:
    """Convert API error response to user-friendly message.

    Args:
        status_code: HTTP status code.
        response_text: Response body text.

    Returns:
        User-friendly error message.
    """
    try:
        import json
        error_data = json.loads(response_text)
        detail = error_data.get("detail", response_text)
    except (json.JSONDecodeError, TypeError):
        detail = response_text

    # Map common errors to user-friendly messages
    if "Failed to extract text" in detail:
        return "📄 Could not read the file. Make sure it's a valid PDF or EPUB."
    if "Failed to process embeddings" in detail or "Connection error" in detail:
        return "🔌 Could not connect to the embedding service. Please try again."
    if "No text could be extracted" in detail:
        return "📄 The file appears to be empty or has no readable text."
    if "Unsupported file type" in detail:
        return "❌ Only PDF and EPUB files are supported."

    return f"❌ An error occurred (code: {status_code})"


# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Upload Book", "My Books", "Search", "RAG Chat"],
)

if page == "Upload Book":
    st.header("Upload a New Book")

    uploaded_file = st.file_uploader("Choose a PDF or EPUB file", type=["pdf", "epub"])
    title = st.text_input("Title (optional)")
    author = st.text_input("Author (optional)")
    description = st.text_area("Description (optional)")

    if st.button("Upload", disabled=uploaded_file is None):
        if uploaded_file is not None:
            with st.spinner("Processing book... This may take a minute."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {}
                if title:
                    data["title"] = title
                if author:
                    data["author"] = author
                if description:
                    data["description"] = description

                try:
                    with _api_client() as client:
                        response = client.post("/books/", files=files, data=data)
                    if response.status_code == 201:
                        book = response.json()
                        st.success(f"✅ Uploaded '{book['title']}' successfully!")
                    else:
                        st.error(_get_error_message(response.status_code, response.text))
                except httpx.ConnectError:
                    st.error("🔌 Could not connect to the API server. Is it running?")
                except httpx.TimeoutException:
                    st.error("⏱️ The request timed out. The book may be too large.")
                except Exception as exc:
                    st.error(f"❌ An unexpected error occurred: {exc!s}")

elif page == "My Books":
    st.header("My Books")

    try:
        with _api_client() as client:
            response = client.get("/books/")
        if response.status_code == 200:
            data = response.json()
            books = data.get("books", [])
            total = data.get("total", 0)
            st.write(f"Total books: {total}")

            for book in books:
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.subheader(book["title"])
                        st.caption(f"Author: {book.get('author') or 'Unknown'}")
                        st.caption(f"Type: {book['file_type'].upper()}")
                        if book.get("description"):
                            st.write(book["description"])
                    with col2:
                        if st.button("Delete", key=f"del_{book['id']}"):
                            try:
                                with _api_client() as client:
                                    del_resp = client.delete(f"/books/{book['id']}")
                                if del_resp.status_code == 204:
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.error(_get_error_message(del_resp.status_code, del_resp.text))
                            except httpx.ConnectError:
                                st.error("🔌 Could not connect to the API server.")
                            except Exception as exc:
                                st.error(f"❌ An unexpected error occurred: {exc!s}")
        else:
            st.error(_get_error_message(response.status_code, response.text))
    except httpx.ConnectError:
        st.error("🔌 Could not connect to the API server. Is it running?")
    except Exception as exc:
        st.error(f"❌ An unexpected error occurred: {exc!s}")

elif page == "Search":
    st.header("Semantic Search")

    query = st.text_input("Search query")
    limit = st.slider("Number of results", min_value=1, max_value=20, value=5)

    if st.button("Search", disabled=not query):
        with st.spinner("Searching..."):
            try:
                with _api_client() as client:
                    response = client.get(
                        "/search/",
                        params={"q": query, "limit": limit},
                    )
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    st.write(f"Found {len(results)} results for _{data['query']}_")

                    for i, res in enumerate(results, 1):
                        with st.container(border=True):
                            st.markdown(f"**#{i} — {res['book_title']}**")
                            st.markdown(f"Score: `{res['score']:.4f}`")
                            st.text(res["chunk_text"][:1000])
                else:
                    st.error(_get_error_message(response.status_code, response.text))
            except httpx.ConnectError:
                st.error("🔌 Could not connect to the API server.")
            except httpx.TimeoutException:
                st.error("⏱️ Search request timed out.")
            except Exception as exc:
                st.error(f"❌ An unexpected error occurred: {exc!s}")

elif page == "RAG Chat":
    st.header("💬 RAG Chat")
    st.markdown("Ask questions about your books and get AI-powered answers.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your books..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    with _api_client() as client:
                        response = client.post(
                            "/rag/",
                            json={"question": prompt, "top_k": 5},
                        )
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            with st.expander("📎 Sources"):
                                for i, src in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}** — {src['book_title']}")
                                    st.text(src["chunk_text"][:500])

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(_get_error_message(response.status_code, response.text))
                except httpx.ConnectError:
                    st.error("🔌 Could not connect to the API server.")
                except httpx.TimeoutException:
                    st.error("⏱️ RAG request timed out. The service may be overloaded.")
                except Exception as exc:
                    st.error(f"❌ An unexpected error occurred: {exc!s}")
