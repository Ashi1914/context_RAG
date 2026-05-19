"""
Frontend Application for RAG System
Streamlit UI for:
- Document upload
- Vector search
- RAG content generation
"""

import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom modules
from document_loader import load_and_split
from vector_store import (
    VectorStore,
    create_vector_store
)
from generator import (
    create_rag_generator,
    generate_content_with_context
)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_documents"

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="RAG Content Generator",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

def apply_custom_css():

    st.markdown("""
    <style>

    html, body, [class*="css"]  {
        font-family: Arial, sans-serif;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """, unsafe_allow_html=True)


apply_custom_css()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🤖 RAG System")

page = st.sidebar.radio(
    "Navigation",
    [
        "📚 Upload Documents",
        "🔍 Search",
        "✨ Generate",
        "📊 Dashboard"
    ]
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

def init_session_state():
    """Initialize Streamlit session state."""

    defaults = {
        "documents": [],
        "vector_store": None,
        "generator": None,
        "search_results": [],
        "uploaded_files": set()
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value

    # Auto-load persisted vector DB
    if (
        st.session_state.vector_store is None
        and os.path.exists(CHROMA_DIR)
    ):

        try:

            vector_store = VectorStore(
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_DIR
            )

            st.session_state.vector_store = vector_store

            st.session_state.generator = (
                create_rag_generator(
                    vector_store
                )
            )

        except Exception as e:

            st.warning(
                f"Auto-load failed: {str(e)}"
            )

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file safely.

    Returns:
        Temporary file path
    """

    suffix = (
        ".pdf"
        if uploaded_file.name.endswith(".pdf")
        else ".docx"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())

        return tmp.name


def build_vector_store():
    """Build vector database."""

    vector_store = create_vector_store(
        documents=st.session_state.documents,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        clear_existing=False
    )

    st.session_state.vector_store = vector_store

    st.session_state.generator = (
        create_rag_generator(vector_store)
    )

# ---------------------------------------------------
# UPLOAD PAGE
# ---------------------------------------------------

def page_upload_documents():

    st.header("📚 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:

        total_chunks = 0

        for uploaded_file in uploaded_files:

            if (
                uploaded_file.name
                in st.session_state.uploaded_files
            ):

                st.info(
                    f"Skipped duplicate: "
                    f"{uploaded_file.name}"
                )

                continue

            temp_path = None

            try:

                temp_path = save_uploaded_file(
                    uploaded_file
                )

                chunks = load_and_split(temp_path)

                # Add metadata
                for chunk in chunks:

                    chunk.metadata["file_name"] = (
                        uploaded_file.name
                    )

                st.session_state.documents.extend(
                    chunks
                )

                st.session_state.uploaded_files.add(
                    uploaded_file.name
                )

                total_chunks += len(chunks)

            except Exception as e:

                st.error(
                    f"Failed processing "
                    f"{uploaded_file.name}: {str(e)}"
                )

            finally:

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)

        st.success(
            f"Added {total_chunks} chunks."
        )

    # Build vector store
    if st.session_state.documents:

        st.info(
            f"Ready for indexing: "
            f"{len(st.session_state.documents)} chunks"
        )

        if st.button("🚀 Build Index"):

            with st.spinner(
                "Building vector embeddings..."
            ):

                try:

                    build_vector_store()

                    st.success(
                        "Vector DB created successfully."
                    )

                except Exception as e:

                    st.error(str(e))

# ---------------------------------------------------
# SEARCH PAGE
# ---------------------------------------------------

def page_search():

    st.header("🔍 Semantic Search")

    if st.session_state.vector_store is None:

        st.warning(
            "Please build vector DB first."
        )

        return

    query = st.text_input(
        "Search Query"
    )

    if st.button("Search"):

        if not query.strip():

            st.warning(
                "Please enter a query."
            )

            return

        try:

            results = (
                st.session_state.vector_store
                .similarity_search_with_score(
                    query=query,
                    k=5
                )
            )

            st.session_state.search_results = (
                results
            )

            if not results:

                st.warning("No results found.")

                return

            for i, (doc, score) in enumerate(
                results,
                start=1
            ):

                st.markdown(f"""

                **File:**  
                {doc.metadata.get("file_name", "Unknown")}

                **Content:**  
                {doc.page_content[:500]}
                """)

                st.divider()

        except Exception as e:

            st.error(f"Search failed: {str(e)}")

# ---------------------------------------------------
# GENERATION PAGE
# ---------------------------------------------------

def page_generate():

    st.header("✨ Content Generation")

    if st.session_state.vector_store is None:

        st.warning(
            "Please build vector DB first."
        )

        return

    content_type = st.selectbox(
        "Content Type",
        [
            "Summary",
            "FAQ",
            "Article",
            "Report"
        ]
    )

    tone = st.selectbox(
        "Tone",
        [
            "Professional",
            "Creative",
            "Analytical",
            "Concise"
        ]
    )

    topic = st.text_input(
        "Topic"
    )

    keywords_input = st.text_input(
        "Keywords (comma-separated)"
    )

    if st.button("Generate"):

        if not topic.strip():

            st.warning(
                "Please enter topic."
            )

            return

        keywords = [k.strip() for k in keywords_input.split(",")] if keywords_input.strip() else []

        try:

            # Recreate generator
            st.session_state.generator = (
                create_rag_generator(
                    st.session_state.vector_store,
                    max_tokens=1000
                )
            )

            retriever = (
                st.session_state.vector_store
                .as_retriever(
                    search_kwargs={"k": 5}
                )
            )

            response = (
                generate_content_with_context(
                    content_type=content_type,
                    tone=tone,
                    topic=topic,
                    keywords=keywords,
                    retriever=retriever
                )
            )

            st.success("Generation complete.")

            st.markdown("## Output")

            st.write(response)

            st.download_button(
                label="📥 Download",
                data=response,
                file_name="generated_content.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error(
                f"Generation failed: {str(e)}"
            )

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------

def page_dashboard():

    st.header("📊 Dashboard")

    total_chunks = len(
        st.session_state.documents
    )

    indexed_chunks = 0

    try:

        if st.session_state.vector_store:

            indexed_chunks = (
                st.session_state.vector_store
                .get_collection_count()
            )

    except Exception:
        indexed_chunks = 0

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Loaded Chunks",
            total_chunks
        )

    with col2:

        st.metric(
            "Indexed Chunks",
            indexed_chunks
        )

    with col3:

        st.metric(
            "Search Results",
            len(st.session_state.search_results)
        )

    st.divider()

    st.subheader("System Status")

    statuses = {
        "Vector Store":
            st.session_state.vector_store is not None,

        "Generator":
            st.session_state.generator is not None,

        "GROQ API":
            bool(os.getenv("GROQ_API_KEY"))
    }

    for key, value in statuses.items():

        status = (
            "✅ Connected"
            if value
            else "❌ Missing"
        )

        st.write(f"**{key}:** {status}")

    st.divider()

    if st.button("🗑️ Clear Database"):

        try:

            if st.session_state.vector_store:

                st.session_state.vector_store.delete_all()

            st.session_state.documents = []

            st.session_state.search_results = []

            st.session_state.uploaded_files = set()

            st.success("Database cleared.")

        except Exception as e:

            st.error(str(e))

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    init_session_state()

    st.title("🤖 RAG Content Engine")

    st.markdown(
        "RAG System using "
        "LangChain + ChromaDB + Groq"
    )

    if page == "📚 Upload Documents":

        page_upload_documents()

    elif page == "🔍 Search":

        page_search()

    elif page == "✨ Generate":

        page_generate()

    elif page == "📊 Dashboard":

        page_dashboard()


if __name__ == "__main__":

    main()