import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pdf_processing import extract_pdf_pages, chunk_pages
from src.embeddings import (
    load_embedding_model,
    create_embeddings,
    build_faiss_index,
)
from src.retrieval import load_reranker, retrieve
from src.generation import generate_answer


load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents. "
    "Answers are generated only from retrieved document evidence."
)


@st.cache_resource
def load_models():
    embedding_model = load_embedding_model()
    reranker = load_reranker()
    return embedding_model, reranker


@st.cache_resource(show_spinner=False)
def process_document(pdf_bytes):
    pages = extract_pdf_pages(pdf_bytes)
    chunks = chunk_pages(pages)

    embeddings = create_embeddings(
        chunks,
        embedding_model,
    )

    index = build_faiss_index(embeddings)

    return pages, chunks, index


embedding_model, reranker = load_models()


def get_api_key():
    local_key = os.getenv("OPENAI_API_KEY")

    if local_key:
        return local_key

    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


api_key = get_api_key()


with st.sidebar:
    st.header("About")

    st.write(
        "This application uses a Retrieval-Augmented Generation pipeline:"
    )

    st.markdown(
        """
        1. PDF text extraction  
        2. Chunking with overlap  
        3. Sentence embeddings  
        4. FAISS vector search  
        5. Cross-Encoder reranking  
        6. Grounded LLM generation  
        """
    )

    st.caption(
        "The system is instructed to refuse questions that are not "
        "supported by the document."
    )


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("Processing document..."):
        pages, chunks, index = process_document(pdf_bytes)

    st.success(
        f"Document ready — {len(pages)} pages with text, "
        f"{len(chunks)} searchable chunks."
    )

    question = st.text_input(
        "Ask a question about the document",
        placeholder="Example: What causes lift on an airplane wing?",
    )

    if st.button(
        "Ask",
        type="primary",
        disabled=not question.strip(),
    ):
        if not api_key:
            st.error(
                "OPENAI_API_KEY is not configured."
            )
            st.stop()

        with st.spinner("Searching the document..."):
            results = retrieve(
                question,
                chunks,
                embedding_model,
                index,
                reranker,
                top_k=3,
                candidate_k=10,
            )

        with st.spinner("Generating grounded answer..."):
            answer = generate_answer(
                question,
                results,
                api_key,
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved evidence")

        for rank, result in enumerate(results, start=1):
            with st.expander(
                f"Source {rank} — Page {result['page']}"
            ):
                st.write(result["text"])

                st.caption(
                    f"Semantic score: "
                    f"{result['semantic_score']:.3f} · "
                    f"Reranker score: "
                    f"{result['rerank_score']:.3f}"
                )

else:
    st.info("Upload a PDF to begin.")
