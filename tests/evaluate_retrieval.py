from pathlib import Path

from src.pdf_processing import extract_pdf_pages, chunk_pages
from src.embeddings import (
    load_embedding_model,
    create_embeddings,
    build_faiss_index,
)
from src.retrieval import load_reranker, retrieve


PDF_PATH = Path("data/pdfs/nasa-aeronautics-guide.pdf")

TEST_CASES = [
    {
        "question": "What causes lift on an airplane wing?",
        "keywords": ["pressure", "lift"],
    },
    {
        "question": "What is thrust?",
        "keywords": ["thrust", "forward"],
    },
    {
        "question": "How does air pressure affect flight?",
        "keywords": ["pressure", "lift"],
    },
    {
        "question": "What is drag?",
        "keywords": ["drag", "resistance"],
    },
    {
        "question": "What is a glider?",
        "keywords": ["glider", "engine"],
    },
]


pages = extract_pdf_pages(PDF_PATH.read_bytes())
chunks = chunk_pages(pages)

embedding_model = load_embedding_model()
embeddings = create_embeddings(chunks, embedding_model)
index = build_faiss_index(embeddings)
reranker = load_reranker()

hits = 0

for test in TEST_CASES:
    results = retrieve(
        test["question"],
        chunks,
        embedding_model,
        index,
        reranker,
        top_k=3,
        candidate_k=10,
    )

    combined_text = " ".join(
        result["text"].lower()
        for result in results
    )

    success = all(
        keyword.lower() in combined_text
        for keyword in test["keywords"]
    )

    if success:
        hits += 1

    print("=" * 90)
    print("QUESTION:", test["question"])
    print("RESULT:", "PASS" if success else "FAIL")
    print(
        "TOP 3 PAGES:",
        [result["page"] for result in results]
    )

print()
print("=" * 90)
print(f"Retrieval Hit@3: {hits}/{len(TEST_CASES)}")
print(
    f"Retrieval accuracy: "
    f"{hits / len(TEST_CASES) * 100:.1f}%"
)
