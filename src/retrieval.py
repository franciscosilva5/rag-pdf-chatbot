from sentence_transformers import CrossEncoder


RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    return CrossEncoder(RERANKER_NAME)


def retrieve(
    query,
    chunks,
    embedding_model,
    index,
    reranker,
    top_k=3,
    candidate_k=10,
):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    candidate_k = min(candidate_k, len(chunks))

    semantic_scores, indices = index.search(
        query_embedding,
        candidate_k,
    )

    candidates = []

    for semantic_score, index_position in zip(
        semantic_scores[0],
        indices[0],
    ):
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        candidates.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "semantic_score": float(semantic_score),
        })

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    for candidate, rerank_score in zip(
        candidates,
        rerank_scores,
    ):
        candidate["rerank_score"] = float(rerank_score)

    candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return candidates[:top_k]
