import logging
from sentence_transformers import CrossEncoder
from configs.config import TOP_K_RERANK, RERANKER_MODEL, RETRIEVAL_SCORE_THRESHOLD, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

reranker = CrossEncoder(RERANKER_MODEL)

def _clean_text(text: str) -> str:
    """
    Cleans extracted PDF text before reranking.
    Removes excessive whitespace and newlines from table extraction.
    """
    import re
    # Replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def rerank_chunks(query: str, chunks: list) -> list:
    """
    Reranks retrieved chunks using CrossEncoder model.
    CrossEncoder reads query and chunk together for precise relevance scoring.
    Returns top_k most relevant chunks above score threshold.
    At Siemens same CrossEncoder pattern used with enterprise Qdrant results.
    """
    if not chunks:
        logger.warning("No chunks to rerank — returning empty list")
        return []

    # Clean text before reranking
    pairs = [[query, _clean_text(chunk["text"])] for chunk in chunks]

    # Score each pair
    scores = reranker.predict(pairs)

    # Attach scores to chunks
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    # Sort by score descending
    ranked_chunks = sorted(
        chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Filter by threshold and keep top_k
    filtered_chunks = [
        chunk for chunk in ranked_chunks
        if chunk["rerank_score"] >= RETRIEVAL_SCORE_THRESHOLD
    ][:TOP_K_RERANK]

    logger.info(
        f"Reranked {len(chunks)} chunks → "
        f"kept {len(filtered_chunks)} above threshold {RETRIEVAL_SCORE_THRESHOLD}"
    )

    return filtered_chunks