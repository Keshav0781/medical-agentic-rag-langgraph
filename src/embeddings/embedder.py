import logging
from sentence_transformers import SentenceTransformer
from configs.config import EMBEDDING_MODEL, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def generate_embeddings(chunks: list) -> list:
    """
    Converts text chunks into vector embeddings.
    Each chunk gets a vector of 768 numbers representing its meaning.
    Uses sentence-transformers all-mpnet-base-v2 locally.
    """
    logger.info(f"Generating embeddings for {len(chunks)} chunks")

    texts = [chunk["text"] for chunk in chunks]

    vectors = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = vectors[i].tolist()

    logger.info(f"Generated {len(vectors)} embeddings successfully")

    return chunks