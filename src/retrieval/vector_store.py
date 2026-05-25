import logging
from qdrant_client import QdrantClient
from configs.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    LOG_LEVEL
)
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection():
    """
    Creates Qdrant collection if it doesn't already exist.
    Collection stores 768-dim vectors with cosine similarity.
    At Siemens collection was pre-created by infrastructure team.
    """
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    if QDRANT_COLLECTION_NAME in collection_names:
        logger.info(f"Collection '{QDRANT_COLLECTION_NAME}' already exists")
        return

    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=Distance.COSINE
        )
    )
    logger.info(f"Created collection '{QDRANT_COLLECTION_NAME}'")


def create_payload_index():
    """
    Creates payload index on 'source' field for metadata filtering.
    Required for summariser to fetch chunks by document name.
    At Siemens payload indexes were created during collection setup
    by infrastructure team — engineers never did this manually.
    """
    try:
        qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logger.info("Payload index created on 'source' field")
    except Exception as e:
        logger.warning(f"Payload index creation: {e}")


def store_chunks(chunks: list, batch_size: int = 100):
    """
    Stores chunks with embeddings in Qdrant collection.
    Uploads in batches to respect Qdrant Cloud payload size limits.
    Includes basic checkpoint logging for resumability.
    At Siemens full checkpoint system tracked completed batches
    to avoid re-processing on failure with 1M+ chunks.
    """
    points = []

    for i, chunk in enumerate(chunks):
        point = PointStruct(
            id=i,
            vector=chunk["embedding"],
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "total_pages": chunk["total_pages"],
                "document_type": chunk["document_type"],
                "chunk_index": chunk["chunk_index"],
            }
        )
        points.append(point)

    total_batches = (len(points) + batch_size - 1) // batch_size
    successful_batches = 0

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        batch_number = i // batch_size + 1

        try:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=batch
            )
            successful_batches += 1
            logger.info(
                f"Batch {batch_number}/{total_batches} uploaded successfully"
            )

        except Exception as e:
            logger.error(
                f"Batch {batch_number}/{total_batches} failed: {e}\n"
                f"Resume from batch {batch_number} — "
                f"{successful_batches * batch_size} chunks already stored"
            )
            raise

    logger.info(
        f"Stored {len(points)} chunks in "
        f"{successful_batches} batches successfully"
    )


def search(query_vector: list, top_k: int) -> list:
    """
    Searches Qdrant for most similar chunks to query vector.
    Returns top_k most similar chunks with scores and metadata.
    """
    results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    chunks = []
    for result in results:
        chunk = result.payload
        chunk["score"] = result.score
        chunks.append(chunk)

    logger.info(f"Retrieved {len(chunks)} chunks for query")

    return chunks