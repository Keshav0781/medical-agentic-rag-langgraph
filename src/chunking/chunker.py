import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.config import CHUNK_SIZE, CHUNK_OVERLAP, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

def chunk_documents(pages: list) -> list:
    """
    Splits pages into smaller chunks for embedding and retrieval.
    Uses RecursiveCharacterTextSplitter to preserve natural language boundaries.
    Returns list of chunk dictionaries with text and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    for page in pages:
        text = page["text"]
        chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(chunks):
            chunk = {
                "text": chunk_text,
                "chunk_index": i,
                "source": page["source"],
                "page_number": page["page_number"],
                "total_pages": page["total_pages"],
                "document_type": page["document_type"],
            }
            all_chunks.append(chunk)

    logger.info(
        f"Created {len(all_chunks)} chunks from {len(pages)} pages"
    )

    return all_chunks