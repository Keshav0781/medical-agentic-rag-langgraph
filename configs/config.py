from dotenv import load_dotenv
import os

# Open the locker and read all keys into memory
load_dotenv()

# Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "medical_rd_documents")

# Groq LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
RAW_DOCS_PATH = "data/raw_docs"
PROCESSED_PATH = "data/processed"


# Retrieval Configuration
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 20))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
RETRIEVAL_SCORE_THRESHOLD = 1.5

# LangSmith Tracing
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "medical-agentic-rag")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")

# Set LangSmith environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Embedding Model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
)
EMBEDDING_DIMENSION = 768

def validate_config():
    """
    Validates critical environment variables at startup.
    Fails fast if anything critical is missing.
    """
    missing = []
    
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
        
    if missing:
        raise ValueError(
            f"Missing critical environment variables: {missing}\n"
            f"Please check your .env file."
        )
    
    if not LANGSMITH_API_KEY:
        print("WARNING: LANGSMITH_API_KEY not set — tracing disabled")
    
    return True