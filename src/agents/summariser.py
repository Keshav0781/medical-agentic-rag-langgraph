"""
Document Summariser for medical-agentic-rag-langgraph pipeline.

Map-reduce summarisation pattern:
1. Detect which document user wants to summarise
2. Fetch chunks from that document via Qdrant metadata filter
3. Summarise chunks in batches — map step
4. Combine batch summaries into structured output — reduce step

At Siemens used for quick R&D report overviews.
Researchers got structured summary instead of reading 150 pages.

Token note: Full document summarisation requires enterprise API tier.
Groq free tier: limited to first 100 chunks per document.
Azure OpenAI enterprise (Siemens): full corpus summarisation in ~30s.
"""

import logging
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client.models import Filter, FieldCondition, MatchValue
from configs.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LOG_LEVEL,
    QDRANT_COLLECTION_NAME
)
from src.retrieval.vector_store import qdrant_client

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ── LLM initialization ───────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL
)

# ── Document name mapping ────────────────────────────────────────────────────
# Maps casual descriptions to exact filenames in our corpus
# At Siemens with 5000+ documents — dynamic registry from Qdrant used
# Here 8 documents hardcoded — same concept, smaller scale
DOCUMENT_MAPPING = {
    "annual report 2023": "siemens-healthineers-ir-annual-report-2023.pdf",
    "annual report 2022": "siemens-healthineers-ir-annual-report-2022.pdf",
    "sustainability 2022": "Siemens-Healthineers_Sustainability_Report_FY-2022.pdf",
    "sustainability 2025": "Siemens-healthineers-SustainabilityImpact-2025.pdf",
    "capital markets 2025": "siemens-healthineers-ir-cmd-2025.pdf",
    "management 2023": "siemens-healthineers-ir-Mtm-2023_Segment-Presentations_Dec-7.pdf",
    "atellica workflow": "cls-atellica-id-srs-workflow-whitepaper.pdf",
    "sigma metrics": "cls-global-sigma-metrics-chemistry-immunochemistry-assays-atellica-portfolio-whitepaper.pdf",
}

# Token limit management for Groq free tier
# At Siemens no limit — Azure OpenAI enterprise handles full corpus
MAX_CHUNKS_FOR_SUMMARY = 100


# ── Function 1: Document Detection ──────────────────────────────────────────
def detect_document(query: str) -> str:
    """
    Matches user query to actual document filename using LLM.
    Returns filename or empty string if no match found.

    At Siemens documents had structured IDs — matching was exact.
    Here we use LLM for fuzzy matching on casual descriptions.
    With 5000+ docs at Siemens — dynamic Qdrant registry used.
    """
    doc_list = "\n".join([
        f"- {key}: {value}"
        for key, value in DOCUMENT_MAPPING.items()
    ])

    messages = [
        SystemMessage(content=f"""
You are a document matcher for Siemens Healthineers R&D system.

Available documents:
{doc_list}

Match the user query to the most relevant document.
Respond with ONLY the exact filename — nothing else.
If no document matches — respond with: NO_MATCH
        """),
        HumanMessage(content=f"Query: {query}")
    ]

    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            filename = response.content.strip()

            if filename == "NO_MATCH":
                logger.warning(f"No document match for: {query}")
                return ""

            logger.info(f"Document detected: {filename}")
            return filename

        except Exception as e:
            if attempt < 2:
                logger.warning(
                    f"Document detection attempt {attempt + 1} "
                    f"failed: {e} — retrying in 2 seconds"
                )
                time.sleep(2)
            else:
                logger.error(f"Document detection failed: {e}")
                return ""


# ── Function 2: Fetch Document Chunks ────────────────────────────────────────
def fetch_document_chunks(filename: str) -> list:
    """
    Fetches chunks from a specific document using metadata filter.
    Uses Qdrant scroll — not vector search.
    Returns chunks sorted by page number.

    scroll() vs query_points():
    - query_points(): find similar vectors — semantic search
    - scroll(): fetch all matching records — database query

    At Siemens same pattern with pagination for large documents.
    Here limit=500 sufficient for our 8 document corpus.
    """
    try:
        results = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename)
                    )
                ]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False
        )

        chunks = []
        for point in results[0]:
            chunks.append(point.payload)

        # Sort by page number for coherent summarisation
        chunks.sort(key=lambda x: x.get("page_number", 0))

        logger.info(
            f"Fetched {len(chunks)} chunks from {filename}"
        )
        return chunks

    except Exception as e:
        logger.error(f"Failed to fetch chunks for {filename}: {e}")
        return []


# ── Function 3: Summarise Document Node ──────────────────────────────────────
def summarise_document_node(state: dict) -> dict:
    """
    LangGraph node — orchestrates document summarisation.

    Map-reduce pattern:
    1. Detect which document user wants
    2. Fetch chunks — limited to MAX_CHUNKS_FOR_SUMMARY for token efficiency
    3. Summarise chunks in batches of 10 — map step
    4. Combine batch summaries into structured output — reduce step

    Token management:
    - Groq free tier: limited to first 100 chunks (~12,000 tokens)
    - Azure OpenAI enterprise (Siemens): full corpus, no limit

    At Siemens saved researchers 2-3 hours per week
    by providing instant structured overviews of R&D reports.
    """
    query = state["query"]

    # Step 1 — Detect document
    filename = detect_document(query)

    if not filename:
        return {
            "answer": """I could not identify which document you want summarised.

Please specify more clearly, for example:
- Summarise the 2023 Annual Report
- Give me an overview of the Sustainability Report 2022
- Summarise the Capital Markets presentation 2025
- Summarise the Atellica workflow whitepaper""",
            "follow_ups": []
        }

    # Step 2 — Fetch chunks
    all_chunks = fetch_document_chunks(filename)

    if not all_chunks:
        return {
            "answer": f"Could not retrieve content from {filename}. "
                      f"Please try again.",
            "follow_ups": []
        }

    # Limit chunks for Groq free tier token management
    chunks = all_chunks[:MAX_CHUNKS_FOR_SUMMARY]
    if len(all_chunks) > MAX_CHUNKS_FOR_SUMMARY:
        logger.info(
            f"Limited to first {MAX_CHUNKS_FOR_SUMMARY} of "
            f"{len(all_chunks)} chunks for token efficiency"
        )

    # Step 3 — Map: summarise chunks in batches of 10
    batch_size = 10
    batch_summaries = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_text = "\n\n".join([
            chunk.get("text", "") for chunk in batch
        ])

        messages = [
            SystemMessage(content="""
You are summarising sections of a Siemens Healthineers document.
Extract key facts, figures, and findings from this section.
Be concise — 3-5 sentences maximum.
            """),
            HumanMessage(content=f"Section content:\n{batch_text}")
        ]

        for attempt in range(3):
            try:
                response = llm.invoke(messages)
                batch_summaries.append(response.content.strip())
                logger.info(
                    f"Summarised batch {i//batch_size + 1}/"
                    f"{len(chunks)//batch_size + 1}"
                )
                time.sleep(2)  # Groq free tier rate limit management
                break
            except Exception as e:
                if attempt < 2:
                    logger.warning(
                        f"Batch {i//batch_size + 1} attempt "
                        f"{attempt + 1} failed — retrying in 3s"
                    )
                    time.sleep(3)
                else:
                    logger.error(
                        f"Batch {i//batch_size + 1} failed: {e}"
                    )
                    batch_summaries.append("")

    # Step 4 — Reduce: combine into structured summary
    combined = "\n\n".join([s for s in batch_summaries if s])

    if not combined:
        return {
            "answer": "Summary generation failed. Please try again.",
            "follow_ups": []
        }

    messages = [
        SystemMessage(content="""
You are creating a structured summary of a Siemens Healthineers document.
Based on the section summaries provided, create a final structured summary:

OVERVIEW: 2-3 sentence document overview
KEY FINDINGS: 3-5 most important findings or metrics
STRATEGIC HIGHLIGHTS: Key strategic initiatives or goals mentioned
CONCLUSION: 1-2 sentence conclusion

Be precise and factual. Only include information from the summaries.
        """),
        HumanMessage(content=f"""
Document: {filename}
Section summaries:
{combined}
        """)
    ]

    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            final_summary = response.content.strip()

            # Generate follow-up questions
            followup_messages = [
                SystemMessage(content="""
Generate exactly 3 follow-up questions about this document.
Write as:
FOLLOWUP1: <question>
FOLLOWUP2: <question>
FOLLOWUP3: <question>
                """),
                HumanMessage(
                    content=f"Document summary: {final_summary[:500]}"
                )
            ]

            followup_response = llm.invoke(followup_messages)
            follow_ups = []
            for line in followup_response.content.split("\n"):
                if line.startswith("FOLLOWUP"):
                    follow_ups.append(
                        line.split(":", 1)[1].strip()
                        if ":" in line else line
                    )

            logger.info(
                f"Summary complete — {len(batch_summaries)} "
                f"batches processed"
            )

            note = ""
            if len(all_chunks) > MAX_CHUNKS_FOR_SUMMARY:
                note = (
                    f"\n\n*Note: Summary based on first "
                    f"{MAX_CHUNKS_FOR_SUMMARY} of {len(all_chunks)} "
                    f"chunks. Full summarisation requires "
                    f"enterprise API access.*"
                )

            return {
                "answer": f"**Summary of {filename}**\n\n"
                          f"{final_summary}{note}\n\n"
                          f"Source: {filename}",
                "follow_ups": follow_ups
            }

        except Exception as e:
            if attempt < 2:
                logger.warning(
                    f"Final summary attempt {attempt + 1} "
                    f"failed — retrying in 3s"
                )
                time.sleep(3)
            else:
                logger.error(f"Final summary failed: {e}")
                return {
                    "answer": "Summary generation failed. "
                              "Please try again.",
                    "follow_ups": []
                }