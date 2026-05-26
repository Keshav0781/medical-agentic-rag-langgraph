# Medical Agentic RAG — LangGraph Pipeline

> Agentic RAG pipeline over Siemens Healthineers public R&D documents.  
> Built with LangGraph, Qdrant, Groq Llama-3, and deployed on GCP Cloud Run.

**🔴 Live Demo:** https://medical-rag-service-324111066236.europe-west1.run.app

---

## Overview

Intelligent document search and summarisation pipeline over 4,035 chunks from 8 public Siemens Healthineers R&D documents — Annual Reports, Sustainability Reports, Capital Markets presentations, and Technical Whitepapers.

---

## Architecture

```
User Query
    ↓
Input Guardrail Node — semantic safety check (relevance, PII, prompt injection)
    ↓
Router Node — RAG vs Summary path
    ↓
Query Understanding Node — query rewriting with conversation memory
    ↓
Retrieval Node — Qdrant vector search (top 20 chunks)
    ↓
Retrieval Validation Node — score threshold + retry logic
    ↓
Reranking Node — CrossEncoder two-stage retrieval (top 5)
    ↓
Generation Node — Groq Llama-3 with source citations
    ↓
Output Guardrail Node — faithfulness check
    ↓
Update History Node — 5-exchange conversation memory
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph StateGraph |
| Vector Store | Qdrant Cloud (Frankfurt, GCP europe-west3) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 (768 dim) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-12-v2 |
| LLM | Groq Llama-3.3-70b-versatile |
| Evaluation | RAGAS (faithfulness, context precision) |
| API | FastAPI |
| Observability | LangSmith |
| Deployment | GCP Cloud Run (europe-west1) |

---

## Key Features

**Agentic Pipeline:**
- 9 LangGraph nodes with conditional routing and retry logic
- Query rewriting with conversation memory (last 5 exchanges)
- Retrieval validation with automatic retry on poor results
- Reranker fallback — top 3 vector results when CrossEncoder returns 0

**Guardrails:**
- Input: relevance, safety, PII, medical advice, prompt injection, length checks
- Semantic LLM classification — catches idioms and indirect harmful requests
- Output: faithfulness checking against retrieved context

**Document Summarisation:**
- Map-reduce pattern — batch summarisation across full documents
- Document detection from casual descriptions using LLM fuzzy matching
- Structured output: OVERVIEW, KEY FINDINGS, STRATEGIC HIGHLIGHTS, CONCLUSION

**Evaluation:**
- RAGAS evaluation pipeline with 10-question golden dataset
- Metrics: faithfulness, context precision
- Results saved as JSON and CSV with timestamps for trend tracking

---

## Documents

8 Siemens Healthineers public documents:
- Annual Reports 2022, 2023
- Sustainability Reports 2022, 2025
- Capital Markets Day 2025
- Management Presentations 2023
- Atellica Workflow Whitepaper
- Sigma Metrics Whitepaper

**4,035 chunks** stored in Qdrant Cloud Frankfurt

---

## Local Setup

```bash
git clone https://github.com/Keshav0781/medical-agentic-rag-langgraph
cd medical-agentic-rag-langgraph
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`:
```
GROQ_API_KEY=your_key
QDRANT_URL=your_url
QDRANT_API_KEY=your_key
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=medical-agentic-rag
LANGCHAIN_TRACING_V2=true
LOG_LEVEL=INFO
```

Run:
```bash
uvicorn app:app --reload --port 8000
```

Open: http://localhost:8000

---

## Evaluation

```bash
python -m src.evaluation.ragas_eval
```

Note: Full evaluation requires OpenAI or Azure OpenAI API. Groq free tier token limits affect RAGAS scoring reliability.

---

## Project Structure

```
├── app.py                    # FastAPI + UI serving
├── configs/config.py         # Central configuration
├── src/
│   ├── ingestion/            # Document loading and chunking
│   ├── embeddings/           # Sentence-transformers embedder
│   ├── retrieval/            # Qdrant vector store + CrossEncoder reranker
│   ├── agents/               # LangGraph pipeline + document summariser
│   └── evaluation/           # RAGAS evaluation + golden dataset
├── static/                   # CSS and JavaScript
├── templates/                # HTML template
├── Dockerfile                # Multi-stage Docker build
└── requirements.txt
```

---

## Note on Token Limits

This project uses Groq free tier (100,000 tokens/day):
- RAG queries use ~500-1,000 tokens each
- Summarisation uses ~12,000 tokens
- For best experience test summary requests with fresh daily quota