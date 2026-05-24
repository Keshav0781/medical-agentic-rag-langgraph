"""
RAGAS evaluation for medical-agentic-rag-langgraph pipeline.
Measures faithfulness, answer relevancy, context precision.
Runs on 5 questions — respects Groq free tier limits.

At Siemens Azure OpenAI enterprise used — no rate limits.
Full 50-100 question evaluation ran weekly as scheduled job.
"""

import json
import logging
import time
import os
import pandas as pd
from datasets import Dataset
from datetime import datetime

from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.agents.rag_agent import rag_graph
from src.evaluation.golden_dataset import GOLDEN_DATASET
from configs.config import GROQ_API_KEY, GROQ_MODEL, EMBEDDING_MODEL, LOG_LEVEL

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ── Configure RAGAS with Groq ─────────────────────────────────────────────────
# Using LangchainLLMWrapper — confirmed working with Groq
# RunConfig limits concurrency to avoid rate limits
ragas_llm = LangchainLLMWrapper(ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL
))

ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
)

# Limit concurrency for Groq free tier
# At Siemens no limit needed — Azure enterprise tier
run_config = RunConfig(
    max_workers=1,      # sequential — no parallel calls
    timeout=120,        # 2 min timeout per call
    max_retries=3       # retry on failure
)


def run_evaluation(
    output_dir: str = "logs",
    num_questions: int = 5
) -> dict:
    """
    Runs RAGAS evaluation on golden dataset.
    Measures faithfulness, answer relevancy, context precision.
    Saves results to logs/ folder for trend tracking.

    Args:
        output_dir: folder to save results
        num_questions: number of questions — default 5 for Groq limits
    """
    dataset_slice = GOLDEN_DATASET[:num_questions]

    logger.info(
        f"Starting RAGAS evaluation on {len(dataset_slice)} questions"
    )

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(dataset_slice):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(
            f"Processing question {i+1}/{len(dataset_slice)}: "
            f"{question[:60]}"
        )

        try:
            result = rag_graph.invoke({
                "query": question,
                "rewritten_query": "",
                "route": "",
                "chunks": [],
                "reranked_chunks": [],
                "answer": "",
                "follow_ups": [],
                "retry_count": 0,
                "is_safe": False,
                "error": "",
                "conversation_history": []
            })

            answer = result.get("answer", "")
            reranked_chunks = result.get("reranked_chunks", [])
            context = [chunk["text"] for chunk in reranked_chunks]

            if not result.get("is_safe"):
                logger.warning(
                    f"Question {i+1} rejected by guardrail — skipping"
                )
                continue

            if not context:
                logger.warning(
                    f"Question {i+1} returned no context — skipping"
                )
                continue

            questions.append(question)
            answers.append(answer)
            contexts.append(context)
            ground_truths.append(ground_truth)

            logger.info(f"Question {i+1} processed successfully")

            # Rate limit — wait between pipeline calls
            if i < len(dataset_slice) - 1:
                logger.info("Waiting 5s between questions...")
                time.sleep(5)

        except Exception as e:
            logger.error(f"Question {i+1} failed: {e}")
            continue

    if not questions:
        logger.error("No questions processed successfully")
        return {}

    logger.info(
        f"Running RAGAS scoring on {len(questions)} questions..."
    )
    logger.info("Note: Sequential mode — this takes ~10 minutes")

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    try:
        results = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                context_precision
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config
        )

        # Select only numeric columns
        scores = results.to_pandas().select_dtypes(
            include='number'
        ).mean().to_dict()

        logger.info("=== RAGAS EVALUATION RESULTS ===")
        logger.info(
            f"Faithfulness:      {scores.get('faithfulness', 0):.4f}"
        )
        logger.info(
            f"Answer Relevancy:  {scores.get('answer_relevancy', 0):.4f}"
        )
        logger.info(
            f"Context Precision: {scores.get('context_precision', 0):.4f}"
        )

        # Siemens threshold alerts
        if scores.get('faithfulness', 0) < 0.8:
            logger.warning(
                "ALERT: Faithfulness below 0.8 — investigation needed"
            )
        if scores.get('context_precision', 0) < 0.75:
            logger.warning(
                "ALERT: Context precision below 0.75 threshold"
            )
        if scores.get('answer_relevancy', 0) < 0.7:
            logger.warning(
                "ALERT: Answer relevancy below 0.7 threshold"
            )

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = f"{output_dir}/ragas_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "num_questions": len(questions),
                "scores": scores
            }, f, indent=2)

        csv_path = f"{output_dir}/ragas_detailed_{timestamp}.csv"
        results.to_pandas().to_csv(csv_path, index=False)

        logger.info(f"Results saved to {json_path}")
        logger.info(f"Detailed results saved to {csv_path}")

        return scores

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {}


if __name__ == "__main__":
    scores = run_evaluation()
    if scores:
        print("\n=== FINAL SCORES ===")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
    else:
        print("Evaluation failed — check logs for details")