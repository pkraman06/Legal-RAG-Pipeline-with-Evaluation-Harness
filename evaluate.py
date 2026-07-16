"""
evaluate.py
Runs the golden CUAD QA set through each retrieval mode (dense_only, hybrid,
hybrid_rerank), generates answers, and scores them with RAGAS metrics:
  - context_precision
  - context_recall
  - faithfulness
  - answer_relevancy

Produces results.csv / results.json with one row per retrieval mode — this
is the comparison table for your resume/portfolio. Run this yourself and
use YOUR numbers; do not copy example numbers from elsewhere.

Usage:
    python evaluate.py --n_samples 30
"""

import argparse
import json
import pickle
import pandas as pd
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from rag_pipeline import LegalRAGPipeline

INDEX_DIR = "indices"
MODES = ["dense_only", "hybrid", "hybrid_rerank"]


def load_eval_set(n_samples: int):
    with open(f"{INDEX_DIR}/qa_records.pkl", "rb") as f:
        records = pickle.load(f)

    # Keep only records that actually have a gold answer (CUAD includes
    # unanswerable questions by design, which we exclude from this eval).
    answerable = [r for r in records if r.answer]
    sample = answerable[:n_samples]
    print(f"[evaluate] Using {len(sample)} answerable QA pairs for evaluation.")
    return sample


def run_eval_for_mode(mode: str, eval_records, pipeline: LegalRAGPipeline):
    questions, answers, contexts, ground_truths = [], [], [], []

    for rec in eval_records:
        result = pipeline.answer(rec.question, retrieval_mode=mode, top_k=5)
        questions.append(rec.question)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["retrieved_chunks"]])
        ground_truths.append(rec.answer)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = ragas_evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    return scores.to_pandas()


def main(n_samples: int):
    eval_records = load_eval_set(n_samples)
    pipeline = LegalRAGPipeline()

    all_rows = []
    for mode in MODES:
        print(f"\n[evaluate] Running mode: {mode}")
        df = run_eval_for_mode(mode, eval_records, pipeline)
        summary = {
            "mode": mode,
            "context_precision": df["context_precision"].mean(),
            "context_recall": df["context_recall"].mean(),
            "faithfulness": df["faithfulness"].mean(),
            "answer_relevancy": df["answer_relevancy"].mean(),
        }
        print(summary)
        all_rows.append(summary)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("results.csv", index=False)
    with open("results.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    print("\n=== Final comparison table (also saved to results.csv) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=30,
                         help="Number of QA pairs to evaluate per retrieval mode")
    args = parser.parse_args()
    main(args.n_samples)
