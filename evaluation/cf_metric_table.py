import json
import argparse
import pandas as pd
import sys
import os
import re

from collections import Counter

# Allow importing utils.metric from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.metric import exact_match, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ================ Text Helpers ================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return normalize_text(text).split()


# ================ Semantic, Correctness, Completeness ================
def semantic_similarity(pred: str, gold: str) -> float:
    """Lightweight semantic similarity using TF-IDF cosine similarity."""
    pred = pred.strip()
    gold = gold.strip()
    if not pred or not gold:
        return 0.0

    vec = TfidfVectorizer()
    tfidf = vec.fit_transform([gold, pred])  # row 0: gold, row 1: pred
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    # Ensure it's a plain float
    return float(sim)


def correctness_completeness(pred: str, gold: str):
    """
    correctness ~ precision  = |overlap| / |pred_tokens|
    completeness ~ recall    = |overlap| / |gold_tokens|
    """
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not gold_tokens:
        return 0.0, 0.0
    if not pred_tokens:
        return 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    # Overlap counted with min frequency for each token
    overlap = 0
    for tok, gc in gold_counts.items():
        overlap += min(gc, pred_counts.get(tok, 0))

    correctness = overlap / len(pred_tokens)   # precision-like
    completeness = overlap / len(gold_tokens)  # recall-like

    return float(correctness), float(completeness)


# ================ Load Dataset ================
def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM output file not found: {path}")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif path.endswith(".csv"):
        df = pd.read_csv(path, encoding="utf-8")
        return df.to_dict(orient="records")
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")


# ================ Argument Parser ================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate compound_flooding QA (No Retrieval vs Hypercube-RAG) with F1, semantic, correctness, completeness."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo", "deepseek", "llama3", "llama4", "gemma", "qwen"],
        help="Model name used in output paths (e.g., gpt-4o)."
    )
    return parser.parse_args()


# ================ Metric Helper ================
def evaluate_file(samples):
    """
    Return averaged metrics over a list of QA dicts:
    - F1      (from utils.metric)
    - Semantic (TF-IDF cosine)
    - Correctness (precision-style)
    - Completeness (recall-style)
    """
    total_f1 = 0.0
    total_semantic = 0.0
    total_correctness = 0.0
    total_completeness = 0.0
    n = len(samples)

    for s in samples:
        gold = s["answer_gold"]
        pred = s["answer_pred"]

        # F1 from your existing implementation
        total_f1 += f1_score(pred, gold)

        # Lightweight semantic similarity
        total_semantic += semantic_similarity(pred, gold)

        # Correctness & Completeness
        c_corr, c_comp = correctness_completeness(pred, gold)
        total_correctness += c_corr
        total_completeness += c_comp

    return {
        "F1":           total_f1 / n,
        "Semantic":     total_semantic / n,
        "Correctness":  total_correctness / n,
        "Completeness": total_completeness / n,
        "N":            n,
    }


# ================ Main ================
def main():
    args = parse_args()

    dataset = "compound_flooding"

    # Read directly from the QA folder where qa_rag_CF.py saves
    qa_base = os.path.join("QA", dataset)

    no_rag_path    = os.path.join(qa_base, f"{dataset}_{args.model}_none.json")
    hypercube_path = os.path.join(qa_base, f"{dataset}_{args.model}_hypercube.json")

    print(f"Loading No Retrieval outputs from: {no_rag_path}")
    no_rag_samples = load_dataset(no_rag_path)

    print(f"Loading Hypercube-RAG outputs from: {hypercube_path}")
    hypercube_samples = load_dataset(hypercube_path)

    # Evaluate both
    no_rag_metrics    = evaluate_file(no_rag_samples)
    hypercube_metrics = evaluate_file(hypercube_samples)

    # Build and print F1 + Semantic + Correctness + Completeness table
    rows = [
        {
            "Method": "GPT-4o (No Retrieval)",
            "F1": round(no_rag_metrics["F1"] * 100, 1),
            "Semantic": round(no_rag_metrics["Semantic"] * 100, 1),
            "Correctness": round(no_rag_metrics["Correctness"] * 100, 1),
            "Completeness": round(no_rag_metrics["Completeness"] * 100, 1),
        },
        {
            "Method": "Hypercube-RAG (GPT-4o)",
            "F1": round(hypercube_metrics["F1"] * 100, 1),
            "Semantic": round(hypercube_metrics["Semantic"] * 100, 1),
            "Correctness": round(hypercube_metrics["Correctness"] * 100, 1),
            "Completeness": round(hypercube_metrics["Completeness"] * 100, 1),
        },
    ]

    df = pd.DataFrame(
        rows,
        columns=["Method", "F1", "Semantic", "Correctness", "Completeness"]
    )

    print("\n=== Compound Flooding QA – Automatic Metrics ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
