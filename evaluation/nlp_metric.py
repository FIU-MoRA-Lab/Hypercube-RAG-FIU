import json
import argparse
import pandas as pd
import sys
import os

# Allow importing utils.metric from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.metric import exact_match, f1_score, bleu_score, rouge_score, semantic_score


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
        description=(
            "Evaluate compound_flooding QA (No Retrieval vs Hypercube-RAG vs "
            "Multimodal Hypercube-RAG) and print a small metrics table."
        )
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
    """Return averaged metrics over a list of QA dicts."""
    total_em = total_f1 = total_bleu = total_rouge = total_semantic = 0.0
    n = len(samples)

    for s in samples:
        gold = s["answer_gold"]
        pred = s["answer_pred"]

        total_em       += exact_match(pred, gold)
        total_f1       += f1_score(pred, gold)
        total_bleu     += bleu_score(pred, gold)
        total_rouge    += rouge_score(pred, gold)
        total_semantic += semantic_score(pred, gold)

    return {
        "EM":        total_em / n,
        "F1":        total_f1 / n,
        "BLEU":      total_bleu / n,
        "ROUGE":     total_rouge / n,
        "Semantic":  total_semantic / n,
        "N":         n,
    }


# ================ Main ================
def main():
    args = parse_args()

    dataset = "compound_flooding"
    base_dir = os.path.join("output", dataset, args.model)

    # Files we care about
    no_rag_path      = os.path.join(base_dir, "llm_output_no_rag.json")
    unimodal_path    = os.path.join(base_dir, "llm_output_hypercube.json")
    multimodal_path  = os.path.join(base_dir, f"compound_flooding_{args.model}_hypercube_multimodal.json")

    rows = []

    # ---------- No Retrieval ----------
    print(f"Loading No Retrieval outputs from: {no_rag_path}")
    no_rag_samples = load_dataset(no_rag_path)
    no_rag_metrics = evaluate_file(no_rag_samples)

    rows.append({
        "Method": f"{args.model} (No Retrieval)",
        "F1": round(no_rag_metrics["F1"] * 100, 1),
        "Semantic": round(no_rag_metrics["Semantic"] * 100, 1),
    })

    # ---------- Unimodal Hypercube (optional) ----------
    if os.path.exists(unimodal_path):
        print(f"Loading Unimodal Hypercube-RAG outputs from: {unimodal_path}")
        unimodal_samples = load_dataset(unimodal_path)
        unimodal_metrics = evaluate_file(unimodal_samples)

        rows.append({
            "Method": f"Hypercube-RAG ({args.model})",
            "F1": round(unimodal_metrics["F1"] * 100, 1),
            "Semantic": round(unimodal_metrics["Semantic"] * 100, 1),
        })
    else:
        print(f"Unimodal Hypercube file not found, skipping: {unimodal_path}")

    # ---------- Multimodal Hypercube (optional) ----------
    if os.path.exists(multimodal_path):
        print(f"Loading Multimodal Hypercube-RAG outputs from: {multimodal_path}")
        multimodal_samples = load_dataset(multimodal_path)
        multimodal_metrics = evaluate_file(multimodal_samples)

        rows.append({
            "Method": f"Multimodal Hypercube-RAG ({args.model})",
            "F1": round(multimodal_metrics["F1"] * 100, 1),
            "Semantic": round(multimodal_metrics["Semantic"] * 100, 1),
        })
    else:
        print(f"Multimodal Hypercube file not found, skipping: {multimodal_path}")

    # ---------- Print table ----------
    df = pd.DataFrame(rows, columns=["Method", "F1", "Semantic"])

    print("\n=== Compound Flooding QA – Automatic Metrics (F1 / Semantic) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
