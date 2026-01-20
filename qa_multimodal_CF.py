#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qa_multimodal_CF.py
-------------
Hypercube-RAG runner for the compound_flooding dataset, now with
multimodal support (text + physical dimensions).

Example usage (from repo root):

  python qa_multimodal_CF.py --data compound_flooding --model gpt-4o --retrieval_method hypercube --save true

This script assumes:

1) Corpus:
   corpus/compound_flooding/body_SciData_CF_filtered.txt
   (one ScienceDaily-style article per line, same file used by ner_CF.py)

2) Hypercube dimensions (produced by ner_CF.py):
   Text dims:
     hypercube/compound_flooding/date.txt
     hypercube/compound_flooding/event.txt
     hypercube/compound_flooding/location.txt
     hypercube/compound_flooding/organization.txt
     hypercube/compound_flooding/person.txt
     hypercube/compound_flooding/theme.txt

   Physical dims (Option A multimodal per-doc bins):
     hypercube/compound_flooding/coops_level.txt
     hypercube/compound_flooding/imerg_precip.txt
     hypercube/compound_flooding/mrms_precip.txt

3) QA pairs (adjust QA_PATH if your file name differs):
   QA/compound_flooding/synthetic_qa.json

   Expected format (list of dicts):
   [
       {
           "id": "q1",
           "question": "What is compound flooding?",
           "answer": "Ground-truth answer text ..."
       },
       ...
   ]
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any

import spacy
from tqdm import tqdm

# -----------------------------
#  CONFIG
# -----------------------------

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

CORPUS_PATH = os.path.join(
    DATA_ROOT,
    "corpus",
    "compound_flooding",
    "body_SciData_CF_filtered.txt",
)

HYPERCUBE_DIR = os.path.join(
    DATA_ROOT,
    "hypercube",
    "compound_flooding",
)

# Adjust this path if your QA file name or folder is different
QA_PATH = os.path.join(
    DATA_ROOT,
    "QA",
    "compound_flooding",
    "synthetic_qa.json",
)

OUTPUT_DIR = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Hypercube-RAG-FIU\QA\compound_flooding"

# --- Dimensions ---

# Text-based dimensions (original)
TEXT_DIMENSIONS = ["date", "event", "location", "organization", "person", "theme"]

# Physical dimensions added by new ner_CF.py
PHYSICAL_DIMENSIONS = ["coops_level", "imerg_precip", "mrms_precip"]

# All dimensions present in hypercube (for loading / inverted index)
DIMENSIONS = TEXT_DIMENSIONS + PHYSICAL_DIMENSIONS

SPACY_TO_DIM = {
    "DATE": "date",
    "EVENT": "event",
    "GPE": "location",
    "LOC": "location",
    "ORG": "organization",
    "PERSON": "person",
}

# -----------------------------
#  OpenAI wrapper
# -----------------------------

def build_llm_caller():
    """
    Support both the newer `openai` client and the legacy API,
    so it works with more environments.
    """
    try:
        # New style (openai>=1.0)
        from openai import OpenAI
        client = OpenAI()

        def call_llm(model: str, messages: List[Dict[str, str]]) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()

        return call_llm

    except ImportError:
        # Legacy style (openai<1.0)
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")

        def call_llm(model: str, messages: List[Dict[str, str]]) -> str:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()

        return call_llm


CALL_LLM = build_llm_caller()

# -----------------------------
#  Data loading
# -----------------------------

def load_corpus(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


def load_qa(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_hypercube(hypercube_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Load the hypercube JSONL files into an inverted index.

    Returns:
        hypercube[dimension][entity] -> list of doc_ids
    """
    hypercube: Dict[str, Dict[str, List[int]]] = {dim: defaultdict(list) for dim in DIMENSIONS}

    for dim in DIMENSIONS:
        dim_path = os.path.join(hypercube_dir, f"{dim}.txt")
        if not os.path.exists(dim_path):
            raise FileNotFoundError(f"Missing hypercube file: {dim_path}")

        with open(dim_path, "r", encoding="utf-8") as f:
            for doc_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # record: { "entity_string": count, ... }
                for ent in record.keys():
                    hypercube[dim][ent].append(doc_id)

    return hypercube


def load_physical_records(hypercube_dir: str) -> Dict[str, List[Dict[str, int]]]:
    """
    Load per-doc label dictionaries for physical dimensions, so we can
    print them into the context given to the LLM.

    Returns:
        physical_records[dim][doc_id] -> {label -> count}
    """
    physical_records: Dict[str, List[Dict[str, int]]] = {}

    for dim in PHYSICAL_DIMENSIONS:
        dim_path = os.path.join(hypercube_dir, f"{dim}.txt")
        if not os.path.exists(dim_path):
            # If you haven't created some physical dims yet, skip gracefully
            print(f"[WARN] Physical dimension file not found: {dim_path}, skipping.")
            physical_records[dim] = []
            continue

        records: List[Dict[str, int]] = []
        with open(dim_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    records.append({})
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        records.append({})
                except json.JSONDecodeError:
                    records.append({})
        physical_records[dim] = records

    return physical_records


# -----------------------------
#  Query decomposition (NER)
# -----------------------------

def init_nlp():
    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")
    return nlp


def decompose_query(text: str, nlp) -> Dict[str, Dict[str, int]]:
    """
    Extract entities per dimension from the query text.
    Mirrors ner_CF.py logic so retrieval is aligned with how the hypercube was built.
    Only text dimensions are populated; physical dims are left empty (used for context).
    """
    doc = nlp(text)
    # Initialize all dims so we can safely loop over them later
    dim_entities = {dim: {} for dim in DIMENSIONS}

    # Named entities -> text dims
    for ent in doc.ents:
        dim = SPACY_TO_DIM.get(ent.label_)
        if dim is None:
            continue
        key = ent.text.strip()
        if not key:
            continue
        dim_entities[dim][key] = dim_entities[dim].get(key, 0) + 1

    # Theme: noun chunks that are not already counted
    seen = {k.lower() for dim in TEXT_DIMENSIONS for k in dim_entities[dim].keys()}

    for chunk in doc.noun_chunks:
        key = chunk.text.strip()
        if not key:
            continue
        if key.lower() in seen:
            continue
        if len(key) < 3:
            continue
        dim_entities["theme"][key] = dim_entities["theme"].get(key, 0) + 1

    # physical dims remain empty in dim_entities
    return dim_entities


# -----------------------------
#  Retrieval
# -----------------------------

def retrieve_hypercube_docs(
    question: str,
    nlp,
    hypercube: Dict[str, Dict[str, List[int]]],
    top_k: int = 5,
) -> List[int]:
    """
    Simple Hypercube-RAG retrieval:
    - Decompose question into entities per dimension (TEXT dims only).
    - For each matched entity, each doc containing it gets +1 score.
    - Return top_k doc ids by score.

    Physical dimensions are part of the hypercube but are NOT used for
    matching here, because the query usually does not mention numeric
    bin labels like '(0.10, 0.50]'. They are used later as context.
    """
    query_entities = decompose_query(question, nlp)
    scores: Dict[int, int] = defaultdict(int)

    for dim in TEXT_DIMENSIONS:
        ent_dict = query_entities.get(dim, {})
        for ent in ent_dict.keys():
            if ent not in hypercube[dim]:
                continue
            for doc_id in hypercube[dim][ent]:
                scores[doc_id] += 1

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    doc_ids = [doc_id for doc_id, _ in ranked[:top_k]]
    return doc_ids


def build_context_from_docs(
    doc_ids: List[int],
    corpus: List[str],
    physical_records: Dict[str, List[Dict[str, int]]] = None,
    max_chars: int = 6000,
) -> str:
    """
    Concatenate the selected documents into a single context string,
    truncated to max_chars.

    If physical_records is provided, prepend a [PHYSICAL] section for
    each doc summarizing its physical bins (e.g., coops_level / imerg_precip).
    """
    if physical_records is None:
        physical_records = {}

    chunks = []
    current_len = 0

    for doc_id in doc_ids:
        if not (0 <= doc_id < len(corpus)):
            continue

        text = corpus[doc_id].strip()
        if not text:
            continue

        # --- Build physical summary for this doc ---
        phys_lines = []
        for dim in PHYSICAL_DIMENSIONS:
            rec_list = physical_records.get(dim, [])
            if doc_id >= len(rec_list):
                continue
            rec = rec_list[doc_id]
            if not rec:
                continue
            # Get top few labels by count
            sorted_items = sorted(rec.items(), key=lambda kv: kv[1], reverse=True)
            top_labels = [lbl for lbl, _ in sorted_items[:3]]
            if top_labels:
                phys_lines.append(f"{dim}: {', '.join(top_labels)}")

        if phys_lines:
            phys_block = "[PHYSICAL]\n" + "\n".join(phys_lines) + "\n"
        else:
            phys_block = ""

        doc_block = f"[DOC {doc_id}]\n{phys_block}{text}"
        if current_len + len(doc_block) + 50 > max_chars:
            break

        chunks.append(doc_block)
        current_len += len(doc_block) + 50

    return "\n\n".join(chunks)


# -----------------------------
#  QA Loop
# -----------------------------

def answer_question_with_rag(
    question: str,
    model: str,
    retrieval_method: str,
    nlp,
    corpus: List[str],
    hypercube: Dict[str, Dict[str, List[int]]],
    physical_records: Dict[str, List[Dict[str, int]]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run one QA example with the chosen retrieval method.
    Returns a dict with answer, context, and retrieved doc ids.
    """

    # --------- Decide which docs to use ----------
    if retrieval_method == "hypercube":
        doc_ids = retrieve_hypercube_docs(question, nlp, hypercube, top_k=top_k)
        if not doc_ids:
            # Fallback: if no hits, just use first document as weak context
            doc_ids = [0] if corpus else []
    elif retrieval_method in ("full", "full_corpus"):
        doc_ids = list(range(min(top_k, len(corpus))))
    else:
        # "none" → no retrieval / no context
        doc_ids = []

    # --------- Build context string ----------
    if doc_ids:
        context = build_context_from_docs(
            doc_ids,
            corpus,
            physical_records=physical_records,
            max_chars=6000,
        )
    else:
        context = ""

    # --------- Build messages depending on retrieval method ----------
    def build_messages(q: str, ctx: str, method: str) -> List[Dict[str, str]]:
        if method == "none":
            # Baseline: let GPT-4o use its own knowledge
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable assistant answering questions about "
                        "compound flooding, coastal flooding, storm surge, tides, "
                        "rainfall, and related hydrological processes. "
                        "Answer clearly and concisely using your own knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": q.strip(),
                },
            ]
        else:
            # RAG: use context if helpful, but do NOT refuse due to context
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable assistant answering questions about "
                        "compound flooding, coastal flooding, storm surge, tides, "
                        "rainfall, and related hydrological and climate processes. "
                        "You are given some context from scientific articles and "
                        "summaries of physical sensor data (water level, rainfall, etc.). "
                        "First, try to use the context. If the context is unclear or incomplete, "
                        "you may rely on your own knowledge of the topic to give the best possible answer. "
                        "Do NOT say that the context does not contain the information; instead, give "
                        "your best explanation of the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {q.strip()}\n\n"
                        f"Context (text + physical):\n{ctx.strip()}\n\n"
                        "Answer the question as accurately and completely as you can."
                    ),
                },
            ]

    messages = build_messages(question, context, retrieval_method)
    answer = CALL_LLM(model, messages)

    return {
        "question": question,
        "answer_pred": answer,
        "retrieved_doc_ids": doc_ids,
        "context": context,
    }


# -----------------------------
#  Main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hypercube-RAG for compound_flooding dataset (multimodal).")
    parser.add_argument("--data", type=str, default="compound_flooding",
                        help="Dataset name (for bookkeeping only).")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI chat model name.")
    parser.add_argument("--retrieval_method", type=str, default="hypercube",
                        choices=["hypercube", "full", "full_corpus", "none"],
                        help="Retrieval method to use.")
    parser.add_argument("--save", type=str, default="true",
                        help="Whether to save outputs to JSON (true/false).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optionally cap the number of QA pairs to run.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of documents to retrieve.")
    return parser.parse_args()


def main():
    args = parse_args()
    save_flag = str(args.save).lower() in ("true", "1", "yes", "y")

    print(f"=== Running QA for dataset: {args.data} ===")
    print(f"Model: {args.model}")
    print(f"Retrieval method: {args.retrieval_method}")
    print(f"Save outputs: {save_flag}")
    print()

    # Load data
    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(corpus)} documents.\n")

    print("Loading QA pairs...")
    qa_data = load_qa(QA_PATH)
    if args.max_samples is not None:
        qa_data = qa_data[: args.max_samples]
    print(f"Loaded {len(qa_data)} QA pairs.\n")

    # Load hypercube and NLP only if we need retrieval or context
    hypercube = None
    physical_records = None
    nlp = None
    if args.retrieval_method in ("hypercube", "full", "full_corpus"):
        print("Loading hypercube dimensions (including physical dims)...")
        hypercube = load_hypercube(HYPERCUBE_DIR)
        print("Hypercube loaded.\n")

        print("Loading physical per-doc records for context...")
        physical_records = load_physical_records(HYPERCUBE_DIR)
        print("Physical records loaded.\n")

        print("Initializing spaCy for query decomposition...")
        nlp = init_nlp()
        print()

    results = []

    print("Starting QA loop...\n")
    for ex in tqdm(qa_data):
        q = ex.get("question", "")
        gold = ex.get("answer", "")

        rag_result = answer_question_with_rag(
            question=q,
            model=args.model,
            retrieval_method=args.retrieval_method if args.retrieval_method != "none" else "none",
            nlp=nlp,
            corpus=corpus,
            hypercube=hypercube,
            physical_records=physical_records,
            top_k=args.top_k,
        )

        out_item = {
            "id": ex.get("id", None),
            "question": q,
            "answer_gold": gold,
            "answer_pred": rag_result["answer_pred"],
            "retrieved_doc_ids": rag_result["retrieved_doc_ids"],
            # You can comment out context if files get too big
            "context": rag_result["context"],
        }
        results.append(out_item)

    if save_flag:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_name = f"{args.data}_{args.model}_{args.retrieval_method}_multimodal.json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} results to: {out_path}\n")
    else:
        print("\nNot saving outputs (save=false).")

    print("Done.")


if __name__ == "__main__":
    main()
