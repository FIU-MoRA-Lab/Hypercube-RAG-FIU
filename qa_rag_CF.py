#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qa_rag_CF.py
-------------
Hypercube-RAG runner for the compound_flooding dataset.

Example usage (from repo root):

  python qa_rag_CF.py \
      --data compound_flooding \
      --model gpt-4o \
      --retrieval_method hypercube \
      --save true

This script assumes:

1) Corpus:
   corpus/compound_flooding/body_SciData_CF.txt
   (one ScienceDaily-style article per line, same file used by ner_CF.py)

2) Hypercube dimensions (produced by ner_CF.py):
   hypercube/compound_flooding/date.txt
   hypercube/compound_flooding/event.txt
   hypercube/compound_flooding/location.txt
   hypercube/compound_flooding/organization.txt
   hypercube/compound_flooding/person.txt
   hypercube/compound_flooding/theme.txt

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
    "body_SciData_CF.txt",
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

OUTPUT_DIR = OUTPUT_DIR = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Hypercube-RAG\QA\compound_flooding"

DIMENSIONS = ["date", "event", "location", "organization", "person", "theme"]

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
    """
    doc = nlp(text)
    dim_entities = {dim: {} for dim in DIMENSIONS}

    # Named entities
    for ent in doc.ents:
        dim = SPACY_TO_DIM.get(ent.label_)
        if dim is None:
            continue
        key = ent.text.strip()
        if not key:
            continue
        dim_entities[dim][key] = dim_entities[dim].get(key, 0) + 1

    # Theme: noun chunks that are not already counted
    seen = {k.lower() for dim in DIMENSIONS for k in dim_entities[dim].keys()}

    for chunk in doc.noun_chunks:
        key = chunk.text.strip()
        if not key:
            continue
        if key.lower() in seen:
            continue
        if len(key) < 3:
            continue
        dim_entities["theme"][key] = dim_entities["theme"].get(key, 0) + 1

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
    - Decompose question into entities per dimension.
    - For each matched entity, each doc containing it gets +1 score.
    - Return top_k doc ids by score.
    """
    query_entities = decompose_query(question, nlp)
    scores: Dict[int, int] = defaultdict(int)

    for dim in DIMENSIONS:
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


def build_context_from_docs(doc_ids: List[int], corpus: List[str], max_chars: int = 6000) -> str:
    """
    Concatenate the selected documents into a single context string,
    truncated to max_chars to avoid over-long prompts.
    """
    chunks = []
    current_len = 0
    for doc_id in doc_ids:
        if 0 <= doc_id < len(corpus):
            text = corpus[doc_id].strip()
            if not text:
                continue
            if current_len + len(text) + 50 > max_chars:  # +50 for separators
                break
            chunks.append(f"[DOC {doc_id}]\n{text}")
            current_len += len(text) + 50
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
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run one QA example with the chosen retrieval method.
    Returns a dict with answer, context, and retrieved doc ids.
    """
    if retrieval_method == "hypercube":
        doc_ids = retrieve_hypercube_docs(question, nlp, hypercube, top_k=top_k)
        if not doc_ids:
            # Fallback: if no hits, just use first document as weak context
            doc_ids = [0]
    elif retrieval_method in ("full", "full_corpus"):
        doc_ids = list(range(min(top_k, len(corpus))))
    else:
        # No retrieval: no context
        doc_ids = []

    context = build_context_from_docs(doc_ids, corpus) if doc_ids else ""

    system_prompt = (
        "You are an expert assistant on compound flooding, coastal hazards, "
        "and related climate and hydrology topics. Use ONLY the provided context "
        "to answer the user's question. If the context is insufficient, say so explicitly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                "Please provide a concise, accurate answer grounded in the context."
            ),
        },
    ]

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
    parser = argparse.ArgumentParser(description="Hypercube-RAG for compound_flooding dataset.")
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

    # Load hypercube and NLP only if we need retrieval
    hypercube = None
    nlp = None
    if args.retrieval_method in ("hypercube", "full", "full_corpus"):
        print("Loading hypercube dimensions...")
        hypercube = load_hypercube(HYPERCUBE_DIR)
        print("Hypercube loaded.\n")

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
        out_name = f"{args.data}_{args.model}_{args.retrieval_method}.json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} results to: {out_path}\n")
    else:
        print("\nNot saving outputs (save=false).")

    print("Done.")


if __name__ == "__main__":
    main()
