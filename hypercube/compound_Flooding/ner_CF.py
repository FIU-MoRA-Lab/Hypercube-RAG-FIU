#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ner_CF.py
---------
Build Hypercube-RAG dimension files for the compound_flooding dataset.

Input:
    body_SciData_CF.txt  (one article per line)

Output (in this exact order):
    date.txt
    event.txt
    location.txt
    organization.txt
    person.txt
    theme.txt

Output directory:
    hypercube/compound_flooding/
"""

import os
import json
from typing import Dict, List
from tqdm import tqdm
import spacy

# -------------------------------------------------------
#  HARD-WIRED PATHS FOR YOUR SYSTEM
# -------------------------------------------------------

CORPUS_FILE = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Hypercube-RAG\corpus\compound_flooding\body_SciData_CF.txt"

DATASET = "compound_flooding"

HYPERCUBE_ROOT = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Hypercube-RAG\hypercube"

# -------------------------------------------------------
#  DIMENSIONS (ordered exactly as you requested)
# -------------------------------------------------------

DIMENSIONS = [
    "date",
    "event",
    "location",
    "organization",
    "person",
    "theme",
]

# Map spaCy entity types → dimensions
SPACY_TO_DIM = {
    "DATE": "date",
    "EVENT": "event",
    "GPE": "location",
    "LOC": "location",
    "ORG": "organization",
    "PERSON": "person",
}


def load_corpus(path: str) -> List[str]:
    print(f"Loading corpus from:\n{path}\n")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} documents.\n")
    return lines


def extract_entities(text: str, nlp) -> Dict[str, Dict[str, int]]:
    doc = nlp(text)
    dim_entities = {dim: {} for dim in DIMENSIONS}

    # Named entities → date, event, location, organization, person
    for ent in doc.ents:
        dim = SPACY_TO_DIM.get(ent.label_)
        if dim is None:
            continue
        key = ent.text.strip()
        if key:
            dim_entities[dim][key] = dim_entities[dim].get(key, 0) + 1

    # theme dimension → noun chunks
    seen = {k.lower() for dim in DIMENSIONS for k in dim_entities[dim]}
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


def write_hypercube(dim_records: Dict[str, List[Dict[str, int]]]):
    out_dir = os.path.join(HYPERCUBE_ROOT, DATASET)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Writing hypercube files to:\n{out_dir}\n")

    for dim in DIMENSIONS:
        out_path = os.path.join(out_dir, f"{dim}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for record in dim_records[dim]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✔ Wrote {out_path}")


def main():
    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    corpus = load_corpus(CORPUS_FILE)

    dim_records = {dim: [] for dim in DIMENSIONS}

    print("Extracting entities and building hypercube...")
    for text in tqdm(corpus):
        record = extract_entities(text, nlp)
        for dim in DIMENSIONS:
            dim_records[dim].append(record.get(dim, {}))

    write_hypercube(dim_records)
    print("\nDONE! Hypercube for compound_flooding is ready.\n")


if __name__ == "__main__":
    main()
