#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ner_CF.py
---------
Build Hypercube-RAG dimension files for the compound_flooding dataset
with BOTH text-based dimensions (NER + noun chunks) and physical data
summarized per document (Option A: true multimodal hypercube).

Each document i in body_SciData_CF_filtered.txt gets:
  - Text dims: date, event, location, organization, person, theme
  - Physical dims (new, optional if metadata is filled):
        coops_level      (binned CO-OPS water level)
        imerg_precip     (binned IMERG half-hour precip)
        mrms_precip      (binned MRMS half-hour precip)

Physical dims are derived by:
  - A doc → (station_id, start_time, end_time) mapping (metadata CSV)
  - Half-hourly physical records filtered by that station & time window
  - Binning continuous values into categories; counts per bin become labels.

Outputs (one JSON-per-line file per dimension, in this exact order):
    date.txt
    event.txt
    location.txt
    organization.txt
    person.txt
    theme.txt
    coops_level.txt
    imerg_precip.txt
    mrms_precip.txt

Output directory:
    hypercube/compound_flooding/
"""

import os
import json
from typing import Dict, List

from tqdm import tqdm
import spacy
import pandas as pd

# -------------------------------------------------------
#  HARD-WIRED PATHS FOR YOUR SYSTEM
# -------------------------------------------------------

# Corpus: one article per line
CORPUS_FILE = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Hypercube-RAG-FIU\corpus\compound_flooding\body_SciData_CF_filtered.txt"

DATASET = "compound_flooding"

HYPERCUBE_ROOT = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Hypercube-RAG-FIU\hypercube"

# Metadata mapping documents → (station_id, start_time, end_time)
# CSV with at least:
#   doc_index, station_id, start_time, end_time
DOC_META_FILE = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Hypercube-RAG-FIU\corpus\compound_flooding\doc_metadata_CF.csv"

DOC_META_DOCIDX_COL = "doc_index"
DOC_META_STATION_COL = "station_id"
DOC_META_STARTTIME_COL = "start_time"
DOC_META_ENDTIME_COL = "end_time"

# Physical data files (half-hour / 30-min)
COOPS_FILE = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Physical Data\co-op_water_levels_halfhour.csv"
IMERG_FILE = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Physical Data\imerg_final_halfhourly_FL_COOPS25mi_20250601_20250630.csv"
MRMS_FILE = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Physical Data\mrms_preciprate_COOPS_30min_20250601_20250630.csv"
USGS_FILE = r"C:\Users\mokol\Desktop\Current FIU Classes\IDC6940 Capstone Project\Data Scraper\Physical Data\usgs_physical_20250601_20250630.tsv"

# -------------------------------------------------------
#  DIMENSIONS (text + physical)
# -------------------------------------------------------

# Text-based dimensions (original)
BASE_TEXT_DIMENSIONS = [
    "date",
    "event",
    "location",
    "organization",
    "person",
    "theme",
]

# Map spaCy entity types → text dimensions
SPACY_TO_DIM = {
    "DATE": "date",
    "EVENT": "event",
    "GPE": "location",
    "LOC": "location",
    "ORG": "organization",
    "PERSON": "person",
}

# Physical variables we want to summarize per document.
# keys: numeric columns in the combined physical dataframe
# values: dimension names in the hypercube
PHYSICAL_VAR_CONFIG = {
    "coops_water_level": "coops_level",
    "imerg_precip_mm": "imerg_precip",
    "mrms_precip_mm": "mrms_precip",
    # Add more if you want, e.g. "usgs_stage": "usgs_stage"
}

PHYSICAL_DIMENSIONS = list(PHYSICAL_VAR_CONFIG.values())

# Final ordered list of all dimensions for the hypercube
DIMENSIONS = BASE_TEXT_DIMENSIONS + PHYSICAL_DIMENSIONS


# =======================================================
#               TEXT NER → HYPERCUBE
# =======================================================

def load_corpus(path: str) -> List[str]:
    print(f"Loading corpus from:\n{path}\n")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} documents.\n")
    return lines


def extract_entities(text: str, nlp) -> Dict[str, Dict[str, int]]:
    """
    Extract text-based dimensions from spaCy:
      - NER: date, event, location, organization, person
      - theme: noun chunks not already captured as entities
    Returns: dict(dim_name -> {label -> count}).
    """
    doc = nlp(text)
    dim_entities = {dim: {} for dim in DIMENSIONS}  # pre-create, physical dims filled later

    # Named entities → date, event, location, organization, person
    for ent in doc.ents:
        dim = SPACY_TO_DIM.get(ent.label_)
        if dim is None:
            continue
        key = ent.text.strip()
        if key:
            dim_entities[dim][key] = dim_entities[dim].get(key, 0) + 1

    # theme dimension → noun chunks
    seen = {k.lower() for dim in BASE_TEXT_DIMENSIONS for k in dim_entities[dim]}
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


# =======================================================
#               DOC METADATA HELPERS
# =======================================================

def load_doc_metadata(path: str, num_docs: int) -> pd.DataFrame:
    """
    Load doc metadata and ensure it has a doc_index column aligned
    with corpus line indices (0..num_docs-1).

    If the file does not exist, create a TEMPLATE CSV with the correct
    columns for the user to fill in and return an empty DataFrame.
    In that case, physical dimensions will be skipped (text-only hypercube).
    """
    if not os.path.exists(path):
        template_path = path
        print(
            "================================================================\n"
            "Doc metadata CSV not found.\n"
            f"  Expected: {path}\n\n"
            "Creating a TEMPLATE metadata file with the following columns:\n"
            f"  {DOC_META_DOCIDX_COL}, {DOC_META_STATION_COL}, "
            f"{DOC_META_STARTTIME_COL}, {DOC_META_ENDTIME_COL}\n"
            "One row is created per document (0..N-1).\n"
            "Fill in station_id, start_time, and end_time for each doc and\n"
            "re-run ner_CF.py to attach physical dimensions.\n"
            "Until then, the script will proceed in TEXT-ONLY mode.\n"
            "================================================================\n"
        )

        tmpl = pd.DataFrame({
            DOC_META_DOCIDX_COL: list(range(num_docs)),
            DOC_META_STATION_COL: ["" for _ in range(num_docs)],
            DOC_META_STARTTIME_COL: ["" for _ in range(num_docs)],
            DOC_META_ENDTIME_COL: ["" for _ in range(num_docs)],
        })
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        tmpl.to_csv(template_path, index=False)
        print(f"✔ Wrote metadata template to:\n  {template_path}\n")

        # Empty index → no physical dims will be attached
        return pd.DataFrame(
            columns=[
                DOC_META_STATION_COL,
                DOC_META_STARTTIME_COL,
                DOC_META_ENDTIME_COL,
            ]
        ).set_index(pd.Index([], name=DOC_META_DOCIDX_COL))

    # ---------- Normal path: metadata exists ----------
    df = pd.read_csv(path)

    if DOC_META_DOCIDX_COL not in df.columns:
        print(
            f"[WARN] {DOC_META_DOCIDX_COL} column missing in metadata; "
            "assuming row order aligns with corpus indices."
        )
        df[DOC_META_DOCIDX_COL] = range(len(df))

    # Parse times; allow blanks
    df[DOC_META_STARTTIME_COL] = pd.to_datetime(
        df[DOC_META_STARTTIME_COL], errors="coerce"
    )
    df[DOC_META_ENDTIME_COL] = pd.to_datetime(
        df[DOC_META_ENDTIME_COL], errors="coerce"
    )

    if df[DOC_META_DOCIDX_COL].max() >= num_docs:
        print(
            "[WARN] Some doc_index values exceed corpus length. "
            "Ensure metadata aligns with corpus ordering."
        )

    return df.set_index(DOC_META_DOCIDX_COL)


# =======================================================
#               PHYSICAL DATA HELPERS
# =======================================================

def _guess_column(df: pd.DataFrame, candidates, kind: str) -> str:
    """
    Try to find a column in df whose name matches one of the candidates
    (case-insensitive). Raise a clear error if none are found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    raise KeyError(
        f"Could not find a '{kind}' column in dataframe.\n"
        f"Tried candidates: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def load_coops_water_levels(path: str) -> pd.DataFrame:
    """
    Load CO-OPS water level data and return:
        station_id | datetime | coops_water_level

    Auto-detects likely station / time / water level columns.
    """
    df = pd.read_csv(path)

    station_col = _guess_column(
        df,
        candidates=["station_id", "station", "id", "gauge_id"],
        kind="CO-OPS station id",
    )
    time_col = _guess_column(
        df,
        candidates=["datetime", "time", "dateTime", "DateTime", "timestamp"],
        kind="CO-OPS datetime",
    )
    wl_col = _guess_column(
        df,
        candidates=["water_level", "water_level_m", "waterlevel", "wl", "value"],
        kind="CO-OPS water level",
    )

    df[time_col] = pd.to_datetime(df[time_col])

    df = df[[station_col, time_col, wl_col]].copy()
    df.rename(
        columns={
            station_col: "station_id",
            time_col: "datetime",
            wl_col: "coops_water_level",
        },
        inplace=True,
    )
    return df


def load_imerg(path: str) -> pd.DataFrame:
    """
    Load IMERG half-hourly rainfall and return:
        station_id | datetime | imerg_precip_mm

    For your current file, we treat:
      - 'point' as a station-like identifier
      - 'time'  as the timestamp
      - 'precip_mm_per_hr' (or similar) as the precipitation column

    NOTE: We keep the values as-is and rename to 'imerg_precip_mm'
    even if the original units are mm/hr; since we only use them
    for relative binning, a constant scale factor does not affect
    the quantile bins. If you want true mm over 30 min, you can
    multiply by 0.5 before renaming.
    """
    df = pd.read_csv(path)

    # Include 'point' as a candidate for station id,
    # and 'precip_mm_per_hr' as a candidate for precipitation.
    station_col = _guess_column(
        df,
        candidates=["station_id", "coops_id", "gauge_id", "station", "point"],
        kind="IMERG station id",
    )
    time_col = _guess_column(
        df,
        candidates=["datetime", "time", "dateTime", "DateTime", "timestamp"],
        kind="IMERG datetime",
    )
    precip_col = _guess_column(
        df,
        candidates=["precip_mm", "precipitation", "precip", "rain_mm", "value", "precip_mm_per_hr"],
        kind="IMERG precipitation",
    )

    df[time_col] = pd.to_datetime(df[time_col])

    # If you want actual mm over a 30-min window instead of mm/hr,
    # uncomment the next line:
    # df[precip_col] = df[precip_col] * 0.5

    df = df[[station_col, time_col, precip_col]].copy()
    df.rename(
        columns={
            station_col: "station_id",
            time_col: "datetime",
            precip_col: "imerg_precip_mm",
        },
        inplace=True,
    )
    return df


def load_mrms(path: str) -> pd.DataFrame:
    """
    Load MRMS 30-min precipitation rate and return:
        station_id | datetime | mrms_precip_mm

    For your current file, we treat:
      - 'station_id' as the station identifier
      - 'time'       as the timestamp
      - 'precip_rate' as the precipitation variable

    We keep the values as-is and rename to 'mrms_precip_mm' for consistency,
    even if the original units are a rate; since we only use them for
    relative binning, a constant scale factor does not affect the bins.
    """
    df = pd.read_csv(path)

    # Include 'precip_rate' as a candidate for precipitation
    station_col = _guess_column(
        df,
        candidates=["station_id", "coops_id", "gauge_id", "station"],
        kind="MRMS station id",
    )
    time_col = _guess_column(
        df,
        candidates=["datetime", "time", "dateTime", "DateTime", "timestamp"],
        kind="MRMS datetime",
    )
    precip_col = _guess_column(
        df,
        candidates=[
            "precip_mm",
            "precipitation",
            "precip",
            "rain_mm",
            "value",
            "precip_rate",
        ],
        kind="MRMS precipitation",
    )

    df[time_col] = pd.to_datetime(df[time_col])

    # If you want to convert a rate to an amount per 30 minutes,
    # you could adjust here (e.g., multiply by 0.5), but it's not
    # necessary for quantile-based binning.
    # df[precip_col] = df[precip_col] * 0.5

    df = df[[station_col, time_col, precip_col]].copy()
    df.rename(
        columns={
            station_col: "station_id",
            time_col: "datetime",
            precip_col: "mrms_precip_mm",
        },
        inplace=True,
    )
    return df



def load_usgs_physical(path: str) -> pd.DataFrame:
    """
    Load USGS physical data (TSV) and return:
        station_id | datetime | usgs_<variable>...

    Assumes a typical USGS/NWIS RDB-like format, where:
      - Lines starting with '#' are comments
      - The first non-comment line contains column headers such as:
          'agency_cd', 'site_no', 'dateTime', ...

    If parsing fails (or we can't find site/time columns), we return
    an EMPTY dataframe but let the pipeline continue.
    """
    # First try: standard NWIS / USGS RDB-style parsing
    try:
        df = pd.read_csv(path, sep="\t", comment="#")
    except Exception as e:
        print(f"[WARN] Could not read USGS file '{path}' as TSV with comments: {e}")
        return pd.DataFrame(columns=["station_id", "datetime"])

    # If we still only have a single weird column, bail out gracefully
    if df.shape[1] == 1 and list(df.columns)[0] == "#":
        print(
            "[WARN] USGS file appears to have only a single '#' column even after "
            "skipping comments. Check the file format. Skipping USGS data for now."
        )
        return pd.DataFrame(columns=["station_id", "datetime"])

    # Try to guess site and time columns
    try:
        site_col = _guess_column(
            df,
            candidates=["site_no", "station_id", "site", "gage_id", "gauge_id"],
            kind="USGS site id",
        )
        time_col = _guess_column(
            df,
            candidates=["datetime", "dateTime", "time", "DateTime", "timestamp"],
            kind="USGS datetime",
        )
    except KeyError as e:
        print(f"[WARN] {e}\nSkipping USGS data for now.")
        return pd.DataFrame(columns=["station_id", "datetime"])

    # Parse time
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    feature_cols = [c for c in df.columns if c not in [site_col, time_col]]

    if not feature_cols:
        print("[WARN] No USGS feature columns found; skipping USGS data.")
        return pd.DataFrame(columns=["station_id", "datetime"])

    base = df[[site_col, time_col] + feature_cols].copy()
    base.rename(
        columns={
            site_col: "station_id",
            time_col: "datetime",
        },
        inplace=True,
    )

    # Prefix USGS features so they don't collide with others
    rename_map = {c: f"usgs_{c}" for c in feature_cols}
    base.rename(columns=rename_map, inplace=True)

    return base

def _normalize_datetime_col(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    """
    Ensure df[col] is timezone-naive datetime64[ns] for consistent merging.
    We parse with utc=True (so 'Z' / timezone strings are handled),
    then drop the timezone information.
    """
    if col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df[col] = df[col].dt.tz_convert(None)
    return df


def build_physical_df() -> pd.DataFrame:
    """
    Outer-join all physical sources on (station_id, datetime),
    after normalizing datetime columns to be timezone-naive.
    """
    print("Loading physical datasets...")

    coops = load_coops_water_levels(COOPS_FILE)
    print(f"  CO-OPS rows: {len(coops)}")

    imerg = load_imerg(IMERG_FILE)
    print(f"  IMERG rows: {len(imerg)}")

    mrms = load_mrms(MRMS_FILE)
    print(f"  MRMS rows: {len(mrms)}")

    usgs = load_usgs_physical(USGS_FILE)
    print(f"  USGS rows: {len(usgs)}")

    # Normalize datetime columns across all sources
    coops = _normalize_datetime_col(coops, "datetime")
    imerg = _normalize_datetime_col(imerg, "datetime")
    mrms = _normalize_datetime_col(mrms, "datetime")
    usgs = _normalize_datetime_col(usgs, "datetime")

    # Outer join so we don't lose rows if one dataset is missing a timestamp
    df = coops.merge(imerg, on=["station_id", "datetime"], how="outer")
    df = df.merge(mrms, on=["station_id", "datetime"], how="outer")
    df = df.merge(usgs, on=["station_id", "datetime"], how="outer")

    df.sort_values(["station_id", "datetime"], inplace=True)
    print(f"Combined physical rows: {len(df)}\n")
    return df



def add_binned_physical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each numeric physical variable named in PHYSICAL_VAR_CONFIG,
    create a binned categorical column "<var>_bin" using quantile bins.

    NOTE: For real use, you may want to replace qcut with physically
    meaningful thresholds (e.g. NWS flood stages).
    """
    for var_col in PHYSICAL_VAR_CONFIG.keys():
        if var_col not in df.columns:
            print(f"[WARN] Physical column '{var_col}' not found; skipping.")
            continue

        series = df[var_col].dropna()
        if series.nunique() < 2:
            print(f"[WARN] Physical column '{var_col}' has <2 unique values; skipping.")
            continue

        try:
            df[f"{var_col}_bin"] = pd.qcut(df[var_col], q=4, duplicates="drop")
            df[f"{var_col}_bin"] = df[f"{var_col}_bin"].astype(str)
            print(f"  Binned '{var_col}' into '{var_col}_bin' using quantiles.")
        except ValueError as e:
            print(f"[WARN] Could not bin '{var_col}': {e}")

    return df


def attach_physical_dims_for_doc(
    doc_idx: int,
    dim_record: Dict[str, Dict[str, int]],
    meta_df_idx: pd.DataFrame,
    physical_df: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """
    For a given document index, use metadata to find (station_id, time window),
    then aggregate binned physical features over that window, populating
    the physical dimensions in dim_record.
    """
    if doc_idx not in meta_df_idx.index:
        # No metadata → no physical dims
        return dim_record

    meta_row = meta_df_idx.loc[doc_idx]
    station_id = meta_row[DOC_META_STATION_COL]
    start_time = meta_row[DOC_META_STARTTIME_COL]
    end_time = meta_row[DOC_META_ENDTIME_COL]

    # If any of these are missing/NaT, skip
    if pd.isna(station_id) or pd.isna(start_time) or pd.isna(end_time):
        return dim_record
    if station_id == "":
        return dim_record

    subset = physical_df[
        (physical_df["station_id"] == station_id)
        & (physical_df["datetime"] >= start_time)
        & (physical_df["datetime"] <= end_time)
    ]

    if subset.empty:
        return dim_record

    for var_col, dim_name in PHYSICAL_VAR_CONFIG.items():
        bin_col = f"{var_col}_bin"
        if bin_col not in subset.columns:
            continue

        counts = subset[bin_col].value_counts()
        if counts.empty:
            continue

        dim_record[dim_name] = {str(k): int(v) for k, v in counts.items()}

    return dim_record


# =======================================================
#                      MAIN
# =======================================================

def main():
    # ------------------ TEXT NER → DIMENSION FILES ------------------
    print("Loading spaCy model (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    corpus = load_corpus(CORPUS_FILE)
    num_docs = len(corpus)

    # Container for all dimensions
    dim_records: Dict[str, List[Dict[str, int]]] = {dim: [] for dim in DIMENSIONS}

    # ------------------ LOAD METADATA ------------------
    print("Loading document metadata (station/time windows)...")
    meta_df_idx = load_doc_metadata(DOC_META_FILE, num_docs)
    has_physical = not meta_df_idx.empty

    if has_physical:
        print("Metadata found; physical dimensions will be attached.\n")
        print("Building combined physical dataframe...")
        physical_df = build_physical_df()
        physical_df = add_binned_physical_columns(physical_df)
    else:
        print("No metadata (or empty). Running in TEXT-ONLY mode.\n")
        physical_df = None

    # ------------------ PROCESS DOCUMENTS ------------------
    print("Extracting entities, attaching physical dims (if available), and building hypercube...")
    for doc_idx, text in tqdm(enumerate(corpus), total=num_docs):
        # Text-based dims
        record = extract_entities(text, nlp)

        # Physical dims (only if metadata + physical_df are available)
        if has_physical and physical_df is not None:
            record = attach_physical_dims_for_doc(
                doc_idx=doc_idx,
                dim_record=record,
                meta_df_idx=meta_df_idx,
                physical_df=physical_df,
            )

        # Append per-dimension records
        for dim in DIMENSIONS:
            dim_records[dim].append(record.get(dim, {}))

    # ------------------ WRITE HYPERCUBE ------------------
    write_hypercube(dim_records)
    print("\nDONE! Hypercube for compound_flooding is ready.\n")


if __name__ == "__main__":
    main()
