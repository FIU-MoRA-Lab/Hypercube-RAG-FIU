import json
import os

# Paths to your new QA files (these were saved after you reran QA)
NO_RAG = r"QA\compound_flooding\compound_flooding_gpt-4o_none.json"
HYP = r"QA\compound_flooding\compound_flooding_gpt-4o_hypercube.json"

# Load files
with open(NO_RAG, "r", encoding="utf-8") as f:
    base = json.load(f)

with open(HYP, "r", encoding="utf-8") as f:
    rag = json.load(f)

same = 0
diff_samples = []  # store sample indices that differ

for i, (b, r) in enumerate(zip(base, rag)):
    pred_base = b.get("answer_pred", "").strip()
    pred_rag = r.get("answer_pred", "").strip()

    if pred_base == pred_rag:
        same += 1
    else:
        diff_samples.append(i)

print(f"\nIdentical predictions: {same} / {len(base)}")
print(f"Different predictions: {len(diff_samples)} / {len(base)}")

# Show 5 examples of different ones
print("\n=== SAMPLE DIFFERENCES (first 5) ===")
for i in diff_samples[:5]:
    print(f"\n--- Sample {i} ---")
    print("Question: ", base[i]["question"])
    print("No Retrieval Pred:", base[i]["answer_pred"])
    print("Hypercube Pred  :", rag[i]["answer_pred"])
