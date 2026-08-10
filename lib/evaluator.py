import os
import sys
import json
import math
from typing import List, Dict, Set, Any

def compute_dcg(relevances: List[float], k: int = 10) -> float:
    dcg = 0.0
    for i, r in enumerate(relevances[:k], start=1):
        dcg += r / math.log2(i + 1)
    return dcg

def compute_ndcg(retrieved_ids: List[str], ground_truth_relevance: Dict[str, float], k: int = 10) -> float:
    retrieved_rels = [ground_truth_relevance.get(cid, 0.0) for cid in retrieved_ids[:k]]
    actual_dcg = compute_dcg(retrieved_rels, k=k)
    
    ideal_rels = sorted(ground_truth_relevance.values(), reverse=True)[:k]
    ideal_dcg = compute_dcg(ideal_rels, k=k)
    
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

def compute_f1_at_3(retrieved_doc_ids: List[str], ground_truth_docs: Set[str]) -> Dict[str, float]:
    top_3 = retrieved_doc_ids[:3]
    relevant_retrieved = len(set(top_3).intersection(ground_truth_docs))
    
    precision = relevant_retrieved / 3.0
    denom_r = min(len(ground_truth_docs), 3)
    recall = (relevant_retrieved / float(denom_r)) if denom_r > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2.0 * precision * recall) / (precision + recall)
        
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_borda_count(scores_ndcg: List[float], scores_f1: List[float]) -> List[int]:
    """
    Computes Borda count total points for N configurations evaluated across NDCG and F1.
    Section 11.2 Conteo de Borda formula: B_i = B_NDCG_i + B_F1_i
    """
    N = len(scores_ndcg)
    # Sort indices descending by NDCG
    rank_ndcg = sorted(range(N), key=lambda i: scores_ndcg[i], reverse=True)
    borda_ndcg = [0] * N
    for p, idx in enumerate(rank_ndcg):
        borda_ndcg[idx] = N - 1 - p
        
    # Sort indices descending by F1
    rank_f1 = sorted(range(N), key=lambda i: scores_f1[i], reverse=True)
    borda_f1 = [0] * N
    for p, idx in enumerate(rank_f1):
        borda_f1[idx] = N - 1 - p
        
    borda_total = [borda_ndcg[i] + borda_f1[i] for i in range(N)]
    return borda_total

def validate_results_schema(jsonl_path: str) -> Dict[str, Any]:
    """
    Validates that output jsonl complies 100% with Section 9 & 10 rules.
    """
    if not os.path.exists(jsonl_path):
        return {"valid": False, "error": f"File {jsonl_path} does not exist"}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != 50:
        return {"valid": False, "error": f"Expected exactly 50 lines, got {len(lines)}"}

    errors = []
    for line_idx, line in enumerate(lines, start=1):
        try:
            obj = json.loads(line)
        except Exception as e:
            errors.append(f"Line {line_idx}: Invalid JSON syntax - {str(e)}")
            continue

        if "query_id" not in obj or "documents" not in obj or "fragments" not in obj:
            errors.append(f"Line {line_idx}: Missing top-level schema fields")
            continue

        docs = obj["documents"]
        frags = obj["fragments"]

        if len(docs) != 3:
            errors.append(f"Line {line_idx}: Expected 3 documents, got {len(docs)}")

        if len(frags) != 10:
            errors.append(f"Line {line_idx}: Expected 10 fragments, got {len(frags)}")

        for f_idx, frag in enumerate(frags, start=1):
            text = frag.get("text", "")
            words = text.split()
            if len(words) > 250:
                errors.append(f"Line {line_idx}, Fragment {f_idx}: Exceeds 250 words limit ({len(words)} words)")

    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True, "num_queries": len(lines)}
