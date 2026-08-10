import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [self._tokenize(c["texto"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and extract alphanumeric words
        return re.findall(r'\w+', text.lower())

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
            
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                item = dict(self.chunks[idx])
                item["score"] = score
                item["idx"] = int(idx)
                results.append(item)
                
        return results

def reciprocal_rank_fusion(results_lists: List[List[Dict[str, Any]]], k0: float = 60.0) -> List[Dict[str, Any]]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
    Section 8.4 equation (7): s_RRF(c) = sum(1 / (k0 + r_j(c)))
    """
    scores_map = {}
    chunk_map = {}

    for r_list in results_lists:
        for rank, item in enumerate(r_list, start=1):
            c_id = item["chunk_id"]
            if c_id not in chunk_map:
                chunk_map[c_id] = item
                scores_map[c_id] = 0.0
            scores_map[c_id] += 1.0 / (k0 + rank)

    # Sort chunks by fused score descending
    sorted_chunk_ids = sorted(scores_map.keys(), key=lambda cid: scores_map[cid], reverse=True)
    
    fused_results = []
    for cid in sorted_chunk_ids:
        item = dict(chunk_map[cid])
        item["rrf_score"] = scores_map[cid]
        fused_results.append(item)

    return fused_results
