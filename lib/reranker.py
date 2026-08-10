from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        if not candidate_chunks:
            return []

        pairs = [[query, c["texto"]] for c in candidate_chunks]
        scores = self.model.predict(pairs)

        reranked = []
        for score, chunk in zip(scores, candidate_chunks):
            c_copy = dict(chunk)
            c_copy["cross_score"] = float(score)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["cross_score"], reverse=True)
        return reranked[:top_k]
