from typing import List, Dict, Any

class CrossEncoderReranker:
    """
    Pass-through Reranker.
    Cross-Encoder models were removed to strictly comply with Section 8.3
    (prohibition of generative/decoder/LLM reranking models).
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        return candidate_chunks[:top_k]

