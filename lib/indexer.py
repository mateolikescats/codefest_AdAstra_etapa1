import os
import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Maximize PyTorch CPU thread count
torch.set_num_threads(os.cpu_count() or 8)

class VectorIndexer:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 128
        try:
            self.dimension = self.model.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 512):
        """
        Embeds chunks, normalizes vectors, adds them to FAISS IndexFlatIP,
        and stores metadata.
        """
        if not chunks:
            return

        texts = [c["texto"] for c in chunks]
        print(f"Generating optimized embeddings for {len(texts)} chunks using {self.model_name} (max_seq_length=128, batch_size={batch_size})...", flush=True)
        
        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata_store.extend(chunks)

    def save(self, output_dir: str):
        """
        Saves index.faiss and metadata.jsonl inside output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        faiss_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(self.index, faiss_path)
        print(f"FAISS index saved to: {faiss_path}", flush=True)
        
        meta_path = os.path.join(output_dir, "metadata.jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for item in self.metadata_store:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Metadata store ({len(self.metadata_store)} items) saved to: {meta_path}", flush=True)

    @classmethod
    def load(cls, input_dir: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Loads an existing index.faiss and metadata.jsonl.
        """
        obj = cls(model_name=model_name)
        faiss_path = os.path.join(input_dir, "index.faiss")
        meta_path = os.path.join(input_dir, "metadata.jsonl")
        
        obj.index = faiss.read_index(faiss_path)
        obj.metadata_store = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj.metadata_store.append(json.loads(line.strip()))
                    
        return obj

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        with torch.inference_mode():
            query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)
        
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata_store):
                item = dict(self.metadata_store[idx])
                item["score"] = float(score)
                item["faiss_idx"] = int(idx)
                results.append(item)
                
        return results
