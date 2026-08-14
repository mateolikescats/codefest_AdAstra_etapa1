import os
import json
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Maximize PyTorch CPU thread count across all available cores
torch.set_num_threads(os.cpu_count() or 8)

class VectorIndexer:
    def __init__(self, model_name: str = "BAAI/bge-m3"):


        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        try:
            self.dimension = self.model.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 128):
        """
        Embeds chunks, normalizes vectors, adds them to FAISS IndexFlatIP,
        and stores metadata.
        """
        if not chunks:
            return

        is_e5 = "e5" in self.model_name.lower()
        is_bge = "bge" in self.model_name.lower()
        if is_e5:
            texts = [f"passage: {c['texto']}" for c in chunks]
        elif is_bge:
            texts = [c["texto"] for c in chunks]  # BGE-M3 no requiere prefijo para passages
        else:
            texts = [c["texto"] for c in chunks]

        print(f"Generating optimized embeddings for {len(texts)} chunks using {self.model_name} (max_seq_length=512, batch_size={batch_size})...", flush=True)
        
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
    def load(cls, input_dir: str, model_name: str = None):
        """
        Loads an existing index.faiss and metadata.jsonl, auto-detecting model_name.
        """
        faiss_path = os.path.join(input_dir, "index.faiss")
        meta_path = os.path.join(input_dir, "metadata.jsonl")

        temp_index = faiss.read_index(faiss_path)
        dim = temp_index.d

        if model_name is None:
            dir_name = os.path.basename(os.path.normpath(input_dir)).lower()
            if "bge" in dir_name or dim == 1024:
                model_name = "BAAI/bge-m3"
            elif "e5_base" in dir_name or dim == 768:
                model_name = "intfloat/multilingual-e5-base"
            elif "e5_small" in dir_name or dim == 384:
                model_name = "intfloat/multilingual-e5-small"
            else:
                model_name = "BAAI/bge-m3"

        obj = cls(model_name=model_name)
        obj.index = temp_index
        obj.metadata_store = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj.metadata_store.append(json.loads(line.strip()))
                    
        assert obj.index.ntotal == len(obj.metadata_store), f"Mismatch between FAISS index ({obj.index.ntotal}) and metadata store ({len(obj.metadata_store)})"
        return obj


    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        is_e5 = "e5" in self.model_name.lower()
        is_bge = "bge" in self.model_name.lower()
        if is_e5:
            query_text = f"query: {query}"
        elif is_bge:
            query_text = f"Represent this sentence for searching relevant passages: {query}"
        else:
            query_text = query

        with torch.inference_mode():
            query_vec = self.model.encode([query_text], normalize_embeddings=True)
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

