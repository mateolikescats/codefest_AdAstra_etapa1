import os
import sys
import time
import json
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))

from lib.indexer import VectorIndexer

def run_fast_indexing():
    start_time = time.time()
    base_dir = os.path.dirname(__file__)
    cache_path = os.path.join(base_dir, "base_vectorial", "chunks_cache.jsonl")
    output_base_dir = os.path.join(base_dir, "base_vectorial")

    print(f"=== STARTING HIGH-EFFICIENCY CODEFEST VECTOR INDEXING ===", flush=True)
    print(f"Reading cached chunks from: {cache_path}", flush=True)

    doc_chunks = defaultdict(list)
    total_raw = 0

    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_raw += 1
                item = json.loads(line.strip())
                text = item["texto"].strip()
                words = text.split()
                # Filter out noise (very short snippets or TOC filler)
                if len(words) >= 20:
                    doc_chunks[item["doc_id"]].append(item)

    print(f"Total raw chunks: {total_raw} across {len(doc_chunks)} documents.", flush=True)

    # Sample up to 12 representative chunks per document to guarantee 100% doc coverage
    sampled_chunks = []
    for doc_id, c_list in doc_chunks.items():
        if len(c_list) <= 12:
            sampled_chunks.extend(c_list)
        else:
            step = len(c_list) / 12.0
            indices = [int(i * step) for i in range(12)]
            sampled_chunks.extend([c_list[idx] for idx in indices])

    print(f"Sampled {len(sampled_chunks)} representative chunks across all {len(doc_chunks)} documents.", flush=True)

    # Encode with SOTA Multilingual MiniLM L12 v2
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    indexer = VectorIndexer(model_name=model_name)
    indexer.build_index(sampled_chunks, batch_size=256)

    # Save to base_vectorial/encoder_paraphrase_multilingual_MiniLM_L12_v2
    encoder_dir = os.path.join(output_base_dir, "encoder_paraphrase_multilingual_MiniLM_L12_v2")
    indexer.save(encoder_dir)

    elapsed = time.time() - start_time
    print(f"\n=== HIGH-EFFICIENCY INDEXING COMPLETE IN {elapsed:.2f} SECONDS ===", flush=True)
    print(f"FAISS Base Vectorial Ready At: {encoder_dir}", flush=True)

if __name__ == "__main__":
    run_fast_indexing()
