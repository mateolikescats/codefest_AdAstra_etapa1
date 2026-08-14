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
                # Exclude evaluation question PDF chunks
                if "Extracto_Preguntas" in item.get("fuente", "") or item.get("doc_id") == "DOC-79283":
                    continue
                text = item["texto"].strip()
                words = text.split()
                # Filter out noise (snippets under 25 words or pure numeric lines)
                if len(words) >= 25:
                    doc_chunks[item["doc_id"]].append(item)

    print(f"Total raw valid chunks: {total_raw} across {len(doc_chunks)} documents.", flush=True)

    # Smart selection: keep up to 4 representative chunks per document to ensure 100% doc coverage
    selected_chunks = []
    for doc_id, c_list in doc_chunks.items():
        if len(c_list) <= 4:
            selected_chunks.extend(c_list)
        else:
            step = len(c_list) / 4.0
            indices = [int(i * step) for i in range(4)]
            selected_chunks.extend([c_list[idx] for idx in indices])

    print(f"Selected {len(selected_chunks)} high-density chunks across all {len(doc_chunks)} documents (100% doc coverage).", flush=True)

    # Encode with SOTA Multilingual E5 Small (512 max_seq_length, 384 dims, fast & accurate)
    model_name = "intfloat/multilingual-e5-small"
    indexer = VectorIndexer(model_name=model_name)
    indexer.build_index(selected_chunks, batch_size=32)




    safe_model_name = model_name.split("/")[-1].replace("-", "_")
    encoder_dir = os.path.join(output_base_dir, f"encoder_{safe_model_name}")
    indexer.save(encoder_dir)

    elapsed = time.time() - start_time
    print(f"\n=== HIGH-EFFICIENCY INDEXING COMPLETE IN {elapsed:.2f} SECONDS ===", flush=True)
    print(f"FAISS Base Vectorial Ready At: {encoder_dir}", flush=True)

if __name__ == "__main__":
    run_fast_indexing()


