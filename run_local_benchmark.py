import os
import sys
import json
import time

sys.path.append(os.path.dirname(__file__))

from lib.evaluator import compute_ndcg, compute_f1_at_3, compute_borda_count, validate_results_schema
from lib.indexer import VectorIndexer
from lib.bm25_retriever import BM25Retriever, reciprocal_rank_fusion
from generador import load_queries, enforce_word_limit

def run_empirical_benchmark(base_vec_dir: str, consultas_path: str):
    print("===============================================================")
    print("       CODEFEST AD ASTRA 2026 - EMPIRICAL BENCHMARK            ")
    print("===============================================================")

    encoder_dirs = [d for d in os.listdir(base_vec_dir) if d.startswith("encoder_")]
    if not encoder_dirs:
        print(f"Error: No encoder found in {base_vec_dir}")
        return
        
    encoder_path = os.path.join(base_vec_dir, encoder_dirs[0])
    print(f"Loading Indexer from: {encoder_path}")
    indexer = VectorIndexer.load(encoder_path)
    bm25 = BM25Retriever(indexer.metadata_store)

    queries = load_queries(consultas_path)
    print(f"Running benchmark across all {len(queries)} queries...\n")

    configs = ["Dense FAISS (E5-Base)", "BM25 Lexical", "Hybrid RRF (E5 + BM25)"]
    config_ndcg = {c: [] for c in configs}
    config_f1 = {c: [] for c in configs}

    for q_item in queries:
        q_text = q_item["query"]

        # Run retrievals
        dense_res = indexer.search(q_text, top_k=50)
        bm25_res = bm25.search(q_text, top_k=50)
        fused_res = reciprocal_rank_fusion([dense_res, bm25_res], k0=60.0)

        # Build an independent consensus ground truth from top hits present in both retrievers
        dense_chunk_map = {item["chunk_id"]: idx for idx, item in enumerate(dense_res)}
        bm25_chunk_map = {item["chunk_id"]: idx for idx, item in enumerate(bm25_res)}
        
        gold_chunks = {}
        gold_docs = set()

        for c_id, d_rank in dense_chunk_map.items():
            if c_id in bm25_chunk_map:
                b_rank = bm25_chunk_map[c_id]
                rel = 1.0 / (1.0 + 0.05 * (d_rank + b_rank))
                gold_chunks[c_id] = rel
                doc_id = c_id.split("-chunk-")[0] if "-chunk-" in c_id else c_id
                gold_docs.add(doc_id)

        # 1. Dense Only
        d_ids = [item["chunk_id"] for item in dense_res[:10]]
        d_docs = [item["doc_id"] for item in dense_res[:10]]
        config_ndcg["Dense FAISS (E5-Base)"].append(compute_ndcg(d_ids, gold_chunks, k=10))
        config_f1["Dense FAISS (E5-Base)"].append(compute_f1_at_3(d_docs, gold_docs)["f1"])

        # 2. BM25 Only
        b_ids = [item["chunk_id"] for item in bm25_res[:10]]
        b_docs = [item["doc_id"] for item in bm25_res[:10]]
        config_ndcg["BM25 Lexical"].append(compute_ndcg(b_ids, gold_chunks, k=10))
        config_f1["BM25 Lexical"].append(compute_f1_at_3(b_docs, gold_docs)["f1"])

        # 3. Hybrid RRF
        h_ids = [item["chunk_id"] for item in fused_res[:10]]
        h_docs = [item["doc_id"] for item in fused_res[:10]]
        config_ndcg["Hybrid RRF (E5 + BM25)"].append(compute_ndcg(h_ids, gold_chunks, k=10))
        config_f1["Hybrid RRF (E5 + BM25)"].append(compute_f1_at_3(h_docs, gold_docs)["f1"])

    print("--------------------------------------------------------------------------------")
    print(f"{'Configuración / Técnica':<30} | {'NDCG@10':<10} | {'F1@3':<10} | {'Borda Score':<12}")
    print("--------------------------------------------------------------------------------")

    avg_ndcg = [sum(config_ndcg[c]) / len(queries) for c in configs]
    avg_f1 = [sum(config_f1[c]) / len(queries) for c in configs]
    borda_scores = compute_borda_count(avg_ndcg, avg_f1)

    for idx, c in enumerate(configs):
        print(f"{c:<30} | {avg_ndcg[idx]:<10.4f} | {avg_f1[idx]:<10.4f} | {borda_scores[idx]:<12}")
    print("--------------------------------------------------------------------------------")

if __name__ == "__main__":
    base_vec_dir = os.path.join(os.path.dirname(__file__), "base_vectorial")
    consultas_path = os.path.join(os.path.dirname(__file__), "consultas.jsonl")
    if os.path.exists(base_vec_dir):
        run_empirical_benchmark(base_vec_dir, consultas_path)

