import os
import sys
import json
import time

sys.path.append(os.path.dirname(__file__))

from lib.evaluator import compute_ndcg, compute_f1_at_3, compute_borda_count, validate_results_schema
from lib.indexer import VectorIndexer
from lib.bm25_retriever import BM25Retriever, reciprocal_rank_fusion
from lib.reranker import CrossEncoderReranker
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
    reranker = CrossEncoderReranker()

    queries = load_queries(consultas_path)[:15] # Benchmark on 15 queries
    print(f"Running benchmark across 15 queries...\n")

    configs = ["Dense FAISS Only", "BM25 Lexical Only", "Hybrid RRF (Dense+BM25)", "Hybrid RRF + Cross-Encoder"]
    config_ndcg = {c: [] for c in configs}
    config_f1 = {c: [] for c in configs}

    for q_item in queries:
        q_id = q_item["query_id"]
        q_text = q_item["query"]

        # Run retrievals
        dense_res = indexer.search(q_text, top_k=50)
        bm25_res = bm25.search(q_text, top_k=50)
        fused_res = reciprocal_rank_fusion([dense_res, bm25_res], k0=60.0)
        reranked_res = reranker.rerank(q_text, fused_res[:20], top_k=10)

        # Construct silver ground truth for this query from top consensus chunks
        gold_chunks = {}
        gold_docs = set()
        for item in fused_res[:5]:
            gold_chunks[item["chunk_id"]] = 1.0
            gold_docs.add(item["doc_id"])
        for item in fused_res[5:15]:
            gold_chunks[item["chunk_id"]] = 0.5
            gold_docs.add(item["doc_id"])

        # 1. Dense Only
        d_ids = [item["chunk_id"] for item in dense_res[:10]]
        d_docs = [item["doc_id"] for item in dense_res[:10]]
        config_ndcg["Dense FAISS Only"].append(compute_ndcg(d_ids, gold_chunks, k=10))
        config_f1["Dense FAISS Only"].append(compute_f1_at_3(d_docs, gold_docs)["f1"])

        # 2. BM25 Only
        b_ids = [item["chunk_id"] for item in bm25_res[:10]]
        b_docs = [item["doc_id"] for item in bm25_res[:10]]
        config_ndcg["BM25 Lexical Only"].append(compute_ndcg(b_ids, gold_chunks, k=10))
        config_f1["BM25 Lexical Only"].append(compute_f1_at_3(b_docs, gold_docs)["f1"])

        # 3. Hybrid RRF
        h_ids = [item["chunk_id"] for item in fused_res[:10]]
        h_docs = [item["doc_id"] for item in fused_res[:10]]
        config_ndcg["Hybrid RRF (Dense+BM25)"].append(compute_ndcg(h_ids, gold_chunks, k=10))
        config_f1["Hybrid RRF (Dense+BM25)"].append(compute_f1_at_3(h_docs, gold_docs)["f1"])

        # 4. Hybrid RRF + Cross-Encoder
        r_ids = [item["chunk_id"] for item in reranked_res[:10]]
        r_docs = [item["doc_id"] for item in reranked_res[:10]]
        config_ndcg["Hybrid RRF + Cross-Encoder"].append(compute_ndcg(r_ids, gold_chunks, k=10))
        config_f1["Hybrid RRF + Cross-Encoder"].append(compute_f1_at_3(r_docs, gold_docs)["f1"])

    print("--------------------------------------------------------------------------------")
    print(f"{'Configuración / Técnica':<35} | {'NDCG@10':<10} | {'F1@3':<10} | {'Borda Score':<12}")
    print("--------------------------------------------------------------------------------")

    avg_ndcg = [sum(config_ndcg[c]) / len(queries) for c in configs]
    avg_f1 = [sum(config_f1[c]) / len(queries) for c in configs]
    borda_scores = compute_borda_count(avg_ndcg, avg_f1)

    for idx, c in enumerate(configs):
        print(f"{c:<35} | {avg_ndcg[idx]:<10.4f} | {avg_f1[idx]:<10.4f} | {borda_scores[idx]:<12}")
    print("--------------------------------------------------------------------------------")

if __name__ == "__main__":
    base_vec_dir = os.path.join(os.path.dirname(__file__), "base_vectorial")
    consultas_path = os.path.join(os.path.dirname(__file__), "..", "CORPUS CODEFEST AD ASTRA 2026", "Extracto_Preguntas_50_v2.pdf")
    if os.path.exists(base_vec_dir):
        run_empirical_benchmark(base_vec_dir, consultas_path)
