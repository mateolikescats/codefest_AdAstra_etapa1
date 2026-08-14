import os
import sys
import json
import argparse
import pypdf
from typing import List, Dict, Any

# Add current directory to path so lib modules can be imported
sys.path.append(os.path.dirname(__file__))

from lib.indexer import VectorIndexer
from lib.bm25_retriever import BM25Retriever, reciprocal_rank_fusion

def load_queries(consultas_path: str) -> List[Dict[str, str]]:
    """
    Loads 50 queries either from a .jsonl file or by parsing PDF / text extract.
    Expected schema in JSONL: {"query_id": "q001", "query": "..."} or {"query_id": "q001", "texto": "..."}
    """
    queries = []
    
    if os.path.exists(consultas_path) and consultas_path.endswith(".jsonl"):
        with open(consultas_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    q_id = item.get("query_id") or item.get("id")
                    q_text = item.get("query") or item.get("texto") or item.get("text")
                    queries.append({"query_id": q_id, "query": q_text})
        if len(queries) >= 50:
            return queries[:50]

    # Fallback: check Extracto_Preguntas_50_v2.pdf in parent or corpus directory
    pdf_paths = [
        consultas_path,
        os.path.join(os.path.dirname(__file__), "..", "CORPUS CODEFEST AD ASTRA 2026", "Extracto_Preguntas_50_v2.pdf"),
        os.path.join(os.path.dirname(__file__), "..", "Extracto_Preguntas_50_v2.pdf")
    ]
    
    pdf_target = None
    for p in pdf_paths:
        if os.path.exists(p) and p.endswith(".pdf"):
            pdf_target = p
            break
            
    if pdf_target:
        reader = pypdf.PdfReader(pdf_target)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        import re
        matches = re.findall(r'(q\d{3})\s+(.*?)(?=(?:q\d{3})|\Z)', full_text, re.DOTALL)
        for q_id, q_text in matches:
            clean_q = re.sub(r'\s+', ' ', q_text).strip()
            queries.append({"query_id": q_id, "query": clean_q})
            
    if not queries:
        raise FileNotFoundError(f"Could not load queries from {consultas_path}")
        
    return queries[:50]

def enforce_word_limit(text: str, max_words: int = 250) -> str:
    """
    Ensures text does not exceed max_words and respects sentence completeness.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    # Trim to max_words and find last sentence boundary
    trimmed_words = words[:max_words]
    trimmed_text = " ".join(trimmed_words)
    
    # Match last sentence end (. ! ?)
    last_punct = max(trimmed_text.rfind('.'), trimmed_text.rfind('!'), trimmed_text.rfind('?'))
    if last_punct > 50:
        return trimmed_text[:last_punct + 1]
    return trimmed_text

def run_generator(consultas_path: str, base_vectorial_path: str, salida_path: str):
    print(f"=== CODEFEST AD ASTRA 2026 GENERATOR ===")
    print(f"Consultas Path: {consultas_path}")
    print(f"Base Vectorial: {base_vectorial_path}")
    print(f"Salida Path:    {salida_path}")

    # 1. Discover encoder directory inside base_vectorial
    encoder_dirs = [d for d in os.listdir(base_vectorial_path) if d.startswith("encoder_")]
    if not encoder_dirs:
        raise FileNotFoundError(f"No encoder subfolder found in {base_vectorial_path}")
    
    target_encoder_dir = os.path.join(base_vectorial_path, encoder_dirs[0])
    print(f"Loading FAISS Index from: {target_encoder_dir}")
    
    # 2. Load FAISS Vector Indexer
    indexer = VectorIndexer.load(target_encoder_dir)
    print(f"Vector Index Loaded. Total vectors: {indexer.index.ntotal}")

    # 3. Load BM25 Retriever
    print("Initializing BM25 Lexical Retriever...")
    bm25 = BM25Retriever(indexer.metadata_store)

    # 4. Load Queries
    queries = load_queries(consultas_path)
    print(f"Loaded {len(queries)} queries for evaluation.")

    output_lines = []

    for q_idx, q_item in enumerate(queries, start=1):
        q_id = q_item["query_id"]
        q_text = q_item["query"]

        # Dense FAISS Search (Top 50)
        dense_results = indexer.search(q_text, top_k=50)

        # Lexical BM25 Search (Top 50)
        lexical_results = bm25.search(q_text, top_k=50)

        # Pure Hybrid Reciprocal Rank Fusion (RRF k0=60)
        fused_candidates = reciprocal_rank_fusion([dense_results, lexical_results], k0=60.0)

        # Top 10 Chunks
        top_10_chunks = fused_candidates[:10]

        # Format Top 10 Fragments
        formatted_fragments = []
        doc_scores = {}

        for r_idx, frag in enumerate(top_10_chunks, start=1):
            doc_id = frag["doc_id"]
            chunk_id = frag["chunk_id"]
            text_cleaned = enforce_word_limit(frag["texto"], max_words=250)
            score = frag.get("rrf_score", 0.0)

            formatted_fragments.append({
                "rank": r_idx,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text_cleaned
            })

            # Max pooling score per document
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        # Format Top 3 Documents
        sorted_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
        
        # If fewer than 3 docs in top 10 fragments, pull from wider fused candidates
        if len(sorted_docs) < 3:
            for cand in fused_candidates:
                d_id = cand["doc_id"]
                if d_id not in sorted_docs:
                    sorted_docs.append(d_id)
                if len(sorted_docs) == 3:
                    break

        formatted_docs = [
            {"rank": d_idx, "doc_id": d_id}
            for d_idx, d_id in enumerate(sorted_docs[:3], start=1)
        ]

        query_result_obj = {
            "query_id": q_id,
            "documents": formatted_docs,
            "fragments": formatted_fragments
        }

        output_lines.append(query_result_obj)

    # Save to resultados.jsonl
    os.makedirs(os.path.dirname(os.path.abspath(salida_path)), exist_ok=True)
    with open(salida_path, "w", encoding="utf-8") as f:
        for obj in output_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nSUCCESS! Generated {len(output_lines)} evaluation lines in {salida_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODEFEST AD ASTRA 2026 - Generador de Resultados (Etapa 1)")
    
    default_base_vec = os.path.join(os.path.dirname(__file__), "base_vectorial")
    default_salida = os.path.join(os.path.dirname(__file__), "resultados.jsonl")
    default_consultas = os.path.join(os.path.dirname(__file__), "consultas.jsonl")

    parser.add_argument("--consultas", type=str, default=default_consultas, help="Ruta al archivo de consultas (.jsonl o .pdf)")
    parser.add_argument("--base-vectorial", type=str, default=default_base_vec, help="Ruta al directorio raíz de la base vectorial")
    parser.add_argument("--salida", type=str, default=default_salida, help="Ruta del archivo de resultados.jsonl a generar")

    args = parser.parse_args()
    
    run_generator(
        consultas_path=args.consultas,
        base_vectorial_path=args.base_vectorial,
        salida_path=args.salida
    )

