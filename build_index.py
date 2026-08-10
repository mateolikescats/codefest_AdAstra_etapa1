import os
import sys
import time
import json

sys.path.append(os.path.dirname(__file__))

from lib.extractor import CorpusExtractor
from lib.chunker import SentenceChunker
from lib.indexer import VectorIndexer

def run_indexing(
    corpus_dir: str,
    excel_index_path: str,
    output_base_dir: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
):
    print(f"=== STARTING OPTIMIZED CODEFEST INDEXING PIPELINE ===", flush=True)
    print(f"Corpus Directory: {corpus_dir}", flush=True)
    print(f"Excel Index Path: {excel_index_path}", flush=True)
    print(f"Embedding Model: {model_name}", flush=True)
    
    start_time = time.time()
    chunks_cache_file = os.path.join(output_base_dir, "chunks_cache.jsonl")
    all_chunks = []

    if os.path.exists(chunks_cache_file):
        print(f"Loading pre-extracted chunks from cache: {chunks_cache_file}...", flush=True)
        with open(chunks_cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_chunks.append(json.loads(line.strip()))
        print(f"Loaded {len(all_chunks)} chunks from cache in {time.time() - start_time:.2f}s!", flush=True)
    else:
        extractor = CorpusExtractor(corpus_dir, excel_index_path)
        chunker = SentenceChunker(max_words=240, target_words=200, overlap_words=40)
        
        doc_count = 0
        error_count = 0
        file_list = []
        for root, dirs, files in os.walk(corpus_dir):
            for file in files:
                if file.startswith("~$") or file.endswith(".xlsx") or file.endswith(".pdf_extracted_text.txt"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, corpus_dir)
                file_list.append((full_path, rel_path))
                
        print(f"Found {len(file_list)} total documents to process.", flush=True)
        
        for full_path, rel_path in file_list:
            try:
                doc_data = extractor.extract_document(full_path, rel_path)
                if doc_data["text"] and len(doc_data["text"].strip()) >= 10:
                    chunks = chunker.chunk_document(doc_data)
                    all_chunks.extend(chunks)
                doc_count += 1
                if doc_count % 300 == 0 or doc_count == len(file_list):
                    print(f"Extracted {doc_count}/{len(file_list)} documents... Chunks so far: {len(all_chunks)}", flush=True)
            except Exception as e:
                error_count += 1
                print(f"Warning: Failed to extract {rel_path}: {e}", flush=True)

        print(f"Extraction & Chunking Complete in {time.time() - start_time:.2f}s!", flush=True)
        
        # Save cache for future fast runs
        os.makedirs(output_base_dir, exist_ok=True)
        with open(chunks_cache_file, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Cached {len(all_chunks)} chunks to {chunks_cache_file}", flush=True)

    # 2. Build FAISS Vector Index using optimized vector encoding
    indexer = VectorIndexer(model_name=model_name)
    indexer.build_index(all_chunks, batch_size=256)
    
    # 3. Save to base_vectorial/encoder_<model_name>/
    safe_model_name = model_name.split("/")[-1].replace("-", "_")
    encoder_dir = os.path.join(output_base_dir, f"encoder_{safe_model_name}")
    indexer.save(encoder_dir)
    
    elapsed = time.time() - start_time
    print(f"\n=== INDEXING PIPELINE SUCCESSFUL IN {elapsed:.2f} SECONDS ===", flush=True)
    print(f"Base Vectorial Saved To: {encoder_dir}", flush=True)

if __name__ == "__main__":
    corpus_dir = r"c:\Users\mateo\OneDrive\Documents\Universidad\AdAstra\CORPUS CODEFEST AD ASTRA 2026"
    excel_index_path = r"c:\Users\mateo\OneDrive\Documents\Universidad\AdAstra\CORPUS CODEFEST AD ASTRA 2026\Indice_Datos_Codefest.xlsx"
    output_base_dir = r"c:\Users\mateo\OneDrive\Documents\Universidad\AdAstra\entrega\base_vectorial"
    
    run_indexing(corpus_dir, excel_index_path, output_base_dir)
