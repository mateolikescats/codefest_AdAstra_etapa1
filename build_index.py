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
    model_name: str = "BAAI/bge-m3"
):
    print(f"=== STARTING OPTIMIZED CODEFEST INDEXING PIPELINE ===", flush=True)
    print(f"Corpus Directory: {corpus_dir}", flush=True)
    print(f"Excel Index Path: {excel_index_path}", flush=True)
    print(f"Embedding Model:  {model_name}", flush=True)
    
    start_time = time.time()
    chunks_cache_file = os.path.join(output_base_dir, "chunks_cache.jsonl")
    all_chunks = []

    # If cache exists, check if it contains polluted question chunks
    use_cache = False
    if os.path.exists(chunks_cache_file):
        print(f"Checking pre-extracted chunks from cache: {chunks_cache_file}...", flush=True)
        cached = []
        has_question_pdf = False
        with open(chunks_cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if "Extracto_Preguntas" in item.get("fuente", "") or item.get("doc_id") == "DOC-79283":
                        has_question_pdf = True
                    else:
                        cached.append(item)
        if not has_question_pdf and len(cached) > 0:
            all_chunks = cached
            use_cache = True
            print(f"Loaded {len(all_chunks)} clean chunks from cache in {time.time() - start_time:.2f}s!", flush=True)
        else:
            print("Cache contained evaluation questions PDF or was invalid. Re-running extraction...", flush=True)

    if not use_cache:
        extractor = CorpusExtractor(corpus_dir, excel_index_path)
        chunker = SentenceChunker(max_words=240, target_words=200, overlap_words=40)
        
        doc_count = 0
        error_count = 0
        file_list = []
        seen_filenames = set()
        for root, dirs, files in os.walk(corpus_dir):
            for file in files:
                # Exclude temporary files, Excel indexes, extracted text logs, AND the questions PDF
                if file.startswith("~$") or file.endswith(".xlsx") or file.endswith(".pdf_extracted_text.txt") or "Extracto_Preguntas" in file:
                    continue
                # Solo procesar si el archivo está en el inventario oficial y no lo hemos visto antes
                if file not in extractor.doc_index:
                    continue
                if file in seen_filenames:
                    continue
                seen_filenames.add(file)
                
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
        
        # Save clean cache for future runs
        os.makedirs(output_base_dir, exist_ok=True)
        with open(chunks_cache_file, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"Cached {len(all_chunks)} clean chunks to {chunks_cache_file}", flush=True)

    # 2. Build FAISS Vector Index using optimized vector encoding
    indexer = VectorIndexer(model_name=model_name)
    indexer.build_index(all_chunks, batch_size=128)
    
    # 3. Save to base_vectorial/encoder_<model_name>/
    safe_model_name = model_name.split("/")[-1].replace("-", "_")
    encoder_dir = os.path.join(output_base_dir, f"encoder_{safe_model_name}")
    indexer.save(encoder_dir)

    # 4. Build Knowledge Graph (bonus component)
    try:
        from lib.knowledge_graph import KnowledgeGraph
        print("\n=== BUILDING KNOWLEDGE GRAPH (BONUS) ===", flush=True)
        kg = KnowledgeGraph()
        kg.build_from_chunks(all_chunks, batch_size=500)
        grafo_dir = os.path.join(output_base_dir, "grafo")
        kg.save(grafo_dir)
        print(f"Knowledge Graph saved to: {grafo_dir}", flush=True)
    except ImportError as e:
        print(f"Skipping Knowledge Graph (missing dependency): {e}", flush=True)
    except Exception as e:
        print(f"Warning: Knowledge Graph construction failed: {e}", flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n=== INDEXING PIPELINE SUCCESSFUL IN {elapsed:.2f} SECONDS ===", flush=True)
    print(f"Base Vectorial Saved To: {encoder_dir}", flush=True)

if __name__ == "__main__":
    # Autodetección de rutas locales para Mateo y Cristian
    cristian_corpus = r"d:\Usuarios\Cristian\Desktop\ad astra"
    if os.path.exists(cristian_corpus):
        corpus_dir = cristian_corpus
        excel_index_path = os.path.join(cristian_corpus, "Indice_Datos_Codefest.xlsx")
        output_base_dir = os.path.join(os.path.dirname(__file__), "base_vectorial")
    else:
        corpus_dir = r"c:\Users\mateo\OneDrive\Documents\Universidad\AdAstra\CORPUS CODEFEST AD ASTRA 2026"
        excel_index_path = r"c:\Users\mateo\OneDrive\Documents\Universidad\AdAstra\CORPUS CODEFEST AD ASTRA 2026\Indice_Datos_Codefest.xlsx"
        output_base_dir = r"c:\Users\mateo\OneDrive\Documents\Universidad\AdAstra\entrega\base_vectorial"
        
    run_indexing(corpus_dir, excel_index_path, output_base_dir)

