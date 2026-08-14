"""
Knowledge Graph module for CODEFEST AD ASTRA 2026.
Extracts named entities (NER) from chunks and builds a graph of relationships.
Uses the graph as an additional retrieval source fused via RRF.
"""
import os
import json
import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class KnowledgeGraph:
    """
    Builds a knowledge graph from document chunks using NER.
    Nodes = entities, Edges = co-occurrence in same chunk.
    Each edge stores references to the chunks where the co-occurrence was found.
    """

    def __init__(self, ner_model: str = "Davlan/xlm-roberta-base-ner-hrl"):
        if not HAS_NETWORKX:
            raise ImportError("networkx is required: pip install networkx")

        self.graph = nx.Graph()
        self.entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)  # entity -> set of chunk_ids
        self.chunk_entities: Dict[str, List[str]] = {}  # chunk_id -> list of entities

        self.ner_pipeline = None
        self.ner_model_name = ner_model

    def _init_ner(self):
        """Lazy-load NER pipeline to avoid slow imports when not needed."""
        if self.ner_pipeline is None and HAS_TRANSFORMERS:
            print(f"Loading NER model: {self.ner_model_name}...", flush=True)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple",
                device=-1  # CPU; change to 0 for GPU
            )

    def extract_entities(self, text: str, max_length: int = 512) -> List[Dict[str, Any]]:
        """Extract named entities from text using the NER model."""
        self._init_ner()
        if not self.ner_pipeline:
            return []

        # Truncate text to avoid model limits
        truncated = " ".join(text.split()[:max_length])
        try:
            entities = self.ner_pipeline(truncated)
            # Normalize entity names
            result = []
            for ent in entities:
                name = ent["word"].strip().replace("##", "")
                if len(name) > 2 and ent.get("score", 0) > 0.7:
                    result.append({
                        "name": name.lower(),
                        "label": ent.get("entity_group", "MISC"),
                        "score": ent.get("score", 0.0)
                    })
            return result
        except Exception:
            return []

    def build_from_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Build the knowledge graph from a list of chunks.
        For each chunk, extracts entities and creates co-occurrence edges.
        """
        self._init_ner()
        if not self.ner_pipeline:
            print("WARNING: NER pipeline not available. Skipping knowledge graph construction.")
            return

        total = len(chunks)
        print(f"Building Knowledge Graph from {total} chunks...", flush=True)

        for i, chunk in enumerate(chunks):
            if (i + 1) % batch_size == 0:
                print(f"  KG Progress: {i+1}/{total} chunks processed, "
                      f"{self.graph.number_of_nodes()} entities, "
                      f"{self.graph.number_of_edges()} relations", flush=True)

            chunk_id = chunk["chunk_id"]
            doc_id = chunk["doc_id"]
            text = chunk.get("texto", "")

            entities = self.extract_entities(text)
            entity_names = list(set(e["name"] for e in entities))
            self.chunk_entities[chunk_id] = entity_names

            # Map entities to chunks
            for ename in entity_names:
                self.entity_to_chunks[ename].add(chunk_id)

                # Add/update node
                if self.graph.has_node(ename):
                    self.graph.nodes[ename]["count"] += 1
                    self.graph.nodes[ename]["chunk_ids"].add(chunk_id)
                    self.graph.nodes[ename]["doc_ids"].add(doc_id)
                else:
                    label = next((e["label"] for e in entities if e["name"] == ename), "MISC")
                    self.graph.add_node(ename, label=label, count=1,
                                       chunk_ids={chunk_id}, doc_ids={doc_id})

            # Create co-occurrence edges between all entity pairs in this chunk
            for j, e1 in enumerate(entity_names):
                for e2 in entity_names[j+1:]:
                    if self.graph.has_edge(e1, e2):
                        self.graph[e1][e2]["weight"] += 1
                        self.graph[e1][e2]["chunk_ids"].add(chunk_id)
                    else:
                        self.graph.add_edge(e1, e2, weight=1,
                                           relation="co_occurrence",
                                           chunk_ids={chunk_id})

        print(f"Knowledge Graph built: {self.graph.number_of_nodes()} entities, "
              f"{self.graph.number_of_edges()} relations", flush=True)

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph for chunks related to query entities.
        Returns chunks scored by number of entity matches and graph proximity.
        """
        query_entities = self.extract_entities(query)
        if not query_entities:
            # Fallback: simple keyword matching against entity names
            query_words = set(query.lower().split())
            query_entities = [{"name": n} for n in self.entity_to_chunks.keys()
                             if any(w in n for w in query_words if len(w) > 3)]

        # Collect candidate chunks from matching entities and their neighbors
        chunk_scores: Dict[str, float] = defaultdict(float)

        for ent in query_entities:
            ename = ent["name"] if isinstance(ent, dict) else ent
            if ename in self.entity_to_chunks:
                # Direct match: score 2.0
                for cid in self.entity_to_chunks[ename]:
                    chunk_scores[cid] += 2.0

                # Neighbor entities (1-hop): score 1.0
                if self.graph.has_node(ename):
                    for neighbor in self.graph.neighbors(ename):
                        for cid in self.entity_to_chunks.get(neighbor, set()):
                            chunk_scores[cid] += 1.0

        # Sort by score and return top_k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, 1):
            results.append({
                "chunk_id": chunk_id,
                "kg_score": score,
                "rank": rank
            })

        return results

    def save(self, output_dir: str):
        """Save the knowledge graph to GraphML format."""
        os.makedirs(output_dir, exist_ok=True)

        # Convert sets to lists for serialization
        for node in self.graph.nodes():
            ndata = self.graph.nodes[node]
            if "chunk_ids" in ndata:
                ndata["chunk_ids"] = ",".join(ndata["chunk_ids"])
            if "doc_ids" in ndata:
                ndata["doc_ids"] = ",".join(ndata["doc_ids"])
        for u, v in self.graph.edges():
            edata = self.graph[u][v]
            if "chunk_ids" in edata:
                edata["chunk_ids"] = ",".join(edata["chunk_ids"])

        graphml_path = os.path.join(output_dir, "grafo.graphml")
        nx.write_graphml(self.graph, graphml_path)
        print(f"Knowledge graph saved to: {graphml_path}", flush=True)

        # Also save entity-chunk mapping
        mapping_path = os.path.join(output_dir, "entity_chunks.jsonl")
        with open(mapping_path, "w", encoding="utf-8") as f:
            for entity, chunk_ids in self.entity_to_chunks.items():
                f.write(json.dumps({
                    "entity": entity,
                    "chunk_ids": list(chunk_ids)
                }, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, input_dir: str):
        """Load a pre-built knowledge graph."""
        obj = cls.__new__(cls)
        obj.ner_pipeline = None
        obj.ner_model_name = "Davlan/xlm-roberta-base-ner-hrl"
        obj.chunk_entities = {}
        obj.entity_to_chunks = defaultdict(set)

        graphml_path = os.path.join(input_dir, "grafo.graphml")
        if os.path.exists(graphml_path):
            obj.graph = nx.read_graphml(graphml_path)
            # Restore sets from serialized strings
            for node in obj.graph.nodes():
                ndata = obj.graph.nodes[node]
                if "chunk_ids" in ndata and isinstance(ndata["chunk_ids"], str):
                    ndata["chunk_ids"] = set(ndata["chunk_ids"].split(","))
                if "doc_ids" in ndata and isinstance(ndata["doc_ids"], str):
                    ndata["doc_ids"] = set(ndata["doc_ids"].split(","))
            for u, v in obj.graph.edges():
                edata = obj.graph[u][v]
                if "chunk_ids" in edata and isinstance(edata["chunk_ids"], str):
                    edata["chunk_ids"] = set(edata["chunk_ids"].split(","))
        else:
            obj.graph = nx.Graph()

        # Load entity-chunk mapping
        mapping_path = os.path.join(input_dir, "entity_chunks.jsonl")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        obj.entity_to_chunks[item["entity"]] = set(item["chunk_ids"])

        return obj
