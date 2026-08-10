# CODEFEST AD ASTRA 2026 - Etapa 1: Base de Conocimiento

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/FAISS-FlatIP-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NDCG@10](https://img.shields.io/badge/NDCG%4010-1.0000-brightgreen.svg)]()

Repositorio oficial con la entrega del sistema de **Recuperación Semántica Híbrida y Vectorización Avanzada** para la **Etapa 1 (Base de Conocimiento)** del reto clasificatorio **CODEFEST AD ASTRA 2026**, apoyado por **Aval Digital Labs (ADL)**.

---

## 📌 Contexto del Proyecto

El proyecto desarrolla una arquitectura de recuperación de información (IR) inteligente para el análisis estratégico de fuentes abiertas en el dominio aéreo, espacial y territorial. El corpus abarca **1,826 documentos (~3.14 GB)** distribuidos en tres fenómenos principales:

1. **F1: IA y Capacidades Estratégicas** (459 documentos): Impacto de la IA en prevención NBQR, operaciones militares, enjambres de drones, semiconductores e infraestructura de defensa.
2. **F2: Seguridad del Entorno Espacial** (479 documentos): Operaciones contraespaciales, basura orbital, interferencia cibernética satelital, spoofing, guerra electrónica y maniobras RPO.
3. **F3: Dinámicas Territoriales en América Latina** (888 documentos): Gobernanza, sustitución del Estado por grupos armados (GAO/GAOR/GDO), minería ilegal, narcotráfico y control social en regiones colombianas (Chocó, Cauca, Arauca, Norte de Santander).

---

## 🚀 Arquitectura Técnica y Tecnologías Utilizadas

La solución implementa una **búsqueda híbrida multilingüe de doble canal** (Vectorial Densa + Léxica BM25) con reordenamiento cross-encoder y agregación por max pooling:

```
[ Consulta de Evaluación (q001 - q050) ]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐┌──────────────────┐
│  FAISS Vectorial ││   BM25 Léxico    │
│ (MiniLM-L12-v2)  ││ (Rank-BM25)      │
└────────┬─────────┘└────────┬─────────┘
         │ Top-50            │ Top-50
         └─────────┬─────────┘
                   ▼
┌──────────────────────────────────────┐
│  Reciprocal Rank Fusion (RRF k0=60)  │
└──────────────────┬───────────────────┘
                   ▼ Top-20
┌──────────────────────────────────────┐
│ Cross-Encoder Reranker (ms-marco)    │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│ Formateador Lingüístico (<=250 words)│
└──────────────────┬───────────────────┘
                   ▼
[ 10 Fragmentos + 3 Documentos (resultados.jsonl) ]
```

### Tecnologías Clave:
- **Lenguaje Principal**: Python 3.10+
- **Indexación Vectorial**: **FAISS (`faiss-cpu`)** utilizando `IndexFlatIP` (similitud coseno con vectores normalizados $L_2$).
- **Embedding Model (Encoder)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (soporte nativo multilingüe en Español, Inglés y Portugués).
- **Recuperación Léxica**: `rank_bm25` (BM25Okapi) para la coincidencia de términos técnicos y acrónimos específicos (*GAO*, *RPO*, *NBQR*, *GEO*).
- **Fusión de Rankings**: *Reciprocal Rank Fusion (RRF)* con constante de suavizado $k_0 = 60$.
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` para scoring bidireccional consulta-fragmento.
- **Extracción de Documentos**: `PyMuPDF` (PDFs), `pandas`/`openpyxl` (Tablas), `BeautifulSoup`/regex (HTML/JSON/TXT).

---

## 📊 Resultados del Benchmark de Validación Local

Se construyó un benchmark local empírico para medir cuantitativamente el rendimiento según las métricas oficiales del reto:
- **$NDCG@10$** (evaluación a nivel fragmento)
- **$F1@3$** (evaluación a nivel documento)
- **Conteo de Borda Total** ($B_i = B_i^{NDCG} + B_i^{F1}$)

### Tabla Comparativa de Resultados:

| Configuración / Técnica | NDCG@10 (Fragmentos) | F1@3 (Documentos) | Conteo de Borda |
| :--- | :---: | :---: | :---: |
| **FAISS Vectorial Único** | 0.8290 | 0.8889 | 5 pts |
| **BM25 Léxico Único** | 0.7512 | 0.5111 | 0 pts |
| **Recuperación Híbrida RRF (Dense + BM25)** | **1.0000** | 0.7556 | **5 pts (Líder NDCG)** |
| **Recuperación Híbrida + Cross-Encoder** | 0.7558 | 0.6000 | 2 pts |

---

## 📋 Estructura del Repositorio de Entrega

La estructura del proyecto cumple estrictamente con el esquema exigido por la organización:

```
.
├── generador.py                  # Script principal ejecutable sin argumentos
├── resultados.jsonl              # 50 líneas JSONL con resultados de evaluación
├── informe_tecnico.pdf           # Documento técnico en PDF (8 páginas)
├── consultas.jsonl               # 50 consultas de entrada (q001-q050)
├── build_fast_index.py           # Script para construir/actualizar la base vectorial
├── run_local_benchmark.py        # Script para reproducir el benchmark local
├── base_vectorial/
│   └── encoder_paraphrase_multilingual_MiniLM_L12_v2/
│       ├── index.faiss           # Archivo binario del índice FAISS
│       └── metadata.jsonl        # Almacén de metadata asociado (Tabla 2)
└── lib/
    ├── extractor.py              # Extractor multiformato con mapeo oficial DOC_ID
    ├── chunker.py                # Segmentador lingüístico (oraciones completas, <=250 palabras)
    ├── indexer.py                # Módulo de administración de vectores FAISS
    ├── bm25_retriever.py         # Módulo BM25 + Reciprocal Rank Fusion
    ├── reranker.py               # Reordenador Cross-Encoder
    └── evaluator.py              # Validador sintáctico y calculador de métricas
```

---

## 🛠️ Instrucciones de Ejecución

### Requisitos Previos
Instalar las dependencias del proyecto:
```bash
pip install faiss-cpu sentence-transformers rank-bm25 PyMuPDF openpyxl pandas fpdf2
```

### Reproducción de Resultados (Requisito Obligatorio)
Para generar el archivo de resultados `resultados.jsonl` a partir del índice entregado:
```bash
python generador.py
```

Parámetros opcionales soportados:
```bash
python generador.py --consultas consultas.jsonl --base-vectorial ./base_vectorial --salida ./resultados.jsonl
```

### Ejecutar Benchmark Local
```bash
python run_local_benchmark.py
```

---

## ⚖️ Cumplimiento de Reglas Obligatorias

1. **Cero Modelos Generativos**: No se utiliza ningún decoder (GPT/LLaMA/Claude) en ninguna etapa de inferencia o recuperacion (cumple Sección 8.3).
2. **Completitud Lingüística**: Los fragmentos respetan fronteras oracionales completas sin cortes arbitrarios (cumple Sección 3.3).
3. **Límite de Palabras**: Todos los fragmentos tienen $\le 250$ palabras (cumple Sección 9.2.1).
4. **Mapeo Oficial de Documentos**: `doc_id` alineado 1:1 con el inventario oficial `Indice_Datos_Codefest.xlsx`.
5. **Formato JSON Lines**: Exactamente 50 líneas válidas con 3 documentos y 10 fragmentos cada una (cumple Sección 9.3).

---
*Desarrollado para el CODEFEST AD ASTRA 2026.*
