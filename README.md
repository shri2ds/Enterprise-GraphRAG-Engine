# Enterprise Graph RAG Engine

A production-grade, hybrid retrieval-augmented generation (RAG) engine that bridges the gap between unstructured document text and multi-hop relational data. Designed to solve both local precision lookups and global macro-analysis in enterprise environments.

## Architecture Overview

```text
 [ User Query ] 
       │
       ├────────────────────────────────────────┬────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
│       Sparse Retrieval       │ │       Dense Retrieval        │ │      Graph Traversal         │
│         (BM25 / SPLADE)      │ │     (Vector DB / Cosine)     │ │     (Neo4j Cypher / KG)      │
└──────────────┬───────────────┘ └──────────────┬───────────────┘ └──────────────┬───────────────┘
       │                                        │                                │
       └────────────────────────┬───────────────┘                                │
                                ▼                                                │
                 ┌─────────────────────────────┐                                 │
                 │ Reciprocal Rank Fusion(RRF) │                                 │
                 │     (Merges Top-100 Pool)   │                                 │
                 └──────────────┬──────────────┘                                 │
                                │                                                │
                                ▼                                                │
                 ┌─────────────────────────────┐                                 │
                 │   Cross-Encoder Re-ranker   │                                 │
                 │  (Narrows Top-100 to Top-5) │                                 │
                 └──────────────┬──────────────┘                                 │
                                │                                                │
                                └────────────────────────┬───────────────────────┘
                                                         ▼
                                          ┌─────────────────────────────┐
                                          │      LLM Context Window     │
                                          │  (Synthesized Final Answer) │
                                          └─────────────────────────────┘
```

## Repository Structure

```text
enterprise-graph-rag/
├── README.md                          # Architecture overview, setup guide, and benchmarks
├── requirements.txt                   # Dependencies (sentence-transformers, rank-bm25, neo4j, ragas, etc.)
├── config.yaml                        # Configuration (embedding models, reranker model, top_k limits, RRF k)
├── data/                              # Sample raw enterprise documents / knowledge base
├── retrieval/                         # Step 1: Hybrid Retrieval & Re-ranking (Today's Code)
│   ├── __init__.py
│   └── hybrid_retrieval_rrf.py        # BM25, Dense search, RRF, and Cross-Encoder re-ranker
├── graph/                             # Step 2: Knowledge Graph & Traversal (Coming up)
│   ├── __init__.py
│   ├── kg_builder.py                  # LLM entity/relation extraction pipeline
│   ├── neo4j_client.py                # Cypher query execution and graph traversal
│   └── community_detection.py         # Leiden/Louvain clustering & global summarization
├── pipeline/                          # Step 3: Orchestration & Serving
│   ├── __init__.py
│   └── rag_engine.py                  # End-to-end orchestration (Query -> Hybrid Retrieval -> Graph -> LLM)
├── evaluation/                        # Step 4: RAGAS Evaluation Harness
│   ├── __init__.py
│   └── evaluate_pipeline.py           # Automated evaluation script for Faithfulness, Recall, Relevance
└── tests/                             # Unit & Integration Tests
    └── test_retrieval.py
```

### Key Features
- **Hybrid Search with RRF** : Combines lexical sparse retrieval (BM25) with dense vector semantic search, normalized via Reciprocal Rank Fusion.
- **Cross-Encoder Re-ranking**: Deep semantic re-ranking using state-of-the-art transformer cross-encoders (bge-reranker-large) to optimize top-k precision.
- **Graph RAG Integration**: Neo4j knowledge graph backend supporting multi-hop relational traversal and Leiden community clustering for global macro-summaries.
- **Automated Evaluation**: Built-in RAGAS evaluation harness measuring Faithfulness, Answer Relevance, and Context Recall.


