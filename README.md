## I'm building a Production-Grade Enterprise Hybrid & Graph RAG Engine with Automated RAGAS Evaluation and Community Summarization.


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
