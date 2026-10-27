from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class AdvancedRetrievalPipeline:
    def __init__(self, corpus: List[Dict[str, Any]], cross_encoder_model: str = "BAAI/bge-reranker-large"):
        """
        corpus: List of dicts containing {'id': str, 'text': str, 'embedding': List[float]}
        """
        self.corpus = corpus
        self.corpus_texts = [doc['text'] for doc in corpus]
        
        # 1. Initialize Sparse Index (BM25)
        tokenized_corpus = [text.lower().split() for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Initialize Cross-Encoder for Re-ranking
        self.reranker = CrossEncoder(cross_encoder_model)

    def sparse_search(self, query: str, top_k: int = 50) -> List[tuple]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def dense_search(self, query_embedding: List[float], top_k: int = 50) -> List[tuple]:
        # Vector similarity mock (Cosine Similarity)
        scores = []
        q_emb = np.array(query_embedding)
        for idx, doc in enumerate(self.corpus):
            doc_emb = np.array(doc['embedding'])
            cos_sim = np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb) + 1e-8)
            scores.append((idx, float(cos_sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def reciprocal_rank_fusion(self, sparse_results: List[tuple], dense_results: List[tuple], k: int = 60) -> List[int]:
        rrf_scores = {}
        
        def accumulate_ranks(results):
            for rank, (doc_id, _) in enumerate(results):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (k + (rank + 1))

        accumulate_ranks(sparse_results)
        accumulate_ranks(dense_results)
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs]

    def retrieve_and_rerank(self, query: str, query_embedding: List[float], final_top_k: int = 5) -> List[Dict[str, Any]]:
        # Stage 1: Broad Retrieval
        sparse_res = self.sparse_search(query, top_k=50)
        dense_res = self.dense_search(query_embedding, top_k=50)
        
        # Stage 2: Score Fusion via RRF
        fused_doc_ids = self.reciprocal_rank_fusion(sparse_res, dense_res, k=60)[:20] # Take top 20 candidates
        
        # Stage 3: Cross-Encoder Re-ranking
        candidate_pairs = [[query, self.corpus[doc_id]['text']] for doc_id in fused_doc_ids]
        rerank_scores = self.reranker.predict(candidate_pairs)
        
        scored_candidates = list(zip(fused_doc_ids, rerank_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        final_selected_ids = [doc_id for doc_id, score in scored_candidates[:final_top_k]]
        return [self.corpus[doc_id] for doc_id in final_selected_ids]
