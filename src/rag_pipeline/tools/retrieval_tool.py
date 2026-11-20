from rank_bm25 import BM25Okapi
from rag_pipeline.tools.base import Tool
from rag_pipeline.embeddings.embedder import embed, rerank
from rag_pipeline.vectorstore.pinecone_client import query as pinecone_query
from rag_pipeline.config import DENSE_WEIGHT, SPARSE_WEIGHT, TOP_K


def _combine_scores(query_text, chunks, index, top_k):
    query_embedding = embed(query_text)
    dense_matches = pinecone_query(index, query_embedding, top_k=top_k * 2)
    dense_scores = {m['metadata']['text']: m['score'] * DENSE_WEIGHT for m in dense_matches}

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query_text.lower().split())
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1e-9

    chunk_scores = dict(dense_scores)
    for i, score in enumerate(bm25_scores):
        normalized = score / max_bm25
        chunk_scores[chunks[i]] = chunk_scores.get(chunks[i], 0) + normalized * SPARSE_WEIGHT

    return chunk_scores
class RetrievalTool(Tool):
    name = "search_documents"
    description = "Search the indexed documents for relevant chunks."

    def __init__(self, index, all_chunks):
        self.index = index
        self.all_chunks = all_chunks

    def run(self, query_text, top_k=TOP_K):
        chunk_scores = _combine_scores(query_text, self.all_chunks, self.index, top_k)
        ranked_chunks = sorted(chunk_scores, key=chunk_scores.get, reverse=True)

        rerank_scores = rerank(query_text, ranked_chunks)
        reranked = sorted(zip(rerank_scores, ranked_chunks), reverse=True)

        return [chunk for score, chunk in reranked[:top_k]]
