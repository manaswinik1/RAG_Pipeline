from sentence_transformers import SentenceTransformer, CrossEncoder
from rag_pipeline.config import EMBEDDING_MODEL, RERANKER_MODEL

_embedding_model = SentenceTransformer(EMBEDDING_MODEL)
_reranker_model = CrossEncoder(RERANKER_MODEL)


def embed(texts):
    return _embedding_model.encode(texts).tolist()


def rerank(query, chunks):
    pairs = [(query, chunk) for chunk in chunks]
    return _reranker_model.predict(pairs)
