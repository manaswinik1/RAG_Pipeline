from pinecone import Pinecone, ServerlessSpec
from rag_pipeline.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_DIMENSION

_pc = Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index():
    if PINECONE_INDEX_NAME not in _pc.list_indexes().names():
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return _pc.Index(PINECONE_INDEX_NAME)


def upsert_chunks(index, chunks, embeddings):
    vectors = [
        (f"chunk_{i}", emb, {"text": chunk, "chunk_index": i})
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    index.upsert(vectors=vectors)


def query(index, query_embedding, top_k=5):
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results['matches']
