from rag_pipeline.ingestion.loaders import load_file
from rag_pipeline.ingestion.chunking import fixed_size_chunks
from rag_pipeline.embeddings.embedder import embed
from rag_pipeline.vectorstore.pinecone_client import get_or_create_index, upsert_chunks
from rag_pipeline.config import CHUNK_SIZE, CHUNK_OVERLAP

DATA_PATH = "data/wealth_of_nations_clean.txt"

if __name__ == "__main__":
    text = load_file(DATA_PATH)
    chunks = fixed_size_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = embed(chunks)

    index = get_or_create_index()
    upsert_chunks(index, chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks")
