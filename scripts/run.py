from rag_pipeline.ingestion.loaders import load_file
from rag_pipeline.ingestion.chunking import fixed_size_chunks
from rag_pipeline.vectorstore.pinecone_client import get_or_create_index
from rag_pipeline.tools.retrieval_tool import RetrievalTool
from rag_pipeline.pipelines import combined
from rag_pipeline.config import CHUNK_SIZE, CHUNK_OVERLAP

DATA_PATH = "data/wealth_of_nations_clean.txt"

if __name__ == "__main__":
    text = load_file(DATA_PATH)
    chunks = fixed_size_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    index = get_or_create_index()
    retrieval_tool = RetrievalTool(index, chunks)

    query = input("Enter query: ")
    result = combined.answer(query, retrieval_tool)
    print(result)
