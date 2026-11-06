import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-haiku-4-5-20251001"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
TOP_K = 5