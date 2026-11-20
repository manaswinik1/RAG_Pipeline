from rag_pipeline.llm.client import ask
from rag_pipeline.llm.prompts import RELEVANCE_PROMPT_TEMPLATE


def evaluate_chunk_relevance(query, chunk):
    prompt = RELEVANCE_PROMPT_TEMPLATE.format(query=query, chunk=chunk)
    return ask(messages=[{"role": "user", "content": prompt}])
