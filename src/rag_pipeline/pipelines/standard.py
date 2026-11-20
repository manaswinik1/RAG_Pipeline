from rag_pipeline.llm.client import ask


def answer(query_text, retrieval_tool):
    chunks = retrieval_tool.run(query_text)
    context = "\n\n".join(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer using only the context above."
    return ask(messages=[{"role": "user", "content": prompt}])
