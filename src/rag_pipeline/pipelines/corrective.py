from rag_pipeline.grading.relevance_grader import evaluate_chunk_relevance

STOP_WORDS = {"what","when","where","why","how","did","do","does","is","are",
              "was","were","the","a","an","in","on","at","to","for","of","and",
              "i","my","me","this","that","throughout","much","many"}


def extract_key_terms(query_text):
    words = query_text.lower().replace("?", " ").split()
    key_terms = [w for w in words if w not in STOP_WORDS]
    return " ".join(key_terms)


def retrieve_with_grading(query_text, retrieval_tool):
    chunks = retrieval_tool.run(query_text)
    relevance = [evaluate_chunk_relevance(query_text, c) for c in chunks]
    good_chunks = [c for c, r in zip(chunks, relevance) if r in ("RELEVANT", "PARTIAL")]

    if not good_chunks:
        broader_query = extract_key_terms(query_text)
        good_chunks = retrieval_tool.run(broader_query)

    return good_chunks
