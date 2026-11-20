from rag_pipeline.pipelines.corrective import retrieve_with_grading
from rag_pipeline.pipelines import agentic


def answer(query_text, retrieval_tool):
    good_chunks = retrieve_with_grading(query_text, retrieval_tool)

    if not good_chunks:
        return "Could not find anything relevant about this in documents."

    return agentic.run(query_text, good_chunks, retrieval_tool)
