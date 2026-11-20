from rag_pipeline.llm.client import ask
from rag_pipeline.llm.prompts import AGENT_SYSTEM_PROMPT


def run(query_text, initial_chunks, retrieval_tool, max_steps=5):
    numbered_context = ""
    for i, chunk in enumerate(initial_chunks):
        numbered_context += f"[{i}] {chunk}\n\n"

    gathered = list(initial_chunks)
    messages = [{
        "role": "user",
        "content": (
            f"Question: {query_text}\n\nInitially retrieved chunks:\n{numbered_context}\n\n"
            "When answering, always start with 'Based on [X]...' where X is the chunk number(s) used. "
            "Example: Based on [2] and [3]: The answer is ... "
            "If the context doesn't contain the answer, respond with: "
            "'This information is not available in the provided context.' "
            "Keep answers clear and concise.\n\nContext: None yet. What should I search for?"
        )
    }]

    for step in range(max_steps):
        output = ask(messages=messages, system=AGENT_SYSTEM_PROMPT).strip()

        if output.startswith("ANSWER:"):
            return output[len("ANSWER:"):].strip()

        if output.startswith("SEARCH:"):
            search_query = output[len("SEARCH:"):].strip()
            retrieved = retrieval_tool.run(search_query)
            gathered.extend(retrieved)
            messages.append({"role": "assistant", "content": output})
            messages.append({
                "role": "user",
                "content": "Retrieved:\n" + "\n---\n".join(retrieved) +
                           f"\nHave {len(gathered)} chunks now. Enough to answer, or search more?"
            })
        else:
            return output

    return output
