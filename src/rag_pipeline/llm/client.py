import anthropic
from rag_pipeline.config import API_KEY, LLM_MODEL

_client = anthropic.Anthropic(api_key=API_KEY)


def ask(messages, system=None, max_tokens=1234):
    kwargs = {"model": LLM_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    response = _client.messages.create(**kwargs)
    return response.content[0].text

