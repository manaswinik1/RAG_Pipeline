RELEVANCE_PROMPT_TEMPLATE = """Is this chunk relevant to answering the question? Question: {query}, chunk: {chunk}. Answer with only: RELEVANT, PARTIAL, OR IRRELEVANT"""

AGENT_SYSTEM_PROMPT = """Your entire response must be exactly one line, with no other text before or after it.
Do not explain what you're about to do.
Respond with ONLY:
SEARCH:[query] or ANSWER:[text]
"""
