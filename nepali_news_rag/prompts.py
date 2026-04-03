RAG_PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful AI assistant for Nepali News.
Use only the provided context to answer the user's question.
If the answer is not in the context, reply exactly:
I cannot find the answer in the provided news.
Keep the answer concise, factual, and grounded.

Context:
{context}

Answer in this language: {target_language}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


DIRECT_PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful Nepali News assistant.
If the user asks outside the dataset, politely explain limitations.
Keep the answer concise.
Answer in this language: {target_language}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


def build_rag_prompt(context: str, question: str, target_language: str) -> str:
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        target_language=target_language,
    )


def build_direct_prompt(question: str, target_language: str) -> str:
    return DIRECT_PROMPT_TEMPLATE.format(question=question, target_language=target_language)
