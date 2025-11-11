# prompt_builder.py
"""
Builds prompts for each task mode: general, answer, summary.
"""

def build_general_prompt(query: str) -> str:
    return (
        "You are my personal research assistant. Answer using general knowledge only.\n"
        "Do not fabricate citations or refer to documents unless they are canonical.\n"
        "Be clear, concise, and informative. Use bullet points or short paragraphs if helpful.\n\n"
        f"User question: {query}\n\n"
        "Answer:"
    )

def build_answer_prompt(query: str, documents: list[dict]) -> str:
    prompt = (
        "You are my personal research assistant. Use only the information provided below to answer the question.\n"
        "Cite document numbers in brackets (e.g., [2]) to indicate the source of each claim.\n"
        "Do not speculate, summarize external knowledge, or invent citations.\n"
        "Structure your answer clearly. Use bullet points or numbered steps if appropriate.\n\n"
        "Retrieved documents:\n"
    )
    for i, doc in enumerate(documents):
        prompt += f"[{i+1}] {doc['text'].strip()}\n\n"
    prompt += f"User question: {query}\n\nAnswer:"
    return prompt

def build_summary_prompt(documents: list[dict]) -> str:
    prompt = (
        "You are summarizing the following documents. Focus on clarity, conciseness, and structure.\n"
        "Do not add external information or commentary. Use bullet points or short paragraphs.\n\n"
        "Documents to summarize:\n"
    )
    for i, doc in enumerate(documents):
        prompt += f"[{i+1}] {doc['text'].strip()}\n\n"
    prompt += "Summary:"
    return prompt