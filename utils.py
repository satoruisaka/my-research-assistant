# utils.py
"""
Utility functions for token counting, history management, response cleanup, and citation formatting.
"""

from datetime import datetime

MAX_TOKENS = 3000  # adjust based on model context window

def count_tokens(messages):
    """
    Rough token count using whitespace splitting.
    """
    return sum(len(m["content"].split()) for m in messages)

def manage_history(history):
    """
    Trims conversation history to stay within token limits.
    """
    while count_tokens(history) > MAX_TOKENS:
        history.pop(0)
    return history

def build_search_query(history):
    """
    Constructs a smart search query using the latest user input.
    """
    recent_user_turns = [h["content"] for h in history if h["role"] == "user"]
    if recent_user_turns:
        return f"latest updates on {recent_user_turns[-1]}"
    return "latest news"

def format_citation_map(chunks):
    """
    Returns a formatted citation map string for Markdown logging.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        filename = chunk.get("filename", "Unknown file")
        source_type = chunk.get("source_type", "Unknown type")
        section = chunk.get("section", "Unknown section")
        score = round(chunk.get("score", 0.0), 3)
        lines.append(f"[{i}] {filename} ({source_type}) — Section: {section} — Score: {score}")
    return "\n".join(lines)

def clean_response(text: str) -> str:
    """
    Cleans LLM output by trimming whitespace and removing common artifacts.
    """
    return text.strip().replace("Answer:", "").replace("Response:", "").strip()

