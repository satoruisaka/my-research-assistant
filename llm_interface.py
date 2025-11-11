# llm_interface.py
"""
Handles interaction with the local LLM via Ollama's HTTP API.
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def ask_llm(prompt: str, model: str = MODEL_NAME, timeout: int = 30) -> str:
    """
    Sends a prompt to the local LLM via Ollama and returns the full streamed response.
    Includes basic error handling and timeout protection.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt},
            stream=True,
            timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"[MRA] ⚠️ LLM request failed: {e}"

    output = ""
    for line in response.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                output += chunk.get("response", "")
            except json.JSONDecodeError:
                continue
    return output


