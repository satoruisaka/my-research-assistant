# conversation.py
"""
Manages the interactive conversation loop, keyword-based task selection, prompt assembly, and logging.
"""

import datetime
from retrieval import search, rerank
from llm_interface import ask_llm
from prompt_builder import (
    build_general_prompt,
    build_answer_prompt,
    build_summary_prompt
)
from utils import clean_response
from utils import format_citation_map
from markdown_logger import MarkdownLogger
from web_search import global_search, format_web_results

def show_task_menu(retrieval_ready):
    if not retrieval_ready:
        print("\nChoose your next task:")
        print("[1] General knowledge (@general)")
        print("[2] Search documents (@search)")
        print("[3] Web search (@websearch)")  # ✅ NEW
        return {
            "1": "@general",
            "2": "@search",
            "3": "@websearch"
        }
    else:
        print("\nChoose your next task:")
        print("[1] Answer using retrieved documents (@answer)")
        print("[2] Summarize a document (@summary)")
        print("[3] Search new documents (@search)")
        print("[4] General knowledge (@general)")
        print("[5] Web search (@websearch)")  # ✅ NEW
        return {
            "1": "@answer",
            "2": "@summary",
            "3": "@search",
            "4": "@general",
            "5": "@websearch"
        }

def start_conversation(index, metadata):
    retrieval_ready = False
    retrieved_documents = []
    log_path = f"logs/MRAlog_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    logger = MarkdownLogger()

    print("\n🧠 Welcome to My Research Assistant (MRA)")
    print("💬 Type 'exit' to quit at any time.")

    while True:
        keyword_map = show_task_menu(retrieval_ready)
        choice = input("Enter number or keyword: ").strip()
        if choice.lower() in ["exit", "quit"]:
            print("\n👋 Session ended.")
            break

        keyword = keyword_map.get(choice, choice if choice.startswith("@") else None)
        if keyword not in ["@general", "@search", "@answer", "@summary", "@websearch"]:
            print("Invalid task. Try again.")
            continue

        if keyword == "@general":
            query = input("Enter your query or topic: ").strip()
            prompt = build_general_prompt(query)
            response = ask_llm(prompt)
            cleaned = clean_response(response)
            print(f"\n[MRA] (General)\n{cleaned}")
            logger.log_turn("@general", query, prompt, cleaned, [], log_path)
            retrieval_ready = False

        elif keyword == "@search":
            query = input("Enter your query or topic: ").strip()
            retrieved_documents = search(query, index, metadata, top_k=20)
            ranked_documents = rerank(query, retrieved_documents, top_k=20)
            print("\nTop documents:")
            print(format_citation_map(ranked_documents))
            retrieved_documents = ranked_documents
            citations = format_citation_map(retrieved_documents)
            logger.log_turn("@search", query, " ", citations, [], log_path)
            retrieval_ready = True

        elif keyword == "@answer":
            if not retrieved_documents:
                print("No documents retrieved yet. Use @search first.")
                continue
            query = input("Enter your query or topic: ").strip()
            prompt = build_answer_prompt(query, retrieved_documents)
            response = ask_llm(prompt)
            cleaned = clean_response(response)
            print(f"\n[MRA] (Answer)\n{cleaned}")
            citations = format_citation_map(retrieved_documents)
            logger.log_turn("@answer", query, prompt, cleaned, citations, log_path)

        elif keyword == "@summary":
            if not retrieved_documents:
                print("No documents retrieved yet. Use @search first.")
                continue
            print("Enter comma-separated document numbers to summarize:")
            doc_ids = input().strip()
            selected = [int(i) for i in doc_ids.split(",") if i.isdigit()]
            selected_documents = [retrieved_documents[i - 1] for i in selected if 0 < i <= len(retrieved_documents)]
            prompt = build_summary_prompt(selected_documents)
            cleaned = clean_response(ask_llm(prompt))
            print("\n📄 Summary:\n")
            print(cleaned)
            citations = format_citation_map(selected_documents)
            top_chunks = [f"[{i+1}] {doc['filename']}" for i, doc in enumerate(selected_documents)]
            logger.log_turn("@summary", query, prompt, cleaned, citations, log_path)

        elif keyword == "@websearch":  # ✅ NEW
            query = input("Enter your web search query: ").strip()
            print(f"\n🔎 Searching the web for: {query}")
            results = global_search(query)
            print(f"\n✅ Displaying {len(results)} clean results")
            print(format_web_results(results))
            logger.log_turn("@websearch", query, "web_search", "Displayed web results", [], log_path)

        print("\n🔁 Ready for your next task.")