# web_search.py
"""
Performs global web search and formats results for display.
"""

import requests
from bs4 import BeautifulSoup
import trafilatura
import textwrap
from config import BRAVE_API_KEY

# === Brave Search ===
def brave_search(query, max_results=20, api_key=BRAVE_API_KEY):
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "count": max_results
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("web", {}).get("results", []):
        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("description")
        })
    return results

# === Content Filtering ===
def is_useless_content(text):
    keywords = {
        "javascript", "js", "ad block", "adblock", "enable js", "disable ad blocker",
        "please enable", "your browser does not support", "legal statement",
        "copyright", "all rights reserved", "terms of use",
        "subscribe to continue", "sign in to read", "access denied",
        "404 error", "page not found", "content not available",
        "error while loading", "relod this page"
    }
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)

# === Scrapers ===
def scrape_with_trafilatura(url, max_chars=1000):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text.strip()[:max_chars] if text else ""
        return ""
    except Exception as e:
        return f"[Trafilatura error: {e}]"

def scrape_with_bs4(url, max_chars=1000):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs)
        return text.strip()[:max_chars]
    except Exception as e:
        return f"[BS4 error: {e}]"

def scrape_content(url, max_chars=1000):
    text = scrape_with_trafilatura(url, max_chars)
    if not text.strip():
        text = scrape_with_bs4(url, max_chars)
    return text

# === Main Search Pipeline ===
def global_search(query):
    raw_results = brave_search(query)
    clean_results = []

    for r in raw_results:
        text = scrape_content(r["url"])
        if is_useless_content(text):
            continue
        clean_results.append({
            "title": r["title"],
            "url": r["url"],
            "content": text
        })
        if len(clean_results) == 10:
            break

    return clean_results

# === CLI-Friendly Formatter ===
def format_web_results(results):
    output = "\n🌐 Filtered Web Search Results:\n"
    for i, r in enumerate(results, 1):
        summary = textwrap.fill(r["content"].strip(), width=100)
        output += f"\n[{i}] 📝 {r['title'].strip()}\n🔗 {r['url'].strip()}\n📄 Summary:\n{summary[:500]}...\n"
    return output