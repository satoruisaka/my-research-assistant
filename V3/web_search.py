"""
web_search.py - Web Search with BRAVE API + DuckDuckGo Fallback

Provides web search capabilities with automatic caching to data/web_cache
for later FAISS indexing.

Search flow:
1. Try BRAVE API (primary, requires API key)
2. Fall back to DuckDuckGo if BRAVE fails
3. Cache results to data/web_cache/*.json
4. Return structured results

Cache structure:
data/web_cache/
├── query_hash_timestamp.json
└── query_hash_timestamp.md  (optional, for indexing)
"""

import os
import json
import hashlib
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# Import config constants
try:
    from config import BRAVE_API_URL, USER_AGENTS, WEB_FETCH_TIMEOUT, WEB_FETCH_MAX_CHARS
except ImportError:
    # Fallback defaults if config not available
    BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
    USER_AGENTS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"]
    WEB_FETCH_TIMEOUT = 10
    WEB_FETCH_MAX_CHARS = 10000


@dataclass
class SearchResult:
    """Single web search result."""
    title: str
    url: str
    snippet: str
    source: str  # 'brave' or 'duckduckgo'
    timestamp: str  # ISO-8601
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_found: int
    source: str  # Which API was used
    fallback_used: bool
    cached: bool
    cache_path: Optional[str]
    query_time_ms: float
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'total_found': self.total_found,
            'source': self.source,
            'fallback_used': self.fallback_used,
            'cached': self.cached,
            'cache_path': self.cache_path,
            'query_time_ms': self.query_time_ms
        }


class WebSearchError(Exception):
    """Base exception for web search errors."""
    pass


class BraveAPIError(WebSearchError):
    """Raised when BRAVE API fails."""
    pass


class DuckDuckGoError(WebSearchError):
    """Raised when DuckDuckGo fails."""
    pass


class WebSearchClient:
    """
    Web search client with BRAVE (primary) and DuckDuckGo (fallback).
    
    Usage:
        client = WebSearchClient(
            brave_api_key="your_key",
            cache_dir="data/web_cache"
        )
        
        response = client.search(
            query="quantum computing applications",
            num_results=10,
            use_cache=True
        )
        
        for result in response.results:
            print(f"{result.title}: {result.url}")
    """
    
    def __init__(
        self,
        brave_api_key: Optional[str] = None,
        cache_dir: str = "data/web_cache",
        verbose: bool = False
    ):
        """
        Initialize web search client.
        
        Args:
            brave_api_key: BRAVE API key (reads from env if None)
            cache_dir: Directory for caching results
            verbose: Enable debug logging
        """
        self.brave_api_key = brave_api_key or os.getenv('BRAVE_API_KEY', '')
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # BRAVE API endpoint (from config)
        self.brave_url = BRAVE_API_URL
        
        # Fetch settings
        self.fetch_timeout = WEB_FETCH_TIMEOUT
        self.fetch_max_chars = WEB_FETCH_MAX_CHARS
        
        self._log(f"WebSearchClient initialized (BRAVE: {bool(self.brave_api_key)})")
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the pool."""
        return random.choice(USER_AGENTS)
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[WebSearchClient] {message}")
    
    def _query_hash(self, query: str) -> str:
        """Generate hash for query (for cache filenames)."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, query: str) -> Path:
        """
        Get cache file path for query.
        
        Args:
            query: Search query
            
        Returns:
            Path to cache file
        """
        query_hash = self._query_hash(query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{query_hash}_{timestamp}.json"
        return self.cache_dir / filename
    
    def _save_to_cache(
        self,
        query: str,
        results: List[SearchResult]
    ) -> str:
        """
        Save search results to cache.
        
        Args:
            query: Search query
            results: Search results
            
        Returns:
            Path to cache file
        """
        cache_path = self._get_cache_path(query)
        
        cache_data = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        self._log(f"Cached {len(results)} results to {cache_path.name}")
        return str(cache_path)
    
    def _search_brave(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        Search using BRAVE API.
        
        Args:
            query: Search query
            num_results: Number of results to fetch
            
        Returns:
            List of search results
            
        Raises:
            BraveAPIError: If BRAVE API fails
        """
        if not self.brave_api_key:
            raise BraveAPIError("BRAVE API key not configured")
        
        self._log(f"Searching BRAVE for: '{query}'")
        
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.brave_api_key,
            'User-Agent': self._get_random_user_agent()
        }
        
        params = {
            'q': query,
            'count': num_results,
            'text_decorations': False,
            'search_lang': 'en'
        }
        
        try:
            response = requests.get(
                self.brave_url,
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse BRAVE response
            results = []
            web_results = data.get('web', {}).get('results', [])
            
            for item in web_results[:num_results]:
                result = SearchResult(
                    title=item.get('title', 'No title'),
                    url=item.get('url', ''),
                    snippet=item.get('description', ''),
                    source='brave',
                    timestamp=datetime.now().isoformat()
                )
                results.append(result)
            
            self._log(f"BRAVE returned {len(results)} results")
            return results
        
        except requests.exceptions.RequestException as e:
            self._log(f"BRAVE API error: {e}")
            raise BraveAPIError(f"BRAVE API request failed: {e}")
        except Exception as e:
            self._log(f"BRAVE parsing error: {e}")
            raise BraveAPIError(f"BRAVE response parsing failed: {e}")
    
    def _search_duckduckgo(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        Search using DuckDuckGo (fallback).
        
        Args:
            query: Search query
            num_results: Number of results to fetch
            
        Returns:
            List of search results
            
        Raises:
            DuckDuckGoError: If DuckDuckGo fails
        """
        self._log(f"Searching DuckDuckGo for: '{query}'")
        
        try:
            ddgs = DDGS()
            raw_results = ddgs.text(query, max_results=num_results)
            
            results = []
            for item in raw_results:
                result = SearchResult(
                    title=item.get('title', 'No title'),
                    url=item.get('href', item.get('link', '')),
                    snippet=item.get('body', item.get('snippet', '')),
                    source='duckduckgo',
                    timestamp=datetime.now().isoformat()
                )
                results.append(result)
            
            self._log(f"DuckDuckGo returned {len(results)} results")
            return results
        
        except Exception as e:
            self._log(f"DuckDuckGo error: {e}")
            raise DuckDuckGoError(f"DuckDuckGo search failed: {e}")
    
    def fetch_url_content(
        self,
        url: str,
        max_chars: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Fetch and extract clean text content from a URL using BeautifulSoup.
        
        Args:
            url: URL to fetch
            max_chars: Maximum characters to extract (uses config default if None)
            
        Returns:
            Dict with keys: url, title, text, metadata
            
        Raises:
            Exception: If fetch or parsing fails
        """
        max_chars = max_chars or self.fetch_max_chars
        
        try:
            # Fetch with timeout and rotating user agent
            headers = {"User-Agent": self._get_random_user_agent()}
            response = requests.get(url, headers=headers, timeout=self.fetch_timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else url
            
            # Remove script, style, and navigation elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract text from main content areas (prioritize)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            
            if main_content:
                text = main_content.get_text(separator='\\n', strip=True)
            else:
                # Fallback to body
                text = soup.get_text(separator='\\n', strip=True)
            
            # Clean up text: remove excessive newlines
            lines = [line.strip() for line in text.split('\\n') if line.strip()]
            text = '\\n'.join(lines)
            
            # Truncate if too long
            if len(text) > max_chars:
                text = text[:max_chars] + "\\n\\n[Content truncated...]"
            
            metadata = {
                "content_type": response.headers.get('Content-Type', 'unknown'),
                "length": len(text),
                "fetched_at": datetime.now().isoformat()
            }
            
            self._log(f"Fetched {len(text)} chars from {url}")
            
            return {
                "url": url,
                "title": title,
                "text": text,
                "metadata": metadata
            }
            
        except requests.RequestException as e:
            self._log(f"Failed to fetch {url}: {e}")
            raise Exception(f"Failed to fetch URL: {e}")
        except Exception as e:
            self._log(f"Error processing {url}: {e}")
            raise Exception(f"Error processing URL: {e}")
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        use_cache: bool = True,
        force_fallback: bool = False
    ) -> SearchResponse:
        """
        Search with BRAVE (primary) or DuckDuckGo (fallback).
        
        Args:
            query: Search query
            num_results: Number of results to fetch
            use_cache: Save results to cache
            force_fallback: Skip BRAVE, use DuckDuckGo only
            
        Returns:
            SearchResponse with results and metadata
            
        Raises:
            WebSearchError: If both BRAVE and DuckDuckGo fail
        """
        start_time = time.time()
        results = []
        source = None
        fallback_used = False
        
        # Try BRAVE first
        if not force_fallback and self.brave_api_key:
            try:
                results = self._search_brave(query, num_results)
                source = 'brave'
            except BraveAPIError as e:
                self._log(f"BRAVE failed, falling back to DuckDuckGo: {e}")
                fallback_used = True
        else:
            fallback_used = True
        
        # Fall back to DuckDuckGo
        if fallback_used or not results:
            try:
                results = self._search_duckduckgo(query, num_results)
                source = 'duckduckgo'
            except DuckDuckGoError as e:
                if not results:  # Both failed
                    raise WebSearchError(f"All search methods failed. Last error: {e}")
        
        # Cache results
        cache_path = None
        if use_cache and results:
            cache_path = self._save_to_cache(query, results)
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            source=source,
            fallback_used=fallback_used,
            cached=use_cache,
            cache_path=cache_path,
            query_time_ms=query_time_ms
        )
    
    def get_cached_queries(self) -> List[Dict]:
        """
        List all cached queries.
        
        Returns:
            List of dicts with query info
        """
        cached = []
        
        for cache_file in self.cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached.append({
                        'query': data.get('query'),
                        'timestamp': data.get('timestamp'),
                        'results_count': len(data.get('results', [])),
                        'file': cache_file.name
                    })
            except Exception as e:
                self._log(f"Error reading cache file {cache_file}: {e}")
        
        return cached
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached results.
        
        Args:
            older_than_days: Only clear files older than N days (None = all)
        """
        count = 0
        now = datetime.now()
        
        for cache_file in self.cache_dir.glob('*.json'):
            if older_than_days:
                age_days = (now - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
                if age_days < older_than_days:
                    continue
            
            cache_file.unlink()
            count += 1
        
        self._log(f"Cleared {count} cache files")


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Web Search Client')
    parser.add_argument('--query', type=str, required=True,
                       help='Search query')
    parser.add_argument('--num-results', type=int, default=10,
                       help='Number of results')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--force-ddg', action='store_true',
                       help='Force DuckDuckGo (skip BRAVE)')
    parser.add_argument('--cache-dir', type=str, default='data/web_cache',
                       help='Cache directory')
    parser.add_argument('--list-cache', action='store_true',
                       help='List cached queries')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all cached queries')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Initialize client
    client = WebSearchClient(
        cache_dir=args.cache_dir,
        verbose=args.verbose
    )
    
    # List cache
    if args.list_cache:
        print("Cached queries:")
        cached = client.get_cached_queries()
        for item in cached:
            print(f"  - {item['query']} ({item['results_count']} results) - {item['timestamp']}")
        exit(0)
    
    # Clear cache
    if args.clear_cache:
        client.clear_cache()
        print("Cache cleared")
        exit(0)
    
    # Search
    print(f"Searching for: '{args.query}'")
    print(f"{'='*60}\n")
    
    try:
        response = client.search(
            query=args.query,
            num_results=args.num_results,
            use_cache=not args.no_cache,
            force_fallback=args.force_ddg
        )
        
        print(f"Source: {response.source}")
        print(f"Fallback used: {response.fallback_used}")
        print(f"Cached: {response.cached}")
        print(f"Query time: {response.query_time_ms:.2f}ms")
        print(f"\nResults ({response.total_found}):\n")
        
        for i, result in enumerate(response.results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet[:100]}...")
            print()
        
        if response.cache_path:
            print(f"Cached to: {response.cache_path}")
    
    except WebSearchError as e:
        print(f"❌ Search failed: {e}")
        exit(1)
