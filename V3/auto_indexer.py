"""
auto_indexer.py

Automatic indexing for sessions and web cache.

Called by chat_manager.py and web_search.py to immediately index
new content during MRA_v3 runtime.

This is NOT for reference papers (those use update_paper_indices.py manually).
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class AutoIndexer:
    """
    Handles automatic indexing of sessions and web cache.
    
    Called when:
    - Chat session ends → auto-index session log
    - Web result saved → auto-index cached page
    """
    
    def __init__(self):
        """Initialize auto-indexer (lazy-load processors)"""
        self._document_processor = None
        self._faiss_builder = None
    
    @property
    def document_processor(self):
        """Lazy-load document processor"""
        if self._document_processor is None:
            from utils.document_processor import process_document
            self._document_processor = process_document
        return self._document_processor
    
    @property
    def faiss_builder(self):
        """Lazy-load FAISS builder"""
        if self._faiss_builder is None:
            from utils.faiss_builder import add_documents_to_index
            self._faiss_builder = add_documents_to_index
        return self._faiss_builder
    
    def index_session_log(self, log_path: Path, session_id: str) -> bool:
        """
        Automatically index a session log after chat ends.
        
        Args:
            log_path: Path to session markdown file
            session_id: Unique session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Auto-indexing session log: {log_path.name}")
            
            # Process session log (markdown → chunks → embeddings)
            result = self.document_processor(
                file_path=str(log_path),
                index_name='sessions',
                source_type='session_log',
                metadata={
                    'session_id': session_id,
                    'auto_indexed': True,
                    'indexed_at': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"✅ Session log indexed: {result['chunks_created']} chunks")
            return True
        
        except Exception as e:
            logger.error(f"Failed to auto-index session log: {e}", exc_info=True)
            return False
    
    def index_web_cache(self, cache_path: Path, url: str, query: str) -> bool:
        """
        Automatically index a cached web page.
        
        Args:
            cache_path: Path to cached markdown file
            url: Original URL
            query: Search query that found this page
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Auto-indexing web cache: {cache_path.name}")
            
            # Process cached web page (markdown → chunks → embeddings)
            result = self.document_processor(
                file_path=str(cache_path),
                index_name='web_cache',
                source_type='web_cache',
                metadata={
                    'url': url,
                    'search_query': query,
                    'auto_indexed': True,
                    'indexed_at': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"✅ Web cache indexed: {result['chunks_created']} chunks")
            return True
        
        except Exception as e:
            logger.error(f"Failed to auto-index web cache: {e}", exc_info=True)
            return False
    
    def batch_index_sessions(self, session_logs: list) -> Dict[str, int]:
        """
        Batch index multiple session logs (startup recovery).
        
        Used when MRA_v3 starts and finds un-indexed session logs.
        
        Args:
            session_logs: List of (path, session_id) tuples
            
        Returns:
            {'success': N, 'failed': M}
        """
        results = {'success': 0, 'failed': 0}
        
        for log_path, session_id in session_logs:
            if self.index_session_log(log_path, session_id):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def batch_index_web_cache(self, cache_files: list) -> Dict[str, int]:
        """
        Batch index multiple cached web pages (startup recovery).
        
        Args:
            cache_files: List of (path, url, query) tuples
            
        Returns:
            {'success': N, 'failed': M}
        """
        results = {'success': 0, 'failed': 0}
        
        for cache_path, url, query in cache_files:
            if self.index_web_cache(cache_path, url, query):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        return results


# Global instance (singleton pattern)
_auto_indexer = None

def get_auto_indexer() -> AutoIndexer:
    """Get singleton auto-indexer instance"""
    global _auto_indexer
    if _auto_indexer is None:
        _auto_indexer = AutoIndexer()
    return _auto_indexer
