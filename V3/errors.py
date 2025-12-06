"""
errors.py

MRA_v3 Error Handling Framework

Centralized exception definitions and error handling utilities.
"""
from enum import Enum
from typing import Optional, Dict, Any
import logging
import traceback

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification"""
    FILE_IO = "file_io"
    NETWORK = "network"
    PROCESSING = "processing"
    USER_INPUT = "user_input"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"


class MRAError(Exception):
    """Base exception for all MRA_v3 errors"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        self.message = message
        self.category = category
        self.details = details or {}
        self.recoverable = recoverable
        super().__init__(message)
    
    def to_dict(self) -> dict:
        """Convert error to JSON-serializable dict"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "details": self.details,
            "recoverable": self.recoverable
        }


# === File I/O Errors ===

class FileNotFoundError(MRAError):
    """File or directory not found"""
    def __init__(self, path: str, details: Optional[Dict] = None):
        super().__init__(
            f"File not found: {path}",
            ErrorCategory.FILE_IO,
            details or {"path": path},
            recoverable=True
        )


class FileProcessingError(MRAError):
    """Error processing file (PDF, DOCX, etc.)"""
    def __init__(self, path: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Failed to process file {path}: {reason}",
            ErrorCategory.PROCESSING,
            details or {"path": path, "reason": reason},
            recoverable=True
        )


class DiskFullError(MRAError):
    """Disk full during write operation"""
    def __init__(self, path: str, details: Optional[Dict] = None):
        super().__init__(
            f"Disk full, cannot write to: {path}",
            ErrorCategory.FILE_IO,
            details or {"path": path},
            recoverable=False
        )


# === Network Errors ===

class OllamaConnectionError(MRAError):
    """Cannot connect to Ollama server"""
    def __init__(self, url: str, details: Optional[Dict] = None):
        super().__init__(
            f"Ollama server not responding at {url}",
            ErrorCategory.NETWORK,
            details or {"url": url, "solution": "Run: ollama serve"},
            recoverable=False
        )


class TwistedPairConnectionError(MRAError):
    """Cannot connect to TwistedPair V2 server"""
    def __init__(self, url: str, details: Optional[Dict] = None):
        super().__init__(
            f"TwistedPair V2 not responding at {url}",
            ErrorCategory.NETWORK,
            details or {"url": url, "solution": "Run: cd TwistedPair/V2 && uvicorn server:app"},
            recoverable=True  # Can continue without distortion
        )


class WebSearchError(MRAError):
    """Web search failed (BRAVE/DDG)"""
    def __init__(self, query: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Web search failed for '{query}': {reason}",
            ErrorCategory.NETWORK,
            details or {"query": query, "reason": reason},
            recoverable=True
        )


class WebFetchError(MRAError):
    """Failed to fetch web page content"""
    def __init__(self, url: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Failed to fetch {url}: {reason}",
            ErrorCategory.NETWORK,
            details or {"url": url, "reason": reason},
            recoverable=True
        )


# === Processing Errors ===

class ChunkingError(MRAError):
    """Error during text chunking"""
    def __init__(self, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Chunking failed: {reason}",
            ErrorCategory.PROCESSING,
            details or {"reason": reason},
            recoverable=True
        )


class DocumentProcessingError(MRAError):
    """Error during document processing pipeline"""
    def __init__(self, stage: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Document processing failed at {stage}: {reason}",
            ErrorCategory.PROCESSING,
            details or {"stage": stage, "reason": reason},
            recoverable=True
        )


class EmbeddingError(MRAError):
    """Error during text embedding"""
    def __init__(self, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Embedding failed: {reason}",
            ErrorCategory.PROCESSING,
            details or {"reason": reason},
            recoverable=False
        )


class FAISSIndexError(MRAError):
    """FAISS index operation failed"""
    def __init__(self, operation: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"FAISS {operation} failed: {reason}",
            ErrorCategory.PROCESSING,
            details or {"operation": operation, "reason": reason},
            recoverable=False
        )


class IndexCorruptionError(MRAError):
    """FAISS index file corrupted"""
    def __init__(self, index_name: str, details: Optional[Dict] = None):
        super().__init__(
            f"Index corrupted: {index_name}. Rebuild required.",
            ErrorCategory.PROCESSING,
            details or {"index_name": index_name, "solution": "Run: python utils/rebuild_indices.py"},
            recoverable=False
        )


# === User Input Errors ===

class InvalidQueryError(MRAError):
    """Invalid user query"""
    def __init__(self, query: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Invalid query '{query}': {reason}",
            ErrorCategory.USER_INPUT,
            details or {"query": query, "reason": reason},
            recoverable=True
        )


class InvalidIndexError(MRAError):
    """Requested index does not exist"""
    def __init__(self, index_name: str, available: list, details: Optional[Dict] = None):
        super().__init__(
            f"Index '{index_name}' not found. Available: {', '.join(available)}",
            ErrorCategory.USER_INPUT,
            details or {"index_name": index_name, "available": available},
            recoverable=True
        )


# === Configuration Errors ===

class ConfigurationError(MRAError):
    """Configuration error"""
    def __init__(self, key: str, reason: str, details: Optional[Dict] = None):
        super().__init__(
            f"Configuration error for '{key}': {reason}",
            ErrorCategory.CONFIGURATION,
            details or {"key": key, "reason": reason},
            recoverable=False
        )


class MissingDependencyError(MRAError):
    """Required dependency not available"""
    def __init__(self, dependency: str, solution: str, details: Optional[Dict] = None):
        super().__init__(
            f"Missing dependency: {dependency}",
            ErrorCategory.DEPENDENCY,
            details or {"dependency": dependency, "solution": solution},
            recoverable=False
        )


# === Error Handler Utilities ===

def handle_error(error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Central error handler that converts exceptions to structured responses.
    
    Args:
        error: Exception that occurred
        context: Additional context (operation, file, etc.)
        
    Returns:
        Structured error dict for API responses
    """
    context = context or {}
    
    # Log error with traceback
    logger.error(
        f"Error in {context.get('operation', 'unknown')}: {error}",
        exc_info=True
    )
    
    # Convert to MRAError if not already
    if isinstance(error, MRAError):
        mra_error = error
    else:
        # Wrap unknown errors
        mra_error = MRAError(
            message=str(error),
            category=ErrorCategory.PROCESSING,
            details={"original_error": error.__class__.__name__, **context},
            recoverable=False
        )
    
    # Build response
    response = mra_error.to_dict()
    response["context"] = context
    response["traceback"] = traceback.format_exc() if logger.level == logging.DEBUG else None
    
    return response


def safe_operation(operation_name: str):
    """
    Decorator for safe operation execution with error handling.
    
    Usage:
        @safe_operation("embedding_text")
        def embed_text(text):
            return model.encode(text)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_response = handle_error(e, {"operation": operation_name})
                logger.error(f"Operation '{operation_name}' failed: {error_response}")
                raise
        return wrapper
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying operations that may fail transiently.
    
    Usage:
        @retry_on_failure(max_retries=3, delay=2.0, exceptions=(NetworkError,))
        def fetch_url(url):
            return requests.get(url)
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        return wrapper
    return decorator
