"""
server.py - FastAPI REST API for MRA_v3


Endpoints:
- POST /api/chat/message - Process chat message with RAG
- POST /api/chat/end-session - End session
- GET /api/sessions - List all sessions
- GET /api/sessions/{id} - Get session details
- POST /api/search - Direct search endpoint
- POST /api/web-search - Direct web search
- POST /api/distort - Direct distortion endpoint
- GET /api/health - Service health check
- Static file serving for Web UI
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from chat_manager import ChatManager, SessionSettings
from retrieval_manager import RetrievalManager, SearchScope
from web_search import WebSearchClient
from ollama_client import OllamaClient
from twistedpair_client import TwistedPairClient, DistortionMode, DistortionTone


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class ChatMessageRequest(BaseModel):
    """Request for /api/chat/message"""
    session_id: str = Field(..., description="Session ID or 'new' for new session")
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    
    # LLM settings
    model: str = "mistral:latest"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4000, ge=100, le=128000)
    
    # Retrieval settings
    use_rag: bool = True
    top_k_retrieval: int = Field(20, ge=1, le=50)
    search_scope: Dict[str, bool] = Field(default_factory=lambda: {
        'reference_papers': True,
        'my_papers': True,
        'sessions': False,
        'web_cache': False
    })
    
    # Web search settings
    use_web_search: bool = False
    
    # Distortion settings
    use_distortion: bool = False
    use_ensemble_distortion: bool = False  # Get all 6 modes
    include_conversation_context: bool = True  # Include recent messages in distortion
    distortion_mode: str = "CUCUMB_ER"
    distortion_tone: str = "NEUTRAL"
    distortion_gain: int = Field(5, ge=1, le=10)
    
    @validator('message')
    def sanitize_message(cls, v):
        return v.strip()


class ChatMessageResponse(BaseModel):
    """Response for /api/chat/message"""
    session_id: str
    message_id: str
    assistant_response: str
    context_used: List[Dict[str, Any]]
    distorted: bool
    ensemble_outputs: Optional[List[Dict[str, Any]]] = None  # Array of 6 outputs when ensemble enabled
    timestamp: str


class EndSessionRequest(BaseModel):
    """Request for /api/chat/end-session"""
    session_id: str


class EndSessionResponse(BaseModel):
    """Response for /api/chat/end-session"""
    status: str
    session_id: str
    messages_count: int
    log_path: str


class SearchRequest(BaseModel):
    """Request for /api/search"""
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(10, ge=1, le=50)
    search_scope: Dict[str, bool] = Field(default_factory=lambda: {
        'reference_papers': True,
        'my_papers': True,
        'sessions': False,
        'web_cache': False
    })


class WebSearchRequest(BaseModel):
    """Request for /api/web-search"""
    query: str = Field(..., min_length=1, max_length=500)
    num_results: int = Field(10, ge=1, le=50)
    use_cache: bool = True


class DistortRequest(BaseModel):
    """Request for /api/distort"""
    text: str = Field(..., min_length=1, max_length=10000)
    mode: str = "CUCUMB_ER"
    tone: str = "NEUTRAL"
    gain: int = Field(5, ge=1, le=10)
    model: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for /api/health"""
    status: str
    timestamp: str
    services: Dict[str, Any]
    indices: Dict[str, Any]


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="MRA_v3 - My Research Assistant",
    description="Local-first research assistant with RAG, web search, and distortion",
    version="3.0.0"
)

# CORS middleware (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Instances
# ============================================================================

# Initialize components
chat_manager: Optional[ChatManager] = None
retrieval_manager: Optional[RetrievalManager] = None
web_search: Optional[WebSearchClient] = None
ollama_client: Optional[OllamaClient] = None
twistedpair_client: Optional[TwistedPairClient] = None


@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global chat_manager, retrieval_manager, web_search, ollama_client, twistedpair_client
    
    logger.info("Starting MRA_v3 server...")
    
    try:
        # Initialize clients (verbose=True for debugging)
        retrieval_manager = RetrievalManager(verbose=True)
        web_search = WebSearchClient(verbose=False)
        ollama_client = OllamaClient(verbose=True)
        twistedpair_client = TwistedPairClient(verbose=False)
        
        # Initialize chat manager with all clients
        chat_manager = ChatManager(
            retrieval_manager=retrieval_manager,
            web_search=web_search,
            ollama_client=ollama_client,
            twistedpair_client=twistedpair_client,
            sessions_dir="data/sessions",
            verbose=True
        )
        
        logger.info("All components initialized successfully")
        
        # Check service health
        ollama_health = ollama_client.is_healthy()
        twistedpair_health = twistedpair_client.is_healthy()
        
        if not ollama_health:
            logger.warning("Ollama service not available - LLM features will fail")
        
        if not twistedpair_health:
            logger.warning("TwistedPair service not available - distortion features will fail")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/chat/message", response_model=ChatMessageResponse)
async def chat_message(request: ChatMessageRequest):
    """
    Process chat message with RAG, web search, and optional distortion.
    
    This is the main endpoint used by the Web UI.
    """
    try:
        # Build session settings from request
        settings = SessionSettings(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_k_retrieval=request.top_k_retrieval,
            use_rag=request.use_rag,
            use_web_search=request.use_web_search,
            use_distortion=request.use_distortion,
            use_ensemble_distortion=request.use_ensemble_distortion,
            include_conversation_context=request.include_conversation_context,
            search_scope=request.search_scope,
            distortion_mode=request.distortion_mode,
            distortion_tone=request.distortion_tone,
            distortion_gain=request.distortion_gain
        )
        
        # Process message
        response = chat_manager.process_message(
            session_id=request.session_id,
            message=request.message,
            settings=settings
        )
        
        return ChatMessageResponse(**response)
        
    except Exception as e:
        logger.error(f"Chat message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/end-session", response_model=EndSessionResponse)
async def end_session(request: EndSessionRequest):
    """End chat session and save final state."""
    try:
        result = chat_manager.end_session(request.session_id)
        return EndSessionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"End session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List all saved sessions."""
    try:
        sessions = chat_manager.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}
        
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific session details."""
    try:
        session = chat_manager.get_session(session_id)
        return session.to_dict()
        
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/search")
async def search(request: SearchRequest):
    """Direct search endpoint (bypasses chat)."""
    try:
        scope = SearchScope(**request.search_scope)
        results = retrieval_manager.unified_search(
            query=request.query,
            scope=scope,
            k=request.k
        )
        
        return {
            "query": request.query,
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "parent_text": r.parent_text,
                    "child_text": r.child_text,
                    "score": r.score,
                    "source": r.source,
                    "filename": r.filename,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/web-search")
async def web_search_endpoint(request: WebSearchRequest):
    """Direct web search endpoint."""
    try:
        response = web_search.search(
            query=request.query,
            num_results=request.num_results,
            use_cache=request.use_cache
        )
        
        return {
            "query": response.query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "timestamp": r.timestamp
                }
                for r in response.results
            ],
            "total_found": response.total_found,
            "source": response.source,
            "fallback_used": response.fallback_used,
            "cached": response.cached,
            "query_time_ms": response.query_time_ms
        }
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/distort")
async def distort(request: DistortRequest):
    """Direct distortion endpoint."""
    try:
        mode = DistortionMode(request.mode)
        tone = DistortionTone(request.tone)
        
        result = twistedpair_client.distort(
            text=request.text,
            mode=mode,
            tone=tone,
            gain=request.gain,
            model=request.model
        )
        
        return {
            "output": result.output,
            "mode": result.mode.value,
            "tone": result.tone.value,
            "gain": result.gain,
            "provenance": result.provenance
        }
        
    except Exception as e:
        logger.error(f"Distort error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Service health check with status for all components.
    
    Used by Web UI to populate model dropdown and show service status.
    """
    try:
        # Check Ollama
        ollama_healthy = ollama_client.is_healthy()
        ollama_models = []
        if ollama_healthy:
            try:
                ollama_models = ollama_client.list_models()
            except:
                pass
        
        # Check TwistedPair
        twistedpair_healthy = twistedpair_client.is_healthy()
        
        # Get index stats
        index_stats = retrieval_manager.get_index_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            services={
                "ollama": {
                    "status": "online" if ollama_healthy else "offline",
                    "models": ollama_models
                },
                "twistedpair": {
                    "status": "online" if twistedpair_healthy else "offline"
                }
            },
            indices={
                "reference_papers": index_stats.get('reference_papers', {}),
                "my_papers": index_stats.get('my_papers', {}),
                "sessions": index_stats.get('sessions', {}),
                "web_cache": index_stats.get('web_cache', {})
            }
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
            services={"error": str(e)},
            indices={}
        )


# ============================================================================
# Static File Serving (Web UI)
# ============================================================================

# Serve static files from ./static directory
static_dir = Path(__file__).parent / "static"


@app.get("/")
async def serve_index():
    """Serve index.html at root."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/styles.css")
async def serve_styles():
    """Serve styles.css at root."""
    css_path = static_dir / "styles.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    else:
        raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js")
async def serve_app_js():
    """Serve app.js at root."""
    js_path = static_dir / "app.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="app.js not found")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "timestamp": datetime.now().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": str(exc),
                "timestamp": datetime.now().isoformat()
            }
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
