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
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator
import asyncio
import json as json_lib

from chat_manager import ChatManager, SessionSettings
from retrieval_manager import RetrievalManager, SearchScope
from web_search import WebSearchClient
from ollama_client import OllamaClient
from twistedpair_client import TwistedPairClient, DistortionMode, DistortionTone
from config import NUM_CTX, DEFAULT_MODEL, MAX_OUTPUT_TOKENS, MAX_TEMPERATURE, MAX_TOP_P, MAX_TOP_K


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
    message: str = Field(..., min_length=1, max_length=1000000, description="User message (max 1,000,000 characters)")
    
    # LLM settings
    model: str = DEFAULT_MODEL
    temperature: float = Field(0.7, ge=0.0, le=MAX_TEMPERATURE)
    top_p: float = Field(0.9, ge=0.0, le=MAX_TOP_P)
    top_k: int = Field(40, ge=0, le=MAX_TOP_K)
    max_tokens: int = Field(MAX_OUTPUT_TOKENS, ge=100, le=MAX_OUTPUT_TOKENS)
    num_ctx: int = Field(NUM_CTX, ge=1000, le=NUM_CTX)
    
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
    
    # Uploaded document context (optional)
    uploaded_context: Optional[List[Dict[str, str]]] = None
    
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
    text: str = Field(..., min_length=1, max_length=100000)
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

@app.post("/api/chat/message/stream")
async def chat_message_stream(request: ChatMessageRequest):
    """
    Process chat message with streaming response (Server-Sent Events).
    
    Returns:
        StreamingResponse with text/event-stream
    """
    try:
        # Convert request settings to SessionSettings
        settings = SessionSettings(
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            num_ctx=request.num_ctx,
            top_k_retrieval=request.top_k_retrieval,
            use_rag=request.use_rag,
            use_web_search=request.use_web_search,
            use_distortion=request.use_distortion,
            use_ensemble_distortion=request.use_ensemble_distortion,
            include_conversation_context=request.include_conversation_context,
            search_scope=request.search_scope,
            distortion_mode=request.distortion_mode.lower(),
            distortion_tone=request.distortion_tone.lower(),
            distortion_gain=request.distortion_gain
        )
        
        # Get or create session
        session = chat_manager.get_session(request.session_id, settings)
        session.add_message('user', request.message)
        
        # Perform retrieval and web search (same as non-streaming)
        context_items = []
        if settings.use_rag:
            from retrieval_manager import SearchScope
            scope = SearchScope(**settings.search_scope)
            retrieval_results = chat_manager.retrieval.unified_search(
                query=request.message,
                scope=scope,
                k=settings.top_k_retrieval
            )
            
            from chat_manager import ContextItem
            for result in retrieval_results:
                context_items.append(ContextItem(
                    source=result.source,
                    title=result.filename,
                    snippet=result.child_text[:300],
                    score=result.score,
                    doc_id=result.doc_id
                ))
        
        if settings.use_web_search:
            try:
                web_response = chat_manager.web_search.search(
                    query=request.message,
                    num_results=10
                )
                from chat_manager import ContextItem
                for result in web_response.results:
                    context_items.append(ContextItem(
                        source='web_search',
                        title=result.title,
                        snippet=result.snippet,
                        url=result.url
                    ))
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Add uploaded document context if provided
        if settings.use_rag and request.uploaded_context:
            logger.info(f"Including {len(request.uploaded_context)} uploaded documents in context")
            from chat_manager import ContextItem
            for doc in request.uploaded_context:
                context_items.append(ContextItem(
                    source='uploaded_document',
                    title=doc.get('filename', 'Uploaded Document'),
                    snippet=doc.get('content', '')[:500],  # First 500 chars as snippet
                    full_content=doc.get('content', '')  # Store full content
                ))
        
        # Build prompt
        if context_items:
            session.add_context(context_items)
            context_summary = session.get_context_summary()
            
            # Add full content of uploaded documents to system prompt
            uploaded_docs_content = ""
            for item in context_items:
                if item.source == 'uploaded_document' and hasattr(item, 'full_content'):
                    uploaded_docs_content += f"\n\n=== Uploaded Document: {item.title} ===\n{item.full_content}\n"
            
            system_prompt = (
                f"You are a helpful research assistant. "
                f"Answer based on the following context:\n\n{context_summary}{uploaded_docs_content}\n\n"
                f"If the context doesn't contain relevant information, say so."
            )
        else:
            system_prompt = "You are a helpful research assistant."
        
        messages = session.get_messages_for_llm(include_system=False)
        messages.insert(0, {'role': 'system', 'content': system_prompt})
        
        async def event_generator():
            """Generate Server-Sent Events for streaming."""
            try:
                # Send session ID first
                yield f"data: {json_lib.dumps({'type': 'session_id', 'session_id': session.session_id})}\n\n"
                
                # Send context
                if context_items:
                    yield f"data: {json_lib.dumps({'type': 'context', 'context': [c.to_dict() for c in context_items]})}\n\n"
                
                # Stream tokens
                full_response = ""
                for token in chat_manager.ollama.chat_stream(
                    messages=messages,
                    model=settings.model,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    top_k=settings.top_k,
                    max_tokens=settings.max_tokens,
                    num_ctx=settings.num_ctx
                ):
                    full_response += token
                    yield f"data: {json_lib.dumps({'type': 'token', 'content': token})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
                
                # Apply distortion if enabled (after streaming completes)
                ensemble_outputs = None
                if settings.use_distortion:
                    try:
                        from twistedpair_client import DistortionTone
                        tone = DistortionTone(settings.distortion_tone)
                        
                        # Build text for distortion (response + optional conversation context)
                        distortion_text = full_response
                        if settings.include_conversation_context:
                            # Include last 3 user-assistant exchanges for context
                            recent_messages = []
                            for msg in session.messages[-7:-1]:  # Skip the current assistant response
                                if msg.role in ['user', 'assistant']:
                                    recent_messages.append(f"{msg.role.upper()}: {msg.content}")
                            
                            if recent_messages:
                                context = "\n".join(recent_messages)
                                distortion_text = f"CONVERSATION CONTEXT:\n{context}\n\nCURRENT RESPONSE:\n{full_response}"
                        
                        # Ensemble mode - all 6 perspectives
                        if settings.use_ensemble_distortion:
                            ensemble_result = chat_manager.twistedpair.distort_ensemble(
                                text=distortion_text,
                                tone=tone,
                                gain=settings.distortion_gain,
                                model=settings.model
                            )
                            
                            # Store all outputs for frontend display
                            ensemble_outputs = [
                                {
                                    'mode': output.mode,
                                    'tone': output.tone,
                                    'gain': output.gain,
                                    'response': output.output
                                }
                                for output in ensemble_result.outputs
                            ]
                            
                            # Send ensemble outputs
                            yield f"data: {json_lib.dumps({'type': 'ensemble', 'outputs': ensemble_outputs})}\n\n"
                        
                        # Manual mode - single perspective (optional: could override full_response)
                        else:
                            from twistedpair_client import DistortionMode
                            mode = DistortionMode(settings.distortion_mode)
                            distortion_result = chat_manager.twistedpair.distort(
                                text=distortion_text,
                                mode=mode,
                                tone=tone,
                                gain=settings.distortion_gain,
                                model=settings.model
                            )
                            
                            # Send single distortion output
                            yield f"data: {json_lib.dumps({'type': 'distortion', 'content': distortion_result.output})}\n\n"
                            
                    except Exception as e:
                        logger.error(f"Distortion failed: {e}")
                        yield f"data: {json_lib.dumps({'type': 'error', 'message': f'Distortion failed: {str(e)}'})}\n\n"
                
                # Save to session
                session.add_message('assistant', full_response, metadata={
                    'distorted': settings.use_distortion,
                    'context_count': len(context_items),
                    'ensemble': ensemble_outputs is not None
                })
                session.save()
                
                # Send done signal
                yield f"data: {json_lib.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json_lib.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            num_ctx=request.num_ctx,
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
                logger.info(f"Ollama models: {ollama_models}")
            except Exception as e:
                logger.error(f"Failed to list Ollama models: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
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
# File Upload Endpoint
# ============================================================================

@app.post("/api/upload")
async def upload_file(request: Request):
    """
    Upload a file (PDF, TXT, CSV, MD) for conversion and chat context.
    
    Returns:
        - filename: Original filename
        - file_type: File extension
        - markdown_content: Converted markdown text
        - token_count: Token count
        - saved_path: Path where file was saved (if saved)
    """
    from fastapi import UploadFile, File, Form
    from MRA_v3_1_upload_to_md import process_uploaded_file
    from utils.document_processor import DocumentProcessor
    import tempfile
    import shutil
    
    try:
        # Parse multipart form data
        form = await request.form()
        uploaded_file = form.get('file')
        
        if not uploaded_file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.filename).suffix) as tmp_file:
            content = await uploaded_file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Process the file (convert to markdown)
            md_content, saved_path = process_uploaded_file(
                file_path=tmp_path,
                save_to_disk=True,  # Save to user_uploads directory
                include_metadata=True
            )
            
            # Get token count
            processor = DocumentProcessor()
            token_count = processor.count_tokens(md_content)
            
            # Store in chat context (add to state for current session)
            # Note: This is a simplified approach - in production, you'd want to
            # associate the uploaded doc with the specific session
            
            return JSONResponse({
                "filename": uploaded_file.filename,
                "file_type": Path(uploaded_file.filename).suffix,
                "markdown_content": md_content,
                "token_count": token_count,
                "saved_path": str(saved_path) if saved_path else None,
                "message": "File uploaded and converted successfully"
            })
            
        finally:
            # Cleanup temp file
            try:
                tmp_path.unlink()
            except:
                pass
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom validation error handler for clearer error messages."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        error_type = error["type"]
        
        # Provide human-readable messages for common validation errors
        if "max_length" in error_type:
            max_val = error.get("ctx", {}).get("limit_value", "unknown")
            errors.append(f"{field}: Text too long (max {max_val:,} characters)")
        elif "min_length" in error_type:
            min_val = error.get("ctx", {}).get("limit_value", "unknown")
            errors.append(f"{field}: Text too short (min {min_val} characters)")
        elif "greater_than_equal" in error_type or "less_than_equal" in error_type:
            limit = error.get("ctx", {}).get("limit_value", "unknown")
            errors.append(f"{field}: Value out of range ({msg})")
        else:
            errors.append(f"{field}: {msg}")
    
    logger.warning(f"Validation error on {request.url.path}: {errors}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": errors,
                "timestamp": datetime.now().isoformat()
            }
        }
    )


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
