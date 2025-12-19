"""
chat_manager.py - Chat Session Management

Manages multi-turn conversations with:
- Session persistence (JSON log files)
- Message history (user/assistant/system)
- Context tracking (retrieved docs, web results)
- Integration with retrieval_manager, web_search, ollama_client, twistedpair_client
- Auto-save on session end

Session structure:
data/sessions/
├── session_uuid_timestamp.json  (full conversation log)
└── session_uuid_timestamp.md    (markdown summary for indexing)
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

from retrieval_manager import RetrievalManager, SearchScope
from web_search import WebSearchClient
from ollama_client import OllamaClient
from twistedpair_client import TwistedPairClient, DistortionMode, DistortionTone
from config import NUM_CTX, DEFAULT_MODEL


@dataclass
class Message:
    """Single message in conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str  # ISO-8601
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class ContextItem:
    """Context item (retrieved document or web result)."""
    source: str  # 'reference_papers', 'web_search', 'uploaded_document', etc.
    title: str
    snippet: str
    score: Optional[float] = None
    url: Optional[str] = None
    doc_id: Optional[str] = None
    full_content: Optional[str] = None  # Full content for uploaded documents
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class SessionSettings:
    """Session-level settings."""
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = NUM_CTX
    top_k_retrieval: int = 20
    use_rag: bool = True
    use_web_search: bool = False
    use_distortion: bool = False
    use_ensemble_distortion: bool = False  # If True, returns all 6 modes
    include_conversation_context: bool = True  # Include recent messages in distortion
    search_scope: Dict[str, bool] = field(default_factory=lambda: {
        'reference_papers': True,
        'my_papers': True,
        'sessions': False,
        'web_cache': False
    })
    distortion_mode: str = "cucumb_er"
    distortion_tone: str = "neutral"
    distortion_gain: int = 5
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


class ChatSession:
    """
    Single chat session with message history and context.
    
    Usage:
        session = ChatSession(session_id="new")
        
        # Add user message
        session.add_message("user", "What is quantum computing?")
        
        # Add context from retrieval
        session.add_context([...])
        
        # Add assistant response
        session.add_message("assistant", "Quantum computing is...")
        
        # Save session
        session.save()
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        settings: Optional[SessionSettings] = None,
        sessions_dir: str = "data/sessions"
    ):
        """
        Initialize chat session.
        
        Args:
            session_id: Session ID (generates new if None)
            settings: Session settings (uses defaults if None)
            sessions_dir: Directory for session logs
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.settings = settings or SessionSettings()
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        self.messages: List[Message] = []
        self.context: List[ContextItem] = []
        self.title: str = "New Session"  # Auto-generated from first message
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
        # Try to load existing session
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load session from disk if it exists."""
        session_files = list(self.sessions_dir.glob(f"{self.session_id}_*.json"))
        
        if session_files:
            # Load most recent
            session_file = sorted(session_files)[-1]
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._from_dict(data)
    
    def _from_dict(self, data: Dict):
        """Load session from dict."""
        self.messages = [Message(**m) for m in data.get('messages', [])]
        # Don't load context - it's per-turn only, not persisted
        self.context = []
        self.title = data.get('title', 'Untitled Session')
        self.created_at = data.get('created_at', self.created_at)
        self.updated_at = data.get('updated_at', self.updated_at)
        
        if 'settings' in data:
            self.settings = SessionSettings(**data['settings'])
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add message to session.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: Message content
            metadata: Optional metadata
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = message.timestamp
        
        # Auto-generate title from first user message
        if role == 'user' and len(self.messages) == 1:
            # Use first 50 chars of first user message as title
            self.title = content[:50] + ('...' if len(content) > 50 else '')
    
    def add_context(self, context_items: List[ContextItem]):
        """
        Add context items (retrieved docs, web results).
        Replaces previous context (context is per-turn, not accumulated).
        
        Args:
            context_items: List of ContextItem objects
        """
        self.context = context_items  # Replace, don't extend
        self.updated_at = datetime.now().isoformat()
    
    def get_messages_for_llm(
        self,
        include_system: bool = True,
        max_history: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM (Ollama chat format).
        
        Args:
            include_system: Include system messages
            max_history: Maximum messages to include (recent)
            
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = []
        
        # Get recent messages
        recent = self.messages[-max_history:] if max_history else self.messages
        
        for msg in recent:
            if msg.role == 'system' and not include_system:
                continue
            messages.append({
                'role': msg.role,
                'content': msg.content
            })
        
        return messages
    
    def get_context_summary(self, max_items: int = 5) -> str:
        """
        Get formatted context summary for LLM prompt.
        
        Args:
            max_items: Maximum context items to include
            
        Returns:
            Formatted context string
        """
        if not self.context:
            return ""
        
        lines = ["Retrieved Context:"]
        for i, item in enumerate(self.context[:max_items], 1):
            lines.append(f"\n{i}. [{item.source}] {item.title}")
            lines.append(f"   {item.snippet[:200]}...")
            if item.score:
                lines.append(f"   Score: {item.score:.3f}")
        
        return "\n".join(lines)
    
    def save(self) -> str:
        """
        Save session to disk.
        
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_id}_{timestamp}.json"
        filepath = self.sessions_dir / filename
        
        data = {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'settings': self.settings.to_dict(),
            'messages': [m.to_dict() for m in self.messages]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def to_dict(self) -> Dict:
        """Convert full session to dict including messages."""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'metadata': msg.metadata
                }
                for msg in self.messages
            ],
            'message_count': len(self.messages),
            'settings': self.settings.to_dict()
        }


class ChatManager:
    """
    Manages chat sessions with RAG, web search, and distortion.
    
    Integrates:
    - RetrievalManager (FAISS search)
    - WebSearchClient (BRAVE + DuckDuckGo)
    - OllamaClient (LLM generation)
    - TwistedPairClient (optional distortion)
    
    Usage:
        manager = ChatManager()
        
        # Process user message
        response = manager.process_message(
            session_id="uuid-or-new",
            message="What is quantum entanglement?",
            settings=SessionSettings(use_rag=True)
        )
        
        print(response['assistant_response'])
    """
    
    def __init__(
        self,
        retrieval_manager: Optional[RetrievalManager] = None,
        web_search: Optional[WebSearchClient] = None,
        ollama_client: Optional[OllamaClient] = None,
        twistedpair_client: Optional[TwistedPairClient] = None,
        sessions_dir: str = "data/sessions",
        verbose: bool = False
    ):
        """
        Initialize chat manager.
        
        Args:
            retrieval_manager: RetrievalManager instance (creates new if None)
            web_search: WebSearchClient instance (creates new if None)
            ollama_client: OllamaClient instance (creates new if None)
            twistedpair_client: TwistedPairClient instance (creates new if None)
            sessions_dir: Directory for session logs
            verbose: Enable debug logging
        """
        self.retrieval = retrieval_manager or RetrievalManager(verbose=verbose)
        self.web_search = web_search or WebSearchClient(verbose=verbose)
        self.ollama = ollama_client or OllamaClient(verbose=verbose)
        self.twistedpair = twistedpair_client or TwistedPairClient(verbose=verbose)
        self.sessions_dir = Path(sessions_dir)
        self.verbose = verbose
        
        # Active sessions cache
        self.sessions: Dict[str, ChatSession] = {}
        
        self._log("ChatManager initialized")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[ChatManager] {message}")
    
    def get_session(
        self,
        session_id: str,
        settings: Optional[SessionSettings] = None
    ) -> ChatSession:
        """
        Get or create session.
        
        Args:
            session_id: Session ID ('new' creates new session)
            settings: Session settings (for new sessions)
            
        Returns:
            ChatSession instance
        """
        if session_id == 'new':
            session = ChatSession(
                session_id=None,
                settings=settings,
                sessions_dir=str(self.sessions_dir)
            )
            self.sessions[session.session_id] = session
            self._log(f"Created new session: {session.session_id}")
            return session
        
        if session_id not in self.sessions:
            session = ChatSession(
                session_id=session_id,
                settings=settings,
                sessions_dir=str(self.sessions_dir)
            )
            self.sessions[session_id] = session
            self._log(f"Loaded session: {session_id}")
        
        return self.sessions[session_id]
    
    def process_message(
        self,
        session_id: str,
        message: str,
        settings: Optional[SessionSettings] = None
    ) -> Dict[str, Any]:
        """
        Process user message with RAG, web search, LLM, and optional distortion.
        
        Args:
            session_id: Session ID ('new' for new session)
            message: User message
            settings: Session settings (uses session defaults if None)
            
        Returns:
            Response dict with assistant_response, context_used, session_id, etc.
        """
        # Get or create session
        session = self.get_session(session_id, settings)
        
        # Update settings if provided
        if settings:
            session.settings = settings
        
        # Add user message
        session.add_message('user', message)
        self._log(f"Processing message in session {session.session_id}")
        
        # 1. Retrieval (if enabled)
        context_items = []
        if session.settings.use_rag:
            self._log("Performing RAG search...")
            scope = SearchScope(**session.settings.search_scope)
            retrieval_results = self.retrieval.unified_search(
                query=message,
                scope=scope,
                k=session.settings.top_k_retrieval
            )
            
            self._log(f"Got {len(retrieval_results)} retrieval results")
            
            for result in retrieval_results:
                context_items.append(ContextItem(
                    source=result.source,
                    title=result.filename,
                    snippet=result.child_text[:300],
                    score=result.score,
                    doc_id=result.doc_id
                ))
            
            self._log(f"Built {len(context_items)} context items from retrieval")
        
        # 2. Web search (if enabled)
        if session.settings.use_web_search:
            self._log("Performing web search...")
            try:
                web_response = self.web_search.search(
                    query=message,
                    num_results=10
                )
                
                for result in web_response.results:
                    context_items.append(ContextItem(
                        source='web_search',
                        title=result.title,
                        snippet=result.snippet,
                        url=result.url
                    ))
            except Exception as e:
                self._log(f"Web search failed: {e}")
        
        # Add context to session
        if context_items:
            session.add_context(context_items)
        
        # 3. Build prompt with context
        context_summary = session.get_context_summary()
        
        if context_summary:
            system_prompt = (
                f"You are a helpful research assistant. "
                f"Answer based on the following context:\n\n{context_summary}\n\n"
                f"If the context doesn't contain relevant information, say so."
            )
        else:
            system_prompt = "You are a helpful research assistant."
        
        # 4. Generate response with Ollama
        self._log(f"Generating with {session.settings.model}...")
        try:
            # Use chat API for multi-turn
            messages = session.get_messages_for_llm(include_system=False)
            
            # Prepend system message with context
            messages.insert(0, {'role': 'system', 'content': system_prompt})
            
            # Non-streaming for now (streaming handled by separate endpoint)
            assistant_response = self.ollama.chat(
                messages=messages,
                model=session.settings.model,
                temperature=session.settings.temperature,
                max_tokens=session.settings.max_tokens
            )
        except Exception as e:
            self._log(f"Ollama generation failed: {e}")
            assistant_response = f"Error generating response: {e}"
        
        # 5. Apply distortion (if enabled)
        distorted = False
        ensemble_outputs = None
        
        if session.settings.use_distortion:
            self._log("Applying distortion...")
            try:
                tone = DistortionTone(session.settings.distortion_tone)
                
                # Build text for distortion (response + optional conversation context)
                distortion_text = assistant_response
                if session.settings.include_conversation_context:
                    # Include last 3 user-assistant exchanges for context
                    recent_messages = []
                    for msg in session.messages[-7:-1]:  # Skip the current assistant response
                        if msg.role in ['user', 'assistant']:
                            recent_messages.append(f"{msg.role.upper()}: {msg.content}")
                    
                    if recent_messages:
                        context = "\n".join(recent_messages)
                        distortion_text = f"CONVERSATION CONTEXT:\n{context}\n\nCURRENT RESPONSE:\n{assistant_response}"
                        self._log(f"Including {len(recent_messages)} context messages in distortion")
                
                # Ensemble mode - all 6 perspectives
                if session.settings.use_ensemble_distortion:
                    self._log("Using ensemble distortion (all 6 modes)...")
                    ensemble_result = self.twistedpair.distort_ensemble(
                        text=distortion_text,
                        tone=tone,
                        gain=session.settings.distortion_gain,
                        model=session.settings.model
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
                    
                    # Use first mode's output as the main assistant response
                    # (or could use the selected mode if not ensemble)
                    assistant_response = ensemble_result.outputs[0].output if ensemble_result.outputs else assistant_response
                    distorted = True
                    self._log(f"Ensemble complete: {len(ensemble_outputs)} perspectives")
                
                # Manual mode - single perspective
                else:
                    mode = DistortionMode(session.settings.distortion_mode)
                    distortion_result = self.twistedpair.distort(
                        text=distortion_text,
                        mode=mode,
                        tone=tone,
                        gain=session.settings.distortion_gain,
                        model=session.settings.model
                    )
                    
                    assistant_response = distortion_result.output
                    distorted = True
                    self._log("Single distortion complete")
                    
            except Exception as e:
                self._log(f"Distortion failed: {e}")
        
        # Add assistant message
        session.add_message('assistant', assistant_response, metadata={
            'distorted': distorted,
            'context_count': len(context_items),
            'ensemble': ensemble_outputs is not None
        })
        
        # Save session
        session.save()
        
        # Debug: log what we're returning
        self._log(f"Returning {len(context_items)} context items")
        if context_items:
            for i, ctx in enumerate(context_items[:3]):
                score_str = f"{ctx.score:.2f}" if ctx.score is not None else "N/A"
                self._log(f"  [{i}] {ctx.title} (score={score_str})")
        
        # Return response
        return {
            'session_id': session.session_id,
            'message_id': str(uuid.uuid4()),
            'assistant_response': assistant_response,
            'context_used': [c.to_dict() for c in context_items],
            'distorted': distorted,
            'ensemble_outputs': ensemble_outputs,  # Array of 6 outputs or None
            'timestamp': datetime.now().isoformat()
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End session and save final state.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary dict
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.sessions[session_id]
        log_path = session.save()
        
        # Remove from active cache
        del self.sessions[session_id]
        
        self._log(f"Ended session {session_id}")
        
        return {
            'status': 'success',
            'session_id': session_id,
            'messages_count': len(session.messages),
            'log_path': log_path
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.
        
        Returns:
            List of session summary dicts
        """
        sessions = []
        
        # Group by session_id (multiple saves per session)
        session_files = {}
        for filepath in self.sessions_dir.glob('*.json'):
            # Parse filename: session_id_timestamp.json
            parts = filepath.stem.split('_')
            if len(parts) >= 2:
                session_id = parts[0]
                if session_id not in session_files:
                    session_files[session_id] = []
                session_files[session_id].append(filepath)
        
        # Get most recent save for each session
        for session_id, files in session_files.items():
            latest = sorted(files)[-1]
            try:
                with open(latest, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both missing title and empty string
                    title = data.get('title', '') or 'Untitled Session'
                    sessions.append({
                        'session_id': session_id,
                        'title': title,
                        'created_at': data.get('created_at'),
                        'updated_at': data.get('updated_at'),
                        'message_count': len(data.get('messages', [])),
                        'settings': data.get('settings', {})
                    })
            except Exception as e:
                self._log(f"Error reading session {session_id}: {e}")
        
        return sorted(sessions, key=lambda x: x['updated_at'], reverse=True)


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Chat Manager')
    parser.add_argument('--message', type=str, required=True,
                       help='User message')
    parser.add_argument('--session-id', type=str, default='new',
                       help='Session ID (or "new")')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help='Ollama model')
    parser.add_argument('--use-rag', action='store_true',
                       help='Enable RAG')
    parser.add_argument('--use-web', action='store_true',
                       help='Enable web search')
    parser.add_argument('--use-distortion', action='store_true',
                       help='Enable distortion')
    parser.add_argument('--list-sessions', action='store_true',
                       help='List all sessions')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = ChatManager(verbose=args.verbose)
    
    # List sessions
    if args.list_sessions:
        print("Saved sessions:")
        sessions = manager.list_sessions()
        for session in sessions:
            print(f"  - {session['session_id']} ({session['message_count']} messages)")
            print(f"    Updated: {session['updated_at']}")
        exit(0)
    
    # Process message
    settings = SessionSettings(
        model=args.model,
        use_rag=args.use_rag,
        use_web_search=args.use_web,
        use_distortion=args.use_distortion
    )
    
    print(f"Processing: {args.message}")
    print(f"Session: {args.session_id}")
    print(f"{'='*60}\n")
    
    response = manager.process_message(
        session_id=args.session_id,
        message=args.message,
        settings=settings
    )
    
    print(f"Assistant: {response['assistant_response']}\n")
    
    if response['context_used']:
        print(f"Context ({len(response['context_used'])} items):")
        for i, ctx in enumerate(response['context_used'][:3], 1):
            print(f"  {i}. [{ctx['source']}] {ctx['title']}")
            print(f"     {ctx['snippet'][:100]}...")
    
    print(f"\nSession ID: {response['session_id']}")
    print(f"Distorted: {response['distorted']}")
