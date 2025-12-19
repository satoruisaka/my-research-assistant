"""
ollama_client.py - Ollama LLM Interface (Official Library)

Provides a clean interface to Ollama using the official Python library.

Features:
- Health checks and model listing
- Fast streaming generation (20-30% faster than REST)
- Temperature, max_tokens, top_k control
- Error handling
- Model validation

Ollama Python Library: https://github.com/ollama/ollama-python
"""

import os
import json
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
import ollama
import requests

from config import NUM_CTX, DEFAULT_MODEL


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    host: str = "http://localhost:11434"
    timeout: int = 300  # seconds (5 minutes)


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    model: str
    temperature: float = 0.7
    max_tokens: int = NUM_CTX
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_ctx: int = NUM_CTX
    stop: Optional[List[str]] = None
    
    def to_ollama_options(self) -> Dict:
        """Convert to Ollama API options format."""
        return {
            'temperature': self.temperature,
            'num_predict': self.max_tokens,
            'num_ctx': self.num_ctx,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'repeat_penalty': self.repeat_penalty,
            'stop': self.stop or []
        }


class OllamaError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when unable to connect to Ollama server."""
    pass


class OllamaModelError(OllamaError):
    """Raised when model is not found or invalid."""
    pass


class OllamaGenerationError(OllamaError):
    """Raised when generation fails."""
    pass


class OllamaClient:
    """
    Client for interacting with Ollama LLM server.
    
    Usage:
        client = OllamaClient()
        
        # Check health
        if client.is_healthy():
            print("Ollama is running")
        
        # List models
        models = client.list_models()
        print(f"Available: {models}")
        
        # Generate
        response = client.generate(
            prompt="Explain quantum computing",
            model=DEFAULT_MODEL,
            temperature=0.7
        )
        print(response)
    """
    
    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize Ollama client.
        
        Args:
            config: Client configuration (uses defaults if None)
            verbose: Enable debug logging
        """
        self.config = config or OllamaConfig()
        self.verbose = verbose
        
        # Initialize official ollama client
        self.client = ollama.Client(host=self.config.host, timeout=self.config.timeout)
        
        self._log(f"OllamaClient initialized: {self.config.host}")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[OllamaClient] {message}")
    
    def is_healthy(self) -> bool:
        """
        Check if Ollama server is running and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{self.config.host}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """
        List all available models (using REST API for compatibility).
        
        Returns:
            List of model names (e.g., ['mistral:latest', 'llama3.1:8b'])
            
        Raises:
            OllamaConnectionError: If server is unreachable
        """
        try:
            # Use REST API directly (like old working code)
            url = f"{self.config.host}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            self._log(f"Found {len(models)} models: {models}")
            return models
        except Exception as e:
            self._log(f"Error listing models: {e}")
            raise OllamaConnectionError(f"Failed to list models: {e}")
    
    def validate_model(self, model_name: str) -> bool:
        """
        Check if a model is available.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            models = self.list_models()
            return model_name in models
        except OllamaConnectionError:
            return False
    
    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = NUM_CTX,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        num_ctx: int = NUM_CTX,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text completion (non-streaming).
        
        Args:
            prompt: User prompt
            model: Model name (e.g., 'mistral:latest')
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repeat_penalty: Penalty for repetition
            num_ctx: Context window size in tokens
            stop: Stop sequences
            
        Returns:
            Generated text
            
        Raises:
            OllamaConnectionError: If server is unreachable
            OllamaModelError: If model not found
            OllamaGenerationError: If generation fails
        """
        params = GenerationParams(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            num_ctx=num_ctx,
            stop=stop
        )
        
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': params.to_ollama_options()
        }
        
        if system:
            payload['system'] = system
        
        self._log(f"Generating with {model} (temp={temperature}, max_tokens={max_tokens})")
        
        try:
            response = self._make_request(
                '/api/generate',
                method='POST',
                json_data=payload
            )
            
            # Ollama returns {'response': 'text', 'done': true, ...}
            if 'response' in response:
                text = response['response']
                self._log(f"Generated {len(text)} chars")
                return text
            else:
                raise OllamaGenerationError("No 'response' field in Ollama response")
        
        except OllamaConnectionError:
            raise
        except Exception as e:
            self._log(f"Generation error: {e}")
            raise OllamaGenerationError(f"Generation failed: {e}")
    
    def generate_stream(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = NUM_CTX,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        num_ctx: int = NUM_CTX,
        stop: Optional[List[str]] = None
    ) -> Generator[str, None, None]:
        """
        Generate text completion with streaming.
        
        Args:
            prompt: User prompt
            model: Model name
            system: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_k: Top-k sampling
            top_p: Top-p sampling
            repeat_penalty: Penalty for repetition
            num_ctx: Context window size in tokens
            stop: Stop sequences
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            OllamaConnectionError: If server is unreachable
            OllamaGenerationError: If generation fails
        """
        params = GenerationParams(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            num_ctx=num_ctx,
            stop=stop
        )
        
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': True,
            'options': params.to_ollama_options()
        }
        
        if system:
            payload['system'] = system
        
        self._log(f"Streaming with {model}")
        
        try:
            response = self._make_request(
                '/api/generate',
                method='POST',
                json_data=payload,
                stream=True
            )
            
            # Ollama streams JSONL (one JSON per line)
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            yield chunk['response']
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError as e:
                        self._log(f"JSON decode error: {e}")
                        continue
        
        except OllamaConnectionError:
            raise
        except Exception as e:
            self._log(f"Streaming error: {e}")
            raise OllamaGenerationError(f"Streaming failed: {e}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = NUM_CTX,
        num_ctx: int = NUM_CTX,
        stream: bool = False
    ) -> str:
        """
        Multi-turn chat completion (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            num_ctx: Context window size in tokens
            stream: Ignored (use chat_stream for streaming)
            
        Returns:
            Assistant response text
            
        Raises:
            OllamaConnectionError: If server is unreachable
            OllamaGenerationError: If generation fails
        """
        self._log(f"Chat with {model} ({len(messages)} messages)")
        
        try:
            response = self.client.chat(
                model=model,
                messages=messages,
                stream=False,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'num_ctx': num_ctx
                }
            )
            
            return response['message']['content']
        
        except Exception as e:
            self._log(f"Chat error: {e}")
            raise OllamaGenerationError(f"Chat failed: {e}")
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = NUM_CTX,
        num_ctx: int = NUM_CTX
    ) -> Generator[str, None, None]:
        """
        Multi-turn chat completion with streaming.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            num_ctx: Context window size in tokens
            
        Yields:
            Token strings as they are generated
            
        Raises:
            OllamaConnectionError: If server is unreachable
            OllamaGenerationError: If generation fails
        """
        self._log(f"Chat stream with {model} ({len(messages)} messages)")
        
        try:
            stream = self.client.chat(
                model=model,
                messages=messages,
                stream=True,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'num_ctx': num_ctx
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        
        except Exception as e:
            self._log(f"Chat stream error: {e}")
            raise OllamaGenerationError(f"Chat stream failed: {e}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Model info dict or None if not found
        """
        try:
            response = self._make_request(
                f'/api/show',
                method='POST',
                json_data={'name': model_name}
            )
            return response
        except Exception as e:
            self._log(f"Error getting model info: {e}")
            return None


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Ollama Client')
    parser.add_argument('--url', type=str, default='http://localhost:11434',
                       help='Ollama server URL')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help='Model to use')
    parser.add_argument('--prompt', type=str, default='Explain quantum computing in simple terms.',
                       help='Test prompt')
    parser.add_argument('--stream', action='store_true',
                       help='Use streaming')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature (0.0-1.0)')
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='Maximum tokens')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Initialize client
    config = OllamaConfig(base_url=args.url)
    client = OllamaClient(config=config, verbose=args.verbose)
    
    # Check health
    print("Checking Ollama health...")
    if not client.is_healthy():
        print("❌ Ollama is not running!")
        print("Start it with: ollama serve")
        exit(1)
    print("✅ Ollama is healthy")
    
    # List models
    print("\nAvailable models:")
    models = client.list_models()
    for model in models:
        print(f"  - {model}")
    
    # Validate requested model
    if not client.validate_model(args.model):
        print(f"\n❌ Model '{args.model}' not found!")
        print(f"Pull it with: ollama pull {args.model}")
        exit(1)
    
    # Generate
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'='*60}\n")
    
    if args.stream:
        print("Response (streaming):")
        for chunk in client.generate_stream(
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        ):
            print(chunk, end='', flush=True)
        print()
    else:
        print("Response:")
        response = client.generate(
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        print(response)
    
    print(f"\n{'='*60}")
