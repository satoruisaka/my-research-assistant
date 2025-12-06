"""
ollama_client.py - Ollama LLM Interface

Provides a clean interface to Ollama's REST API for local LLM inference.

Features:
- Health checks and model listing
- Streaming and non-streaming generation
- Temperature, max_tokens, top_k control
- Error handling and retries
- Model validation

Ollama API: http://localhost:11434/api/*
"""

import os
import json
import time
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    timeout: int = 120  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    
    def to_ollama_options(self) -> Dict:
        """Convert to Ollama API options format."""
        return {
            'temperature': self.temperature,
            'num_predict': self.max_tokens,
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
            model="mistral:latest",
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
        
        # Ensure base_url doesn't end with /
        self.config.base_url = self.config.base_url.rstrip('/')
        
        self._log(f"OllamaClient initialized: {self.config.base_url}")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[OllamaClient] {message}")
    
    def _make_request(
        self,
        endpoint: str,
        method: str = 'GET',
        json_data: Optional[Dict] = None,
        stream: bool = False
    ) -> Any:
        """
        Make HTTP request to Ollama API.
        
        Args:
            endpoint: API endpoint (e.g., '/api/tags')
            method: HTTP method (GET, POST)
            json_data: JSON payload for POST
            stream: Enable streaming response
            
        Returns:
            Response object or parsed JSON
            
        Raises:
            OllamaConnectionError: If connection fails
        """
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.retry_attempts):
            try:
                self._log(f"{method} {url} (attempt {attempt + 1})")
                
                if method == 'GET':
                    response = requests.get(
                        url,
                        timeout=self.config.timeout
                    )
                elif method == 'POST':
                    response = requests.post(
                        url,
                        json=json_data,
                        timeout=self.config.timeout,
                        stream=stream
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                
                if stream:
                    return response  # Return response object for streaming
                else:
                    return response.json()
                
            except ConnectionError as e:
                self._log(f"Connection error: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise OllamaConnectionError(
                        f"Failed to connect to Ollama at {self.config.base_url}. "
                        f"Is Ollama running? Try: ollama serve"
                    )
                time.sleep(self.config.retry_delay)
            
            except Timeout as e:
                self._log(f"Timeout error: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise OllamaConnectionError(f"Request timeout after {self.config.timeout}s")
                time.sleep(self.config.retry_delay)
            
            except RequestException as e:
                self._log(f"Request error: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise OllamaConnectionError(f"Request failed: {e}")
                time.sleep(self.config.retry_delay)
    
    def is_healthy(self) -> bool:
        """
        Check if Ollama server is running and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Ollama doesn't have a dedicated health endpoint,
            # so we use /api/tags which is lightweight
            self._make_request('/api/tags', method='GET')
            return True
        except OllamaConnectionError:
            return False
    
    def list_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of model names (e.g., ['mistral:latest', 'llama3.1:8b'])
            
        Raises:
            OllamaConnectionError: If server is unreachable
        """
        try:
            response = self._make_request('/api/tags', method='GET')
            models = [model['name'] for model in response.get('models', [])]
            self._log(f"Found {len(models)} models: {models}")
            return models
        except OllamaConnectionError:
            raise
        except Exception as e:
            self._log(f"Error listing models: {e}")
            return []
    
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
        model: str = "mistral:latest",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
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
        model: str = "mistral:latest",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
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
        model: str = "mistral:latest",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Multi-turn chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [
                         {'role': 'system', 'content': 'You are helpful'},
                         {'role': 'user', 'content': 'Hello'},
                         {'role': 'assistant', 'content': 'Hi there!'},
                         {'role': 'user', 'content': 'How are you?'}
                     ]
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Enable streaming (not implemented yet)
            
        Returns:
            Assistant response text
            
        Raises:
            OllamaConnectionError: If server is unreachable
            OllamaGenerationError: If generation fails
        """
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': temperature,
                'num_predict': max_tokens
            }
        }
        
        self._log(f"Chat with {model} ({len(messages)} messages)")
        
        try:
            response = self._make_request(
                '/api/chat',
                method='POST',
                json_data=payload
            )
            
            # Ollama returns {'message': {'role': 'assistant', 'content': '...'}}
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                raise OllamaGenerationError("Invalid chat response format")
        
        except OllamaConnectionError:
            raise
        except Exception as e:
            self._log(f"Chat error: {e}")
            raise OllamaGenerationError(f"Chat failed: {e}")
    
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
    parser.add_argument('--model', type=str, default='mistral:latest',
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
