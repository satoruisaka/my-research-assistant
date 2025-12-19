"""
twistedpair_client.py - TwistedPair V4 REST API Client

Provides interface to TwistedPair's /distort-manual endpoint for
rhetorical distortion of text.

TwistedPair Parameters:
- 6 Modes: INVERT_ER, SO_WHAT_ER, ECHO_ER, WHAT_IF_ER, CUCUMB_ER, ARCHIV_ER
- 5 Tones: NEUTRAL, TECHNICAL, PRIMAL, POETIC, SATIRICAL
- Gain: 1-10 (intensity level)

API: http://localhost:8000/distort-manual
"""

import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime


class DistortionMode(Enum):
    """TwistedPair distortion modes."""
    INVERT_ER = "invert_er"      # Negates signals, flips polarity
    SO_WHAT_ER = "so_what_er"    # Questions implications
    ECHO_ER = "echo_er"          # Amplifies positives
    WHAT_IF_ER = "what_if_er"    # Explores alternatives
    CUCUMB_ER = "cucumb_er"      # Cool academic analysis
    ARCHIV_ER = "archiv_er"      # Historical context


class DistortionTone(Enum):
    """TwistedPair tone styles."""
    NEUTRAL = "neutral"          # Clear, standard
    TECHNICAL = "technical"      # Precise, jargon-heavy
    PRIMAL = "primal"           # Short, punchy
    POETIC = "poetic"           # Lyrical, metaphorical
    SATIRICAL = "satirical"     # Witty, ironic


@dataclass
class DistortionConfig:
    """Configuration for TwistedPair client."""
    base_url: str = "http://localhost:8001"
    timeout: int = 240  # seconds (distortion can be slow)
    retry_attempts: int = 3
    retry_delay: float = 2.0  # seconds


@dataclass
class DistortionRequest:
    """Request parameters for distortion."""
    text: str
    mode: DistortionMode
    tone: DistortionTone
    gain: int  # 1-10
    model: Optional[str] = None  # Override LLM model
    
    def to_api_dict(self) -> Dict[str, Any]:
        """Convert to API request format."""
        payload = {
            'text': self.text,
            'mode': self.mode.value,
            'tone': self.tone.value,
            'gain': self.gain
        }
        # Only include model if it's not None
        if self.model is not None:
            payload['model'] = self.model
        return payload


@dataclass
class DistortionResult:
    """Result from distortion."""
    output: str
    mode: str
    tone: str
    gain: int
    provenance: Dict[str, Any]  # Model info, temperature, etc.
    
    def __str__(self) -> str:
        return self.output


@dataclass
class EnsembleResult:
    """Result from ensemble distortion (all 6 modes)."""
    outputs: list[DistortionResult]
    signal_id: str
    provenance: Dict[str, Any]
    
    def get_mode(self, mode: str) -> Optional[DistortionResult]:
        """Get output for specific mode."""
        for output in self.outputs:
            if output.mode == mode:
                return output
        return None


class TwistedPairError(Exception):
    """Base exception for TwistedPair client errors."""
    pass


class TwistedPairConnectionError(TwistedPairError):
    """Raised when unable to connect to TwistedPair server."""
    pass


class TwistedPairDistortionError(TwistedPairError):
    """Raised when distortion fails."""
    pass


class TwistedPairClient:
    """
    Client for TwistedPair V4 distortion API.
    
    Usage:
        client = TwistedPairClient()
        
        # Check health
        if client.is_healthy():
            print("TwistedPair is running")
        
        # Distort text
        result = client.distort(
            text="AI will revolutionize education",
            mode=DistortionMode.INVERT_ER,
            tone=DistortionTone.TECHNICAL,
            gain=7
        )
        print(result.output)
    """
    
    def __init__(
        self,
        config: Optional[DistortionConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize TwistedPair client.
        
        Args:
            config: Client configuration (uses defaults if None)
            verbose: Enable debug logging
        """
        self.config = config or DistortionConfig()
        self.verbose = verbose
        
        # Ensure base_url doesn't end with /
        self.config.base_url = self.config.base_url.rstrip('/')
        
        self._log(f"TwistedPairClient initialized: {self.config.base_url}")
    
    def _log(self, message: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[TwistedPairClient] {message}")
    
    def _make_request(
        self,
        endpoint: str,
        method: str = 'GET',
        json_data: Optional[Dict] = None
    ) -> Dict:
        """
        Make HTTP request to TwistedPair API.
        
        Args:
            endpoint: API endpoint (e.g., '/distort-manual')
            method: HTTP method (GET, POST)
            json_data: JSON payload for POST
            
        Returns:
            Response JSON dict
            
        Raises:
            TwistedPairConnectionError: If connection fails
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
                        timeout=self.config.timeout
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                self._log(f"Connection error: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise TwistedPairConnectionError(
                        f"Failed to connect to TwistedPair at {self.config.base_url}. "
                        f"Is TwistedPair running? Try: uvicorn server:app --port 8000"
                    )
                time.sleep(self.config.retry_delay)
            
            except requests.exceptions.Timeout as e:
                self._log(f"Timeout error: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise TwistedPairConnectionError(
                        f"Request timeout after {self.config.timeout}s"
                    )
                time.sleep(self.config.retry_delay)
            
            except requests.exceptions.RequestException as e:
                self._log(f"Request error: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise TwistedPairConnectionError(f"Request failed: {e}")
                time.sleep(self.config.retry_delay)
    
    def is_healthy(self) -> bool:
        """
        Check if TwistedPair server is running and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to hit the root endpoint or health endpoint
            response = requests.get(
                f"{self.config.base_url}/",
                timeout=5
            )
            return response.status_code in [200, 404]  # 404 is ok (no root handler)
        except Exception:
            return False
    
    def distort(
        self,
        text: str,
        mode: DistortionMode = DistortionMode.CUCUMB_ER,
        tone: DistortionTone = DistortionTone.NEUTRAL,
        gain: int = 5,
        model: Optional[str] = None
    ) -> DistortionResult:
        """
        Distort text with specified parameters.
        
        Args:
            text: Input text to distort
            mode: Distortion mode (default: CUCUMB_ER)
            tone: Tone style (default: NEUTRAL)
            gain: Intensity level 1-10 (default: 5)
            model: Override LLM model (optional)
            
        Returns:
            DistortionResult with output and metadata
            
        Raises:
            TwistedPairConnectionError: If server is unreachable
            TwistedPairDistortionError: If distortion fails
            ValueError: If gain not in 1-10 range
        """
        # Validate gain
        if not 1 <= gain <= 10:
            raise ValueError(f"Gain must be 1-10, got {gain}")
        
        # Build request
        request = DistortionRequest(
            text=text,
            mode=mode,
            tone=tone,
            gain=gain,
            model=model
        )
        
        self._log(f"Distorting with {mode.value}/{tone.value}/gain={gain}")
        
        # Debug: log the exact payload
        payload = request.to_api_dict()
        self._log(f"Request payload: {payload}")
        
        try:
            response = self._make_request(
                '/distort-manual',
                method='POST',
                json_data=payload
            )
            
            # Parse V2 response format:
            # {"signal_id": "...", "output": {"response": "...", "mode": "...", ...}}
            output_data = response.get('output', {})
            
            result = DistortionResult(
                output=output_data.get('response', ''),
                mode=output_data.get('mode', mode.value),
                tone=output_data.get('tone', tone.value),
                gain=output_data.get('gain', gain),
                provenance=response.get('provenance', {})
            )
            
            self._log(f"Distortion complete: {len(result.output)} chars")
            return result
            
        except TwistedPairConnectionError:
            raise
        except Exception as e:
            self._log(f"Distortion error: {e}")
            raise TwistedPairDistortionError(f"Distortion failed: {e}")
    
    def distort_ensemble(
        self,
        text: str,
        tone: DistortionTone = DistortionTone.NEUTRAL,
        gain: int = 5,
        model: Optional[str] = None
    ) -> EnsembleResult:
        """
        Distort text with all 6 modes (ensemble mode).
        
        Args:
            text: Input text to distort
            tone: Tone style (default: NEUTRAL)
            gain: Intensity level 1-10 (default: 5)
            model: Override LLM model (optional)
            
        Returns:
            EnsembleResult with outputs from all 6 modes
            
        Raises:
            TwistedPairConnectionError: If server is unreachable
            TwistedPairDistortionError: If distortion fails
            ValueError: If gain not in 1-10 range
        """
        # Validate gain
        if not 1 <= gain <= 10:
            raise ValueError(f"Gain must be 1-10, got {gain}")
        
        # Build request payload matching TwistedPair V3 /distort endpoint
        payload = {
            'text': text,
            'source': 'mra-v3-client',
            'captured_at': datetime.now().isoformat(),
            'tags': [],
            'tone': tone.value,
            'gain': gain
        }
        
        if model is not None:
            payload['model'] = model
        
        self._log(f"Ensemble distorting with {tone.value}/gain={gain}")
        self._log(f"Request payload: {payload}")
        
        try:
            response = self._make_request(
                '/distort',
                method='POST',
                json_data=payload
            )
            
            # Parse V3 ensemble response format:
            # {"signal_id": "...", "outputs": [{"response": "...", "mode": "...", ...}, ...], "provenance": {...}}
            outputs_data = response.get('outputs', [])
            
            results = []
            for output_data in outputs_data:
                result = DistortionResult(
                    output=output_data.get('response', ''),
                    mode=output_data.get('mode', ''),
                    tone=output_data.get('tone', tone.value),
                    gain=output_data.get('gain', gain),
                    provenance={}
                )
                results.append(result)
            
            ensemble = EnsembleResult(
                outputs=results,
                signal_id=response.get('signal_id', ''),
                provenance=response.get('provenance', {})
            )
            
            self._log(f"Ensemble complete: {len(results)} modes")
            return ensemble
            
        except TwistedPairConnectionError:
            raise
        except Exception as e:
            self._log(f"Ensemble distortion error: {e}")
            raise TwistedPairDistortionError(f"Ensemble distortion failed: {e}")
    
    def get_available_modes(self) -> list[str]:
        """
        Get list of available distortion modes.
        
        Returns:
            List of mode names
        """
        return [mode.value for mode in DistortionMode]
    
    def get_available_tones(self) -> list[str]:
        """
        Get list of available tone styles.
        
        Returns:
            List of tone names
        """
        return [tone.value for tone in DistortionTone]
    
    def get_mode_description(self, mode: DistortionMode) -> str:
        """
        Get description of a distortion mode.
        
        Args:
            mode: Distortion mode
            
        Returns:
            Mode description
        """
        descriptions = {
            DistortionMode.INVERT_ER: "Negates signals, flips polarity, points out missing info",
            DistortionMode.SO_WHAT_ER: "Questions signals, explores implications and consequences",
            DistortionMode.ECHO_ER: "Amplifies positives, reverberates opportunities",
            DistortionMode.WHAT_IF_ER: "Hypothesizes new ideas, explores alternative scenarios",
            DistortionMode.CUCUMB_ER: "Cool-headed academic analysis",
            DistortionMode.ARCHIV_ER: "Brings historical context, prior works, literature parallels"
        }
        return descriptions.get(mode, "Unknown mode")
    
    def get_tone_description(self, tone: DistortionTone) -> str:
        """
        Get description of a tone style.
        
        Args:
            tone: Tone style
            
        Returns:
            Tone description
        """
        descriptions = {
            DistortionTone.NEUTRAL: "Clear, standard English",
            DistortionTone.TECHNICAL: "Precise, jargon-heavy, scientific/engineering register",
            DistortionTone.PRIMAL: "Short, punchy, aggressive",
            DistortionTone.POETIC: "Lyrical, metaphor-rich, mystical",
            DistortionTone.SATIRICAL: "Witty, ironic, humorous"
        }
        return descriptions.get(tone, "Unknown tone")


# Example usage and testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TwistedPair Client')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='TwistedPair server URL')
    parser.add_argument('--text', type=str, 
                       default='Artificial intelligence will transform education in the next decade.',
                       help='Text to distort')
    parser.add_argument('--mode', type=str, default='CUCUMB_ER',
                       choices=[m.value for m in DistortionMode],
                       help='Distortion mode')
    parser.add_argument('--tone', type=str, default='NEUTRAL',
                       choices=[t.value for t in DistortionTone],
                       help='Tone style')
    parser.add_argument('--gain', type=int, default=5,
                       help='Gain level (1-10)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Initialize client
    config = DistortionConfig(base_url=args.url)
    client = TwistedPairClient(config=config, verbose=args.verbose)
    
    # Check health
    print("Checking TwistedPair health...")
    if not client.is_healthy():
        print("❌ TwistedPair is not running!")
        print("Start it with: uvicorn server:app --port 8000")
        exit(1)
    print("✅ TwistedPair is healthy")
    
    # Show available options
    print("\nAvailable Modes:")
    for mode in DistortionMode:
        desc = client.get_mode_description(mode)
        print(f"  - {mode.value}: {desc}")
    
    print("\nAvailable Tones:")
    for tone in DistortionTone:
        desc = client.get_tone_description(tone)
        print(f"  - {tone.value}: {desc}")
    
    # Distort
    print(f"\n{'='*60}")
    print(f"Text: {args.text}")
    print(f"Mode: {args.mode}")
    print(f"Tone: {args.tone}")
    print(f"Gain: {args.gain}")
    print(f"{'='*60}\n")
    
    mode = DistortionMode(args.mode)
    tone = DistortionTone(args.tone)
    
    result = client.distort(
        text=args.text,
        mode=mode,
        tone=tone,
        gain=args.gain
    )
    
    print("Distorted Output:")
    print(result.output)
    
    if result.provenance:
        print(f"\nProvenance:")
        for key, value in result.provenance.items():
            print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")
