"""HTTP client for ConversaVoice backend API."""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with ConversaVoice backend API."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the backend API (defaults to BACKEND_API_URL env var)
        """
        self.base_url = base_url or os.getenv("BACKEND_API_URL", "http://localhost:8000")
        self.session_id: Optional[str] = None
        logger.info(f"Initialized API client with base URL: {self.base_url}")
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise exceptions for errors.

        Args:
            response: Response from API

        Returns:
            JSON response data

        Raises:
            Exception: If API returns an error
        """
        try:
            response.raise_for_status()
            if not response.text:
                raise Exception("Empty response from server")
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"API error: {e}")
            try:
                error_detail = response.json().get("detail", str(e)) if response.text else str(e)
            except:
                error_detail = response.text[:200] if response.text else str(e)
            raise Exception(f"API error: {error_detail}")
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {response.text[:200]}")
            raise Exception(f"Server returned invalid response: {response.text[:100] if response.text else 'empty'}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Health status information
        """
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            return self._handle_response(response)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            Session ID
        """
        try:
            response = requests.post(f"{self.base_url}/api/session", timeout=5)
            data = self._handle_response(response)
            self.session_id = data["session_id"]
            logger.info(f"Created session: {self.session_id}")
            return self.session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Ensure we have a session
            if not self.session_id:
                self.create_session()
            
            with open(audio_path, "rb") as audio_file:
                files = {"audio": audio_file}
                data = {"session_id": self.session_id}
                
                response = requests.post(
                    f"{self.base_url}/api/transcribe",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            result = self._handle_response(response)
            logger.info(f"Transcribed audio: {result['text']}")
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def chat(self, text: str) -> Dict[str, Any]:
        """
        Send chat message to LLM.
        
        Args:
            text: User input text
            
        Returns:
            Chat response with metadata
        """
        try:
            # Ensure we have a session
            if not self.session_id:
                self.create_session()
            
            payload = {
                "text": text,
                "session_id": self.session_id
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            
            result = self._handle_response(response)
            logger.info(f"Chat response: {result['response'][:50]}...")
            return result
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def synthesize_speech(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            style: Emotional style label
            pitch: Pitch adjustment
            rate: Speech rate adjustment
            
        Returns:
            URL to download audio file
        """
        try:
            payload = {
                "text": text,
                "style": style,
                "pitch": pitch,
                "rate": rate
            }
            
            response = requests.post(
                f"{self.base_url}/api/synthesize",
                json=payload,
                timeout=30
            )
            
            result = self._handle_response(response)
            audio_url = self.base_url + result["audio_url"]
            logger.info(f"Synthesized speech: {audio_url}")
            return audio_url
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise
    
    def download_audio(self, audio_url: str, output_path: str):
        """
        Download audio file from URL.
        
        Args:
            audio_url: URL to audio file
            output_path: Path to save audio file
        """
        try:
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Downloaded audio to: {output_path}")
        except Exception as e:
            logger.error(f"Audio download failed: {e}")
            raise
    
    def delete_session(self):
        """Delete the current session."""
        if not self.session_id:
            return
        
        try:
            response = requests.delete(
                f"{self.base_url}/api/session/{self.session_id}",
                timeout=5
            )
            self._handle_response(response)
            logger.info(f"Deleted session: {self.session_id}")
            self.session_id = None
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
