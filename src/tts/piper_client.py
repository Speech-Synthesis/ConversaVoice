"""
Piper TTS client for ConversaVoice.

Provides local text-to-speech synthesis as a fallback when Azure TTS is unavailable.
Piper is a fast, local neural TTS engine.
"""

import os
import subprocess
import tempfile
import wave
import logging
from typing import Optional, Callable, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class PiperTTSError(Exception):
    """Exception raised when Piper TTS synthesis fails."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.details = details
        super().__init__(message)


class PiperTTSClient:
    """
    Local TTS client using Piper.

    Piper is a fast, local neural text-to-speech engine that works offline.
    Used as a fallback when Azure TTS is unavailable.
    """

    def __init__(
        self,
        piper_path: Optional[str] = None,
        model_path: Optional[str] = None,
        voice: str = "en_US-lessac-medium"
    ):
        """
        Initialize Piper TTS client.

        Args:
            piper_path: Path to piper executable (defaults to PIPER_PATH env var or 'piper')
            model_path: Path to voice model (defaults to PIPER_MODEL_PATH env var)
            voice: Voice model name (used if model_path not specified)
        """
        self.piper_path = piper_path or os.getenv("PIPER_PATH", "piper")
        self.model_path = model_path or os.getenv("PIPER_MODEL_PATH")
        self.voice = voice
        self._available = None

    def is_available(self) -> bool:
        """
        Check if Piper is available on the system.

        Returns:
            True if Piper can be executed, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            result = subprocess.run(
                [self.piper_path, "--help"],
                capture_output=True,
                timeout=5
            )
            self._available = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            self._available = False

        if not self._available:
            logger.warning("Piper TTS not available on this system")

        return self._available

    def _get_model_args(self) -> list[str]:
        """Get model-related command line arguments."""
        if self.model_path:
            return ["--model", self.model_path]
        return ["--model", self.voice]

    def synthesize_to_file(
        self,
        text: str,
        filepath: str,
        rate: Optional[float] = None
    ) -> str:
        """
        Synthesize text and save to WAV file.

        Args:
            text: Text to synthesize
            filepath: Output file path
            rate: Speech rate multiplier (optional)

        Returns:
            Path to the output file

        Raises:
            PiperTTSError: If synthesis fails
        """
        if not self.is_available():
            raise PiperTTSError("Piper TTS is not available")

        try:
            cmd = [self.piper_path]
            cmd.extend(self._get_model_args())
            cmd.extend(["--output_file", filepath])

            if rate:
                cmd.extend(["--length_scale", str(1.0 / rate)])

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            stdout, stderr = process.communicate(input=text.encode("utf-8"), timeout=30)

            if process.returncode != 0:
                raise PiperTTSError(
                    f"Piper synthesis failed with code {process.returncode}",
                    details=stderr.decode("utf-8", errors="replace")
                )

            return filepath

        except subprocess.TimeoutExpired:
            process.kill()
            raise PiperTTSError("Piper synthesis timed out")
        except Exception as e:
            raise PiperTTSError(f"Piper synthesis error: {e}")

    def synthesize_to_bytes(
        self,
        text: str,
        rate: Optional[float] = None
    ) -> bytes:
        """
        Synthesize text and return audio bytes.

        Args:
            text: Text to synthesize
            rate: Speech rate multiplier (optional)

        Returns:
            WAV audio data as bytes

        Raises:
            PiperTTSError: If synthesis fails
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            self.synthesize_to_file(text, temp_path, rate=rate)
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def speak(
        self,
        text: str,
        style: Optional[str] = None,
        rate: Optional[float] = None
    ) -> None:
        """
        Synthesize text and play through speakers.

        Note: Style parameter is accepted for API compatibility but
        Piper doesn't support emotional styles like Azure.

        Args:
            text: Text to synthesize
            style: Ignored (for API compatibility with Azure)
            rate: Speech rate multiplier
        """
        if not self.is_available():
            raise PiperTTSError("Piper TTS is not available")

        try:
            # Try to use sounddevice for playback
            import sounddevice as sd
            import numpy as np

            audio_bytes = self.synthesize_to_bytes(text, rate=rate)

            # Parse WAV data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                with wave.open(temp_path, "rb") as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    audio_data = wf.readframes(wf.getnframes())

                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels)

                # Play audio
                sd.play(audio_array, sample_rate)
                sd.wait()
            finally:
                os.unlink(temp_path)

        except ImportError:
            logger.warning("sounddevice not available, falling back to system player")
            self._play_with_system(text, rate)

    def _play_with_system(self, text: str, rate: Optional[float] = None) -> None:
        """Play audio using system audio player."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            self.synthesize_to_file(text, temp_path, rate=rate)

            # Try different system players
            if os.name == "nt":  # Windows
                os.startfile(temp_path)
            elif os.name == "posix":
                for player in ["aplay", "paplay", "afplay"]:
                    try:
                        subprocess.run([player, temp_path], check=True)
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
        finally:
            # Note: File might still be playing, so we don't delete immediately on Windows
            if os.name != "nt" and os.path.exists(temp_path):
                os.unlink(temp_path)

    def speak_with_llm_params(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> None:
        """
        Synthesize using parameters from LLM response.

        Note: Piper has limited prosody control compared to Azure.
        Style and pitch are ignored; rate is converted to length_scale.

        Args:
            text: Reply text from LLM
            style: Ignored (Piper doesn't support styles)
            pitch: Ignored (Piper doesn't support pitch control)
            rate: Rate from LLM (e.g., "0.85" or "1.1")
        """
        rate_float = None
        if rate:
            try:
                rate_float = float(rate)
            except ValueError:
                pass

        self.speak(text, rate=rate_float)

    def speak_chunked(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        on_sentence_start: Optional[Callable[[str, int], None]] = None,
        on_sentence_complete: Optional[Callable[[int], None]] = None
    ) -> None:
        """
        Synthesize text sentence-by-sentence.

        Args:
            text: Full text to synthesize
            style: Ignored (for API compatibility)
            pitch: Ignored (for API compatibility)
            rate: Speech rate
            on_sentence_start: Callback when sentence synthesis starts
            on_sentence_complete: Callback when sentence completes
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        rate_float = None
        if rate:
            try:
                rate_float = float(rate)
            except ValueError:
                pass

        for i, sentence in enumerate(sentences):
            if on_sentence_start:
                on_sentence_start(sentence, i)

            self.speak(sentence, rate=rate_float)

            if on_sentence_complete:
                on_sentence_complete(i)

    def synthesize_chunks_generator(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> Generator[bytes, None, None]:
        """
        Generate audio chunks for each sentence.

        Args:
            text: Full text to synthesize
            style: Ignored (for API compatibility)
            pitch: Ignored (for API compatibility)
            rate: Speech rate

        Yields:
            Audio bytes for each sentence
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        rate_float = None
        if rate:
            try:
                rate_float = float(rate)
            except ValueError:
                pass

        for sentence in sentences:
            audio_bytes = self.synthesize_to_bytes(sentence, rate=rate_float)
            yield audio_bytes
