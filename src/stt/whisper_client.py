"""Whisper client for speech-to-text transcription."""

import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")
warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")
logging.getLogger("transformers").setLevel(logging.ERROR)


class STTError(Exception):
    """Exception raised for STT errors."""
    pass


class WhisperClient:
    """
    Whisper-based speech-to-text client for the orchestrator.

    Wraps Distil-Whisper for real-time microphone transcription
    with voice activity detection (VAD).
    """

    def __init__(self, model_id: str = "distil-whisper/distil-large-v3"):
        """
        Initialize the Whisper client.

        Args:
            model_id: HuggingFace model ID for Whisper
        """
        self.model_id = model_id
        self.pipe = None
        self.device = None
        self.dtype = None
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds per chunk
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5
        self._is_loaded = False

    def load_model(self) -> None:
        """Load the Whisper model onto GPU."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading Whisper model on {self.device}...")

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(self.model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.dtype,
                device=self.device,
            )

            self._is_loaded = True
            print("Whisper model loaded successfully!")

        except Exception as e:
            raise STTError(f"Failed to load Whisper model: {e}")

    def transcribe_audio(self, audio_array, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy audio array.

        Args:
            audio_array: Numpy array of audio samples
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        pass  # To be implemented in next commit

    def start_listening(self, callback=None) -> None:
        """
        Start listening from microphone with VAD.

        Args:
            callback: Function to call with transcribed text
        """
        pass  # To be implemented in next commit

    def stop_listening(self) -> None:
        """Stop listening from microphone."""
        pass  # To be implemented in next commit
