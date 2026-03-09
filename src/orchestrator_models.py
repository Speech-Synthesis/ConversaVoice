from dataclasses import dataclass
from enum import Enum
from typing import Optional

class PipelineState(Enum):
    """States of the voice assistant pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

@dataclass
class PipelineResult:
    """Result from a single pipeline cycle."""
    user_input: str
    assistant_response: str
    style: Optional[str] = None
    is_repetition: bool = False
    latency_ms: float = 0.0
    pitch: Optional[str] = None
    rate: Optional[str] = None
