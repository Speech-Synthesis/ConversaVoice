# ConversaVoice

Context-aware voice assistant with emotional intelligence, powered by Distil-Whisper, Llama 3, and Azure Neural TTS.

## Current Features

- **Voice input mode** - Real-time microphone transcription with Whisper STT
- **Hybrid Architecture** - Automatic fallback to local services (Ollama/Piper) when APIs fail
- **Intelligent responses** - Groq API with Llama 3 for emotionally-aware replies
- **Conversation memory** - Redis-based session storage and context persistence
- **NLP & Context** - Sentiment analysis, preference tracking, and frustration detection
- **Emotional speech** - Azure Neural TTS with SSML prosody control
- **Streaming Pipeline** - Token-by-token LLM streaming and chunked TTS for low latency
- **GPU accelerated** - Optimized for NVIDIA GPUs (CUDA)

## Quick Start

```bash
# Activate environment
.\venv\Scripts\activate

# Start Redis (required)
docker run -d --name redis -p 6379:6379 redis

# Run the full voice assistant (interactive mode)
python main.py

# Run the Web UI (Streamlit)
streamlit run app.py

# Run with voice input (microphone)
python main.py --voice

# Process a single message
python main.py --text "Hello, how are you?"

# Use a specific session ID
python main.py --session my-session-123
```

## Transcription Only

```bash
# Real-time microphone transcription
python transcribe.py --realtime

# Transcribe audio file
python transcribe.py audio.wav --timestamps
```

## LLM Usage (Groq)

```python
from src.llm import GroqClient

client = GroqClient()
response = client.get_emotional_response("Where is my order?")

print(response.reply)          # "I understand your concern..."
print(response.style)          # "empathetic" (prosody fetched from Redis)
print(response.emphasis_words) # ["order"] (words to stress in speech)
```

## Memory Usage (Redis)

```python
from src.memory import RedisClient, VectorStore

# Session management
redis = RedisClient()
redis.create_session("user-123")
redis.add_message("user-123", "user", "Hello!")
context = redis.get_context_string("user-123")

# Repetition detection
vectors = VectorStore(redis)
result = vectors.check_repetition("user-123", "Where is my order?")
if result.is_repetition:
    print(f"User is repeating (similarity: {result.score:.0%})")
```

## Prosody Profiles (Redis)

Emotional prosody parameters are stored in Redis and fetched by style label:

```python
from src.memory import RedisClient

redis = RedisClient()

# Initialize default profiles (called automatically on startup)
redis.init_prosody_profiles()

# Get prosody for a style
prosody = redis.get_prosody("empathetic")
# Returns: {"pitch": "-5%", "rate": "0.85", "volume": "medium"}

# Update a prosody profile
redis.set_prosody("empathetic", pitch="-8%", rate="0.8", volume="soft")
```

**Default Profiles:**

| Style | Pitch | Rate | Volume |
|-------|-------|------|--------|
| neutral | 0% | 1.0 | medium |
| cheerful | +5% | 1.1 | medium |
| patient | -5% | 0.9 | medium |
| empathetic | -5% | 0.85 | medium |
| de_escalate | -10% | 0.8 | soft |

## TTS Usage (Azure)

```python
from src.tts import AzureTTSClient, ProsodyProfile

# Initialize client
tts = AzureTTSClient()

# Speak with emotional profile
tts.speak("I understand your concern.", profile=ProsodyProfile.EMPATHETIC)

# Speak with style (prosody fetched from Redis by orchestrator)
tts.speak_with_llm_params(
    text="I hear you.",
    style="empathetic",
    pitch="-5%",  # These come from Redis lookup
    rate="0.85"
)

# Save to file
tts.synthesize_to_file("Hello world", "output.mp3")
```

## SSML Features

ConversaVoice uses advanced SSML (Speech Synthesis Markup Language) for expressive speech output.

### Express-As Styles

Azure Neural voices support emotional styles via `<mstts:express-as>`:

```python
from src.tts import SSMLBuilder

builder = SSMLBuilder()

# Build SSML with style and intensity
ssml = builder.build(
    text="I understand how frustrating this must be.",
    style="empathetic",
    styledegree=1.3  # 1.0 = normal, 1.5 = intense, 0.7 = subtle
)
```

**Supported Styles:** empathetic, cheerful, calm, angry, sad, excited, friendly, hopeful, and [30+ more Azure styles](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice).

**Style Intensity (styledegree):**
- `0.01 - 0.99`: Subtle expression
- `1.0`: Default intensity
- `1.01 - 2.0`: Intensified expression

### Word Emphasis

Add stress to key words for clearer communication:

```python
from src.tts import SSMLBuilder, EmphasisLevel

builder = SSMLBuilder()

# Emphasize specific words
ssml = builder.build_with_emphasis(
    text="Your order will arrive by Friday.",
    emphasis_words=["Friday"],
    emphasis_level=EmphasisLevel.STRONG
)

# Or use inline markers in text
ssml = builder.apply_emphasis_markers(
    "Your order will arrive by *Friday*."  # *strong*, _moderate_, ~reduced~
)
```

**Emphasis Levels:**
| Level | Effect | Marker |
|-------|--------|--------|
| STRONG | Loudest stress | `*word*` |
| MODERATE | Normal emphasis | `_word_` |
| REDUCED | Softer/quieter | `~word~` |

### LLM Integration

The LLM returns `emphasis_words` for automatic speech emphasis:

```python
from src.llm import GroqClient

client = GroqClient()
response = client.get_emotional_response("What laptop should I buy for AI?")

print(response.reply)          # "I recommend the NVIDIA RTX 4060..."
print(response.style)          # "cheerful"
print(response.emphasis_words) # ["NVIDIA", "RTX 4060"]
```

## Local Fallback (Offline Mode)

ConversaVoice provides a robust fallback system to ensure availability even without an internet connection or when cloud APIs are down.

### Fallback Stack

| Service | Cloud (Primary) | Local (Fallback) |
|---------|-----------------|------------------|
| LLM | Groq (Llama 3) | Ollama (Llama 3 / Mistral) |
| TTS | Azure Neural TTS | Piper (ONNX) |

### How it Works

The `FallbackManager` tracks consecutive failures for cloud services. If the threshold (default: 2) is reached, it automatically switches to the local service. It also supports "Auto-Recovery," where it periodically tries to switch back to the cloud after a certain number of successful local calls.

### Setup Local Services

1. **Ollama (LLM):**
   ```bash
   # Install Ollama and pull model
   ollama pull llama3
   ```
2. **Piper (TTS):**
   ```bash
   # Piper is included in requirements, but you need to download a voice model
   # Default: en_US-lessac-medium.onnx
   ```

## NLP & Context Enhancements

The system goes beyond simple chat by analyzing user state and persisting information in Redis.

- **Sentiment Analysis:** Detects emotions (Happy, Frustrated, Confused, Angry) to adapt assistant behavior.
- **Preference Tracking:** Automatically extracts and saves user preferences (e.g., "I use Python for AI") to provide better recommendations.
- **Context Labels:** Tracks if a user is a `first_time` visitor, `continuing` a conversation, or showing signs of `frustration`.
- **Frustration Policy:** When frustration is detected (via keywords/repetition), the LLM is instructed to stop asking questions and provide direct answers immediately.

## Full Pipeline (Orchestrator)

```python
import asyncio
from src.orchestrator import Orchestrator

async def main():
    # Create orchestrator
    orch = Orchestrator()
    await orch.initialize()

    # Process text and get spoken response
    result = await orch.process_text("Where is my order?")

    print(f"Response: {result.assistant_response}")
    print(f"Style: {result.style}")
    print(f"Repetition: {result.is_repetition}")
    print(f"Latency: {result.latency_ms:.0f}ms")

    await orch.shutdown()

asyncio.run(main())
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- Redis server (local or Docker)
- Groq API key (free at console.groq.com)
- Azure Speech API key (free tier available)

## Installation

```bash
# Clone and setup
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Start Redis (Docker)
docker run -d --name redis -p 6379:6379 redis

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Architecture (Conversia/Zudu.ai)

Building a context-aware voice assistant with emotional intelligence:

### Phase 1: Brain (LLM Integration) - COMPLETE
- Groq API with Llama 3 for intelligent responses
- Context-aware conversation handling
- Emotion detection with prosody parameters (style, pitch, rate)

### Phase 2: Memory (Redis) - COMPLETE
- Session-based conversation history
- Vector similarity for repetition detection (threshold: 0.85)
- Context retrieval for coherent responses

### Phase 3: Voice Output (TTS) - COMPLETE
- Azure Neural TTS integration
- SSML-based emotional prosody control
- Prosody profiles: Empathetic, Patient, Cheerful, De-escalate
- Direct integration with LLM response parameters

### Phase 4: Orchestrator - COMPLETE
- Async pipeline: Input -> LLM -> TTS -> Speaker
- Context-aware responses with memory
- Real-time emotion adaptation
- Interactive text mode for testing

### Phase 5: STT Integration - COMPLETE
- Whisper STT integrated into orchestrator pipeline
- Voice input mode with `--voice` flag
- Full loop: Microphone -> Whisper -> LLM -> TTS -> Speaker

### Phase 6: Context Enhancements - COMPLETE
- Context labels: first_time, continuing, repetition, frustration
- Sentiment analysis for emotion detection (happy, frustrated, confused, angry)
- Session metadata tracking (turn_count, error_count, duration)

### Phase 7: SSML Enhancements - COMPLETE
- Express-as styles with styledegree intensity control (0.01-2.0)
- Word emphasis support (strong, moderate, reduced)
- LLM returns emphasis_words for automatic speech emphasis
- Inline emphasis markers (*strong*, _moderate_, ~reduced~)

### Phase 8: Streaming Pipeline - COMPLETE
- Token-by-token LLM streaming for real-time response display
- Chunked TTS synthesis for faster time-to-first-audio
- Sentence-by-sentence audio generation
- Callbacks for streaming progress (on_token, on_audio_chunk)

## Streaming (Low Latency)

ConversaVoice supports streaming for reduced perceived latency:

### LLM Streaming

```python
from src.llm import GroqClient

client = GroqClient()

# Stream tokens as they arrive
def on_token(token):
    print(token, end="", flush=True)

response = client.get_emotional_response_stream(
    "Tell me about Python",
    on_token=on_token
)
print(f"\n[Style: {response.style}]")
```

### Orchestrator Streaming

```python
import asyncio
from src.orchestrator import Orchestrator

async def main():
    orch = Orchestrator(on_token=lambda t: print(t, end="", flush=True))
    await orch.initialize()

    # Process with streaming (tokens shown as they arrive)
    result = await orch.process_text_stream("What's the weather like?")
    print(f"\n[Latency: {result.latency_ms:.0f}ms]")

asyncio.run(main())
```

### Chunked TTS

```python
from src.tts import AzureTTSClient

tts = AzureTTSClient()

# Speak sentence-by-sentence (faster first audio)
tts.speak_chunked(
    "Hello! This is a long response. It will play sentence by sentence.",
    style="cheerful",
    on_sentence_start=lambda s, i: print(f"Speaking: {s}")
)
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Speech-to-Text | Distil-Whisper Large-v3-Turbo |
| LLM | Groq API (Llama 3) |
| Memory | Redis |
| Text-to-Speech | Azure Neural TTS |
| GPU | CUDA (RTX 4060) |

## License

MIT
