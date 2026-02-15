<p align="center">
  <img src="https://img.icons8.com/fluency/96/microphone.png" alt="ConversaVoice Logo" width="96" height="96">
</p>

<h1 align="center">ConversaVoice</h1>

<p align="center">
  <strong>AI-Powered Voice Assistant with Emotional Intelligence</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#api-usage">API Usage</a> •
  <a href="#tech-stack">Tech Stack</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis">
</p>

---

## Overview

**ConversaVoice** is a context-aware voice assistant that understands emotions and responds with appropriate tone and empathy. It combines cutting-edge AI technologies to create natural, emotionally intelligent conversations.

```
🎤 You speak → 🧠 AI understands → 💬 Smart response → 🔊 Natural voice
```

### Why ConversaVoice?

| Traditional Assistants | ConversaVoice |
|----------------------|---------------|
| Monotone responses | Emotional, expressive speech |
| Forgets context | Remembers conversation history |
| Generic replies | Personalized, context-aware |
| Robotic voice | Natural human-like tone |

---

## Features

### 🎙️ Voice Input & Output
- **Speech-to-Text**: Groq Whisper API for fast, accurate transcription
- **Text-to-Speech**: Azure Neural TTS with emotional expressiveness
- **Real-time**: Low-latency streaming pipeline

### 🧠 Intelligent Responses
- **LLM-Powered**: Groq API with Llama 3.3 70B for smart replies
- **Context-Aware**: Remembers conversation history
- **Emotion Detection**: Adapts tone based on user sentiment

### 💭 Emotional Intelligence
- **Sentiment Analysis**: Detects frustration, happiness, confusion
- **Adaptive Prosody**: Changes pitch, rate, and tone dynamically
- **Empathetic Responses**: De-escalation when user is frustrated

### 🔄 Conversation Memory
- **Redis-Backed**: Persistent session storage
- **Repetition Detection**: Knows when user repeats themselves
- **Preference Tracking**: Remembers user preferences

### 🎨 Expressive Speech (SSML)
- **30+ Emotion Styles**: Cheerful, empathetic, calm, excited...
- **Word Emphasis**: Stress important words naturally
- **Prosody Control**: Fine-tune pitch, rate, and volume

### 🔒 Reliability
- **Fallback System**: Auto-switch to local models if cloud fails
- **Ollama Backup**: Local LLM fallback
- **Piper TTS Backup**: Local voice synthesis

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Redis)
- API Keys:
  - [Groq API](https://console.groq.com) (Free)
  - [Azure Speech](https://azure.microsoft.com/en-us/products/ai-services/speech-services) (Free tier)

### Installation

```bash
# Clone the repository
git clone https://github.com/Speech-Synthesis/ConversaVoice.git
cd ConversaVoice

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Configure Environment

```env
# .env file
GROQ_API_KEY=your_groq_api_key
AZURE_SPEECH_KEY=your_azure_key
AZURE_SPEECH_REGION=eastus
REDIS_HOST=localhost
REDIS_PORT=6379
STT_BACKEND=groq
```

### Run Locally

**1. Start Redis:**
```bash
docker run -d -p 6379:6379 redis
```

**2. Start Backend (Terminal 1):**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**3. Start Frontend (Terminal 2):**
```bash
cd frontend
streamlit run app.py
```

**4. Open Browser:**
```
http://localhost:8501
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ConversaVoice                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Frontend │───▶│ Backend  │───▶│   LLM    │───▶│   TTS    │  │
│  │Streamlit │    │ FastAPI  │    │  Groq    │    │  Azure   │  │
│  └──────────┘    └────┬─────┘    └──────────┘    └──────────┘  │
│                       │                                         │
│       ┌───────────────┼───────────────┐                        │
│       ▼               ▼               ▼                        │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  Redis  │    │   STT    │    │   NLP    │                   │
│  │ Memory  │    │  Groq    │    │Sentiment │                   │
│  └─────────┘    └──────────┘    └──────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User speaks
    │
    ▼
┌─────────────────┐
│  Groq Whisper   │  ← Speech-to-Text
│    (STT)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sentiment +    │  ← Analyze emotion
│  Context Check  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Redis Memory   │  ← Fetch history
│  + Preferences  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Groq LLM       │  ← Generate response
│  (Llama 3.3)    │     with emotion style
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Azure TTS      │  ← Convert to speech
│  (Neural Voice) │     with prosody
└────────┬────────┘
         │
         ▼
    User hears response
```

---

## API Usage

### Python SDK

```python
from src.orchestrator import Orchestrator
import asyncio

async def main():
    # Initialize
    orch = Orchestrator(session_id="user-123")
    await orch.initialize()

    # Process voice/text
    result = await orch.process_text("I'm frustrated with my order!")

    print(f"Response: {result.assistant_response}")
    print(f"Emotion Style: {result.style}")  # "empathetic"
    print(f"Latency: {result.latency_ms}ms")

    await orch.shutdown()

asyncio.run(main())
```

### REST API

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Create Session:**
```bash
curl -X POST http://localhost:8000/api/session
```

**Chat:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "session_id": "your-session-id"}'
```

**Transcribe Audio:**
```bash
curl -X POST http://localhost:8000/api/transcribe \
  -F "audio=@recording.wav" \
  -F "session_id=your-session-id"
```

**Synthesize Speech:**
```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello!", "style": "cheerful"}'
```

---

## Emotional Prosody

ConversaVoice adapts voice characteristics based on context:

| Style | When Used | Voice Effect |
|-------|-----------|--------------|
| `neutral` | Normal conversation | Standard tone |
| `cheerful` | Good news, greetings | Higher pitch, faster |
| `empathetic` | User frustrated/sad | Softer, slower |
| `patient` | Explaining complex topics | Calm, measured |
| `de_escalate` | User very angry | Very soft, slow |

### SSML Example

```xml
<speak version="1.0" xmlns:mstts="http://www.w3.org/2001/mstts">
  <voice name="en-US-JennyNeural">
    <mstts:express-as style="empathetic" styledegree="1.3">
      I understand how frustrating this must be.
      <emphasis level="strong">We'll fix this right away.</emphasis>
    </mstts:express-as>
  </voice>
</speak>
```

---

## Project Structure

```
ConversaVoice/
├── backend/                 # FastAPI backend
│   ├── api/
│   │   ├── routes.py       # API endpoints
│   │   └── models.py       # Pydantic models
│   ├── services/
│   │   └── orchestrator_service.py
│   └── main.py             # App entry point
│
├── frontend/               # Streamlit UI
│   ├── app.py             # Main UI
│   └── api_client.py      # Backend client
│
├── src/                    # Core modules
│   ├── llm/               # LLM clients
│   │   ├── groq_client.py
│   │   └── ollama_client.py
│   ├── tts/               # Text-to-Speech
│   │   ├── azure_client.py
│   │   ├── piper_client.py
│   │   └── ssml_builder.py
│   ├── stt/               # Speech-to-Text
│   │   ├── groq_whisper_client.py
│   │   └── whisper_client.py
│   ├── memory/            # Conversation memory
│   │   ├── redis_client.py
│   │   └── vector_store.py
│   ├── nlp/               # NLP utilities
│   │   └── sentiment.py
│   ├── orchestrator.py    # Main pipeline
│   └── fallback.py        # Fallback manager
│
├── scripts/               # CLI tools
│   ├── main.py           # Interactive CLI
│   └── transcribe.py     # Transcription tool
│
└── .env                   # Configuration
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web UI |
| **Backend** | FastAPI | REST API |
| **LLM** | Groq (Llama 3.3 70B) | Response generation |
| **STT** | Groq Whisper | Speech recognition |
| **TTS** | Azure Neural TTS | Voice synthesis |
| **Memory** | Redis | Conversation storage |
| **Embeddings** | Sentence Transformers | Repetition detection |
| **Fallback LLM** | Ollama | Offline backup |
| **Fallback TTS** | Piper | Offline backup |

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM & STT | Yes |
| `AZURE_SPEECH_KEY` | Azure Speech Services key | Yes |
| `AZURE_SPEECH_REGION` | Azure region (e.g., eastus) | Yes |
| `REDIS_HOST` | Redis server host | Yes |
| `REDIS_PORT` | Redis server port | Yes |
| `STT_BACKEND` | `groq` or `local` | No (default: groq) |
| `BACKEND_API_URL` | Backend URL for frontend | No |

---

## Roadmap

- [x] Voice input with Whisper STT
- [x] Intelligent responses with Llama 3
- [x] Emotional TTS with Azure
- [x] Conversation memory with Redis
- [x] Sentiment analysis
- [x] Fallback to local models
- [x] Web UI with Streamlit
- [ ] Multi-language support
- [ ] Voice cloning
- [ ] Mobile app
- [ ] WebSocket real-time streaming

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made by the ConversaVoice Team
</p>

<p align="center">
  <a href="https://github.com/Speech-Synthesis/ConversaVoice">
    <img src="https://img.shields.io/github/stars/Speech-Synthesis/ConversaVoice?style=social" alt="GitHub stars">
  </a>
</p>
