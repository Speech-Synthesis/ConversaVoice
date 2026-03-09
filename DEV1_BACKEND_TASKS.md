# 🧑‍💻 DEV 1 — Backend & Main Repo (ConversaVoice Core)

> **Your Repo**: The main ConversaVoice repo — Python backend, FastAPI, Streamlit, voice pipeline
> **Deployed on**: Render (two services: backend API + Streamlit frontend)
> **Your job**: Fix all backend bugs, harden the server, add features to the AI engine

---

## 🔴 P0 — Critical Fixes (Do These First)

### 1. Fix the Logger Import Bug
**File**: `frontend/simulation_app.py`
**Problem**: `logger.warning()` is called at line 188 but `import logging` happens at line 193 — this crashes on startup.
**Fix**: Move the logging import and logger definition to the very top of the file.
```python
import logging
logger = logging.getLogger(__name__)
```
**Why**: The app will throw a `NameError` before users can even open it.

---

### 2. Fix the Global Singleton Orchestrator (Session Bleed)
**File**: `api.py` (line 35)
**Problem**: One single `Orchestrator` instance is shared across ALL users. User A's conversation bleeds into User B's. Redis session is hardcoded to `"mobile-app-session"` for everyone.
**Fix**: Use a per-session factory pattern:
```python
orchestrators: Dict[str, Orchestrator] = {}

def get_orchestrator(session_id: str) -> Orchestrator:
    if session_id not in orchestrators:
        orchestrators[session_id] = Orchestrator(session_id=session_id)
    return orchestrators[session_id]
```
Add cleanup logic to remove stale sessions after a TTL (e.g., 30 minutes of inactivity).
**Why**: This is the most critical architectural flaw. Without this, every user shares the same conversation history.

---

### 3. Fix the Bare Except Clause
**File**: `frontend/simulation_app.py` (line 172)
**Problem**: `except:` without a type silently swallows ALL errors including keyboard interrupts.
**Fix**:
```python
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return False
```
**Why**: Hidden errors make debugging impossible.

---

### 4. Fix the Male Name / Female Voice Mismatch
**Problem**: The AI customer persona has a male name but uses a female Azure Neural TTS voice (or vice versa). This breaks immersion during training simulations.
**Fix**:
- Audit all scenario JSON files — check the `persona_name` field
- If the name is male (e.g., "Alex", "David", "James"), assign a male Azure Neural voice like `en-US-GuyNeural` or `en-US-DavisNeural`
- If the name is female (e.g., "Sarah", "Emma"), use `en-US-JennyNeural` or `en-US-AriaNeural`
- Add a `voice_gender` field to scenario configs so voice assignment is explicit, not accidental
- Update the TTS module to read the `voice_gender` from the active scenario before synthesizing

**Why**: Trainees will immediately notice if a "Mr. Johnson" sounds like a woman. It breaks trust in the training tool.

---

### 5. Add API Authentication
**Problem**: All API endpoints are publicly accessible — anyone on the internet can hit your backend and burn through your Groq/Azure API credits.
**Fix**: Add token-based auth middleware:
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.getenv("API_SECRET_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
```
Apply this to all simulation endpoints. The Flutter app (Dev 2) will pass this key in headers.
**Why**: Without auth, you will get unexpected bills and potentially have your API keys abused.

---

### 6. Pin All Dependencies
**File**: `requirements.txt`
**Problem**: No version numbers means any `pip install` can pull a breaking update.
**Fix**: Run `pip freeze > requirements.txt` in your current working environment to lock all versions.
**Why**: Builds randomly break when upstream packages update. This takes 2 minutes to fix.

---

### 7. Fix the Hardcoded Redis Hostname
**File**: `render.yaml` (line 19)
**Problem**: Redis hostname `red-d68scf4r85hc73d1fgkg` is hardcoded instead of using an environment variable.
**Fix**: Replace with `${REDIS_URL}` and set the actual value in Render's environment dashboard.
**Why**: When Render rebuilds your Redis instance, the hostname changes and everything breaks.

---

## 🟡 P1 — Important Improvements

### 8. Add Rate Limiting
Install `slowapi` and add per-IP limits to prevent abuse:
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/simulate/chat")
@limiter.limit("20/minute")
async def chat_endpoint(...):
```
**Why**: Without rate limiting, a single bad actor can exhaust your API budget in minutes.

---

### 9. Add Proper CORS Restriction
**File**: `api.py` (line 20-26)
**Problem**: `allow_origins=["*"]` lets any website call your backend.
**Fix**: Replace with the specific Render frontend URL and the mobile app's domain:
```python
allow_origins=["https://your-frontend.onrender.com"]
```
**Why**: Security best practice. Prevents cross-site request forgery from random websites.

---

### 10. Add More Training Scenarios
**Where**: Your scenario JSON config files
**Add at minimum**:
- **Angry billing complaint** (high difficulty) — customer got double charged
- **Tech support frustration** (medium) — internet keeps dropping
- **Cancellation request** (medium) — customer wants to cancel subscription
- **Language barrier** (easy) — customer struggles to explain their issue
- **Escalation demand** (hard) — customer demands to speak to a manager
- **Compliment/positive** (easy) — happy customer with a simple request

For each scenario include: `persona_name`, `voice_gender`, `difficulty`, `emotional_style`, `opening_line`, `skills_tested`, `expected_techniques`.

**Why**: More scenarios = more training value = more compelling product demo.

---

### 11. Clean Up Audio Files
**File**: `api.py` (line 87-94)
**Problem**: `uploads/` and `generated_audio/` directories grow forever. The server disk will eventually fill up.
**Fix**: Add a cleanup task that deletes files older than 1 hour:
```python
import os, time

def cleanup_old_files(directory: str, max_age_seconds: int = 3600):
    now = time.time()
    for f in os.listdir(directory):
        filepath = os.path.join(directory, f)
        if os.path.getmtime(filepath) < now - max_age_seconds:
            os.remove(filepath)
```
Call this at the start of each new session.
**Why**: Free Render tier has limited disk. This will silently kill your deployment.

---

### 12. Fix Blocking I/O in Async Handlers
**Problem**: `Orchestrator.chat()` calls Groq's synchronous API inside an `async` FastAPI handler, blocking the event loop.
**Fix**: Wrap synchronous calls:
```python
import asyncio
result = await asyncio.get_event_loop().run_in_executor(None, groq_client.chat, ...)
```
**Why**: Under concurrent users, this freezes the entire server for all users while one LLM call is running.

---

### 13. Improve Health Check Endpoint
**Problem**: `/api/health` just returns `"healthy"` — it doesn't actually verify anything.
**Fix**: Make it check Redis connectivity, Groq API reachability, and Azure TTS:
```python
@app.get("/api/health")
async def health_check():
    checks = {
        "redis": check_redis(),
        "groq": check_groq(),
        "azure_tts": check_azure()
    }
    status = "healthy" if all(checks.values()) else "degraded"
    return {"status": status, "checks": checks}
```
**Why**: The Flutter app and Render need real health signals to know if the service is actually working.

---

### 14. Refactor Large Files
**Problem**: `simulation_app.py` (39KB) and `orchestrator.py` (34KB) are too large to maintain.
**Fix**:
- Split `simulation_app.py` into: `ui_components.py`, `simulation_flow.py`, `analysis_display.py`
- Split `orchestrator.py` into: `pipeline_orchestrator.py`, `session_manager.py`, `response_processor.py`

**Why**: Easier to debug, review, and collaborate on.

---

## 🟢 P2 — Future Improvements

### 15. Add Structured Logging
Replace all `print()` statements with proper logging:
```python
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s', level=logging.INFO)
```
Add a `request_id` to each API request and thread it through all log lines.

### 16. Add Request Validation
Before processing, validate that `session_id` is a non-empty string and `scenario_id` exists in your scenario registry. Return a clear 400 error if not.

### 17. Remove Duplicate LLM Response Functions
**Problem**: `_get_llm_response` and `_get_llm_response_stream` have duplicated logic.
**Fix**: Extract shared logic into a base function both can call.

### 18. Add SSE Streaming for Chat Responses
Replace waiting for full LLM response with Server-Sent Events so the Flutter app can show the customer's reply character by character. Use FastAPI's `StreamingResponse`.

### 19. Add Cache for Synthesized Audio
If the same phrase gets synthesized repeatedly (greetings, common responses), cache the audio URL instead of re-calling Azure TTS every time.

### 20. Add Scenario Builder Endpoint
Create a `/api/scenarios/generate` endpoint where a manager can describe a customer persona and the LLM generates a full scenario JSON. This is a major competitive differentiator.

---

## ✅ Summary Checklist

| # | Task | Priority |
|---|------|----------|
| 1 | Fix logger import order | 🔴 P0 |
| 2 | Fix global singleton orchestrator | 🔴 P0 |
| 3 | Fix bare except clause | 🔴 P0 |
| 4 | Fix male name / female voice mismatch | 🔴 P0 |
| 5 | Add API authentication | 🔴 P0 |
| 6 | Pin all dependencies | 🔴 P0 |
| 7 | Fix hardcoded Redis hostname | 🔴 P0 |
| 8 | Add rate limiting | 🟡 P1 |
| 9 | Restrict CORS origins | 🟡 P1 |
| 10 | Add 6+ new training scenarios | 🟡 P1 |
| 11 | Clean up audio files on disk | 🟡 P1 |
| 12 | Fix blocking I/O in async handlers | 🟡 P1 |
| 13 | Improve health check endpoint | 🟡 P1 |
| 14 | Refactor large files | 🟡 P1 |
| 15 | Structured logging | 🟢 P2 |
| 16 | Request validation | 🟢 P2 |
| 17 | Remove duplicate LLM functions | 🟢 P2 |
| 18 | SSE streaming for chat | 🟢 P2 |
| 19 | Audio cache | 🟢 P2 |
| 20 | Scenario builder endpoint | 🟢 P2 |

---

> **Coordinate with Dev 2**: Once you add API authentication (Task 5), share the `API_SECRET_KEY` value so Dev 2 can update the Flutter app to pass it in the `X-API-Key` header. Also notify Dev 2 when you add new scenarios so they can update the scenario selection UI.
