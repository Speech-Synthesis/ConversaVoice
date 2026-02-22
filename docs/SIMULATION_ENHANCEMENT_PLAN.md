# Conversation Simulator Enhancement Plan

**Branch**: `feature/conversation-simulator`
**Date**: 2026-02-22
**Author**: Development Team

---

## Workflow Agreement

### Command Execution Protocol
> **IMPORTANT**: All bash/CMD commands and git commands will be provided by the AI assistant.
> The developer will:
> 1. Copy the command
> 2. Paste and run it in their terminal
> 3. Send the output back
> 4. AI will verify and continue
>
> This ensures transparency and developer control over all system operations.

### Commit Strategy
- Each phase will have dedicated commits
- All commits go to branch: `feature/conversation-simulator`
- Commit message format:
  ```
  feat(component): Short description

  - Detailed change 1
  - Detailed change 2

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```

### Testing Protocol
- Test files will be created in `tests/simulation/`
- After each phase, test commands will be provided
- If tests fail, we fix before moving forward

---

## Current Issues to Fix

### Bug 1: API Endpoint Error on End Simulation
**Symptom**: Error when clicking "End (Resolved)" or "End (Unresolved)"
**Root Causes**:
- Session tracker (Redis) connection issues
- Session not saved before analysis fetch
- Missing error handling

**Fix**:
- Add Redis connection check
- Ensure session is saved synchronously before response
- Add proper try/catch with user-friendly errors

### Bug 2: Scenario Start Glitch/Reload
**Symptom**: UI reloads/flickers when starting scenarios
**Root Causes**:
- LLM blocking while generating opening message
- TTS synthesis blocking UI thread
- Streamlit rerun timing issues

**Fix**:
- Add loading states with spinners
- Use async operations properly
- Cache scenario data

---

## New Features Overview

### Feature 1: Natural Conversation Ending
Customer naturally ends conversation when satisfied (while keeping manual buttons as backup).

### Feature 2: Trainee Voice Emotion Detection
Extract emotion from trainee's voice using acoustic analysis.

### Feature 3: Voice Delivery Scoring
Score HOW the trainee speaks (calmness, confidence, empathy tone).

### Feature 4: Final Evaluation Report
Comprehensive score combining text analysis + voice delivery.

---

## Feature 1: Natural Conversation Ending

### Design Philosophy
**Realistic Behavior**: Different customer types end conversations differently:

| Customer Type | Ending Behavior |
|--------------|-----------------|
| **Angry** | Takes 5-8 good exchanges to calm down. Won't end easily. May say "Fine, whatever" even when satisfied. |
| **Frustrated** | 3-5 exchanges. Needs acknowledgment before accepting solution. |
| **Anxious** | Needs reassurance. May ask "Are you sure?" multiple times before accepting. |
| **Neutral/Calm** | 2-3 exchanges. Quick to accept good solutions. |

### Conversation Completion Triggers

```python
COMPLETION_CONDITIONS = {
    # Condition: (emotion_required, consecutive_turns, phrases)
    "satisfied_goodbye": {
        "emotions": ["satisfied", "delighted"],
        "min_turns_at_emotion": 2,
        "trigger_phrases": ["thank you", "thanks", "that works", "great", "perfect"]
    },
    "reluctant_acceptance": {
        "emotions": ["neutral", "hopeful"],  # For initially angry customers
        "min_turns_at_emotion": 3,
        "trigger_phrases": ["fine", "okay", "alright", "i guess"]
    },
    "frustrated_exit": {
        "emotions": ["frustrated", "angry"],
        "min_turns": 10,  # Long conversation
        "trigger_phrases": ["forget it", "never mind", "i'll go elsewhere"]
    }
}
```

### Angry Customer Realism
Angry customers should:
1. **Not forgive easily** - Even good responses only move them one step (angry → frustrated → anxious → neutral)
2. **Test the trainee** - Throw curveballs even when calming down
3. **Remember past grievances** - Reference earlier issues in conversation
4. **Have patience limits** - May threaten to escalate/leave if no progress after 8+ turns
5. **Accept grudgingly** - Final acceptance may be "Fine. I hope this actually works this time."

### Manual Buttons (Safety Backup)
Keep existing buttons but relabel:
- "🏳️ End (Resolved)" → "🏳️ Mark Resolved & End"
- "❌ End (Unresolved)" → "❌ Mark Unresolved & End"
- "🚪 Abandon" → "🚪 Exit Session"

These serve as override controls when natural ending doesn't happen.

### API Response Changes
```json
{
  "customer_message": "Alright, that actually helps. Thanks.",
  "emotion_state": "satisfied",
  "conversation_complete": false,
  "completion_status": {
    "approaching_end": true,
    "turns_until_natural_end": 1,
    "predicted_ending_type": "satisfied_goodbye"
  }
}
```

When conversation naturally ends:
```json
{
  "customer_message": "Okay, thanks for your help. Goodbye.",
  "emotion_state": "satisfied",
  "conversation_complete": true,
  "final_goodbye": true,
  "auto_end_session": true
}
```

---

## Feature 2: Voice Emotion Detection

### Technical Approach
Use **librosa** for acoustic feature extraction (pure Python, no heavy ML models needed).

### Features to Extract

#### 1. Pitch Analysis
```python
pitch_features = {
    "mean_pitch": 150.0,        # Hz - Average speaking pitch
    "pitch_std": 25.0,          # Pitch variation
    "pitch_range": 80.0,        # Max - Min pitch
    "pitch_contour": "falling"  # Rising/falling/flat pattern
}
```

**Interpretation**:
- High variance = stressed/excited
- Low variance = calm/monotone
- Falling contour = confident statements
- Rising contour = questions/uncertainty

#### 2. Energy Analysis
```python
energy_features = {
    "mean_energy": 0.05,        # RMS energy
    "energy_std": 0.02,         # Energy variation
    "energy_trend": "stable"    # Rising/falling/stable
}
```

**Interpretation**:
- Steady energy = composed
- Erratic energy = nervous/stressed
- Decreasing = losing confidence

#### 3. Tempo Analysis
```python
tempo_features = {
    "speaking_rate": 140,       # Estimated WPM
    "pause_ratio": 0.15,        # Time spent in pauses
    "rhythm_regularity": 0.8    # How consistent the rhythm is
}
```

**Interpretation**:
- 120-150 WPM = good pace
- >160 WPM = rushed
- <100 WPM = hesitant
- High pause ratio = uncertain/thinking

### Voice Emotion Classification

```python
VOICE_EMOTIONS = {
    "calm": {
        "pitch_std": (10, 30),      # Low variance
        "energy_std": (0.01, 0.03), # Steady energy
        "speaking_rate": (110, 150) # Moderate pace
    },
    "stressed": {
        "pitch_std": (35, 60),      # High variance
        "energy_std": (0.04, 0.08), # Erratic energy
        "speaking_rate": (155, 200) # Fast pace
    },
    "confident": {
        "pitch_std": (15, 35),      # Moderate variance
        "energy_trend": "stable",
        "pause_ratio": (0.05, 0.15) # Purposeful pauses
    },
    "hesitant": {
        "pause_ratio": (0.25, 0.5), # Many pauses
        "speaking_rate": (80, 110), # Slow pace
        "pitch_contour": "rising"   # Uncertain endings
    },
    "empathetic": {
        "pitch_std": (20, 40),      # Expressive
        "energy_trend": "soft",     # Gentle energy
        "speaking_rate": (100, 130) # Slower, caring pace
    }
}
```

---

## Feature 3: Voice Delivery Scoring

### Scoring Dimensions (1-10)

| Dimension | What We Measure | Ideal Range | Scoring |
|-----------|----------------|-------------|---------|
| **Calmness** | Pitch stability, energy consistency | Low variance | 10 = rock steady, 1 = shaky |
| **Confidence** | Steady pitch, clear endings, good pauses | Moderate energy | 10 = authoritative, 1 = uncertain |
| **Empathy** | Softer tone, slower pace, warmth | Lower pitch range | 10 = warm, caring, 1 = cold |
| **Pace** | Speaking rate, pause patterns | 120-150 WPM | 10 = perfect pace, 1 = too fast/slow |
| **Clarity** | Articulation (via spectral features) | Clear spectrum | 10 = crystal clear, 1 = mumbled |

### Ideal Customer Service Voice Profile

```python
IDEAL_VOICE_PROFILE = {
    "pitch_mean_range": (100, 180),      # Hz (varies by gender)
    "pitch_std_ideal": 20,               # Expressive but controlled
    "speaking_rate_ideal": 135,          # WPM
    "pause_ratio_ideal": 0.12,           # Natural pauses
    "energy_stability": 0.85,            # Consistent
}
```

---

## Feature 4: Final Evaluation Report

### Combined Scoring Formula

```
FINAL_SCORE = (Content Score × 0.7) + (Voice Score × 0.3)
```

**Content Score** (existing - 70% weight):
- Empathy detection
- De-escalation success
- Problem resolution
- Communication clarity
- Technique usage

**Voice Score** (new - 30% weight):
- Calmness: 20%
- Confidence: 25%
- Empathy tone: 25%
- Pace control: 15%
- Clarity: 15%

### Final Report Structure

```
╔════════════════════════════════════════════════════════╗
║            SIMULATION PERFORMANCE REPORT               ║
╠════════════════════════════════════════════════════════╣
║  Overall Grade: B+ (82/100)                            ║
║                                                        ║
║  ─── Content Analysis (56/70) ───                     ║
║  • Empathy:           8/10  ████████░░                ║
║  • De-escalation:     7/10  ███████░░░                ║
║  • Problem Solving:   8/10  ████████░░                ║
║  • Communication:     7/10  ███████░░░                ║
║  • Efficiency:        6/10  ██████░░░░                ║
║                                                        ║
║  ─── Voice Delivery (26/30) ───                       ║
║  • Calmness:          9/10  █████████░                ║
║  • Confidence:        8/10  ████████░░                ║
║  • Empathetic Tone:   7/10  ███████░░░                ║
║  • Pace Control:      9/10  █████████░                ║
║  • Clarity:           8/10  ████████░░                ║
║                                                        ║
║  ─── Detected Voice Emotion ───                       ║
║  Primary: Calm (85%)                                   ║
║  Secondary: Confident (72%)                            ║
║                                                        ║
║  ─── Strengths ───                                    ║
║  ✓ Maintained calm voice throughout                   ║
║  ✓ Good pacing, customer could follow easily          ║
║  ✓ Showed genuine empathy in tone                     ║
║                                                        ║
║  ─── Areas for Improvement ───                        ║
║  • Could speak slightly slower in complex parts       ║
║  • Add more warmth when acknowledging frustration     ║
║                                                        ║
║  ─── Recommended Training ───                         ║
║  • "Vocal Empathy Workshop"                           ║
║  • "Handling Angry Customers - Advanced"              ║
╚════════════════════════════════════════════════════════╝
```

---

## Implementation Phases

### Phase 1: Bug Fixes (Current Priority)
**Files to modify**:
- `backend/api/simulation_routes.py` - Add error handling
- `src/simulation/session_tracker.py` - Fix Redis connection
- `frontend/simulation_app.py` - Fix UI glitches

**Commit**: `fix(simulation): Fix end session and UI glitches`

**Test**: Manual test of start → conversation → end flow

---

### Phase 2: Natural Conversation Ending
**Files to create/modify**:
- `src/simulation/conversation_flow.py` - NEW: Ending detection logic
- `src/simulation/controller.py` - Add completion detection
- `frontend/simulation_app.py` - Handle natural endings

**Commit**: `feat(simulation): Add natural conversation endings`

**Test**: Test each customer type reaches natural conclusion

---

### Phase 3: Voice Analyzer Core
**Files to create**:
- `src/simulation/voice_analyzer.py` - NEW: Audio analysis
- `tests/simulation/test_voice_analyzer.py` - NEW: Unit tests

**Dependencies to add**:
```
librosa>=0.10.0
```

**Commit**: `feat(simulation): Add voice emotion analyzer`

**Test**: Run unit tests on sample audio files

---

### Phase 4: Integration
**Files to modify**:
- `backend/api/simulation_routes.py` - Add voice analysis endpoint
- `src/simulation/controller.py` - Integrate voice analysis
- `frontend/simulation_app.py` - Display voice scores

**Commit**: `feat(simulation): Integrate voice analysis into flow`

---

### Phase 5: Enhanced Scoring & Report
**Files to modify**:
- `src/simulation/analysis.py` - Combined scoring
- `frontend/simulation_app.py` - Enhanced report UI

**Commit**: `feat(simulation): Add combined voice+content scoring`

---

## File Structure After Implementation

```
src/simulation/
├── __init__.py
├── models.py
├── scenarios.py
├── persona.py
├── controller.py          # Updated
├── session_tracker.py     # Fixed
├── analysis.py            # Updated
├── conversation_flow.py   # NEW: Natural ending logic
└── voice_analyzer.py      # NEW: Audio analysis

tests/simulation/
├── test_voice_analyzer.py # NEW
├── test_conversation_flow.py # NEW
└── test_scenarios.py

docs/
├── SIMULATION_ENHANCEMENT_PLAN.md  # This document
└── TESTING_GUIDE.md                # Test instructions
```

---

## Commands Reference

### Start Development Server
```bash
cd S:\Projects\ConversaVoice
python -m uvicorn backend.main:app --reload --port 8000
```

### Start Simulation UI
```bash
cd S:\Projects\ConversaVoice
streamlit run frontend/simulation_app.py
```

### Run Tests
```bash
cd S:\Projects\ConversaVoice
python -m pytest tests/simulation/ -v
```

### Git Commands Template
```bash
# Stage changes
git add <files>

# Check status
git status

# Commit
git commit -m "type(scope): message"

# View branch
git branch

# Push (when ready)
git push -u origin feature/conversation-simulator
```

---

## Implementation Status

| Phase | Status | Commit |
|-------|--------|--------|
| Phase 1: Bug Fixes | ✅ Complete | `fix(simulation): Fix end session errors and improve reliability` |
| Phase 2: Natural Endings | ✅ Complete | `feat(simulation): Add natural conversation endings` |
| Phase 3: Voice Analyzer | ✅ Complete | `feat(simulation): Add voice emotion analyzer` |
| Phase 4: Integration | ✅ Complete | Integrated with Phase 5 |
| Phase 5: Enhanced Scoring | ✅ Complete | Voice scores in feedback report |

## Testing

To test the simulation:

```bash
# Terminal 1 - Backend
cd S:\Projects\ConversaVoice
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2 - Frontend
cd S:\Projects\ConversaVoice
streamlit run frontend/simulation_app.py
```

## Key Features Implemented

1. **Bug Fixes**: Session tracking works without Redis, graceful error handling
2. **Natural Endings**: Customers end conversations realistically based on emotion
3. **Voice Analysis**: Trainee voice analyzed for emotion and delivery quality
4. **Combined Scoring**: Final report shows content + voice delivery scores

---

## Questions Answered

| Question | Decision |
|----------|----------|
| Voice analysis method | **librosa** (pure acoustic, fast, no heavy models) |
| Voice score weight | **30%** voice + **70%** content |
| Real-time vs post-session | **Both** - light analysis per turn, full analysis at end |
| Angry customer behavior | **Realistic** - won't forgive easily, 5-8 turns minimum |
| Manual buttons | **Keep as backup** - renamed for clarity |

---

*Document will be updated as implementation progresses.*
