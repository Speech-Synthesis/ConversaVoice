"""
Streamlit frontend for Conversation Simulation training system.

Provides:
- Scenario selection
- Live simulation interface with VOICE support
- Post-session feedback display
"""

import sys
import os
import uuid
import tempfile
import streamlit as st
import requests
import time
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from frontend.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="ConversaVoice - Training Simulator",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize API Client for voice features
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()


# ============================================================================
# Audio Playback Component
# ============================================================================

def autoplay_audio(audio_url: str):
    """Auto-play audio with visual wave animation."""
    unique_id = f"audio_{uuid.uuid4().hex[:8]}"

    html_content = f"""
        <style>
            .sim-voice-waves {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
                height: 40px;
                margin-top: 10px;
            }}
            .sim-wave-bar {{
                width: 4px;
                height: 15px;
                background: linear-gradient(180deg, #ef4444 0%, #f97316 100%);
                border-radius: 4px;
                animation: sim-wave-animation 1s ease-in-out infinite;
            }}
            .sim-wave-bar:nth-child(2) {{ animation-delay: 0.1s; height: 25px; }}
            .sim-wave-bar:nth-child(3) {{ animation-delay: 0.2s; height: 20px; }}
            .sim-wave-bar:nth-child(4) {{ animation-delay: 0.3s; height: 30px; }}
            .sim-wave-bar:nth-child(5) {{ animation-delay: 0.4s; height: 20px; }}
            .sim-wave-bar:nth-child(6) {{ animation-delay: 0.5s; height: 25px; }}
            .sim-wave-bar:nth-child(7) {{ animation-delay: 0.6s; height: 15px; }}
            .sim-wave-bar:nth-child(8) {{ animation-delay: 0.7s; height: 20px; }}

            @keyframes sim-wave-animation {{
                0%, 100% {{ transform: scaleY(1); }}
                50% {{ transform: scaleY(1.5); }}
            }}
        </style>
        <div id="{unique_id}_container">
            <audio id="{unique_id}_player" autoplay style="position:absolute; width:0; height:0; opacity:0; pointer-events:none;">
                <source src="{audio_url}" type="audio/wav">
            </audio>
            <div class="sim-voice-waves">
                <div class="sim-wave-bar"></div><div class="sim-wave-bar"></div>
                <div class="sim-wave-bar"></div><div class="sim-wave-bar"></div>
                <div class="sim-wave-bar"></div><div class="sim-wave-bar"></div>
                <div class="sim-wave-bar"></div><div class="sim-wave-bar"></div>
            </div>
            <script>
                (function() {{
                    const container = document.getElementById('{unique_id}_container');
                    const player = document.getElementById('{unique_id}_player');
                    const hide = () => {{
                        if(container) {{
                            container.style.display = 'none';
                        }}
                    }};

                    if (player) {{
                        player.onended = hide;
                        player.onerror = hide;
                        setTimeout(hide, 15000);
                    }} else {{
                        hide();
                    }}
                }})();
            </script>
        </div>
    """
    import streamlit.components.v1 as components
    components.html(html_content, height=60)


def synthesize_customer_voice(text: str, prosody: dict) -> Optional[str]:
    """Synthesize customer speech with emotional prosody."""
    try:
        # Map prosody dict to TTS parameters
        style = prosody.get("style", "angry")  # Customer emotion style
        pitch = prosody.get("pitch", "0%")
        rate = prosody.get("rate", "1.0")

        # Convert rate from float to string percentage if needed
        if isinstance(rate, (int, float)):
            rate = f"{rate}"

        audio_url = st.session_state.api_client.synthesize_speech(
            text=text,
            style=style,
            pitch=pitch,
            rate=rate
        )
        return audio_url
    except Exception as e:
        st.warning(f"Voice synthesis failed: {e}")
        return None


# ============================================================================
# API Client Functions
# ============================================================================

def api_get(endpoint: str) -> Optional[Dict]:
    """Make GET request to API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint: str, data: Dict) -> Optional[Dict]:
    """Make POST request to API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def analyze_voice(audio_data: bytes) -> Optional[Dict]:
    """Analyze voice audio for emotion and delivery scores."""
    try:
        files = {"audio": ("recording.wav", audio_data, "audio/wav")}
        response = requests.post(
            f"{API_BASE_URL}/api/simulation/analyze-voice",
            files=files,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Voice analysis failed: {e}")
        return None


# Import logging for voice analysis
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Styling
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

#MainMenu, footer, header {visibility: hidden;}

.scenario-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.scenario-card:hover {
    border-color: rgba(124, 58, 237, 0.6);
    transform: translateY(-2px);
}

.difficulty-easy { color: #10b981; }
.difficulty-medium { color: #f59e0b; }
.difficulty-hard { color: #ef4444; }
.difficulty-expert { color: #8b5cf6; }

.emotion-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
}

.emotion-angry { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
.emotion-frustrated { background: rgba(249, 115, 22, 0.2); color: #fdba74; }
.emotion-neutral { background: rgba(156, 163, 175, 0.2); color: #d1d5db; }
.emotion-satisfied { background: rgba(34, 197, 94, 0.2); color: #86efac; }
.emotion-delighted { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; }

.chat-customer {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(249, 115, 22, 0.15) 100%);
    border-left: 3px solid #ef4444;
    padding: 1rem;
    border-radius: 0 12px 12px 0;
    margin: 0.5rem 0;
}

.chat-trainee {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
    border-left: 3px solid #3b82f6;
    padding: 1rem;
    border-radius: 0 12px 12px 0;
    margin: 0.5rem 0;
}

.score-bar {
    height: 8px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
}

.score-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

.score-high { background: linear-gradient(90deg, #10b981, #34d399); }
.score-medium { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.score-low { background: linear-gradient(90deg, #ef4444, #f87171); }

.header-container {
    text-align: center;
    padding: 2rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 2rem;
}

.accent {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def ensure_session_state() -> None:
    """Initialize required Streamlit session_state keys."""
    if "sim_state" not in st.session_state:
        st.session_state.sim_state = "select"  # select, active, feedback
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "scenario" not in st.session_state:
        st.session_state.scenario = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_emotion" not in st.session_state:
        st.session_state.current_emotion = None
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = "Customer"
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = True
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None
    if "natural_ending" not in st.session_state:
        st.session_state.natural_ending = False
    if "ending_type" not in st.session_state:
        st.session_state.ending_type = None
    if "last_voice_analysis" not in st.session_state:
        st.session_state.last_voice_analysis = None
    if "voice_analyses" not in st.session_state:
        st.session_state.voice_analyses = []  # Store all voice analyses for final report


# ============================================================================
# Helper Functions
# ============================================================================

def get_emotion_class(emotion: str) -> str:
    """Get CSS class for emotion badge."""
    return f"emotion-{emotion.lower()}"


def get_difficulty_class(difficulty: str) -> str:
    """Get CSS class for difficulty."""
    return f"difficulty-{difficulty.lower()}"


def render_emotion_badge(emotion: str):
    """Render emotion badge."""
    st.markdown(
        f'<span class="emotion-badge {get_emotion_class(emotion)}">{emotion.upper()}</span>',
        unsafe_allow_html=True
    )


def render_score_bar(score: int, label: str):
    """Render a visual score bar."""
    score_class = "score-high" if score >= 7 else "score-medium" if score >= 5 else "score-low"
    st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span>{label}</span>
                <span>{score}/10</span>
            </div>
            <div class="score-bar">
                <div class="score-fill {score_class}" style="width: {score * 10}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Page: Scenario Selection
# ============================================================================

def render_scenario_selection():
    """Render scenario selection page."""
    st.markdown("""
        <div class="header-container">
            <h1>🎭 Training <span class="accent">Simulator</span></h1>
            <p style="opacity: 0.7;">Select a scenario to begin your training session</p>
        </div>
    """, unsafe_allow_html=True)

    # Check API
    if not check_api_health():
        st.error("⚠️ Backend API is not available. Please start the server.")
        st.code("cd backend && uvicorn main:app --reload --port 8000", language="bash")
        return

    # Fetch scenarios
    scenarios = api_get("/api/simulation/scenarios")

    if not scenarios:
        st.warning("No scenarios available. Please add scenarios to the scenarios/ directory.")
        return

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        difficulty_filter = st.selectbox(
            "Filter by Difficulty",
            ["All", "Easy", "Medium", "Hard", "Expert"]
        )
    with col2:
        categories = api_get("/api/simulation/categories")
        category_list = ["All"] + (categories.get("categories", []) if categories else [])
        category_filter = st.selectbox("Filter by Category", category_list)

    # Filter scenarios
    filtered = scenarios
    if difficulty_filter != "All":
        filtered = [s for s in filtered if s["difficulty"].lower() == difficulty_filter.lower()]
    if category_filter != "All":
        filtered = [s for s in filtered if s["category"] == category_filter]

    if not filtered:
        st.info("No scenarios match your filters.")
        return

    # Render scenario cards
    st.markdown("### Available Scenarios")

    for scenario in filtered:
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"#### {scenario['title']}")
                st.markdown(f"*{scenario['description']}*")

                # Tags
                tags_html = " ".join([
                    f'<span class="emotion-badge">{tag}</span>'
                    for tag in scenario.get("tags", [])[:4]
                ])
                st.markdown(tags_html, unsafe_allow_html=True)

            with col2:
                st.markdown(
                    f"<span class='{get_difficulty_class(scenario['difficulty'])}'>"
                    f"⬤ {scenario['difficulty'].upper()}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"⏱️ ~{scenario['estimated_duration']} min")
                st.markdown(
                    f"<span class='{get_emotion_class(scenario['persona_emotion'])}'>"
                    f"😤 {scenario['persona_emotion']}</span>",
                    unsafe_allow_html=True
                )

                if st.button("Start", key=f"start_{scenario['scenario_id']}"):
                    start_simulation(scenario["scenario_id"])

            st.markdown("---")


def start_simulation(scenario_id: str):
    """Start a new simulation."""
    with st.spinner("Starting simulation..."):
        result = api_post("/api/simulation/start", {
            "scenario_id": scenario_id,
            "trainee_id": "streamlit_user"
        })

    if result:
        st.session_state.session_id = result["session_id"]
        st.session_state.scenario = result
        st.session_state.current_emotion = result["initial_emotion"]
        st.session_state.customer_name = result["customer_name"]

        # Synthesize voice for opening message if voice is enabled
        audio_url = None
        if st.session_state.voice_enabled:
            prosody = result.get("prosody", {})
            audio_url = synthesize_customer_voice(result["opening_message"], prosody)

        st.session_state.messages = [{
            "role": "customer",
            "content": result["opening_message"],
            "emotion": result["initial_emotion"],
            "prosody": result.get("prosody", {}),
            "audio_url": audio_url,
            "audio_played": False
        }]
        st.session_state.sim_state = "active"
        st.rerun()


# ============================================================================
# Page: Active Simulation
# ============================================================================

def render_active_simulation():
    """Render active simulation interface."""
    scenario = st.session_state.scenario

    # Header with voice toggle
    col_header, col_voice_toggle = st.columns([3, 1])

    with col_header:
        st.markdown(f"""
            <div>
                <h2>🎭 {scenario['scenario_title']}</h2>
                <p style="opacity: 0.7;">Customer: {st.session_state.customer_name}</p>
            </div>
        """, unsafe_allow_html=True)

    with col_voice_toggle:
        st.session_state.voice_enabled = st.toggle(
            "🔊 Voice",
            value=st.session_state.voice_enabled,
            help="Enable voice input/output"
        )
        st.markdown(
            f'<span class="emotion-badge {get_emotion_class(st.session_state.current_emotion)}">'
            f'{st.session_state.current_emotion.upper()}</span>',
            unsafe_allow_html=True
        )

    # Show completion banner if conversation ended naturally
    if st.session_state.get("natural_ending"):
        ending_type = st.session_state.get("ending_type", "natural_conclusion")
        if ending_type == "satisfied_goodbye":
            banner_color = "rgba(34, 197, 94, 0.2)"
            banner_border = "#22c55e"
            banner_icon = "✅"
            banner_text = "Customer is satisfied! Click 'Mark Resolved & End' to complete."
        elif ending_type == "reluctant_acceptance":
            banner_color = "rgba(234, 179, 8, 0.2)"
            banner_border = "#eab308"
            banner_icon = "😐"
            banner_text = "Customer reluctantly accepted. You may want to ensure they're satisfied."
        elif ending_type == "frustrated_exit":
            banner_color = "rgba(239, 68, 68, 0.2)"
            banner_border = "#ef4444"
            banner_icon = "😤"
            banner_text = "Customer left frustrated. Click 'Mark Unresolved & End' to complete."
        else:
            banner_color = "rgba(99, 102, 241, 0.2)"
            banner_border = "#6366f1"
            banner_icon = "👋"
            banner_text = "Conversation concluded. End the session to see your score."

        st.markdown(f"""
            <div style="background: {banner_color}; border: 2px solid {banner_border};
                        border-radius: 12px; padding: 1rem; margin-bottom: 1rem; text-align: center;">
                <span style="font-size: 1.5rem;">{banner_icon}</span>
                <span style="margin-left: 0.5rem;">{banner_text}</span>
            </div>
        """, unsafe_allow_html=True)

    # Conversation display
    st.markdown("### Conversation")

    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "customer":
            st.markdown(f"""
                <div class="chat-customer">
                    <strong>{st.session_state.customer_name}</strong>
                    {f' <span class="emotion-badge {get_emotion_class(msg.get("emotion", "neutral"))}">{msg.get("emotion", "").upper()}</span>' if msg.get("emotion") else ''}
                    <p style="margin-top: 0.5rem;">{msg["content"]}</p>
                </div>
            """, unsafe_allow_html=True)

            # Play audio for customer message if voice enabled and not played yet
            if st.session_state.voice_enabled and msg.get("audio_url") and not msg.get("audio_played", False):
                autoplay_audio(msg["audio_url"])
                st.session_state.messages[idx]["audio_played"] = True
        else:
            techniques = msg.get("techniques", [])
            techniques_html = " ".join([
                f'<span class="emotion-badge" style="background: rgba(34, 197, 94, 0.2); color: #86efac;">{t}</span>'
                for t in techniques
            ]) if techniques else ""

            # Voice analysis badge if available
            voice_html = ""
            voice_data = msg.get("voice_analysis")
            if voice_data:
                voice_emotion = voice_data.get("emotion", "")
                voice_score = voice_data.get("delivery_scores", {}).get("overall", 0)
                voice_html = f' <span class="emotion-badge" style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc;">🎤 {voice_emotion} ({voice_score}/10)</span>'

            st.markdown(f"""
                <div class="chat-trainee">
                    <strong>You (Trainee)</strong> {techniques_html}{voice_html}
                    <p style="margin-top: 0.5rem;">{msg["content"]}</p>
                </div>
            """, unsafe_allow_html=True)

    # Input area
    st.markdown("---")

    # Input area with voice support
    col_text, col_mic, col_send = st.columns([3.5, 1, 0.5])

    with col_text:
        user_input = st.text_input(
            "Your response:",
            key="trainee_text_input",
            placeholder="Type or speak your response..." if st.session_state.voice_enabled else "Type your response...",
            label_visibility="collapsed"
        )

    with col_mic:
        if st.session_state.voice_enabled:
            audio_bytes = st.audio_input(
                "🎤",
                key="sim_audio_recorder",
                label_visibility="collapsed"
            )
        else:
            audio_bytes = None
            st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)

    with col_send:
        send_clicked = st.button("➤", use_container_width=True, help="Send message")

    # Process voice recording if voice is enabled
    if st.session_state.voice_enabled and audio_bytes is not None:
        audio_data = audio_bytes.getvalue()
        audio_id = hash(audio_data)

        # Only process if new recording
        if audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id
            try:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio.write(audio_data)
                    temp_audio_path = temp_audio.name

                # Transcribe and analyze voice in parallel conceptually
                with st.spinner("Processing your voice... 🎤"):
                    # Transcribe
                    transcribed_text = st.session_state.api_client.transcribe_audio(temp_audio_path)

                    # Analyze voice emotion and delivery
                    voice_analysis = analyze_voice(audio_data)
                    if voice_analysis and voice_analysis.get("analysis_success"):
                        st.session_state.last_voice_analysis = voice_analysis
                        # Show voice emotion badge
                        emotion = voice_analysis.get("primary_emotion", "unknown")
                        delivery = voice_analysis.get("delivery_scores", {})
                        overall = delivery.get("overall", 5)
                        st.toast(f"Voice: {emotion} (delivery: {overall}/10)", icon="🎤")

                # Cleanup
                try:
                    os.remove(temp_audio_path)
                except:
                    pass

                if transcribed_text:
                    # Process the transcribed text
                    process_trainee_response(transcribed_text)
                else:
                    st.warning("No speech detected. Try again.")

            except Exception as e:
                st.error(f"Error processing voice: {e}")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    # Highlight appropriate button based on ending type
    natural_ending = st.session_state.get("natural_ending", False)
    ending_type = st.session_state.get("ending_type", "")

    with col1:
        resolved_label = "✅ Mark Resolved & End" if natural_ending else "🏳️ End (Resolved)"
        if st.button(resolved_label, use_container_width=True, type="primary" if ending_type in ["satisfied_goodbye", "reluctant_acceptance"] else "secondary"):
            end_simulation(resolved=True)

    with col2:
        unresolved_label = "❌ Mark Unresolved & End" if natural_ending else "❌ End (Unresolved)"
        if st.button(unresolved_label, use_container_width=True, type="primary" if ending_type == "frustrated_exit" else "secondary"):
            end_simulation(resolved=False)

    with col3:
        if st.button("🚪 Exit Session", use_container_width=True):
            abandon_simulation()

    # Process input
    if send_clicked and user_input:
        process_trainee_response(user_input)


def process_trainee_response(message: str):
    """Send trainee response and get customer reply."""
    # Get voice analysis if available
    voice_analysis = st.session_state.get("last_voice_analysis")
    voice_data = None
    if voice_analysis:
        voice_data = {
            "emotion": voice_analysis.get("primary_emotion", "unknown"),
            "delivery_scores": voice_analysis.get("delivery_scores", {}),
        }
        # Store for final report
        st.session_state.voice_analyses.append(voice_analysis)
        # Clear for next turn
        st.session_state.last_voice_analysis = None

    # Add trainee message with voice data
    st.session_state.messages.append({
        "role": "trainee",
        "content": message,
        "techniques": [],
        "voice_analysis": voice_data,
    })

    with st.spinner(f"{st.session_state.customer_name} is responding..."):
        result = api_post("/api/simulation/respond", {
            "session_id": st.session_state.session_id,
            "message": message
        })

    if result:
        # Update trainee message with detected techniques
        st.session_state.messages[-1]["techniques"] = result.get("detected_techniques", [])

        # Synthesize voice for customer response if voice enabled
        audio_url = None
        prosody = result.get("prosody", {})
        if st.session_state.voice_enabled:
            audio_url = synthesize_customer_voice(result["customer_message"], prosody)

        # Add customer response with audio
        st.session_state.messages.append({
            "role": "customer",
            "content": result["customer_message"],
            "emotion": result["emotion_state"],
            "prosody": prosody,
            "audio_url": audio_url,
            "audio_played": False
        })

        # Update emotion
        st.session_state.current_emotion = result["emotion_state"]

        # Show emotion change notification
        if result.get("emotion_changed"):
            prev = result.get("previous_emotion", "unknown")
            curr = result["emotion_state"]
            st.toast(f"Customer emotion: {prev} → {curr}")

        # Check for natural conversation ending
        if result.get("conversation_complete"):
            ending_type = result.get("ending_type", "natural_conclusion")
            if ending_type == "satisfied_goodbye":
                st.toast("Customer is satisfied and ending the call.", icon="✅")
            elif ending_type == "reluctant_acceptance":
                st.toast("Customer reluctantly accepts the resolution.", icon="😐")
            elif ending_type == "frustrated_exit":
                st.toast("Customer is frustrated and leaving.", icon="😤")
            else:
                st.toast("Conversation naturally concluding.", icon="👋")

            # Store that conversation ended naturally
            st.session_state.natural_ending = True
            st.session_state.ending_type = ending_type

        # Show approaching end notification
        elif result.get("approaching_end"):
            st.toast("Customer seems ready to wrap up...", icon="💭")

    st.rerun()


def end_simulation(resolved: bool):
    """End the simulation."""
    # Calculate average voice scores
    voice_analyses = st.session_state.get("voice_analyses", [])
    avg_voice_scores = {}
    if voice_analyses:
        for key in ["calmness", "confidence", "empathy", "pace", "clarity", "overall"]:
            scores = [va.get("delivery_scores", {}).get(key, 5) for va in voice_analyses]
            avg_voice_scores[key] = sum(scores) / len(scores) if scores else 5

    # Store local session data for fallback analysis
    st.session_state.local_session_data = {
        "messages": st.session_state.messages.copy(),
        "scenario": st.session_state.scenario,
        "resolution_achieved": resolved,
        "turn_count": len([m for m in st.session_state.messages if m["role"] == "trainee"]),
        "final_emotion": st.session_state.current_emotion,
        "voice_analyses": voice_analyses,
        "avg_voice_scores": avg_voice_scores,
    }

    with st.spinner("Ending simulation..."):
        result = api_post("/api/simulation/end", {
            "session_id": st.session_state.session_id,
            "resolution_achieved": resolved
        })

    if result:
        st.session_state.end_result = result
        st.session_state.sim_state = "feedback"
        st.rerun()
    else:
        # Even if API fails, still go to feedback with local data
        st.warning("Session ended but couldn't save to server. Showing local results.")
        st.session_state.sim_state = "feedback"
        st.rerun()


def abandon_simulation():
    """Abandon the simulation without analysis."""
    st.session_state.sim_state = "select"
    st.session_state.session_id = None
    st.session_state.messages = []
    st.rerun()


# ============================================================================
# Page: Feedback Report
# ============================================================================

def render_feedback():
    """Render post-simulation feedback."""
    st.markdown("""
        <div class="header-container">
            <h1>📊 Performance <span class="accent">Report</span></h1>
            <p style="opacity: 0.7;">Here's how you did in this simulation</p>
        </div>
    """, unsafe_allow_html=True)

    analysis = None

    # Try to fetch analysis from API
    if st.session_state.session_id:
        with st.spinner("Analyzing your performance..."):
            analysis = api_get(f"/api/simulation/analysis/{st.session_state.session_id}")

        if not analysis:
            # Try quick score as fallback
            analysis = api_get(f"/api/simulation/analysis/{st.session_state.session_id}/quick")

    # If API fails, create local fallback analysis
    if not analysis:
        local_data = st.session_state.get("local_session_data", {})
        end_result = st.session_state.get("end_result", {})

        if local_data or end_result:
            st.info("Using local session data for analysis.")
            # Build basic analysis from local data
            messages = local_data.get("messages", [])
            trainee_turns = [m for m in messages if m["role"] == "trainee"]
            customer_turns = [m for m in messages if m["role"] == "customer"]

            # Count techniques used
            techniques_used = []
            for turn in trainee_turns:
                techniques_used.extend(turn.get("techniques", []))

            analysis = {
                "overall_score": 6 if local_data.get("resolution_achieved") else 4,
                "empathy_score": 5 + len([t for t in techniques_used if "empathy" in t.lower()]),
                "de_escalation_score": 5,
                "communication_clarity_score": 6,
                "problem_solving_score": 6 if local_data.get("resolution_achieved") else 4,
                "efficiency_score": max(3, 8 - len(trainee_turns)),
                "resolution_achieved": local_data.get("resolution_achieved", False),
                "de_escalation_success": local_data.get("final_emotion") in ["satisfied", "neutral", "hopeful"],
                "turn_count": len(trainee_turns),
                "duration_seconds": end_result.get("duration_seconds", 0),
                "emotion_changes": end_result.get("emotion_changes", 0),
                "strengths": ["Completed the simulation"],
                "areas_for_improvement": ["Analysis unavailable - practice more scenarios"],
                "specific_feedback": [],
                "recommended_training": [],
            }
        else:
            st.error("Analysis unavailable. No session data found.")
            if st.button("Return to Scenarios"):
                reset_simulation()
            return

    # Overall score
    overall = analysis.get("overall_score", 5)
    grade = "A" if overall >= 9 else "B+" if overall >= 8 else "B" if overall >= 7 else "C+" if overall >= 6 else "C" if overall >= 5 else "D"

    st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: rgba(99, 102, 241, 0.1); border-radius: 16px; margin-bottom: 2rem;">
            <h1 style="font-size: 4rem; margin: 0;">{grade}</h1>
            <p style="font-size: 1.5rem; opacity: 0.8;">Overall Score: {overall}/10</p>
            <p>
                Resolution: {'✅ Achieved' if analysis.get('resolution_achieved') else '❌ Not Achieved'} |
                De-escalation: {'✅ Success' if analysis.get('de_escalation_success') else '❌ Failed'}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Skill scores
    st.markdown("### 📝 Content Analysis")

    col1, col2 = st.columns(2)

    with col1:
        render_score_bar(analysis.get("empathy_score", 5), "Empathy")
        render_score_bar(analysis.get("de_escalation_score", 5), "De-escalation")
        render_score_bar(analysis.get("communication_clarity_score", 5), "Communication")

    with col2:
        render_score_bar(analysis.get("problem_solving_score", 5), "Problem Solving")
        render_score_bar(analysis.get("efficiency_score", 5), "Efficiency")

    # Voice delivery scores (if available)
    local_data = st.session_state.get("local_session_data", {})
    voice_scores = local_data.get("avg_voice_scores", {})

    if voice_scores:
        st.markdown("### 🎤 Voice Delivery")

        col1, col2 = st.columns(2)

        with col1:
            render_score_bar(int(voice_scores.get("calmness", 5)), "Calmness")
            render_score_bar(int(voice_scores.get("confidence", 5)), "Confidence")
            render_score_bar(int(voice_scores.get("empathy", 5)), "Empathetic Tone")

        with col2:
            render_score_bar(int(voice_scores.get("pace", 5)), "Pace Control")
            render_score_bar(int(voice_scores.get("clarity", 5)), "Clarity")

        # Voice emotion summary
        voice_analyses = local_data.get("voice_analyses", [])
        if voice_analyses:
            emotions = [va.get("primary_emotion", "unknown") for va in voice_analyses]
            emotion_counts = {}
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1

            # Get dominant emotion
            dominant = max(emotion_counts, key=emotion_counts.get)
            st.markdown(f"**Primary Voice Emotion:** {dominant.capitalize()} "
                       f"({emotion_counts[dominant]}/{len(emotions)} responses)")

    # Strengths
    st.markdown("### 💪 Strengths")
    for strength in analysis.get("strengths", []):
        st.markdown(f"- {strength}")

    # Areas for improvement
    st.markdown("### 📈 Areas for Improvement")
    for area in analysis.get("areas_for_improvement", []):
        st.markdown(f"- {area}")

    # Specific feedback
    if analysis.get("specific_feedback"):
        st.markdown("### 💡 Specific Feedback")
        for feedback in analysis.get("specific_feedback", []):
            st.info(feedback)

    # Recommended training
    if analysis.get("recommended_training"):
        st.markdown("### 📚 Recommended Training")
        for training in analysis.get("recommended_training", []):
            st.markdown(f"- {training}")

    # Session stats
    st.markdown("### 📊 Session Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Turns", analysis.get("turn_count", 0))
    with col2:
        duration = analysis.get("duration_seconds", 0)
        st.metric("Duration", f"{duration:.0f}s")
    with col3:
        st.metric("Emotion Changes", analysis.get("emotion_changes", 0))

    # Actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Try Again", use_container_width=True):
            # Restart same scenario
            start_simulation(st.session_state.scenario["scenario_id"])

    with col2:
        if st.button("📋 View Transcript", use_container_width=True):
            transcript = api_get(f"/api/simulation/sessions/{st.session_state.session_id}/transcript?format=text")
            if transcript:
                st.text_area("Transcript", transcript.get("transcript", ""), height=300)

    with col3:
        if st.button("🏠 New Scenario", use_container_width=True):
            reset_simulation()


def reset_simulation():
    """Reset simulation state."""
    st.session_state.sim_state = "select"
    st.session_state.session_id = None
    st.session_state.scenario = None
    st.session_state.messages = []
    st.session_state.current_emotion = None
    st.session_state.last_audio_id = None
    st.session_state.natural_ending = False
    st.session_state.ending_type = None
    st.session_state.last_voice_analysis = None
    st.session_state.voice_analyses = []
    st.session_state.local_session_data = {}
    st.rerun()


# ============================================================================
# Main App Router
# ============================================================================

def main():
    """Main app entry point."""
    ensure_session_state()
    if st.session_state.sim_state == "select":
        render_scenario_selection()
    elif st.session_state.sim_state == "active":
        render_active_simulation()
    elif st.session_state.sim_state == "feedback":
        render_feedback()
    else:
        st.session_state.sim_state = "select"
        st.rerun()


if __name__ == "__main__":
    main()
