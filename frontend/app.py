import sys
import os
import uuid
import tempfile
import streamlit as st

# Add parent directory to path to import frontend modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from frontend.api_client import APIClient

# Initialize API Client
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()

# Check API Health
try:
    if "api_health" not in st.session_state:
        health = st.session_state.api_client.health_check()
        st.session_state.api_health = health.get("status") == "healthy"
except Exception:
    st.session_state.api_health = False

# Page configuration
st.set_page_config(
    page_title="ConversaVoice",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Audio Player Component
def autoplay_audio(audio_url: str):
    unique_id = f"audio_{uuid.uuid4().hex[:8]}"
    
    # Use st.components.v1.html for a cleaner execution environment
    html_content = f"""
        <style>
            .voice-waves {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
                height: 40px;
                margin-top: 10px;
            }}
            .wave-bar {{
                width: 4px;
                height: 15px;
                background: linear-gradient(180deg, #6366f1 0%, #a855f7 100%);
                border-radius: 4px;
                animation: wave-animation 1s ease-in-out infinite;
            }}
            .wave-bar:nth-child(2) {{ animation-delay: 0.1s; height: 25px; }}
            .wave-bar:nth-child(3) {{ animation-delay: 0.2s; height: 20px; }}
            .wave-bar:nth-child(4) {{ animation-delay: 0.3s; height: 30px; }}
            .wave-bar:nth-child(5) {{ animation-delay: 0.4s; height: 20px; }}
            .wave-bar:nth-child(6) {{ animation-delay: 0.5s; height: 25px; }}
            .wave-bar:nth-child(7) {{ animation-delay: 0.6s; height: 15px; }}
            .wave-bar:nth-child(8) {{ animation-delay: 0.7s; height: 20px; }}

            @keyframes wave-animation {{
                0%, 100% {{ transform: scaleY(1); }}
                50% {{ transform: scaleY(1.5); }}
            }}
        </style>
        <div id="{unique_id}_container">
            <audio id="{unique_id}_player" autoplay style="position:absolute; width:0; height:0; opacity:0; pointer-events:none;">
                <source src="{audio_url}" type="audio/wav">
            </audio>
            <div class="voice-waves">
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
                <div class="wave-bar"></div><div class="wave-bar"></div>
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
                        // Safety fallback
                        setTimeout(hide, 10000);
                    }} else {{
                        hide();
                    }}
                }})();
            </script>
        </div>
    """
    import streamlit.components.v1 as components
    components.html(html_content, height=60)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #ffffff;
    }

    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 8rem;
        max-width: 900px;
    }

    .stChatMessage {
        background: transparent !important;
        padding: 0.75rem 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        color: #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    div[data-testid="stTextInput"] > label {
        display: none;
    }

    div[data-testid="stTextInputRootElement"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(31, 41, 55, 0.8) !important;
        border: 2px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 26px !important;
        color: #ffffff !important;
        padding: 0 1.5rem !important;
        font-size: 16px !important;
        height: 52px !important;
        line-height: 52px !important;
        display: flex !important;
        align-items: center !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2) !important;
        background: rgba(31, 41, 55, 0.95) !important;
    }

    div[data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
        background: transparent !important;
        border: none !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        outline: none !important;
        border-radius: 50% !important;
        width: 52px !important;
        height: 52px !important;
        min-width: 52px !important;
        max-width: 52px !important;
        padding: 0 !important;
        font-size: 22px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
    }

    .stop-btn .stButton > button {
        background: #1a1f3a !important;
        border: 2px solid #ef4444 !important;
        color: #ef4444 !important;
    }

    .stop-btn .stButton > button:hover {
        background: #ef4444 !important;
        color: white !important;
    }

    .recording-active .stButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        animation: pulse-ring 2s infinite !important;
    }

    .voice-waves {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        height: 52px;
        padding: 0 15px;
        background: rgba(31, 41, 55, 0.6);
        border-radius: 26px;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }

    .wave-bar {
        width: 4px;
        background: linear-gradient(180deg, #6366f1 0%, #a855f7 100%);
        border-radius: 2px;
        animation: wave-animate 0.8s ease-in-out infinite;
    }

    .recording-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%);
        border: 2px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 1.5rem;
        text-align: center;
        animation: pulse-border 2s ease-in-out infinite;
    }

    @keyframes pulse-border {
        0%, 100% { border-color: rgba(239, 68, 68, 0.4); }
        50% { border-color: rgba(239, 68, 68, 0.7); }
    }

    .rec-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #ef4444;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-dot 1s infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
    }

    .accent {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }

    .header-icon {
        font-size: 56px;
        margin-bottom: 1rem;
        filter: drop-shadow(0 4px 12px rgba(124, 58, 237, 0.4));
    }

    .header-title {
        font-size: 36px;
        font-weight: 700;
        margin: 0;
    }

    .header-subtitle {
        font-size: 14px;
        opacity: 0.7;
        margin-top: 0.5rem;
    }

    .welcome-card {
        background: rgba(31, 41, 55, 0.5);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
    }

    .welcome-icon {
        font-size: 72px;
        margin-bottom: 1.5rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .meta-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.2);
        color: #a78bfa;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        margin: 4px 4px 0 0;
        font-weight: 500;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-dot-green 2s infinite;
    }
    
    .status-dot-red {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #ef4444;
        border-radius: 50%;
        margin-right: 8px;
    }

    @keyframes pulse-dot-green {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(31, 41, 55, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(124, 58, 237, 0.5);
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "client_session_id" not in st.session_state:
    # Try to create a session on backend
    try:
        st.session_state.client_session_id = st.session_state.api_client.create_session()
    except:
        st.session_state.client_session_id = None

# Header
status_class = "status-dot" if st.session_state.get("api_health", False) else "status-dot-red"
status_text = "Backend Connected" if st.session_state.get("api_health", False) else "Backend Disconnected"

st.markdown(f"""
    <div class="header-container">
        <div class="header-icon">🎙️</div>
        <h1 class="header-title">Conversa<span class="accent">Voice</span></h1>
        <p class="header-subtitle">
            <span class="{status_class}"></span>
            {status_text}
        </p>
    </div>
""", unsafe_allow_html=True)

# Audio Recording - uses browser's microphone via Streamlit's audio_input

# Chat Container
if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">👋</div>
            <h2 style="font-size: 28px; font-weight: 700; margin-bottom: 1rem;">Welcome to ConversaVoice!</h2>
            <p style="font-size: 16px; opacity: 0.85; line-height: 1.6; max-width: 500px; margin: 0 auto;">
                I'm your AI assistant with emotional intelligence. Click 🎤 to speak or type below.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg and msg["role"] == "assistant":
                meta = msg["metadata"]
                # Only show if meaningful metadata exists
                if meta:
                    st.markdown(f"""
                        <div style="margin-top: 12px;">
                            <span class="meta-badge">🎭 {meta.get('style', 'Neutral')}</span>
                            <span class="meta-badge">⏱️ {meta.get('latency_ms', '0')}ms</span>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Play audio if available (only once)
            if "audio_url" in msg and msg["role"] == "assistant":
                if msg.get("audio_url") and not msg.get("audio_played", False):
                    autoplay_audio(msg["audio_url"])
                    msg["audio_played"] = True

# Input Area
st.markdown("---")

# Callback to handle text submission
def submit_text():
    if st.session_state.text_input:
        st.session_state.pending_text = st.session_state.text_input
        st.session_state.text_input = ""

# Keep input area in a container that we can empty/hide
input_placeholder = st.empty()

if not st.session_state.processing:
    with input_placeholder.container():
        col_input, col_voice = st.columns([8.5, 1.5], gap="small")

        with col_input:
            st.text_input(
                "msg",
                placeholder="💬 Type your message here...",
                key="text_input",
                label_visibility="collapsed",
                on_change=submit_text,
                disabled=not st.session_state.get("api_health", False)
            )

        with col_voice:
            # Browser-based audio recording using Streamlit's audio_input
            audio_bytes = st.audio_input(
                "🎤",
                key="audio_recorder",
                disabled=not st.session_state.get("api_health", False)
            )

# Process Voice Recording from browser
if "audio_recorder" in st.session_state and st.session_state.audio_recorder is not None:
    audio_bytes = st.session_state.audio_recorder
    try:
        # Save audio bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes.getvalue())
            temp_audio_path = temp_audio.name

        # Transcribe via API
        with st.spinner("Transcribing your voice... 🎤"):
            transcribed_text = st.session_state.api_client.transcribe_audio(temp_audio_path)

        # Cleanup temp file
        try:
            os.remove(temp_audio_path)
        except:
            pass

        if transcribed_text:
            st.session_state.pending_text = transcribed_text
            # Clear the audio input
            st.session_state.audio_recorder = None
            st.rerun()
        else:
            st.warning("No speech detected.")
            st.session_state.audio_recorder = None

    except Exception as e:
        st.error(f"Error processing recording: {e}")
        st.session_state.audio_recorder = None

# Process Pending Text (from input or voice)
if "pending_text" in st.session_state and st.session_state.pending_text:
    user_text = st.session_state.pending_text
    del st.session_state.pending_text
    
    st.session_state.processing = True
    input_placeholder.empty()
    
    # 1. Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text
    })
    
    # 2. Call Chat API
    try:
        with st.spinner("Thinking... 🤔"):
            chat_response = st.session_state.api_client.chat(user_text)
            
            response_text = chat_response.get("response", "")
            style = chat_response.get("style")
            
            # 3. Call Synthesize API
            audio_url = None
            if response_text:
                try:
                    with st.spinner("Generating voice... 🔊"):
                        # We pass the full response text for TTS
                        audio_url = st.session_state.api_client.synthesize_speech(
                            text=response_text,
                            style=style,
                            pitch=chat_response.get("pitch"),
                            rate=chat_response.get("rate")
                        )
                except Exception as e:
                    st.warning(f"Voice generation failed: {e}")
            
            # 4. Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "metadata": {
                    "style": style,
                    "latency_ms": chat_response.get("latency_ms", 0)
                },
                "audio_url": audio_url
            })

            
    except Exception as e:
        st.error(f"Error communicating with backend: {e}")
        
    st.session_state.processing = False
    st.rerun()

# Footer
st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; opacity: 0.5; font-size: 12px;">
        <p style="margin: 0;">ConversaVoice Platform • 🎤 Click to speak • 💬 Type to chat</p>
    </div>
""", unsafe_allow_html=True)
