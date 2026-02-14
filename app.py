import sys
print(f"DEBUG: sys.executable = {sys.executable}")
print(f"DEBUG: sys.path = {sys.path}")
import streamlit as st
import asyncio
import tempfile
import os
from src.orchestrator import Orchestrator, PipelineResult

# Page configuration
st.set_page_config(
    page_title="ConversaVoice",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# Custom Audio Player Component
def autoplay_audio(file_path: str):
    import base64
    import uuid
    import wave
    import streamlit.components.v1 as components
    
    # Calculate duration for fallback
    duration = 5.0 # Default
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
    except Exception:
        pass

    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
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
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
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
                            // Optional: Send a heartbeat back to Streamlit if needed
                        }}
                    }};
                    
                    if (player) {{
                        player.onended = hide;
                        player.onerror = hide;
                        // Brute force safety fallback (duration + 1.5s buffer for reliability)
                        setTimeout(hide, {(duration + 1.5) * 1000});
                    }} else {{
                        hide();
                    }}
                }})();
            </script>
        </div>
    """
    # Use a fixed height to avoid scrollbars in the iframe
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

    /* Hide default text input label */
    div[data-testid="stTextInput"] > label {
        display: none;
    }

    /* Targeting the input root more specifically to remove white boxes */
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

    /* Force circular styling on all buttons in the input row and REMOVE white borders */
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

    /* Stop state specifics */
    .stop-btn .stButton > button {
        background: #1a1f3a !important;
        border: 2px solid #ef4444 !important;
        color: #ef4444 !important;
    }

    .stop-btn .stButton > button:hover {
        background: #ef4444 !important;
        color: white !important;
    }

    /* Pulse animation for recording */
    .recording-active .stButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        animation: pulse-ring 2s infinite !important;
    }

    /* Voice waves */
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

    .wave-bar:nth-child(1) { animation-delay: 0s; }
    .wave-bar:nth-child(2) { animation-delay: 0.1s; }
    .wave-bar:nth-child(3) { animation-delay: 0.2s; }
    .wave-bar:nth-child(4) { animation-delay: 0.3s; }
    .wave-bar:nth-child(5) { animation-delay: 0.4s; }
    .wave-bar:nth-child(6) { animation-delay: 0.3s; }
    .wave-bar:nth-child(7) { animation-delay: 0.2s; }
    .wave-bar:nth-child(8) { animation-delay: 0.1s; }

    @keyframes wave-animate {
        0%, 100% { height: 8px; opacity: 0.6; }
        50% { height: 35px; opacity: 1; }
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
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
    # Initialize components lazily but ensure base setup is done
    asyncio.run(st.session_state.orchestrator.initialize())
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "recorder_active" not in st.session_state:
    st.session_state.recorder_active = False
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "processing_voice" not in st.session_state:
    st.session_state.processing_voice = False

# Handle Recording State Synchronization & Background Control
if st.session_state.is_recording and not st.session_state.recorder_active:
    # Trigger background recording start
    async def start_rec():
        try:
            if not st.session_state.orchestrator._stt_client:
                await st.session_state.orchestrator.initialize_stt()
            st.session_state.orchestrator.start_recording_background()
            return True
        except Exception as e:
            st.error(f"Failed to start microphone: {e}")
            return False

    if asyncio.run(start_rec()):
        st.session_state.recorder_active = True
    else:
        st.session_state.is_recording = False
        st.session_state.recorder_active = False

elif not st.session_state.is_recording and st.session_state.recorder_active:
    # Cleanup if somehow desynced
    st.session_state.orchestrator.stop_recording_background()
    st.session_state.recorder_active = False

# Header
st.markdown("""
    <div class="header-container">
        <div class="header-icon">🎙️</div>
        <h1 class="header-title">Conversa<span class="accent">Voice</span></h1>
        <p class="header-subtitle">
            <span class="status-dot"></span>
            AI-Powered Emotional Intelligence Assistant
        </p>
    </div>
""", unsafe_allow_html=True)

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
                st.markdown(f"""
                    <div style="margin-top: 12px;">
                        <span class="meta-badge">🎭 {meta.get('style', 'Neutral')}</span>
                        <span class="meta-badge">⏱️ {meta.get('latency_ms', '0ms')}</span>
                        {f'<span class="meta-badge">🔄 Repetition</span>' if meta.get('is_repetition') else ''}
                    </div>
                """, unsafe_allow_html=True)
            
            # Play audio if available (only once)
            if "audio_file" in msg and msg["role"] == "assistant":
                if msg.get("audio_file") and os.path.exists(msg["audio_file"]) and not msg.get("audio_played", False):
                    autoplay_audio(msg["audio_file"])
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
        # Recording/Processing Banner (Now above search box)
        if st.session_state.is_recording or st.session_state.processing_voice:
            banner_text = "Recording..." if st.session_state.is_recording else "Processing your voice... ⏳"
            status_text = "Speak now" if st.session_state.is_recording else "Almost there"
            dot_style = "background: #ef4444;" if st.session_state.is_recording else "background: #6366f1;"
            
            st.markdown(f"""
                <div class="recording-banner">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 16px;">
                        <span class="rec-dot" style="{dot_style}"></span>
                        <strong style="font-size: 16px;">{banner_text}</strong>
                        <div class="voice-waves">
                            <div class="wave-bar"></div>
                            <div class="wave-bar"></div>
                            <div class="wave-bar"></div>
                            <div class="wave-bar"></div>
                            <div class="wave-bar"></div>
                            <div class="wave-bar"></div>
                            <div class="wave-bar"></div>
                            <div class="wave-bar"></div>
                        </div>
                        <span style="font-size: 13px; opacity: 0.8;">{status_text}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        col_input, col_voice = st.columns([8.8, 1.2], gap="small")

        with col_input:
            st.text_input(
                "msg",
                placeholder="💬 Type your message here...",
                key="text_input",
                label_visibility="collapsed",
                on_change=submit_text
            )

        with col_voice:
            if not st.session_state.processing_voice:
                if st.session_state.is_recording:
                    st.markdown("<div class='stop-btn recording-active'>", unsafe_allow_html=True)
                    if st.button("⏹️", key="stop_rec", help="Stop recording"):
                        st.session_state.is_recording = False
                        st.session_state.recorder_active = False
                        st.session_state.processing_voice = True
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    if st.button("🎤", key="mic_start", help="Start recording"):
                        st.session_state.is_recording = True
                        st.rerun()

# Process Voice Recording
if st.session_state.processing_voice:
    try:
        print("Stopping recording and starting processing...")
        user_text = st.session_state.orchestrator.stop_recording_background()
        print(f"Captured text: '{user_text}'")
        
        if user_text:
            # Set as pending text to be processed
            st.session_state.pending_text = user_text
        else:
            st.warning("No speech detected. Please speak clearly into the microphone.")
            print("No speech detected in buffer")
            
    except Exception as e:
        st.error(f"Error processing recording: {e}")
        print(f"Error stopping recording: {e}")
    finally:
        # Always reset recording/processing states
        st.session_state.is_recording = False
        st.session_state.recorder_active = False
        st.session_state.processing_voice = False
    
    st.rerun()

# Process text input (from text box or voice)
if "pending_text" in st.session_state and st.session_state.pending_text:
    user_text = st.session_state.pending_text
    del st.session_state.pending_text  # Clear immediately to prevent re-processing
    
    st.session_state.processing = True
    input_placeholder.empty() # Clear input area immediately
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_text
    })
    
    async def process_text(text):
        return await st.session_state.orchestrator.process_text(text, speak=False)

    with st.spinner("Thinking... 🤔"):
        result = asyncio.run(process_text(user_text))
    
    # Generate audio file for the response
    audio_file = None
    try:
        with st.spinner("Generating voice response... 🔊"):
            # Create temp file for audio
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio_file = temp_audio.name
            temp_audio.close()
            
            # Get TTS client and synthesize to file with LLM params
            tts_client = st.session_state.orchestrator._tts_client
            
            # Build SSML with LLM params
            ssml = tts_client.ssml_builder.build_from_llm_response(
                text=result.assistant_response,
                style=result.style,
                pitch=result.pitch,
                rate=result.rate
            )
            
            # Synthesize to file
            import azure.cognitiveservices.speech as speechsdk
            audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_file)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=tts_client._speech_config,
                audio_config=audio_config
            )
            
            synthesis_result = synthesizer.speak_ssml_async(ssml).get()
            
            if synthesis_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                st.warning("Could not generate voice response")
                audio_file = None
                
    except Exception as e:
        st.warning(f"Could not generate voice: {e}")
        audio_file = None
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.assistant_response,
        "metadata": {
            "style": result.style,
            "pitch": result.pitch,
            "rate": result.rate,
            "latency_ms": f"{result.latency_ms:.0f}ms",
            "is_repetition": result.is_repetition
        },
        "audio_file": audio_file
    })
    st.session_state.processing = False
    st.rerun()



# Auto refresh during recording for better responsiveness
if st.session_state.is_recording:
    import time
    time.sleep(0.1)
    st.rerun()

# Footer
st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; opacity: 0.5; font-size: 12px;">
        <p style="margin: 0;">ConversaVoice Platform • 🎤 Click to speak • 💬 Type to chat</p>
    </div>
""", unsafe_allow_html=True)
