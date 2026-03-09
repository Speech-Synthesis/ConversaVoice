import streamlit as st
import uuid

def autoplay_audio(audio_url: str):
    """Auto-play audio with visual wave animation."""
    unique_id = f"audio_{uuid.uuid4().hex[:8]}"
    html_content = f"""
        <style>
            .sim-voice-waves {{ display: flex; align-items: center; justify-content: center; gap: 4px; height: 40px; margin-top: 10px; }}
            .sim-wave-bar {{ width: 4px; height: 15px; background: linear-gradient(180deg, #ef4444 0%, #f97316 100%); border-radius: 4px; animation: sim-wave-animation 1s ease-in-out infinite; }}
            .sim-wave-bar:nth-child(2) {{ animation-delay: 0.1s; height: 25px; }}
            .sim-wave-bar:nth-child(3) {{ animation-delay: 0.2s; height: 20px; }}
            .sim-wave-bar:nth-child(4) {{ animation-delay: 0.3s; height: 30px; }}
            .sim-wave-bar:nth-child(5) {{ animation-delay: 0.4s; height: 20px; }}
            .sim-wave-bar:nth-child(6) {{ animation-delay: 0.5s; height: 25px; }}
            .sim-wave-bar:nth-child(7) {{ animation-delay: 0.6s; height: 15px; }}
            .sim-wave-bar:nth-child(8) {{ animation-delay: 0.7s; height: 20px; }}
            @keyframes sim-wave-animation {{ 0%, 100% {{ transform: scaleY(1); }} 50% {{ transform: scaleY(1.5); }} }}
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
                    const hide = () => {{ if(container) container.style.display = 'none'; }};
                    if (player) {{ player.onended = hide; player.onerror = hide; setTimeout(hide, 15000); }} else {{ hide(); }}
                }})();
            </script>
        </div>
    """
    import streamlit.components.v1 as components
    components.html(html_content, height=60)

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

def render_emotion_badge(emotion: str):
    """Render emotion badge."""
    st.markdown(
        f'<span class="emotion-badge emotion-{emotion.lower()}">{emotion.upper()}</span>',
        unsafe_allow_html=True
    )
