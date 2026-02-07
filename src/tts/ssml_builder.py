"""
SSML Builder for emotional prosody control.

Generates SSML markup with prosody profiles for Azure Neural TTS.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ProsodyProfile(Enum):
    """Pre-defined prosody profiles for emotional speech."""

    NEUTRAL = "neutral"
    EMPATHETIC = "empathetic"
    PATIENT = "patient"
    CHEERFUL = "cheerful"
    DE_ESCALATE = "de_escalate"


@dataclass
class ProsodySettings:
    """Prosody settings for speech synthesis."""

    pitch: str  # e.g., "-5%", "+10%", "default"
    rate: str   # e.g., "0.85", "1.1", "default"
    volume: str = "default"  # e.g., "soft", "loud", "default"
    style: Optional[str] = None  # Azure express-as style


# Prosody profiles based on Conversia.md specifications
PROSODY_PROFILES: dict[ProsodyProfile, ProsodySettings] = {
    ProsodyProfile.NEUTRAL: ProsodySettings(
        pitch="default",
        rate="1.0",
        volume="default",
        style=None
    ),
    ProsodyProfile.EMPATHETIC: ProsodySettings(
        pitch="-5%",
        rate="0.85",
        volume="default",
        style="empathetic"
    ),
    ProsodyProfile.PATIENT: ProsodySettings(
        pitch="-3%",
        rate="0.9",
        volume="default",
        style=None
    ),
    ProsodyProfile.CHEERFUL: ProsodySettings(
        pitch="+5%",
        rate="1.1",
        volume="default",
        style="cheerful"
    ),
    ProsodyProfile.DE_ESCALATE: ProsodySettings(
        pitch="-10%",
        rate="0.8",
        volume="soft",
        style="empathetic"
    ),
}


class SSMLBuilder:
    """
    Builder for generating SSML markup with emotional prosody.

    Supports Azure Neural TTS with express-as styles and prosody control.
    """

    # Default Azure Neural voice
    DEFAULT_VOICE = "en-US-JennyNeural"

    def __init__(self, voice: str = DEFAULT_VOICE):
        """
        Initialize SSML builder.

        Args:
            voice: Azure Neural voice name (e.g., "en-US-JennyNeural")
        """
        self.voice = voice

    def build(
        self,
        text: str,
        profile: ProsodyProfile = ProsodyProfile.NEUTRAL,
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        style: Optional[str] = None
    ) -> str:
        """
        Build SSML markup for the given text.

        Args:
            text: The text to synthesize
            profile: Pre-defined prosody profile
            pitch: Override pitch (e.g., "-5%", "+10%")
            rate: Override rate (e.g., "0.85", "1.1")
            volume: Override volume (e.g., "soft", "loud")
            style: Override Azure express-as style

        Returns:
            Complete SSML string for Azure TTS
        """
        # Get profile settings
        settings = PROSODY_PROFILES[profile]

        # Apply overrides if provided
        final_pitch = pitch if pitch is not None else settings.pitch
        final_rate = rate if rate is not None else settings.rate
        final_volume = volume if volume is not None else settings.volume
        final_style = style if style is not None else settings.style

        # Build the inner content with prosody
        prosody_content = self._build_prosody(text, final_pitch, final_rate, final_volume)

        # Wrap with express-as if style is specified
        if final_style:
            inner_content = self._wrap_express_as(prosody_content, final_style)
        else:
            inner_content = prosody_content

        # Build complete SSML
        return self._build_ssml(inner_content)

    def build_from_llm_response(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> str:
        """
        Build SSML from LLM response parameters.

        This method is designed to work directly with the JSON output
        from the Groq LLM (reply, style, pitch, rate).

        Args:
            text: The reply text from LLM
            style: Style from LLM (e.g., "empathetic", "cheerful")
            pitch: Pitch from LLM (e.g., "-5%")
            rate: Rate from LLM (e.g., "0.85")

        Returns:
            Complete SSML string for Azure TTS
        """
        # Map LLM style to profile if possible
        profile = self._style_to_profile(style)

        return self.build(
            text=text,
            profile=profile,
            pitch=pitch,
            rate=rate,
            style=style
        )

    def _style_to_profile(self, style: Optional[str]) -> ProsodyProfile:
        """Map LLM style string to prosody profile."""
        if not style:
            return ProsodyProfile.NEUTRAL

        style_lower = style.lower()
        style_mapping = {
            "empathetic": ProsodyProfile.EMPATHETIC,
            "empathy": ProsodyProfile.EMPATHETIC,
            "patient": ProsodyProfile.PATIENT,
            "cheerful": ProsodyProfile.CHEERFUL,
            "happy": ProsodyProfile.CHEERFUL,
            "calm": ProsodyProfile.DE_ESCALATE,
            "de-escalate": ProsodyProfile.DE_ESCALATE,
            "de_escalate": ProsodyProfile.DE_ESCALATE,
        }

        return style_mapping.get(style_lower, ProsodyProfile.NEUTRAL)

    def _build_prosody(
        self,
        text: str,
        pitch: str,
        rate: str,
        volume: str
    ) -> str:
        """Build prosody tag with settings."""
        attrs = []

        if pitch != "default":
            attrs.append(f'pitch="{pitch}"')
        if rate != "default" and rate != "1.0":
            attrs.append(f'rate="{rate}"')
        if volume != "default":
            attrs.append(f'volume="{volume}"')

        if attrs:
            return f'<prosody {" ".join(attrs)}>{text}</prosody>'
        return text

    def _wrap_express_as(self, content: str, style: str) -> str:
        """Wrap content with Azure express-as tag."""
        return f'<mstts:express-as style="{style}">{content}</mstts:express-as>'

    def _build_ssml(self, content: str) -> str:
        """Build complete SSML document."""
        return f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts"
       xml:lang="en-US">
    <voice name="{self.voice}">
        {content}
    </voice>
</speak>'''
