"""
SSML Builder for emotional prosody control.

Generates SSML markup with prosody profiles for Azure Neural TTS.
Supports express-as styles, prosody control, and emphasis tags.
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


class EmphasisLevel(Enum):
    """SSML emphasis levels for word stress."""

    STRONG = "strong"      # Loudest/most stressed
    MODERATE = "moderate"  # Default emphasis
    REDUCED = "reduced"    # Quieter/less stressed
    NONE = "none"          # No emphasis (explicit)


# Azure Neural Voice express-as styles
# See: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
AZURE_EXPRESS_STYLES = {
    "advertisement_upbeat",
    "affectionate",
    "angry",
    "assistant",
    "calm",
    "chat",
    "cheerful",
    "customerservice",
    "depressed",
    "disgruntled",
    "documentary-narration",
    "embarrassed",
    "empathetic",
    "envious",
    "excited",
    "fearful",
    "friendly",
    "gentle",
    "hopeful",
    "lyrical",
    "narration-professional",
    "narration-relaxed",
    "newscast",
    "newscast-casual",
    "newscast-formal",
    "poetry-reading",
    "sad",
    "serious",
    "shouting",
    "sports_commentary",
    "sports_commentary_excited",
    "whispering",
    "terrified",
    "unfriendly",
}


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

    # Default Azure Neural voices
    DEFAULT_FEMALE_VOICE = "en-US-JennyNeural"
    DEFAULT_MALE_VOICE = "en-US-GuyNeural"

    def __init__(self, voice: Optional[str] = None, voice_gender: str = "female"):
        """
        Initialize SSML builder.

        Args:
            voice: Azure Neural voice name (e.g., "en-US-JennyNeural"). If provided, overrides voice_gender.
            voice_gender: "male" or "female". Used if voice is not provided.
        """
        if voice:
            self.voice = voice
        else:
            self.voice = self.DEFAULT_MALE_VOICE if voice_gender == "male" else self.DEFAULT_FEMALE_VOICE

    def build(
        self,
        text: str,
        profile: ProsodyProfile = ProsodyProfile.NEUTRAL,
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        style: Optional[str] = None,
        styledegree: Optional[float] = None
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
            styledegree: Style intensity (0.01-2.0, default 1.0)

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
            inner_content = self._wrap_express_as(
                prosody_content,
                final_style,
                styledegree=styledegree
            )
        else:
            inner_content = prosody_content

        # Build complete SSML
        return self._build_ssml(inner_content)

    def build_from_llm_response(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        styledegree: Optional[float] = None
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
            styledegree: Style intensity (0.01-2.0, default 1.0)

        Returns:
            Complete SSML string for Azure TTS
        """
        # Map LLM style to profile if possible
        profile = self._style_to_profile(style)

        # Auto-calculate styledegree based on emotion intensity if not provided
        if styledegree is None:
            styledegree = self._get_default_styledegree(style)

        return self.build(
            text=text,
            profile=profile,
            pitch=pitch,
            rate=rate,
            style=style,
            styledegree=styledegree
        )

    def _get_default_styledegree(self, style: Optional[str]) -> Optional[float]:
        """
        Get default styledegree based on emotion/style.

        Higher intensity for strong emotions, lower for calm styles.

        Args:
            style: The emotion/style label

        Returns:
            Default styledegree value or None for default
        """
        if not style:
            return None

        style_lower = style.lower()

        # Strong emotions get higher intensity
        intensity_mapping = {
            "angry": 1.5,
            "excited": 1.4,
            "cheerful": 1.2,
            "happy": 1.2,
            "frustrated": 1.3,
            "empathetic": 1.1,
            "sad": 1.2,
            "calm": 0.8,
            "patient": 0.9,
            "de_escalate": 0.7,
            "gentle": 0.8,
            "neutral": 1.0,
            "friendly": 1.0,
        }

        return intensity_mapping.get(style_lower, 1.0)

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

    def _wrap_express_as(
        self,
        content: str,
        style: str,
        styledegree: Optional[float] = None
    ) -> str:
        """
        Wrap content with Azure express-as tag.

        Args:
            content: The SSML content to wrap
            style: Azure express-as style (e.g., "empathetic", "cheerful")
            styledegree: Style intensity from 0.01 to 2.0 (1.0 = default)
                        Values > 1.0 intensify, < 1.0 soften the style

        Returns:
            SSML string with express-as wrapper
        """
        # Validate and map style to Azure-compatible style
        azure_style = self._map_to_azure_style(style)

        # Build attributes
        attrs = f'style="{azure_style}"'

        if styledegree is not None:
            # Clamp styledegree to valid range (0.01 to 2.0)
            degree = max(0.01, min(2.0, styledegree))
            attrs += f' styledegree="{degree:.2f}"'

        return f'<mstts:express-as {attrs}>{content}</mstts:express-as>'

    def _map_to_azure_style(self, style: str) -> str:
        """
        Map emotion/style labels to Azure express-as styles.

        Args:
            style: Input style (from LLM or internal)

        Returns:
            Valid Azure express-as style name
        """
        if not style:
            return "friendly"

        style_lower = style.lower().replace("-", "_").replace(" ", "_")

        # Direct match with Azure styles
        if style_lower in AZURE_EXPRESS_STYLES:
            return style_lower

        # Map common emotion labels to Azure styles
        emotion_mapping = {
            "empathetic": "empathetic",
            "empathy": "empathetic",
            "patient": "calm",
            "cheerful": "cheerful",
            "happy": "cheerful",
            "excited": "excited",
            "calm": "calm",
            "de_escalate": "calm",
            "frustrated": "disgruntled",
            "angry": "angry",
            "sad": "sad",
            "confused": "empathetic",
            "neutral": "friendly",
            "professional": "narration_professional",
            "casual": "chat",
            "friendly": "friendly",
            "gentle": "gentle",
            "serious": "serious",
            "hopeful": "hopeful",
        }

        return emotion_mapping.get(style_lower, "friendly")

    def _build_ssml(self, content: str) -> str:
        """Build complete SSML document."""
        return f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="https://www.w3.org/2001/mstts"
       xml:lang="en-US">
    <voice name="{self.voice}">
        {content}
    </voice>
</speak>'''

    def add_emphasis(
        self,
        text: str,
        words: list[str],
        level: EmphasisLevel = EmphasisLevel.MODERATE
    ) -> str:
        """
        Add emphasis tags to specific words in text.

        Args:
            text: The input text
            words: List of words to emphasize
            level: Emphasis level (strong, moderate, reduced)

        Returns:
            Text with emphasis SSML tags around specified words
        """
        if not words:
            return text

        result = text
        for word in words:
            # Case-insensitive replacement, preserving original case
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)

            def replace_with_emphasis(match):
                original = match.group(0)
                return f'<emphasis level="{level.value}">{original}</emphasis>'

            result = pattern.sub(replace_with_emphasis, result)

        return result

    def apply_emphasis_markers(self, text: str) -> str:
        """
        Convert emphasis markers in text to SSML emphasis tags.

        Supports markers:
        - *word* -> strong emphasis
        - _word_ -> moderate emphasis
        - ~word~ -> reduced emphasis

        Args:
            text: Text with emphasis markers

        Returns:
            Text with SSML emphasis tags
        """
        import re

        # Strong emphasis: *word*
        text = re.sub(
            r'\*([^*]+)\*',
            r'<emphasis level="strong">\1</emphasis>',
            text
        )

        # Moderate emphasis: _word_
        text = re.sub(
            r'_([^_]+)_',
            r'<emphasis level="moderate">\1</emphasis>',
            text
        )

        # Reduced emphasis: ~word~
        text = re.sub(
            r'~([^~]+)~',
            r'<emphasis level="reduced">\1</emphasis>',
            text
        )

        return text

    def build_with_emphasis(
        self,
        text: str,
        emphasis_words: Optional[list[str]] = None,
        emphasis_level: EmphasisLevel = EmphasisLevel.MODERATE,
        profile: ProsodyProfile = ProsodyProfile.NEUTRAL,
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        style: Optional[str] = None,
        styledegree: Optional[float] = None
    ) -> str:
        """
        Build SSML with emphasis on specific words.

        Args:
            text: The text to synthesize
            emphasis_words: List of words to emphasize
            emphasis_level: Level of emphasis for the words
            profile: Pre-defined prosody profile
            pitch: Override pitch
            rate: Override rate
            volume: Override volume
            style: Override Azure express-as style
            styledegree: Style intensity

        Returns:
            Complete SSML string with emphasis and prosody
        """
        # First apply emphasis to the text
        if emphasis_words:
            text = self.add_emphasis(text, emphasis_words, emphasis_level)

        # Also process any inline markers
        text = self.apply_emphasis_markers(text)

        # Now build the full SSML (emphasis tags are inside the text)
        return self.build(
            text=text,
            profile=profile,
            pitch=pitch,
            rate=rate,
            volume=volume,
            style=style,
            styledegree=styledegree
        )
