"""
Sentiment analysis for ConversaVoice.
Detects user emotions from text input.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Frustration indicators (keywords and phrases)
FRUSTRATION_KEYWORDS = [
    "again", "already said", "already told", "just tell me", "why is this",
    "so hard", "frustrated", "annoying", "annoyed", "ridiculous",
    "waste of time", "doesn't work", "not working", "broken", "useless",
    "terrible", "horrible", "hate", "stupid", "dumb", "impossible"
]

# Confusion indicators
CONFUSION_KEYWORDS = [
    "confused", "don't understand", "what do you mean", "unclear",
    "makes no sense", "i don't get", "explain", "help me understand",
    "lost", "confusing"
]

# Positive indicators
POSITIVE_KEYWORDS = [
    "thanks", "thank you", "great", "awesome", "perfect", "excellent",
    "helpful", "appreciate", "wonderful", "amazing", "good", "nice",
    "love", "happy", "glad", "pleased"
]

# Anger indicators
ANGER_KEYWORDS = [
    "angry", "furious", "outraged", "unacceptable", "demand",
    "speak to manager", "supervisor", "escalate", "complaint",
    "sue", "lawyer", "refund", "cancel"
]


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    emotion: str  # neutral, happy, confused, frustrated, angry
    confidence: float  # 0.0 to 1.0
    polarity: float  # -1.0 (negative) to 1.0 (positive)
    indicators: list  # Keywords that triggered the emotion


class SentimentAnalyzer:
    """
    Lightweight sentiment analyzer for emotion detection.

    Uses keyword matching and TextBlob for polarity analysis.
    Designed for real-time voice assistant use.
    """

    def __init__(self):
        """Initialize the sentiment analyzer."""
        self._textblob_available = False
        try:
            from textblob import TextBlob
            self._textblob_available = True
        except ImportError:
            logger.warning("TextBlob not installed. Using keyword-only analysis.")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment/emotion from text.

        Args:
            text: User input text to analyze.

        Returns:
            SentimentResult with detected emotion and confidence.
        """
        text_lower = text.lower()
        indicators = []

        # Check for anger (highest priority)
        anger_matches = self._find_keywords(text_lower, ANGER_KEYWORDS)
        if anger_matches:
            return SentimentResult(
                emotion="angry",
                confidence=min(0.9, 0.5 + len(anger_matches) * 0.2),
                polarity=-0.8,
                indicators=anger_matches
            )

        # Check for frustration
        frustration_matches = self._find_keywords(text_lower, FRUSTRATION_KEYWORDS)
        if frustration_matches:
            return SentimentResult(
                emotion="frustrated",
                confidence=min(0.9, 0.5 + len(frustration_matches) * 0.15),
                polarity=-0.5,
                indicators=frustration_matches
            )

        # Check for confusion
        confusion_matches = self._find_keywords(text_lower, CONFUSION_KEYWORDS)
        if confusion_matches:
            return SentimentResult(
                emotion="confused",
                confidence=min(0.85, 0.5 + len(confusion_matches) * 0.15),
                polarity=-0.2,
                indicators=confusion_matches
            )

        # Check for positive sentiment
        positive_matches = self._find_keywords(text_lower, POSITIVE_KEYWORDS)
        if positive_matches:
            return SentimentResult(
                emotion="happy",
                confidence=min(0.85, 0.5 + len(positive_matches) * 0.15),
                polarity=0.6,
                indicators=positive_matches
            )

        # Use TextBlob for general polarity if available
        polarity = 0.0
        if self._textblob_available:
            try:
                from textblob import TextBlob
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity

                if polarity < -0.3:
                    return SentimentResult(
                        emotion="frustrated",
                        confidence=0.6,
                        polarity=polarity,
                        indicators=["negative_polarity"]
                    )
                elif polarity > 0.3:
                    return SentimentResult(
                        emotion="happy",
                        confidence=0.6,
                        polarity=polarity,
                        indicators=["positive_polarity"]
                    )
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")

        # Default to neutral
        return SentimentResult(
            emotion="neutral",
            confidence=0.5,
            polarity=polarity,
            indicators=[]
        )

    def _find_keywords(self, text: str, keywords: list) -> list:
        """Find matching keywords in text."""
        matches = []
        for keyword in keywords:
            if keyword in text:
                matches.append(keyword)
        return matches

    def get_emotion_for_context(self, text: str) -> Optional[str]:
        """
        Get emotion label suitable for context labels.

        Args:
            text: User input text.

        Returns:
            Emotion string or None if neutral.
        """
        result = self.analyze(text)
        if result.emotion != "neutral" and result.confidence >= 0.5:
            return result.emotion
        return None
