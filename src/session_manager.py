import logging
from datetime import datetime
from typing import Optional
from .memory import RedisClient, VectorStore
from .nlp import SentimentAnalyzer

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages session-specific conversation history, context, and preferences.
    """
    
    def __init__(self, session_id: str, redis_client: RedisClient, vector_store: VectorStore, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        self.session_id = session_id
        self.redis_client = redis_client
        self.vector_store = vector_store
        self.sentiment_analyzer = sentiment_analyzer

    def _get_external_context(self) -> str:
        """Get current date/time context."""
        now = datetime.now()
        day_name = now.strftime("%A")
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M %p")
        
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
            
        return f"Current time: {time_str} ({time_of_day}), {day_name}, {date_str}"

    def prepare_context(self, user_input: str) -> tuple[str, bool]:
        """
        Processes user input, updates session state, and prepares the conversation context.
        Returns (context_string, is_repetition).
        """
        # Check for repetition
        repetition_result = self.vector_store.check_repetition(self.session_id, user_input)
        is_repetition = repetition_result.is_repetition
        
        # Analyze sentiment
        detected_emotion = None
        if self.sentiment_analyzer:
            detected_emotion = self.sentiment_analyzer.get_emotion_for_context(user_input)
            
        # Update context labels
        self.redis_client.update_context_labels(
            self.session_id,
            is_repetition=is_repetition,
            detected_emotion=detected_emotion
        )
        
        # Detect and store preferences
        detected_prefs = self.redis_client.detect_preferences_from_message(user_input)
        if detected_prefs:
            self.redis_client.set_user_preferences(self.session_id, detected_prefs)
            
        # Add user message to history
        self.redis_client.add_message(self.session_id, "user", user_input)
        
        # Build context string
        context = self.redis_client.get_context_string(self.session_id)
        external_context = self._get_external_context()
        context = f"[{external_context}]\n\n{context}"
        
        # Add hints
        context_hint = self.redis_client.get_context_hint(self.session_id)
        if context_hint:
            context = f"{context}\n\n[Context: {context_hint}]"
            
        prefs_hint = self.redis_client.get_preferences_hint(self.session_id)
        if prefs_hint:
            context = f"{context}\n\n[User Preferences: {prefs_hint}]"
            
        return context, is_repetition

    def add_assistant_response(self, response_text: str):
        """Adds assistant response to conversation history."""
        self.redis_client.add_message(self.session_id, "assistant", response_text)

    def get_prosody(self, style: str) -> dict:
        """Gets prosody parameters for a style."""
        return self.redis_client.get_prosody(style or "neutral")
