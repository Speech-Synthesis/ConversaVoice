"""
Voice Analyzer for the Simulation System.

Extracts acoustic features from trainee audio to detect:
- Voice emotion (calm, stressed, confident, hesitant, empathetic)
- Delivery quality scores (calmness, confidence, empathy, pace, clarity)

Uses librosa for audio feature extraction.
"""

import io
import logging
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Flag to track if librosa is available
LIBROSA_AVAILABLE = False
try:
    import librosa
    import librosa.feature
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("librosa not installed. Voice analysis will use fallback mode.")


class VoiceEmotion(Enum):
    """Detected voice emotions."""
    CALM = "calm"
    STRESSED = "stressed"
    CONFIDENT = "confident"
    HESITANT = "hesitant"
    EMPATHETIC = "empathetic"
    RUSHED = "rushed"
    MONOTONE = "monotone"
    EXCITED = "excited"
    HAPPY = "happy"
    NERVOUS = "nervous"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class AcousticFeatures:
    """Extracted acoustic features from audio."""
    # Pitch features
    pitch_mean: float = 0.0          # Average pitch in Hz
    pitch_std: float = 0.0           # Pitch standard deviation
    pitch_range: float = 0.0         # Max - min pitch
    pitch_contour: str = "flat"      # rising/falling/flat

    # Energy features
    energy_mean: float = 0.0         # RMS energy
    energy_std: float = 0.0          # Energy variation
    energy_trend: str = "stable"     # rising/falling/stable

    # Tempo features
    speaking_rate: float = 0.0       # Estimated words per minute
    pause_ratio: float = 0.0         # Ratio of silence to speech
    rhythm_regularity: float = 0.0   # How consistent the rhythm is

    # Spectral features
    spectral_centroid: float = 0.0   # Brightness of sound
    spectral_rolloff: float = 0.0    # Frequency below which 85% of energy exists
    zero_crossing_rate: float = 0.0  # Rate of sign changes

    # Duration
    duration_seconds: float = 0.0


@dataclass
class DeliveryScores:
    """Voice delivery quality scores (1-10)."""
    calmness: int = 5          # Voice stability, no stress indicators
    confidence: int = 5        # Steady pitch, clear endings
    empathy: int = 5           # Warm tone, appropriate pace
    pace: int = 5              # Speaking rate quality
    clarity: int = 5           # Articulation quality
    overall: int = 5           # Weighted average
    
    # Advanced metrics
    tone_score: float = 5.0
    speaking_style_score: float = 5.0
    final_communication_score: float = 5.0


@dataclass
class VoiceAnalysis:
    """Complete voice analysis result."""
    features: AcousticFeatures
    primary_emotion: VoiceEmotion = VoiceEmotion.UNKNOWN
    secondary_emotion: Optional[VoiceEmotion] = None
    emotion_confidence: float = 0.0
    delivery_scores: DeliveryScores = field(default_factory=DeliveryScores)
    analysis_success: bool = False
    error_message: Optional[str] = None


# Emotion classification thresholds
EMOTION_PROFILES = {
    VoiceEmotion.CALM: {
        "pitch_std": (5, 25),           # Low variance
        "energy_std": (0.005, 0.025),   # Steady energy
        "speaking_rate": (100, 150),    # Moderate pace
        "pause_ratio": (0.1, 0.25),     # Good pauses
    },
    VoiceEmotion.STRESSED: {
        "pitch_std": (30, 80),          # High variance
        "energy_std": (0.03, 0.1),      # Erratic energy
        "speaking_rate": (155, 220),    # Fast pace
        "pause_ratio": (0.0, 0.1),      # Few pauses
    },
    VoiceEmotion.CONFIDENT: {
        "pitch_std": (15, 35),          # Moderate variance
        "energy_mean": (0.02, 0.08),    # Good energy
        "pause_ratio": (0.05, 0.2),     # Purposeful pauses
        "pitch_contour": "falling",     # Declarative
    },
    VoiceEmotion.HESITANT: {
        "pause_ratio": (0.25, 0.6),     # Many pauses
        "speaking_rate": (60, 100),     # Slow pace
        "energy_trend": "falling",      # Losing energy
    },
    VoiceEmotion.EMPATHETIC: {
        "pitch_std": (20, 40),          # Expressive
        "energy_mean": (0.01, 0.04),    # Softer energy
        "speaking_rate": (90, 130),     # Slower, caring pace
    },
    VoiceEmotion.RUSHED: {
        "speaking_rate": (170, 250),    # Very fast
        "pause_ratio": (0.0, 0.08),     # Almost no pauses
    },
    VoiceEmotion.MONOTONE: {
        "pitch_std": (0, 10),           # No variation
        "energy_std": (0, 0.01),        # Flat energy
    },
    VoiceEmotion.EXCITED: {
        "pitch_std": (35, 70),          # High variance and dynamic
        "energy_mean": (0.05, 0.15),    # High energy
        "speaking_rate": (140, 190),    # Fast pace
        "pitch_contour": "rising",      # Excitement
    },
    VoiceEmotion.HAPPY: {
        "pitch_std": (25, 45),          # Positive variation
        "energy_mean": (0.03, 0.09),    # Moderate-high energy
        "speaking_rate": (120, 160),    # Upbeat pace
    },
    VoiceEmotion.NERVOUS: {
        "pitch_std": (40, 90),          # Jittery pitch
        "pause_ratio": (0.15, 0.4),     # Lots of awkward pauses
        "rhythm_regularity": (0.0, 0.4) # Uneven rhythm
    },
    VoiceEmotion.NEUTRAL: {
        "pitch_std": (10, 25),          # Average variation
        "energy_std": (0.01, 0.03),     # Steady energy
        "speaking_rate": (110, 140),    # Normal pace
        "pause_ratio": (0.05, 0.2),     # Normal pauses
    },
}

# Ideal customer service voice profile
IDEAL_PROFILE = {
    "pitch_std": 22,           # Expressive but controlled
    "speaking_rate": 135,      # Clear, not rushed
    "pause_ratio": 0.15,       # Natural pauses
    "energy_stability": 0.85,  # Consistent
}


class VoiceAnalyzer:
    """
    Analyzes trainee voice recordings for emotion and delivery quality.

    Uses librosa for acoustic feature extraction when available,
    falls back to basic analysis otherwise.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize voice analyzer.

        Args:
            sample_rate: Expected audio sample rate.
        """
        self.sample_rate = sample_rate
        self.librosa_available = LIBROSA_AVAILABLE

    def analyze(self, audio_data: bytes) -> VoiceAnalysis:
        """
        Analyze audio data and return voice analysis.

        Args:
            audio_data: Raw audio bytes (WAV format expected).

        Returns:
            VoiceAnalysis with features, emotion, and delivery scores.
        """
        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_array(audio_data)

            if audio_array is None or len(audio_array) == 0:
                return VoiceAnalysis(
                    features=AcousticFeatures(),
                    error_message="Empty or invalid audio data"
                )

            # Extract features
            features = self._extract_features(audio_array)

            # Classify emotion
            primary_emotion, secondary_emotion, confidence = self._classify_emotion(features)

            # Score delivery (pass confidence for final grade computing)
            delivery_scores = self._score_delivery(features, confidence)

            return VoiceAnalysis(
                features=features,
                primary_emotion=primary_emotion,
                secondary_emotion=secondary_emotion,
                emotion_confidence=confidence,
                delivery_scores=delivery_scores,
                analysis_success=True,
            )

        except Exception as e:
            logger.error(f"Voice analysis failed: {e}")
            return VoiceAnalysis(
                features=AcousticFeatures(),
                error_message=str(e)
            )

    def analyze_file(self, file_path: str) -> VoiceAnalysis:
        """
        Analyze audio file.

        Args:
            file_path: Path to audio file.

        Returns:
            VoiceAnalysis result.
        """
        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()
            return self.analyze(audio_data)
        except Exception as e:
            logger.error(f"Failed to read audio file: {e}")
            return VoiceAnalysis(
                features=AcousticFeatures(),
                error_message=f"Failed to read file: {e}"
            )

    def _bytes_to_array(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Convert audio bytes to numpy array."""
        try:
            if self.librosa_available:
                # Use librosa to load from bytes
                audio_array, sr = librosa.load(
                    io.BytesIO(audio_data),
                    sr=self.sample_rate,
                    mono=True
                )
                return audio_array
            else:
                # Fallback: try to parse WAV header and extract data
                return self._parse_wav_bytes(audio_data)
        except Exception as e:
            logger.error(f"Failed to convert audio bytes: {e}")
            return None

    def _parse_wav_bytes(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Parse WAV bytes without librosa."""
        try:
            import wave
            import struct

            with io.BytesIO(audio_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    n_frames = wav_file.getnframes()
                    frames = wav_file.readframes(n_frames)

                    # Convert to numpy array
                    if sample_width == 2:
                        dtype = np.int16
                    elif sample_width == 4:
                        dtype = np.int32
                    else:
                        dtype = np.int16

                    audio_array = np.frombuffer(frames, dtype=dtype)

                    # Convert to mono if stereo
                    if n_channels == 2:
                        audio_array = audio_array[::2]

                    # Normalize to float [-1, 1]
                    audio_array = audio_array.astype(np.float32)
                    audio_array /= np.iinfo(dtype).max

                    return audio_array

        except Exception as e:
            logger.error(f"Failed to parse WAV: {e}")
            return None

    def _extract_features(self, audio: np.ndarray) -> AcousticFeatures:
        """Extract acoustic features from audio array."""
        features = AcousticFeatures()
        features.duration_seconds = len(audio) / self.sample_rate

        if self.librosa_available:
            features = self._extract_features_librosa(audio, features)
        else:
            features = self._extract_features_basic(audio, features)

        return features

    def _extract_features_librosa(
        self,
        audio: np.ndarray,
        features: AcousticFeatures
    ) -> AcousticFeatures:
        """Extract features using librosa."""
        try:
            # Pitch analysis using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            f0_valid = f0[~np.isnan(f0)]

            if len(f0_valid) > 0:
                features.pitch_mean = float(np.mean(f0_valid))
                features.pitch_std = float(np.std(f0_valid))
                features.pitch_range = float(np.max(f0_valid) - np.min(f0_valid))

                # Determine pitch contour
                if len(f0_valid) > 10:
                    first_half = np.mean(f0_valid[:len(f0_valid)//2])
                    second_half = np.mean(f0_valid[len(f0_valid)//2:])
                    if second_half > first_half * 1.05:
                        features.pitch_contour = "rising"
                    elif second_half < first_half * 0.95:
                        features.pitch_contour = "falling"
                    else:
                        features.pitch_contour = "flat"

            # Energy analysis
            rms = librosa.feature.rms(y=audio)[0]
            features.energy_mean = float(np.mean(rms))
            features.energy_std = float(np.std(rms))

            # Energy trend
            if len(rms) > 10:
                first_half = np.mean(rms[:len(rms)//2])
                second_half = np.mean(rms[len(rms)//2:])
                if second_half > first_half * 1.1:
                    features.energy_trend = "rising"
                elif second_half < first_half * 0.9:
                    features.energy_trend = "falling"
                else:
                    features.energy_trend = "stable"

            # Tempo/speaking rate estimation
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sample_rate)
            # Convert tempo (BPM) to rough speaking rate (WPM)
            # Assuming ~2 syllables per word, ~1 syllable per beat
            features.speaking_rate = float(tempo[0] / 2) if len(tempo) > 0 else 0

            # Pause ratio (silence detection)
            non_silent = librosa.effects.split(audio, top_db=30)
            speech_duration = sum(end - start for start, end in non_silent) / self.sample_rate
            total_duration = len(audio) / self.sample_rate
            features.pause_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0

            # Rhythm regularity (onset consistency)
            if len(onset_env) > 0:
                onset_times = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=self.sample_rate,
                    units='time'
                )
                if len(onset_times) > 2:
                    intervals = np.diff(onset_times)
                    features.rhythm_regularity = 1 - (np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            features.spectral_centroid = float(np.mean(spectral_centroids))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            features.spectral_rolloff = float(np.mean(spectral_rolloff))

            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.zero_crossing_rate = float(np.mean(zcr))

        except Exception as e:
            logger.error(f"Librosa feature extraction failed: {e}")

        return features

    def _extract_features_basic(
        self,
        audio: np.ndarray,
        features: AcousticFeatures
    ) -> AcousticFeatures:
        """Extract basic features without librosa."""
        try:
            # Basic energy calculation
            features.energy_mean = float(np.sqrt(np.mean(audio ** 2)))

            # Energy variation
            frame_size = int(0.025 * self.sample_rate)  # 25ms frames
            n_frames = len(audio) // frame_size
            if n_frames > 0:
                frame_energies = []
                for i in range(n_frames):
                    frame = audio[i * frame_size:(i + 1) * frame_size]
                    frame_energies.append(np.sqrt(np.mean(frame ** 2)))
                features.energy_std = float(np.std(frame_energies))

            # Zero crossing rate (rough pitch indicator)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
            features.zero_crossing_rate = zero_crossings / len(audio)

            # Rough speaking rate from energy envelope
            # Count energy peaks as syllables
            threshold = features.energy_mean * 1.5
            above_threshold = audio > threshold
            # Count transitions from below to above threshold
            transitions = np.sum(np.diff(above_threshold.astype(int)) == 1)
            syllables = transitions
            duration_minutes = features.duration_seconds / 60
            # Assume 2 syllables per word
            features.speaking_rate = (syllables / 2) / duration_minutes if duration_minutes > 0 else 0

            # Pause ratio (silence detection)
            silence_threshold = 0.01
            silent_samples = np.sum(np.abs(audio) < silence_threshold)
            features.pause_ratio = silent_samples / len(audio)

        except Exception as e:
            logger.error(f"Basic feature extraction failed: {e}")

        return features

    def _classify_emotion(
        self,
        features: AcousticFeatures
    ) -> Tuple[VoiceEmotion, Optional[VoiceEmotion], float]:
        """
        Classify voice emotion from features.

        Returns:
            Tuple of (primary_emotion, secondary_emotion, confidence).
        """
        scores: Dict[VoiceEmotion, float] = {}

        for emotion, profile in EMOTION_PROFILES.items():
            score = 0.0
            matches = 0
            total_criteria = len(profile)

            for feature_name, criteria in profile.items():
                feature_value = getattr(features, feature_name, None)

                if feature_value is None:
                    continue

                if isinstance(criteria, tuple):
                    # Range check
                    min_val, max_val = criteria
                    if min_val <= feature_value <= max_val:
                        score += 1.0
                        matches += 1
                    elif feature_value < min_val:
                        # Partial credit for being close
                        distance = (min_val - feature_value) / min_val if min_val > 0 else 1
                        score += max(0, 1 - distance)
                        matches += 1
                    elif feature_value > max_val:
                        distance = (feature_value - max_val) / max_val if max_val > 0 else 1
                        score += max(0, 1 - distance)
                        matches += 1
                elif isinstance(criteria, str):
                    # Exact match for string values
                    if feature_value == criteria:
                        score += 1.0
                        matches += 1

            # Normalize score
            if total_criteria > 0:
                scores[emotion] = score / total_criteria

        # Sort by score
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_emotions or sorted_emotions[0][1] < 0.3:
            return VoiceEmotion.UNKNOWN, None, 0.0

        primary = sorted_emotions[0]
        secondary = sorted_emotions[1] if len(sorted_emotions) > 1 and sorted_emotions[1][1] > 0.4 else None

        return (
            primary[0],
            secondary[0] if secondary else None,
            primary[1]
        )

    def _score_delivery(self, features: AcousticFeatures, emotion_confidence: float = 0.5) -> DeliveryScores:
        """Score voice delivery quality."""
        scores = DeliveryScores()

        # Calmness: Low pitch variance, steady energy
        pitch_stability = max(0, 10 - (features.pitch_std / 5))  # Lower is better
        energy_stability = max(0, 10 - (features.energy_std * 100))
        scores.calmness = int(min(10, (pitch_stability + energy_stability) / 2))

        # Confidence: Moderate energy, falling pitch contour, good pauses
        energy_score = 10 if 0.02 < features.energy_mean < 0.06 else 7
        contour_score = 10 if features.pitch_contour == "falling" else 7
        pause_score = 10 if 0.1 < features.pause_ratio < 0.2 else 6
        scores.confidence = int((energy_score + contour_score + pause_score) / 3)

        # Empathy: Softer energy, slower pace, some pitch variation
        soft_energy = 10 if features.energy_mean < 0.04 else 6
        slower_pace = 10 if 90 < features.speaking_rate < 140 else 6
        expressive = 10 if 15 < features.pitch_std < 35 else 6
        scores.empathy = int((soft_energy + slower_pace + expressive) / 3)

        # Pace: Speaking rate in ideal range
        ideal_rate = IDEAL_PROFILE["speaking_rate"]
        rate_diff = abs(features.speaking_rate - ideal_rate)
        scores.pace = int(max(1, 10 - (rate_diff / 15)))

        # Clarity: Higher spectral centroid, good zero crossing rate
        if features.spectral_centroid > 0:
            clarity_score = min(10, features.spectral_centroid / 200)
        else:
            clarity_score = 6  # Default if not available
        scores.clarity = int(clarity_score)

        # Overall (legacy integer base): Weighted average
        scores.overall = int(
            scores.calmness * 0.2 +
            scores.confidence * 0.25 +
            scores.empathy * 0.25 +
            scores.pace * 0.15 +
            scores.clarity * 0.15
        )

        # NEW 1. TONE SCORE (Pitch variation, speech energy, modulation, pace, pauses)
        tone_pitch_var = min(10.0, max(0.0, features.pitch_std / 4.0)) # ~20-30 std is good, maps 5-7.5
        tone_energy = min(10.0, max(0.0, features.energy_mean * 150)) # 0.05 energy = 7.5
        tone_modulation = 8.5 if features.pitch_contour in ["rising", "falling"] else 5.0
        tone_pace = max(0.0, 10.0 - (rate_diff / 10.0))
        tone_pauses = 10.0 if 0.1 < features.pause_ratio < 0.25 else 6.0
        scores.tone_score = round((tone_pitch_var * 0.25) + (tone_energy * 0.2) + (tone_modulation * 0.2) + (tone_pace * 0.2) + (tone_pauses * 0.15), 1)
        scores.tone_score = min(10.0, max(0.0, scores.tone_score))

        # NEW 3. SPEAKING STYLE SCORE (Fluency, clarity, filler words, pace, pronunciation)
        # We proxy fluency via rhythm regularity and pauses
        style_fluency = min(10.0, max(0.0, features.rhythm_regularity * 10.0)) 
        if style_fluency == 0: style_fluency = 7.0 # Default if librosa fails
        style_clarity = clarity_score
        # We proxy filler words via zero crossing rate inconsistency and extra pauses
        style_fillers = max(0.0, 10.0 - (features.pause_ratio * 15)) 
        style_pace = scores.pace
        style_pronunciation = min(10.0, max(0.0, (features.spectral_rolloff / 800))) if features.spectral_rolloff else 7.0
        scores.speaking_style_score = round((style_fluency * 0.25) + (style_clarity * 0.25) + (style_fillers * 0.2) + (style_pace * 0.15) + (style_pronunciation * 0.15), 1)
        scores.speaking_style_score = min(10.0, max(0.0, scores.speaking_style_score))

        # NEW 4. FINAL COMMUNICATION SCORE (Tone -> 30%, Emotion -> 30%, Speaking Style -> 40%)
        # Emotion score based on confidence scale 0-1 mapped to 0-10
        emotion_score = emotion_confidence * 10.0 if emotion_confidence > 0 else 7.5
        scores.final_communication_score = round(
            (scores.tone_score * 0.3) + 
            (emotion_score * 0.3) + 
            (scores.speaking_style_score * 0.4), 
            1
        )

        return scores


# Global analyzer instance
_voice_analyzer: Optional[VoiceAnalyzer] = None


def get_voice_analyzer() -> VoiceAnalyzer:
    """Get the global voice analyzer instance."""
    global _voice_analyzer
    if _voice_analyzer is None:
        _voice_analyzer = VoiceAnalyzer()
    return _voice_analyzer
