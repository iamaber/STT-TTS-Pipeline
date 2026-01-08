import numpy as np
from datetime import datetime

from app.config import settings
from app.core.vad import SileroVAD

# Filler tokens considered as "thinking" sounds
FILLER_WORDS = {
    "hmm",
    "mm",
    "mhm",
    "hm",
    "uh",
    "um",
    "erm",
    "ah",
    "eh",
    "uhh",
    "umm",
    "mmm",
}


def is_filler_only(text: str) -> bool:
    """Return True if all tokens in text are filler tokens."""
    if not text:
        return False
    words = [w for w in text.lower().strip().split() if w]
    return bool(words) and all(w in FILLER_WORDS for w in words)


# VAD singleton
_vad = None


def get_vad() -> SileroVAD:
    """Get or create VAD instance (singleton pattern)"""
    global _vad
    if _vad is None:
        _vad = SileroVAD(
            threshold=settings.vad.threshold,
            min_speech_duration_ms=settings.vad.min_speech_duration_ms,
            min_silence_duration_ms=settings.vad.min_silence_duration_ms,
            sample_rate=settings.streaming.sample_rate,
        )
    return _vad


class StreamingSession:
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.consecutive_silence = 0
        self.silence_trigger_count = settings.streaming.silence_trigger_count

    def reset(self):
        """Reset session state"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.consecutive_silence = 0

    def should_check_transcription(self) -> bool:
        """
        Check if we should transcribe the buffer.
        Triggers on natural pause (silence) detection.
        """
        return self.consecutive_silence >= self.silence_trigger_count

    def should_force_process(self) -> bool:
        """Force processing if buffer exceeds maximum duration (safety fallback)"""
        buffer_duration = len(self.audio_buffer) / settings.streaming.sample_rate
        return buffer_duration >= settings.streaming.max_buffer_seconds

    def add_audio(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> tuple[bool, np.ndarray | None, str]:
        """
        Add audio chunk and determine if we should check for sentence completion.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate of audio

        Returns:
            Tuple of (should_check, audio_chunk, reason)
        """
        vad = get_vad()

        # Resample if needed
        if sample_rate != settings.streaming.sample_rate:
            from scipy import signal as scipy_signal

            num_samples = int(
                len(audio_data) * settings.streaming.sample_rate / sample_rate
            )
            audio_data = scipy_signal.resample(audio_data, num_samples)

        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

        # Check for speech
        is_speech = vad.is_speech(audio_data)

        if is_speech:
            self.is_speaking = True
            self.consecutive_silence = 0
        else:
            if self.is_speaking:
                self.consecutive_silence += 1

        # Check if we should transcribe to detect sentence completion
        if self.should_check_transcription():
            return True, self.audio_buffer.copy(), "pause_detected"

        # Safety: Force process if buffer too long
        if self.should_force_process():
            chunk = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
            self.is_speaking = False
            self.consecutive_silence = 0
            return True, chunk, "max_buffer_reached"

        # Continue buffering
        return False, None, "buffering"

    def clear_buffer(self):
        """Clear the audio buffer after successful processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.consecutive_silence = 0

    def add_transcript(self, text: str, timestamp: str = None):
        """Add transcription with timestamp"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcripts.append(f"[{timestamp}] {text}")

    def get_transcripts(self) -> str:
        """Get all transcripts as formatted string"""
        return "\n".join(self.transcripts) if self.transcripts else ""


def process_streaming_audio(
    session: StreamingSession,
    pipeline,
    audio_data: np.ndarray,
    sample_rate: int,
    speaker_id: int = None,
) -> tuple[str, np.ndarray | None, int | None, bool]:
    if speaker_id is None:
        speaker_id = settings.tts.default_speaker_id

    # Add audio and check if we should transcribe
    # check_type is "transcribe", "speech_start", or None
    should_check, chunk, reason = session.add_audio(audio_data, sample_rate)

    # Check if speech just started (interruption)
    is_speech_start = reason == "speech_start"

    # If nothing to check or no audio chunk, return early
    if not should_check or chunk is None:
        return session.get_transcripts() or "Listening...", None, None, is_speech_start

    # Transcribe the accumulated audio
    transcription = pipeline.process_audio_to_text(
        chunk, settings.streaming.sample_rate
    )

    # Check if transcription is valid (not empty)
    if not transcription or not transcription.strip():
        session.clear_buffer()
        return session.get_transcripts() or "Listening...", None, None, is_speech_start

    # Check if sentence is complete (has ending punctuation)
    text = transcription.strip()

    # If only filler words, keep buffering and keep ASR active
    # Do not finalize or clear the buffer; avoid adding transcript yet
    if is_filler_only(text) and reason != "max_buffer_reached":
        session.consecutive_silence = 0
        session.is_speaking = True
        return session.get_transcripts() or "Listening...", None, None, is_speech_start

    has_sentence_ending = text.endswith((".", "!", "?"))

    # Check if we have any valid text (even single words)
    word_count = len(text.split())
    has_valid_text = word_count >= 1

    # Process if: has punctuation OR has valid text OR hit max buffer
    should_process = (
        has_sentence_ending or has_valid_text or reason == "max_buffer_reached"
    )

    if not should_process:
        session.consecutive_silence = 0
        return session.get_transcripts() or "Listening...", None, None, is_speech_start

    # Process complete sentence
    session.add_transcript(text)
    session.clear_buffer()

    # Generate TTS
    output_audio = None
    output_sr = None

    try:
        if pipeline:
            output_audio = pipeline.process_text_to_audio(text, speaker_id)
            output_sr = settings.tts.sample_rate
    except Exception as e:
        print(f"TTS generation failed: {e}")

    return session.get_transcripts(), output_audio, output_sr, is_speech_start
