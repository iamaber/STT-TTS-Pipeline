import numpy as np
from datetime import datetime
from scipy import signal
from app.config import settings
from app.models.vad import SileroVAD


# Audio constants
SR = 16000
MAX_BUFFER_SECONDS = 60 

# VAD singleton
_vad = None

def get_vad():
    global _vad
    if _vad is None:
        _vad = SileroVAD(
            threshold=0.5,  # Lower threshold to catch more speech
            min_speech_duration_ms=200,  # Shorter bursts OK
            min_silence_duration_ms=300,  # Longer to avoid cutting words
            sample_rate=SR
        )
    return _vad


class StreamingSession:
    """Streaming session with semantic sentence detection"""

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.consecutive_silence = 0
        self.silence_trigger_count = 3  # ~0.375s for stable detection

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.consecutive_silence = 0

    def should_check_transcription(self) -> bool:
        """
        Check if we should transcribe the buffer to see if sentence is complete.
        Triggers on natural pause (silence) detection.
        No minimum buffer needed - semantic detection handles sentence completion.
        """
        return self.consecutive_silence >= self.silence_trigger_count

    def should_force_process(self) -> bool:
        """
        Force processing if buffer is too long (safety fallback).
        """
        buffer_duration = len(self.audio_buffer) / SR
        return buffer_duration >= MAX_BUFFER_SECONDS

    def add_audio(self, audio_data: np.ndarray, sample_rate: int):
        """
        Add audio chunk and determine if we should check for sentence completion.
        
        Returns:
            (should_check, audio_chunk, reason)
        """
        vad = get_vad()
        
        # Resample if needed
        if sample_rate != SR:
            from scipy import signal as scipy_signal
            num_samples = int(len(audio_data) * SR / sample_rate)
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
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcripts.append(f"[{timestamp}] {text}")

    def get_transcripts(self) -> str:
        return "\n".join(self.transcripts) if self.transcripts else ""


def process_streaming_audio(
    session: StreamingSession,
    pipeline,
    audio_data: np.ndarray,
    sample_rate: int,
    speaker_id: int = None,
) -> tuple:


    if speaker_id is None:
        speaker_id = settings.tts.default_speaker_id

    # Add audio and check if we should transcribe
    should_check, chunk, reason = session.add_audio(audio_data, sample_rate)

    if not should_check or chunk is None:
        return session.get_transcripts() or "Listening...", None, None

    # Transcribe the accumulated audio
    transcription = pipeline.process_audio_to_text(chunk, SR)

    # Check if transcription is valid (not empty)
    if not transcription or not transcription.strip():
        # No valid speech detected, clear buffer
        session.clear_buffer()
        return session.get_transcripts() or "Listening...", None, None
    
    # Check if sentence is complete (has ending punctuation)
    text = transcription.strip()
    has_sentence_ending = text.endswith(('.', '!', '?'))
    
    # Also check if we have any valid text (even single words like "perfect", "yes", "no")
    word_count = len(text.split())
    has_valid_text = word_count >= 1  # Process any valid word(s) after a pause
    
    # Process if:
    # 1. Has sentence ending punctuation, OR
    # 2. Has any valid text (1+ words) after a pause, OR
    # 3. Hit max buffer
    should_process = has_sentence_ending or has_valid_text or reason == "max_buffer_reached"
    
    if not should_process:
        # Keep audio in buffer and wait for more
        # Reset silence counter to wait for next pause
        session.consecutive_silence = 0
        return session.get_transcripts() or "Listening...", None, None

    # We have a complete sentence or substantial text - process it!
    session.add_transcript(transcription)
    session.clear_buffer()

    # Generate TTS
    tts_audio = pipeline.process_text_to_audio(transcription, speaker_id)

    return session.get_transcripts(), tts_audio, settings.tts.sample_rate
