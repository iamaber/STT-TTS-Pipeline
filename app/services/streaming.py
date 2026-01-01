import numpy as np
from datetime import datetime
from scipy import signal
from app.config import settings


# Audio constants
SR = 16000  # Target sample rate in Hz
CHUNK_SECONDS = 3  # Process audio every N seconds
ENERGY_THRESHOLD = 0.05  # Minimum energy to consider as speech


class StreamingSession:
    # Manages streaming audio buffer and transcription state

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.silence_chunks = 0
        self.silence_threshold_chunks = 3

    def reset(self):
        # Clear all session state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.silence_chunks = 0

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # Convert audio to 16kHz mono float32
        y = audio_data

        # Convert stereo to mono
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Resample to target sample rate
        if sample_rate != SR:
            y = signal.resample_poly(y, SR, sample_rate)

        # Normalize to float32
        y = y.astype(np.float32)
        if np.abs(y).max() > 0:
            y /= np.abs(y).max() + 1e-9

        return y

    def add_audio(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        # Add audio to buffer, return True when ready to process
        processed = self.preprocess_audio(audio_data, sample_rate)
        self.audio_buffer = np.concatenate([self.audio_buffer, processed])
        return len(self.audio_buffer) >= SR * CHUNK_SECONDS

    def get_chunk(self) -> np.ndarray:
        # Extract next chunk for processing
        if len(self.audio_buffer) < SR * CHUNK_SECONDS:
            return None

        chunk = self.audio_buffer[: SR * CHUNK_SECONDS]
        self.audio_buffer = self.audio_buffer[SR * CHUNK_SECONDS :]
        return chunk

    def has_speech(self, chunk: np.ndarray) -> bool:
        # Check if chunk contains speech based on energy level
        energy = np.sqrt(np.mean(chunk**2))
        return energy > ENERGY_THRESHOLD

    def add_transcript(self, text: str, timestamp: str = None):
        # Add transcript with timestamp
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcripts.append(f"[{timestamp}] {text}")

    def get_transcripts(self) -> str:
        # Return all transcripts as formatted string
        return "\n".join(self.transcripts) if self.transcripts else ""


def process_streaming_audio(
    session: StreamingSession,
    pipeline,
    audio_data: np.ndarray,
    sample_rate: int,
    speaker_id: int = None,
) -> tuple:
    # Process streaming audio chunk through STT-TTS pipeline
    # Returns: (transcription_text, tts_audio, tts_sample_rate)

    # Use default speaker from config if not specified
    if speaker_id is None:
        speaker_id = settings.tts.default_speaker_id

    # Add audio to buffer
    ready = session.add_audio(audio_data, sample_rate)

    if not ready:
        return session.get_transcripts() or "Listening...", None, None

    # Get chunk for processing
    chunk = session.get_chunk()
    if chunk is None:
        return session.get_transcripts() or "Listening...", None, None

    # Skip if no speech detected
    if not session.has_speech(chunk):
        return session.get_transcripts() or "Listening...", None, None

    # Process through STT
    transcription = pipeline.process_audio_to_text(chunk, SR)

    # Skip short or empty transcriptions
    if not transcription or len(transcription.strip()) <= 3:
        return session.get_transcripts() or "Listening...", None, None

    # Add transcript to session
    session.add_transcript(transcription)

    # Generate TTS audio
    tts_audio = pipeline.process_text_to_audio(transcription, speaker_id)

    return session.get_transcripts(), tts_audio, settings.tts.sample_rate
