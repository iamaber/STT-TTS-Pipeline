import numpy as np
from datetime import datetime
from scipy import signal
from app.config import settings
from app.models.vad import SileroVAD


# Audio constants
SR = 16000
MIN_SPEECH_SAMPLES = SR * 3  # 3 seconds minimum buffer
MAX_BUFFER_SAMPLES = SR * 5  # 5 seconds maximum buffer

# VAD singleton
_vad = None

def get_vad():
    global _vad
    if _vad is None:
        _vad = SileroVAD(
            threshold=0.7,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            sample_rate=SR
        )
    return _vad


class StreamingSession:

    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.consecutive_silence = 0
        self.silence_trigger_count = 4

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.is_speaking = False
        self.consecutive_silence = 0

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        y = audio_data
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sample_rate != SR:
            y = signal.resample_poly(y, SR, sample_rate)
        y = y.astype(np.float32)
        if np.abs(y).max() > 0:
            y /= np.abs(y).max() + 1e-9
        return y

    def add_audio(self, audio_data: np.ndarray, sample_rate: int) -> tuple:
        processed = self.preprocess_audio(audio_data, sample_rate)
        self.audio_buffer = np.concatenate([self.audio_buffer, processed])
        
        vad = get_vad()
        has_speech = vad.is_speech(processed)
        
        if has_speech:
            self.is_speaking = True
            self.consecutive_silence = 0
        else:
            if self.is_speaking:
                self.consecutive_silence += 1
                if self.consecutive_silence >= self.silence_trigger_count:
                    if len(self.audio_buffer) >= MIN_SPEECH_SAMPLES:
                        chunk = self.audio_buffer.copy()
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.is_speaking = False
                        self.consecutive_silence = 0
                        return True, chunk
        
        if len(self.audio_buffer) >= MAX_BUFFER_SAMPLES:
            chunk = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
            self.is_speaking = False
            self.consecutive_silence = 0
            return True, chunk
        
        return False, None

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

    # Add audio and check for silence trigger
    should_process, chunk = session.add_audio(audio_data, sample_rate)

    if not should_process or chunk is None:
        return session.get_transcripts() or "Listening...", None, None

    # Process STT
    transcription = pipeline.process_audio_to_text(chunk, SR)

    if not transcription or len(transcription.strip()) <= 3:
        return session.get_transcripts() or "Listening...", None, None

    # Add transcript
    session.add_transcript(transcription)

    # Generate TTS
    tts_audio = pipeline.process_text_to_audio(transcription, speaker_id)

    return session.get_transcripts(), tts_audio, settings.tts.sample_rate
