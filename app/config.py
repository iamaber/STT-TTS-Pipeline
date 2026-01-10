from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ASRConfig(BaseSettings):
    """ASR (Automatic Speech Recognition) configuration"""

    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    sample_rate: int = 16000
    chunk_size_ms: int = 100
    streaming: bool = True
    device: str = "cuda"


class TTSConfig(BaseSettings):
    """TTS (Text-to-Speech) configuration"""

    acoustic_model: str = "models/tts_acoustic/tts_en_fastpitch_multispeaker.nemo"
    vocoder_model: str = "models/tts_vocoder/tts_en_hifitts_hifigan_ft_fastpitch.nemo"
    sample_rate: int = 44100
    device: str = "cuda"
    default_speaker_id: int = 92
    max_text_length: int = 500
    min_audio_duration: float = 0.1  # Minimal padding


class VADConfig(BaseSettings):
    """VAD (Voice Activity Detection) configuration"""

    threshold: float = 0.6
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300


class StreamingConfig(BaseSettings):
    """Real-time streaming configuration"""

    sample_rate: int = 16000
    max_buffer_seconds: int = 30
    silence_trigger_count: int = 1


class LLMConfig(BaseSettings):
    """LLM configuration - Custom Streaming API"""

    # Custom LLM API settings
    api_url: str = "http://192.168.10.2:8000/api/chat/stream"
    max_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


class QueueConfig(BaseSettings):
    """Queue management configuration"""

    max_user_queue_size: int = 5
    max_tts_queue_size: int = 10


class Settings(BaseSettings):
    """Main application settings"""

    asr: ASRConfig = ASRConfig()
    tts: TTSConfig = TTSConfig()
    vad: VADConfig = VADConfig()
    streaming: StreamingConfig = StreamingConfig()
    llm: LLMConfig = LLMConfig()
    queue: QueueConfig = QueueConfig()

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False  # Enable/disable debug logging


settings = Settings()


# API Request Models
class STTTTSRequest(BaseModel):
    """Request model for full STT-TTS pipeline"""

    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    speaker: int | None = Field(None, description="TTS speaker ID (0-12799)")


class StreamingRequest(BaseModel):
    """Request model for streaming audio processing"""

    session_id: str = Field(..., description="Unique session identifier")
    audio: str = Field(..., description="Base64 encoded audio chunk")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    speaker: int | None = Field(None, description="TTS speaker ID (0-12799)")


# API Response Models
class TTSResponse(BaseModel):
    """Response model for TTS generation"""

    transcription: str
    audio: str  # Base64 encoded
    sample_rate: int


class StreamingResponse(BaseModel):
    """Response model for streaming processing"""

    transcription: str
    audio: str | None  # Base64 encoded, None if no TTS generated
    sample_rate: int | None


class ResetResponse(BaseModel):
    """Response model for session reset"""

    status: str
    session_id: str


# Conversation API Models
class ConversationRequest(BaseModel):
    """Request model for conversation with LLM"""

    text: str = Field(..., description="User message text", min_length=1)
    session_id: str = Field(
        ..., description="Unique session identifier", min_length=1, max_length=100
    )
    user_id: str | None = Field(
        None,
        description="Unique user identifier (auto-generated from session_id if not provided)",
        max_length=100,
    )
    speaker_id: int | None = Field(None, description="TTS speaker ID (0-12799)")


class ConversationResponse(BaseModel):
    """Response model for conversation initiation"""

    status: str  # "processing" or "queued"
    response_id: str | None = None
    position: int | None = None


class AudioQueueResponse(BaseModel):
    """Response model for audio queue polling"""

    audio_id: str | None
    audio: str | None  # Base64 encoded
    sample_rate: int | None
    text: str | None


class CleanupResponse(BaseModel):
    """Response model for audio cleanup"""

    status: str
