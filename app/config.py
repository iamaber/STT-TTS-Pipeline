from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ASRConfig(BaseSettings):
    """ASR (Automatic Speech Recognition) configuration"""
    model_name: str = "model_files/asr/parakeet-ctc-1.1b.nemo"
    sample_rate: int = 16000
    chunk_size_ms: int = 100
    streaming: bool = True
    device: str = "cuda"


class TTSConfig(BaseSettings):
    """TTS (Text-to-Speech) configuration"""
    acoustic_model: str = "model_files/tts_acoustic/tts_en_fastpitch_multispeaker.nemo"
    vocoder_model: str = "model_files/tts_vocoder/tts_en_hifitts_hifigan_ft_fastpitch.nemo"
    sample_rate: int = 44100
    device: str = "cuda"
    default_speaker_id: int = 50
    max_text_length: int = 300
    min_audio_duration: float = 0.7  # Minimum audio duration in seconds (for Gradio compatibility)


class VADConfig(BaseSettings):
    """VAD (Voice Activity Detection) configuration"""
    threshold: float = 0.5
    min_speech_duration_ms: int = 200
    min_silence_duration_ms: int = 300


class StreamingConfig(BaseSettings):
    """Real-time streaming configuration"""
    sample_rate: int = 16000
    max_buffer_seconds: int = 60  # Maximum buffer duration before forcing processing
    silence_trigger_count: int = 2  # Number of silence frames before checking transcription (~0.25s)


class Settings(BaseSettings):
    """Main application settings"""
    asr: ASRConfig = ASRConfig()
    tts: TTSConfig = TTSConfig()
    vad: VADConfig = VADConfig()
    streaming: StreamingConfig = StreamingConfig()
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False  # Enable/disable debug logging

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


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
