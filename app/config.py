from pydantic_settings import BaseSettings


class ASRConfig(BaseSettings):
    model_name: str = "models/asr/parakeet-tdt-0.6b-v2.nemo"
    sample_rate: int = 16000
    chunk_size_ms: int = 80
    streaming: bool = True
    device: str = "cuda"


class TTSConfig(BaseSettings):
    acoustic_model: str = "models/tts_acoustic/tts_en_fastpitch_multispeaker.nemo"
    vocoder_model: str = "models/tts_vocoder/tts_en_hifitts_hifigan_ft_fastpitch.nemo"
    sample_rate: int = 44100
    device: str = "cuda"


class VADConfig(BaseSettings):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100


class Settings(BaseSettings):
    asr: ASRConfig = ASRConfig()
    tts: TTSConfig = TTSConfig()
    vad: VADConfig = VADConfig()

    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


settings = Settings()
