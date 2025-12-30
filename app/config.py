from pydantic_settings import BaseSettings
from typing import Literal


class ASRConfig(BaseSettings):
    model_name: str = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
    sample_rate: int = 16000
    chunk_size_ms: int = 80
    streaming: bool = True
    device: str = 'cuda'
    
    
class TTSConfig(BaseSettings):
    acoustic_model: str = 'nvidia/tts_en_fastpitch'
    vocoder_model: str = 'nvidia/tts_hifigan'
    sample_rate: int = 22050
    device: str = 'cuda'
    

class VADConfig(BaseSettings):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    

class Settings(BaseSettings):
    asr: ASRConfig = ASRConfig()
    tts: TTSConfig = TTSConfig()
    vad: VADConfig = VADConfig()
    
    host: str = '0.0.0.0'
    port: int = 8000
    
    class Config:
        env_file = '.env'
        env_nested_delimiter = '__'


settings = Settings()
