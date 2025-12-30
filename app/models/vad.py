import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
from typing import List, Dict


class SileroVAD:
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250, 
                 min_silence_duration_ms: int = 100, sample_rate: int = 16000):
        self.model = load_silero_vad()
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        
    def detect_speech(self, audio: np.ndarray) -> List[Dict[str, int]]:
        audio_tensor = torch.from_numpy(audio).float()
        
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            sampling_rate=self.sample_rate
        )
        
        return speech_timestamps
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        audio_tensor = torch.from_numpy(audio_chunk).float()
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        return speech_prob > self.threshold
