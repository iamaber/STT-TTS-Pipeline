import torch
from silero_vad import load_silero_vad
import numpy as np
from typing import List, Tuple


class SileroVAD:
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250, 
                 min_silence_duration_ms: int = 100, sample_rate: int = 16000):
        self.model = load_silero_vad()
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        
    def is_speech(self, audio: np.ndarray, threshold: float = None) -> bool:
        if threshold is None:
            threshold = self.threshold
        
        chunk_size = 512 if self.sample_rate == 16000 else 256
        
        if len(audio) < chunk_size:
            return False
        
        num_chunks = len(audio) // chunk_size
        speech_chunks = 0
        
        for i in range(num_chunks):
            chunk = audio[i * chunk_size:(i + 1) * chunk_size]
            audio_tensor = torch.from_numpy(chunk).float()
            
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            if speech_prob > threshold:
                speech_chunks += 1
        
        return speech_chunks > (num_chunks * 0.3)
    
    def detect_speech(self, audio: np.ndarray, threshold: float = None) -> List[Tuple[int, int]]:
        if threshold is None:
            threshold = self.threshold
        
        chunk_size = 512 if self.sample_rate == 16000 else 256
        hop_size = chunk_size // 2
        
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(0, len(audio) - chunk_size, hop_size):
            chunk = audio[i:i + chunk_size]
            audio_tensor = torch.from_numpy(chunk).float()
            
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            is_speech_frame = speech_prob > threshold
            
            if is_speech_frame and not in_speech:
                speech_start = i
                in_speech = True
            elif not is_speech_frame and in_speech:
                speech_segments.append((speech_start, i))
                in_speech = False
        
        if in_speech:
            speech_segments.append((speech_start, len(audio)))
        
        return speech_segments
