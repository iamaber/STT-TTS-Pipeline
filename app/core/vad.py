from typing import List, Tuple
import numpy as np
import torch
from silero_vad import load_silero_vad

from app.config import settings


class SileroVAD:
    def __init__(
        self,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        sample_rate: int,
    ):
        self.model = load_silero_vad()
        self.threshold = settings.vad.threshold
        self.min_speech_duration_ms = settings.vad.min_speech_duration_ms
        self.min_silence_duration_ms = settings.vad.min_silence_duration_ms
        self.sample_rate = settings.streaming.sample_rate
        self.window_size_samples = 512 if self.sample_rate == 16000 else 256

    def is_speech(self, audio: np.ndarray, threshold: float = None) -> bool:
        if threshold is None:
            threshold = self.threshold

        if len(audio) < self.window_size_samples:
            return False

        num_windows = len(audio) // self.window_size_samples
        speech_count = 0

        for i in range(num_windows):
            start = i * self.window_size_samples
            end = start + self.window_size_samples
            window = audio[start:end]

            audio_tensor = torch.from_numpy(window).float()

            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()

            if speech_prob > threshold:
                speech_count += 1

        return speech_count > 0

    def get_speech_probability(self, audio: np.ndarray) -> float:
        if len(audio) < self.window_size_samples:
            audio = np.pad(audio, (0, self.window_size_samples - len(audio)))
        elif len(audio) > self.window_size_samples:
            audio = audio[: self.window_size_samples]

        audio_tensor = torch.from_numpy(audio).float()

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        return speech_prob

    def detect_speech(
        self, audio: np.ndarray, threshold: float = None
    ) -> List[Tuple[int, int]]:
        if threshold is None:
            threshold = self.threshold

        hop_size = self.window_size_samples // 2

        speech_segments = []
        in_speech = False
        speech_start = 0

        for i in range(0, len(audio) - self.window_size_samples, hop_size):
            window = audio[i : i + self.window_size_samples]
            audio_tensor = torch.from_numpy(window).float()

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
