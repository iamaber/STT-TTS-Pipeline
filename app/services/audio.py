import numpy as np
import soundfile as sf
import librosa
from typing import Tuple


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(file_path)
    except RuntimeError:
        audio, sr = librosa.load(file_path, sr=None)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 22050):
    sf.write(file_path, audio, sample_rate)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio


def chunk_audio(
    audio: np.ndarray, chunk_size_ms: int, sample_rate: int
) -> list[np.ndarray]:
    chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)
    chunks = []

    for i in range(0, len(audio), chunk_size_samples):
        chunk = audio[i : i + chunk_size_samples]
        if len(chunk) == chunk_size_samples:
            chunks.append(chunk)

    return chunks
