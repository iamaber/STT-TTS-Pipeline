import base64
import numpy as np


def decode_audio(audio_b64: str) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_b64)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def encode_audio(audio_float32: np.ndarray) -> str:
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    audio_int16 = (audio_float32 * 32768.0).astype(np.int16)
    encoded = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

    # Debug logging
    duration = len(audio_float32) / 44100.0
    print(
        f"[ENCODE] {len(audio_float32)} samples ({duration:.2f}s) -> {len(encoded)} b64 chars, {len(audio_int16.tobytes())} bytes"
    )

    return encoded
