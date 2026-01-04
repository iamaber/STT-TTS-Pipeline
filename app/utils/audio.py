import base64
import numpy as np


def decode_audio(audio_b64: str) -> np.ndarray:
    """
    Decode base64 audio to float32 numpy array.

    Args:
        audio_b64: Base64 encoded int16 audio data

    Returns:
        Float32 numpy array normalized to [-1, 1]
    """
    audio_bytes = base64.b64decode(audio_b64)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def encode_audio(audio_float32: np.ndarray) -> str:
    """
    Encode float32 audio to base64.

    Args:
        audio_float32: Float32 numpy array normalized to [-1, 1]

    Returns:
        Base64 encoded int16 audio string
    """
    # Convert float32 [-1, 1] to int16 [-32768, 32767]
    audio_int16 = (audio_float32 * 32768.0).astype(np.int16)
    encoded = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

    # Debug logging
    duration = len(audio_float32) / 44100.0
    print(
        f"[ENCODE] {len(audio_float32)} samples ({duration:.2f}s) -> {len(encoded)} b64 chars, {len(audio_int16.tobytes())} bytes"
    )

    return encoded
