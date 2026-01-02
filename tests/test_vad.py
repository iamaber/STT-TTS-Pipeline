import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.vad import SileroVAD


def test_vad_initialization():
    """Test VAD initialization with default parameters"""
    vad = SileroVAD()
    assert vad is not None
    assert vad.threshold == 0.5
    assert vad.sample_rate == 16000
    print("✓ VAD initialization test passed")


def test_vad_speech_detection():
    """Test VAD speech detection with silence and noise"""
    vad = SileroVAD()

    # Test with silence (should return False)
    silence = np.zeros(16000, dtype=np.float32)
    assert not vad.is_speech(silence), "Silence should not be detected as speech"
    print("✓ Silence detection test passed")

    # Test with random noise (should return bool)
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    result = vad.is_speech(noise)
    assert isinstance(result, bool), "Result should be boolean"
    print("✓ Noise detection test passed")


def test_vad_get_speech_probability():
    """Test VAD speech probability calculation"""
    vad = SileroVAD()

    # Test with silence
    silence = np.zeros(512, dtype=np.float32)
    prob = vad.get_speech_probability(silence)
    assert 0.0 <= prob <= 1.0, "Probability should be between 0 and 1"
    assert prob < 0.3, "Silence should have low speech probability"
    print("✓ Speech probability test passed")


if __name__ == "__main__":
    print("Running VAD tests...\n")
    test_vad_initialization()
    test_vad_speech_detection()
    test_vad_get_speech_probability()
    print("\n✅ All VAD tests passed!")
