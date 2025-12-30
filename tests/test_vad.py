import pytest
from app.models.vad import SileroVAD
import numpy as np


def test_vad_initialization():
    vad = SileroVAD()
    assert vad is not None
    assert vad.threshold == 0.5


def test_vad_speech_detection():
    vad = SileroVAD()
    
    silence = np.zeros(16000)
    assert not vad.is_speech(silence)
    
    noise = np.random.randn(16000) * 0.1
    result = vad.is_speech(noise)
    assert isinstance(result, bool)
