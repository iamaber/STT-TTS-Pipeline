import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.asr import ASRModel
from app.models.tts import TTSModel
from app.config import settings


def test_asr():
    print('Testing ASR (FastConformer Streaming)...')
    asr = ASRModel(
        model_name=settings.asr.model_name,
        device=settings.asr.device
    )
    
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    try:
        result = asr.transcribe_audio(test_audio, 16000)
        print(f'ASR test passed. Result: {result}')
        return True
    except Exception as e:
        print(f'ASR test failed: {e}')
        return False


def test_tts():
    print('Testing TTS (FastPitch Multispeaker + HiFi-GAN)...')
    tts = TTSModel(
        acoustic_model=settings.tts.acoustic_model,
        vocoder_model=settings.tts.vocoder_model,
        device=settings.tts.device
    )
    
    try:
        audio = tts.synthesize('Hello world, this is a test.', speaker=92)
        print(f'TTS test passed. Generated {len(audio)} samples at {settings.tts.sample_rate}Hz')
        return True
    except Exception as e:
        print(f'TTS test failed: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='Test NeMo models')
    parser.add_argument('--component', choices=['asr', 'tts', 'all'], default='all',
                       help='Component to test')
    args = parser.parse_args()
    
    results = {}
    
    if args.component in ['asr', 'all']:
        results['asr'] = test_asr()
    
    if args.component in ['tts', 'all']:
        results['tts'] = test_tts()
    
    print('\nTest Results:')
    for component, passed in results.items():
        status = 'PASSED' if passed else 'FAILED'
        print(f'  {component.upper()}: {status}')
    
    all_passed = all(results.values())
    exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
