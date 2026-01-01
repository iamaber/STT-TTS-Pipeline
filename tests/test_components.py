import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.asr import ASRModel
from app.models.tts import TTSModel
from app.config import settings


def test_asr():
    '''Test Parakeet-TDT ASR model'''
    print('Testing ASR (Parakeet-TDT-0.6B-v2)...')
    try:
        asr = ASRModel(
            model_path=settings.asr.model_name,
            device=settings.asr.device
        )
        
        # Generate test audio (random noise)
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        result = asr.transcribe_audio(test_audio, 16000)
        print(f'✓ ASR test passed. Result: "{result}"')
        return True
    except Exception as e:
        print(f'✗ ASR test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def test_tts():
    '''Test FastPitch + HiFi-GAN TTS models'''
    print('Testing TTS (FastPitch + HiFi-GAN)...')
    try:
        tts = TTSModel(
            acoustic_model=settings.tts.acoustic_model,
            vocoder_model=settings.tts.vocoder_model,
            device=settings.tts.device
        )
        
        test_text = 'Hello world, this is a test of the text to speech system.'
        audio = tts.synthesize(test_text, speaker=92)
        
        duration = len(audio) / tts.tts.sample_rate
        print(f'✓ TTS test passed. Generated {len(audio)} samples ({duration:.2f}s) at {tts.tts.sample_rate}Hz')
        return True
    except Exception as e:
        print(f'✗ TTS test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test NeMo ASR and TTS models')
    parser.add_argument('--component', choices=['asr', 'tts', 'all'], default='all',
                       help='Component to test (default: all)')
    args = parser.parse_args()
    
    print('=' * 60)
    print('NeMo Components Test Suite')
    print('=' * 60)
    print()
    
    results = {}
    
    if args.component in ['asr', 'all']:
        results['asr'] = test_asr()
        print()
    
    if args.component in ['tts', 'all']:
        results['tts'] = test_tts()
        print()
    
    print('=' * 60)
    print('Test Results:')
    print('=' * 60)
    for component, passed in results.items():
        status = '✅ PASSED' if passed else '❌ FAILED'
        print(f'  {component.upper()}: {status}')
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print('✅ All tests passed!')
    else:
        print('❌ Some tests failed')
    
    exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
