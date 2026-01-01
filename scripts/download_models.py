from pathlib import Path
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.tts.models import FastPitchModel, HifiGanModel


def download_models():
    '''Download all required NeMo models for the STT-TTS pipeline'''
    models_dir = Path('models')
    asr_dir = models_dir / 'asr'
    tts_acoustic_dir = models_dir / 'tts_acoustic'
    tts_vocoder_dir = models_dir / 'tts_vocoder'
    
    # Create directories
    for directory in [asr_dir, tts_acoustic_dir, tts_vocoder_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print('=' * 60)
    print('Downloading NeMo Models')
    print('=' * 60)
    print()
    
    # Download ASR model
    print('1/3 Downloading Parakeet-TDT-0.6B-v2 ASR model...')
    print('    Model: nvidia/parakeet-tdt-0.6b-v2')
    print('    Size: ~2.4GB')
    try:
        asr_model = EncDecRNNTBPEModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
        asr_path = asr_dir / 'parakeet_tdt_0.6b_v2.nemo'
        asr_model.save_to(str(asr_path))
        print(f'    ✓ Saved to {asr_path}')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        return False
    print()
    
    # Download TTS acoustic model
    print('2/3 Downloading FastPitch TTS acoustic model...')
    print('    Model: nvidia/tts_en_fastpitch_multispeaker')
    print('    Size: ~100MB')
    try:
        fastpitch = FastPitchModel.from_pretrained('nvidia/tts_en_fastpitch_multispeaker')
        fp_path = tts_acoustic_dir / 'fastpitch.nemo'
        fastpitch.save_to(str(fp_path))
        print(f'    ✓ Saved to {fp_path}')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        return False
    print()
    
    # Download TTS vocoder
    print('3/3 Downloading HiFi-GAN vocoder...')
    print('    Model: nvidia/tts_en_hifitts_hifigan_ft_fastpitch')
    print('    Size: ~50MB')
    try:
        hifigan = HifiGanModel.from_pretrained('nvidia/tts_en_hifitts_hifigan_ft_fastpitch')
        hg_path = tts_vocoder_dir / 'hifigan.nemo'
        hifigan.save_to(str(hg_path))
        print(f'    ✓ Saved to {hg_path}')
    except Exception as e:
        print(f'    ✗ Failed: {e}')
        return False
    print()
    
    print('=' * 60)
    print('✅ All models downloaded successfully!')
    print('=' * 60)
    return True


if __name__ == '__main__':
    import sys
    success = download_models()
    sys.exit(0 if success else 1)
