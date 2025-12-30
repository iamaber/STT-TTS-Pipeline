import os
from pathlib import Path
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.tts.models import FastPitchModel, HifiGanModel


def download_models():
    models_dir = Path('models')
    asr_dir = models_dir / 'asr'
    tts_acoustic_dir = models_dir / 'tts_acoustic'
    tts_vocoder_dir = models_dir / 'tts_vocoder'
    
    for directory in [asr_dir, tts_acoustic_dir, tts_vocoder_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print('Downloading Parakeet-TDT-0.6B-v2 ASR model...')
    asr_model = EncDecRNNTBPEModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
    asr_model.save_to(str(asr_dir / 'parakeet_tdt_0.6b_v2.nemo'))
    print(f'Saved to {asr_dir / "parakeet_tdt_0.6b_v2.nemo"}')
    
    print('\nDownloading FastPitch TTS model...')
    fastpitch = FastPitchModel.from_pretrained('nvidia/tts_en_fastpitch_multispeaker')
    fastpitch.save_to(str(tts_acoustic_dir / 'fastpitch.nemo'))
    print(f'Saved to {tts_acoustic_dir / "fastpitch.nemo"}')
    
    print('\nDownloading HiFi-GAN vocoder...')
    hifigan = HifiGanModel.from_pretrained('nvidia/tts_en_hifitts_hifigan_ft_fastpitch')
    hifigan.save_to(str(tts_vocoder_dir / 'hifigan.nemo'))
    print(f'Saved to {tts_vocoder_dir / "hifigan.nemo"}')
    
    print('\nAll models downloaded successfully!')


if __name__ == '__main__':
    download_models()
