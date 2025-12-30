import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts
from pathlib import Path


def download_models_locally():
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    asr_path = models_dir / 'asr'
    tts_acoustic_path = models_dir / 'tts_acoustic'
    tts_vocoder_path = models_dir / 'tts_vocoder'
    
    print('Downloading ASR model...')
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi'
    )
    asr_model.save_to(str(asr_path / 'fastconformer.nemo'))
    print(f'ASR model saved to {asr_path}')
    
    print('Downloading TTS acoustic model...')
    tts_model = nemo_tts.models.FastPitchModel.from_pretrained(
        'tts_en_fastpitch_multispeaker'
    )
    tts_model.save_to(str(tts_acoustic_path / 'fastpitch.nemo'))
    print(f'TTS acoustic model saved to {tts_acoustic_path}')
    
    print('Downloading TTS vocoder...')
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
        'tts_en_hifitts_hifigan_ft_fastpitch'
    )
    vocoder.save_to(str(tts_vocoder_path / 'hifigan.nemo'))
    print(f'TTS vocoder saved to {tts_vocoder_path}')
    
    print('\nAll models downloaded locally!')
    print(f'Models directory: {models_dir.absolute()}')


if __name__ == '__main__':
    download_models_locally()
