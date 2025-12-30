import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts


def download_models():
    print('Downloading ASR model...')
    asr_model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-rnnt-1.1b')
    print('ASR model downloaded successfully!')
    
    print('Downloading TTS acoustic model...')
    tts_model = nemo_tts.models.FastPitchModel.from_pretrained('nvidia/tts_en_fastpitch')
    print('TTS acoustic model downloaded successfully!')
    
    print('Downloading TTS vocoder...')
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained('nvidia/tts_hifigan')
    print('TTS vocoder downloaded successfully!')
    
    print('All models downloaded!')


if __name__ == '__main__':
    download_models()
