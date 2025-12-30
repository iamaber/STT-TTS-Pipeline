import nemo.collections.asr as nemo_asr
import nemo.collections.tts as nemo_tts


def download_models():
    print('Downloading ASR model...')
    asr_model = nemo_asr.models.ASRModel.from_pretrained('nvidia/stt_en_fastconformer_hybrid_large_streaming_multi')
    print('ASR model downloaded successfully!')
    
    print('Downloading TTS acoustic model...')
    tts_model = nemo_tts.models.FastPitchModel.from_pretrained('tts_en_fastpitch_multispeaker')
    print('TTS acoustic model downloaded successfully!')
    
    print('Downloading TTS vocoder...')
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained('tts_en_hifitts_hifigan_ft_fastpitch')
    print('TTS vocoder downloaded successfully!')
    
    print('All models downloaded!')


if __name__ == '__main__':
    download_models()
