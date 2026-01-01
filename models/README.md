# Models Directory

This directory contains locally downloaded NeMo models.

## Downloaded Models

### ASR (Speech Recognition)
- **Path**: `asr/fastconformer.nemo`
- **Model**: FastConformer Hybrid Large Streaming Multi
- **Size**: 439MB
- **Sample Rate**: 16kHz
- **Features**: Cache-aware streaming, RNNT decoder

### TTS Acoustic (Text to Mel)
- **Path**: `tts_acoustic/fastpitch.nemo`
- **Model**: FastPitch Multispeaker
- **Size**: 198MB
- **Speakers**: 12,800
- **Sample Rate**: 44.1kHz

### TTS Vocoder (Mel to Audio)
- **Path**: `tts_vocoder/hifigan.nemo`
- **Model**: HiFi-GAN
- **Size**: 324MB
- **Sample Rate**: 44.1kHz

## Usage

To use local models, update `app/config.py` or `.env`:

```python
ASR__MODEL_NAME=models/asr/fastconformer.nemo
TTS__ACOUSTIC_MODEL=models/tts_acoustic/fastpitch.nemo
TTS__VOCODER_MODEL=models/tts_vocoder/hifigan.nemo
```

## Re-download

To re-download models:

```bash
uv run python scripts/download_models_local.py
```
