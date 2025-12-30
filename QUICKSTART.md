# Quick Reference

## Change Models

Edit `.env`:
```bash
ASR__MODEL_NAME=models/asr/fastconformer.nemo
TTS__ACOUSTIC_MODEL=models/tts_acoustic/fastpitch.nemo
TTS__VOCODER_MODEL=models/tts_vocoder/hifigan.nemo
```

## Download Models

```bash
uv run python scripts/download_models.py
```

## Test Locally

```bash
uv run python scripts/test_local.py --audio sample.wav --output result.wav
```

## Start Server

```bash
uv run uvicorn main:app --reload
```

## API Endpoints

### Transcribe
```bash
curl -X POST http://localhost:8000/transcribe -F "file=@audio.wav"
```

### Synthesize
```bash
curl -X POST "http://localhost:8000/synthesize?text=Hello%20world" -o output.wav
```

### Full Pipeline
```bash
curl -X POST http://localhost:8000/pipeline -F "file=@audio.wav"
```

## Run Tests

```bash
uv run pytest tests/ -v
```
