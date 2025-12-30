# STT-TTS Pipeline

Real-time Speech-to-Text → LLM → Text-to-Speech pipeline using FastAPI and NeMo 2.0.

## Features

- 🎤 **Streaming ASR**: FastConformer RNNT for real-time speech recognition
- 🤖 **LLM Integration**: Local LLM for intelligent responses
- 🔊 **High-Quality TTS**: FastPitch + HiFi-GAN for natural speech synthesis
- ⚡ **Low Latency**: Optimized for real-time voice interactions
- 🐍 **NeMo 2.0**: Python-based configuration for flexibility

## Architecture

```
Audio Input → Silero VAD → FastConformer RNNT → LLM → FastPitch → HiFi-GAN → Audio Output
```

## Setup

### Prerequisites

- Python 3.10.19+
- CUDA-capable GPU (recommended)
- `uv` package manager

### Installation

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### Download Models

```bash
uv run python scripts/download_models.py
```

## Usage

### Start the server

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Test the pipeline

```bash
uv run python scripts/test_local.py --audio sample.wav
```

## Project Structure

```
STT-TTS-Pipeline/
├── app/                    # Main application
│   ├── models/            # Model wrappers (VAD, ASR, LLM, TTS)
│   ├── services/          # Business logic
│   └── api/               # API endpoints
├── scripts/               # Utility scripts
└── tests/                 # Test suite
```

## License

MIT
