# STT-TTS Pipeline

A production-ready real-time Speech-to-Text and Text-to-Speech pipeline with semantic sentence detection, powered by NVIDIA NeMo models.

## Overview

This project provides a high-performance, GPU-accelerated speech processing pipeline designed for real-time applications. It combines state-of-the-art ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models with intelligent sentence boundary detection to deliver natural, low-latency conversational experiences.

### Key Features

- **Real-time Speech Recognition**: GPU-accelerated ASR using NVIDIA Parakeet-CTC-1.1B model
- **High-Quality Text-to-Speech**: FastPitch acoustic model with HiFiGAN vocoder, supporting 12,800 unique speaker voices
- **Semantic Sentence Detection**: Intelligent sentence boundary detection using punctuation analysis and natural pause detection
- **Low Latency**: Optimized for real-time applications with approximately 1-2 second total latency
- **Modern Web Interface**: Clean, responsive web UI with real-time audio streaming
- **Type-Safe Configuration**: Pydantic-based configuration management with environment variable support
- **Clean Architecture**: Well-organized codebase following clean code principles

## System Requirements

### Hardware
- CUDA-capable NVIDIA GPU (recommended for optimal performance)
- Minimum 8GB GPU VRAM
- 16GB system RAM

### Software
- Python 3.10 or higher
- CUDA Toolkit 11.8+ (for GPU acceleration)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd STT-TTS-Pipeline
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Download Models

```bash
uv run python scripts/download_models.py
```

This downloads the following models:
- **ASR Model**: Parakeet-CTC-1.1B (approximately 1.1GB)
- **TTS Acoustic Model**: FastPitch Multispeaker (approximately 100MB)
- **TTS Vocoder**: HiFiGAN (approximately 50MB)

Models are stored in the `models/` directory.

## Usage

### Starting the Server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`. The web interface is accessible at the root URL.

### Using the Web Interface

1. Navigate to `http://localhost:8000`
2. Select a speaker ID (0-12799)
3. Click "Start Recording" to begin
4. Speak naturally - the system will detect sentence boundaries automatically
5. TTS audio will play back in real-time

## Project Structure

```
STT-TTS-Pipeline/
├── models/                 # Downloaded .nemo model files
│   ├── asr/               # ASR model
│   ├── tts_acoustic/      # FastPitch acoustic model
│   └── tts_vocoder/       # HiFiGAN vocoder
├── app/
│   ├── core/              # Core model classes
│   │   ├── asr.py        # ASR model wrapper
│   │   ├── tts.py        # TTS model wrapper
│   │   └── vad.py        # Voice activity detection
│   ├── services/          # Business logic
│   │   ├── pipeline.py   # Main processing pipeline
│   │   └── streaming.py  # Streaming session management
│   ├── utils/             # Utility functions
│   │   └── audio.py      # Audio encoding/decoding
│   └── config.py          # Configuration management
├── frontend/              # Web interface
│   ├── index.html        # HTML structure
│   ├── style.css         # Styling
│   └── app.js            # Client-side logic
├── tests/                 # Test suite
├── scripts/               # Utility scripts
└── main.py               # FastAPI application entry point
```

## API Reference

### Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "ok"
}
```

### Full STT-TTS Pipeline

Process audio through complete STT-TTS pipeline.

```http
POST /api/stt-tts
Content-Type: application/json
```

**Request Body:**
```json
{
  "audio": "base64_encoded_audio",
  "sample_rate": 16000,
  "speaker": 92
}
```

**Response:**
```json
{
  "transcription": "transcribed text",
  "audio": "base64_encoded_tts_audio",
  "sample_rate": 44100
}
```

### Streaming Audio Processing

Process audio chunks in real-time with session management.

```http
POST /api/stream/process
Content-Type: application/json
```

**Request Body:**
```json
{
  "session_id": "unique-session-identifier",
  "audio": "base64_encoded_audio_chunk",
  "sample_rate": 16000,
  "speaker": 92
}
```

**Response:**
```json
{
  "transcription": "accumulated transcripts",
  "audio": "base64_encoded_tts_audio",
  "sample_rate": 44100
}
```

### Reset Session

Reset or create a new streaming session.

```http
POST /api/stream/reset?session_id=unique-session-identifier
```

**Response:**
```json
{
  "status": "reset",
  "session_id": "unique-session-identifier"
}
```

## Configuration

Configuration is managed through Pydantic models in `app/config.py`. Settings can be overridden using environment variables with the format `SECTION__PARAMETER`.

### ASR Configuration

```bash
ASR__MODEL_NAME=models/asr/parakeet-ctc-1.1b.nemo
ASR__SAMPLE_RATE=16000
ASR__DEVICE=cuda
```

### TTS Configuration

```bash
TTS__ACOUSTIC_MODEL=models/tts_acoustic/tts_en_fastpitch_multispeaker.nemo
TTS__VOCODER_MODEL=models/tts_vocoder/tts_en_hifitts_hifigan_ft_fastpitch.nemo
TTS__SAMPLE_RATE=44100
TTS__DEFAULT_SPEAKER_ID=50
TTS__DEVICE=cuda
TTS__MAX_TEXT_LENGTH=300
TTS__MIN_AUDIO_DURATION=0.7
```

### Streaming Configuration

```bash
STREAMING__SAMPLE_RATE=16000
STREAMING__MAX_BUFFER_SECONDS=60
STREAMING__SILENCE_TRIGGER_COUNT=2
```

### VAD Configuration

```bash
VAD__THRESHOLD=0.5
VAD__MIN_SPEECH_DURATION_MS=200
VAD__MIN_SILENCE_DURATION_MS=300
```

## Technical Details

### Semantic Sentence Detection

The system employs a hybrid approach combining acoustic and linguistic analysis:

1. **Voice Activity Detection (VAD)**: Silero VAD detects speech presence and natural pauses
2. **Semantic Analysis**: Checks for sentence-ending punctuation (period, exclamation mark, question mark)
3. **Word Count Validation**: Ensures valid text content (minimum 1 word)

**Processing Flow:**

```
Audio Input → VAD Analysis → Pause Detection (0.25s silence)
           → ASR Transcription → Sentence Completion Check
           → TTS Generation (if complete) → Audio Output
           → Continue Buffering (if incomplete)
```

### Performance Optimizations

- **CUDA Acceleration**: Both ASR and TTS models utilize GPU acceleration
- **TF32 Precision**: Enabled for faster matrix operations on Ampere GPUs
- **cuDNN Benchmark**: Automatic convolution algorithm selection
- **Greedy Batch Decoding**: Optimized CTC decoding strategy for ASR
- **Audio Padding**: Ensures frontend compatibility for short audio segments

### Model Specifications

#### ASR: Parakeet-CTC-1.1B
- **Architecture**: CTC-based encoder-decoder with BPE tokenization
- **Performance**: Approximately 34 iterations/second on GPU
- **Accuracy**: State-of-the-art for English speech recognition
- **Input**: 16kHz mono audio
- **Provider**: NVIDIA NeMo

#### TTS: FastPitch + HiFiGAN
- **Acoustic Model**: FastPitch with multi-speaker support
- **Vocoder**: HiFiGAN neural vocoder
- **Speakers**: 12,800 unique voice embeddings
- **Output Quality**: 44.1kHz high-fidelity audio
- **Provider**: NVIDIA NeMo

#### VAD: Silero
- **Type**: Lightweight neural voice activity detector
- **Latency**: 5-10ms processing time
- **Accuracy**: Optimized for real-time applications
- **Provider**: Silero Team

## Testing

### Run VAD Tests

```bash
uv run python tests/test_vad.py
```

### Run Component Tests

Test individual components (ASR, TTS):

```bash
# Test all components
uv run python tests/test_components.py --component all

# Test ASR only
uv run python tests/test_components.py --component asr

# Test TTS only
uv run python tests/test_components.py --component tts
```

### Test Full Pipeline

```bash
uv run python tests/test_pipeline.py --audio input.wav --output output.wav --speaker 92
```

## Development

### Code Style

This project uses Black for code formatting:

```bash
black app/ tests/ scripts/
```

### Adding New Features

1. Update configuration models in `app/config.py`
2. Implement core logic in appropriate module (`app/core/`, `app/services/`)
3. Add API endpoint in `main.py` with proper type hints
4. Create or update tests in `tests/`
5. Update documentation

### Project Principles

- **Type Safety**: Use Pydantic models for all configuration and API contracts
- **Clean Code**: Follow single responsibility principle and DRY
- **Documentation**: Comprehensive docstrings for all public functions
- **Testing**: Maintain test coverage for critical components

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

```bash
# Run on CPU instead
ASR__DEVICE=cpu TTS__DEVICE=cpu uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### Models Not Found

Ensure models are downloaded:

```bash
uv run python scripts/download_models.py
```

Verify models exist in the `models/` directory.

## Acknowledgments

- **NVIDIA NeMo**: Providing state-of-the-art ASR and TTS models
- **Silero Team**: Voice activity detection model
- **FastAPI**: Modern web framework for Python
- **PyTorch**: Deep learning framework
