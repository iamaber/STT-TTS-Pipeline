from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.services.pipeline import Pipeline
from app.services.streaming import StreamingSession, process_streaming_audio
from app.config import (
    settings,
    STTTTSRequest,
    StreamingRequest,
    TTSResponse,
    StreamingResponse,
    ResetResponse,
)
from app.utils.audio import decode_audio, encode_audio


# Initialize FastAPI app
app = FastAPI(
    title="STT-TTS Pipeline",
    version="1.0.0",
    description="Real-time speech processing with NVIDIA NeMo models",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pipeline = None
streaming_sessions = {}


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    pipeline = Pipeline()
    print("Backend ready!")


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/api/stt-tts", response_model=TTSResponse)
async def stt_tts(request: STTTTSRequest) -> TTSResponse:
    """
    Full STT-TTS pipeline: transcribe audio and generate TTS response.

    Args:
        request: Audio data and configuration

    Returns:
        Transcription and synthesized audio
    """
    # Decode audio
    audio_data = decode_audio(request.audio)

    # Process through pipeline
    transcription, tts_audio = pipeline.process_full_pipeline(
        audio_data,
        request.sample_rate,
        request.speaker or settings.tts.default_speaker_id,
    )

    # Encode TTS audio
    tts_b64 = encode_audio(tts_audio)

    return TTSResponse(
        transcription=transcription, audio=tts_b64, sample_rate=settings.tts.sample_rate
    )


@app.post("/api/stream/process", response_model=StreamingResponse)
async def stream_process(request: StreamingRequest) -> StreamingResponse:
    """
    Process streaming audio chunk with semantic sentence detection.

    Args:
        request: Session ID, audio chunk, and configuration

    Returns:
        Current transcripts and TTS audio (if sentence complete)
    """
    # Get or create session
    if request.session_id not in streaming_sessions:
        streaming_sessions[request.session_id] = StreamingSession()

    session = streaming_sessions[request.session_id]

    # Decode audio
    audio_data = decode_audio(request.audio)

    # Process streaming audio
    transcripts, tts_audio, tts_sr = process_streaming_audio(
        session, pipeline, audio_data, request.sample_rate, request.speaker
    )

    # Encode TTS if available
    tts_b64 = None
    if tts_audio is not None and len(tts_audio) > 0:
        tts_b64 = encode_audio(tts_audio)

    return StreamingResponse(
        transcription=transcripts, audio=tts_b64, sample_rate=tts_sr
    )


@app.post("/api/stream/reset", response_model=ResetResponse)
async def stream_reset(session_id: str) -> ResetResponse:
    """
    Reset or create a streaming session.

    Args:
        session_id: Session identifier

    Returns:
        Reset confirmation
    """
    if session_id in streaming_sessions:
        streaming_sessions[session_id].reset()
    else:
        streaming_sessions[session_id] = StreamingSession()

    return ResetResponse(status="reset", session_id=session_id)


# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
