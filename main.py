from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import base64
import numpy as np

from app.services.pipeline import Pipeline
from app.services.streaming import StreamingSession, process_streaming_audio
from app.config import settings, STTTTSRequest, StreamingRequest


app = FastAPI(title="STT-TTS Pipeline", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

# Streaming session storage
streaming_sessions = {}


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = Pipeline()
    print("Backend ready!")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/stt-tts")
async def stt_tts(request: STTTTSRequest):
    # Decode base64 audio
    audio_bytes = base64.b64decode(request.audio)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Process through pipeline
    transcription, tts_audio = pipeline.process_full_pipeline(
        audio_data, request.sample_rate, request.speaker
    )

    # Encode TTS audio
    tts_b64 = base64.b64encode(tts_audio.tobytes()).decode("utf-8")

    return {
        "transcription": transcription,
        "audio": tts_b64,
        "sample_rate": settings.tts.sample_rate,
    }


@app.post("/api/stream/process")
async def stream_process(request: StreamingRequest):
    # Get or create session
    if request.session_id not in streaming_sessions:
        streaming_sessions[request.session_id] = StreamingSession()

    session = streaming_sessions[request.session_id]

    # Decode audio
    audio_bytes = base64.b64decode(request.audio)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Process streaming audio
    transcripts, tts_audio, tts_sr = process_streaming_audio(
        session, pipeline, audio_data, request.sample_rate, request.speaker
    )

    # Encode TTS if available
    tts_b64 = None
    if tts_audio is not None and len(tts_audio) > 0:
        print(f"DEBUG API: Encoding {len(tts_audio)} TTS samples")
        tts_b64 = base64.b64encode(tts_audio.tobytes()).decode("utf-8")
    else:
        print(f"DEBUG API: No TTS audio to encode (tts_audio={'None' if tts_audio is None else f'{len(tts_audio)} samples'})")

    return {
        "transcription": transcripts,
        "audio": tts_b64,
        "sample_rate": tts_sr,
    }


@app.post("/api/stream/reset")
async def stream_reset(session_id: str):
    # Reset or create streaming session
    if session_id in streaming_sessions:
        streaming_sessions[session_id].reset()
    else:
        streaming_sessions[session_id] = StreamingSession()

    return {"status": "reset", "session_id": session_id}


# Mount static files AFTER all API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
