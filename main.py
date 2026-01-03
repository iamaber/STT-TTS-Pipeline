from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio

from app.services.pipeline import Pipeline
from app.services.streaming import StreamingSession, process_streaming_audio
from app.core.llm import LLMService
from app.services.conversation import ConversationManager
from app.services.tts_queue import TTSQueueManager
from app.config import (
    settings,
    STTTTSRequest,
    StreamingRequest,
    TTSResponse,
    StreamingResponse,
    ResetResponse,
    ConversationRequest,
    ConversationResponse,
    AudioQueueResponse,
    CleanupResponse,
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
llm_service = None
conversation_manager = None
tts_queue_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline and services on startup"""
    global pipeline, llm_service, conversation_manager, tts_queue_manager

    pipeline = Pipeline()

    # Initialize LLM service if API key is provided
    if settings.llm.api_key:
        try:
            llm_service = LLMService()
            conversation_manager = ConversationManager()
            tts_queue_manager = TTSQueueManager(pipeline)
            print("Backend ready with LLM!")
        except Exception as e:
            print(f"LLM initialization failed: {e}")
            print("Backend ready without LLM")
    else:
        print("Backend ready without LLM (no API key provided)")


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    """Get frontend configuration"""
    return {
        "asr": {
            "sample_rate": settings.asr.sample_rate,
            "chunk_size_ms": settings.asr.chunk_size_ms,
        },
        "tts": {
            "sample_rate": settings.tts.sample_rate,
            "default_speaker_id": settings.tts.default_speaker_id,
        },
        "streaming": {
            "sample_rate": settings.streaming.sample_rate,
            "max_buffer_seconds": settings.streaming.max_buffer_seconds,
        },
        "llm_enabled": llm_service is not None,
    }


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

    # Use default speaker if not provided
    speaker_id = (
        request.speaker
        if request.speaker is not None
        else settings.tts.default_speaker_id
    )

    # Process streaming audio
    transcripts, tts_audio, tts_sr = process_streaming_audio(
        session, pipeline, audio_data, request.sample_rate, speaker_id
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


# Conversation Endpoints
@app.post("/api/conversation/send", response_model=ConversationResponse)
async def send_message(request: ConversationRequest) -> ConversationResponse:
    """
    Send user message to LLM with streaming response.
    Returns immediately with response_id for polling.
    """
    if not llm_service:
        return ConversationResponse(status="error", response_id=None, position=None)

    # Add to conversation queue
    speaker_id = request.speaker_id or settings.tts.default_speaker_id
    queue_status = await conversation_manager.add_user_input(
        request.text,
        request.session_id,
        speaker_id,  # Pass speaker_id to queue
    )

    if queue_status["status"] == "queued":
        return ConversationResponse(
            status="queued", response_id=None, position=queue_status["position"]
        )

    # Process immediately
    import uuid

    response_id = str(uuid.uuid4())
    conversation_manager.current_response_id = response_id

    # Start LLM streaming in background
    import asyncio

    asyncio.create_task(
        process_llm_streaming(
            request.text,
            request.speaker_id or settings.tts.default_speaker_id,
            response_id,
            request.session_id,
        )
    )

    return ConversationResponse(
        status="processing", response_id=response_id, position=0
    )


async def process_llm_streaming(
    user_text: str, speaker_id: int, response_id: str, session_id: str
):
    """Process LLM streaming and queue TTS generation"""
    conversation_manager.is_processing = True
    conversation_manager.current_session_id = session_id
    conversation_manager.add_to_history("user", user_text)

    try:
        # Stream LLM response
        async for sentence in llm_service.generate_streaming(
            user_text, conversation_manager.conversation_history
        ):
            # Add sentence to conversation history
            conversation_manager.add_to_history("assistant", sentence)

            # Queue TTS generation
            await tts_queue_manager.add_to_tts_queue(sentence, speaker_id)

    except Exception as e:
        print(f"LLM streaming error: {e}")

    conversation_manager.is_processing = False

    # Process next queued input if any
    next_input = await conversation_manager.get_next_input()
    if next_input:
        import uuid

        # Use speaker_id from the queued input, or default if not set
        next_speaker_id = (
            next_input.get("speaker_id") or settings.tts.default_speaker_id
        )
        asyncio.create_task(
            process_llm_streaming(
                next_input["text"],
                next_speaker_id,  # Use the correct speaker_id from queued input
                str(uuid.uuid4()),
                next_input["session_id"],
            )
        )


@app.get("/api/conversation/audio/next", response_model=AudioQueueResponse)
async def get_next_audio() -> AudioQueueResponse:
    """Get next audio from playback queue"""
    if not tts_queue_manager:
        return AudioQueueResponse(
            audio_id=None, audio=None, sample_rate=None, text=None
        )

    audio = await tts_queue_manager.get_next_audio()

    if audio:
        # Read audio file
        import soundfile as sf

        audio_data, sr = sf.read(audio["path"])

        # Encode to base64
        audio_b64 = encode_audio(audio_data)

        return AudioQueueResponse(
            audio_id=audio["id"], audio=audio_b64, sample_rate=sr, text=audio["text"]
        )

    return AudioQueueResponse(audio_id=None, audio=None, sample_rate=None, text=None)


@app.post("/api/conversation/audio/cleanup", response_model=CleanupResponse)
async def cleanup_audio(audio_id: str) -> CleanupResponse:
    """Cleanup audio file after playback"""
    if tts_queue_manager:
        tts_queue_manager.cleanup_audio(audio_id)
    return CleanupResponse(status="cleaned")


# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
