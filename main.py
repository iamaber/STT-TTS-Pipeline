from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.services.pipeline import Pipeline
from app.api import http, websocket
from app.config import settings

app = FastAPI(title='STT-TTS Pipeline', version='0.1.0')


# Request models
class STTRequest(BaseModel):
    audio: str
    sample_rate: int = 16000


class STTTTSRequest(BaseModel):
    audio: str
    sample_rate: int = 16000
    speaker: int = 92

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/frontend', StaticFiles(directory='frontend'), name='frontend')

pipeline = None


@app.on_event('startup')
async def startup_event():
    global pipeline
    pipeline = Pipeline()


@app.get('/')
async def root():
    return {
        'message': 'STT-TTS Pipeline API',
        'version': '0.1.0',
        'frontend': 'http://localhost:8000/frontend/index.html',
        'endpoints': {
            'transcribe': '/transcribe',
            'synthesize': '/synthesize',
            'pipeline': '/pipeline',
            'websocket': '/ws'
        }
    }


@app.post('/transcribe')
async def transcribe(file: UploadFile = File(...)):
    return await http.transcribe_audio(pipeline, file)


@app.post('/synthesize')
async def synthesize(text: str, speaker: int = None):
    return await http.synthesize_speech(pipeline, text, speaker)


@app.post('/pipeline')
async def full_pipeline(file: UploadFile = File(...), speaker: int = None):
    return await http.full_pipeline_process(pipeline, file, speaker)


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await websocket.handle_websocket(ws, pipeline)


@app.websocket('/ws/stream')
async def streaming_pipeline_endpoint(ws: WebSocket):
    await websocket.handle_streaming_pipeline(ws, pipeline)


@app.post('/api/stt')
async def stt_only(request: STTRequest):
    """STT-only endpoint for streaming frontend"""
    import base64
    import numpy as np
    
    # Decode base64 audio
    audio_bytes = base64.b64decode(request.audio)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Transcribe
    transcription = pipeline.process_audio_to_text(audio_data, request.sample_rate)
    
    return {
        'transcription': transcription,
        'sample_rate': request.sample_rate
    }


@app.post('/api/stt-tts')
async def stt_tts(request: STTTTSRequest):
    """Full STT-TTS pipeline for non-streaming frontend"""
    import base64
    import numpy as np
    
    # Decode base64 audio
    audio_bytes = base64.b64decode(request.audio)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Process through pipeline
    transcription, tts_audio = pipeline.process_full_pipeline(audio_data, request.sample_rate, request.speaker)
    
    # Encode TTS audio
    tts_b64 = base64.b64encode(tts_audio.tobytes()).decode('utf-8')
    
    return {
        'transcription': transcription,
        'audio': tts_b64,
        'sample_rate': settings.tts.sample_rate
    }

