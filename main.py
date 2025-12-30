from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse
import numpy as np
import tempfile
import os
from app.services.pipeline import Pipeline
from app.services.audio import load_audio, save_audio
from app.config import settings

app = FastAPI(title='STT-TTS Pipeline', version='0.1.0')

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
        'endpoints': {
            'transcribe': '/transcribe',
            'synthesize': '/synthesize',
            'pipeline': '/pipeline'
        }
    }


@app.post('/transcribe')
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        audio, sr = load_audio(tmp_path, target_sr=settings.asr.sample_rate)
        transcription = pipeline.process_audio_to_text(audio, sr)
        
        return {
            'transcription': transcription,
            'sample_rate': sr
        }
    finally:
        os.unlink(tmp_path)


@app.post('/synthesize')
async def synthesize(text: str, speaker: int = None):
    audio = pipeline.process_text_to_audio(text, speaker)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        save_audio(audio, tmp.name, settings.tts.sample_rate)
        tmp_path = tmp.name
    
    return FileResponse(
        tmp_path,
        media_type='audio/wav',
        filename='output.wav'
    )


@app.post('/pipeline')
async def full_pipeline(file: UploadFile = File(...), speaker: int = None):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        audio, sr = load_audio(tmp_path, target_sr=settings.asr.sample_rate)
        transcription, output_audio = pipeline.process_full_pipeline(audio, sr, speaker)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as out_tmp:
            save_audio(output_audio, out_tmp.name, settings.tts.sample_rate)
            out_tmp_path = out_tmp.name
        
        return {
            'transcription': transcription,
            'audio_file': FileResponse(
                out_tmp_path,
                media_type='audio/wav',
                filename='output.wav'
            )
        }
    finally:
        os.unlink(tmp_path)


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio = np.frombuffer(data, dtype=np.float32)
            
            transcription = pipeline.process_audio_to_text(audio, settings.asr.sample_rate)
            
            if transcription:
                await websocket.send_json({'transcription': transcription})
    
    except WebSocketDisconnect:
        pass
