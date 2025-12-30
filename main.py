from fastapi import FastAPI, WebSocket, UploadFile, File
from app.services.pipeline import Pipeline
from app.api import http, websocket

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
