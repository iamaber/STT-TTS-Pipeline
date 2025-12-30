from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.services.pipeline import Pipeline
from app.api import http, websocket

app = FastAPI(title='STT-TTS Pipeline', version='0.1.0')

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
