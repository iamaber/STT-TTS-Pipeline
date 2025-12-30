from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import json
import base64
from app.config import settings


async def handle_websocket(websocket: WebSocket, pipeline):
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


async def handle_streaming_pipeline(websocket: WebSocket, pipeline):
    await websocket.accept()
    
    pipeline.asr.reset()
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get('type') == 'audio':
                audio_bytes = base64.b64decode(data['audio'])
                
                pipeline.asr.feed_audio(audio_bytes)
                
                if pipeline.asr.should_transcribe():
                    transcription = pipeline.asr.transcribe()
                    
                    if transcription:
                        await websocket.send_json({
                            'type': 'transcription',
                            'text': transcription
                        })
                        
                        tts_audio = pipeline.tts.synthesize(transcription)
                        
                        audio_b64 = base64.b64encode(tts_audio.tobytes()).decode('utf-8')
                        await websocket.send_json({
                            'type': 'audio',
                            'audio': audio_b64,
                            'sample_rate': settings.tts.sample_rate
                        })
    
    except WebSocketDisconnect:
        pipeline.asr.reset()
