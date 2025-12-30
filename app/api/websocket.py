from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import json
import base64
from app.config import settings
import time


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
    
    audio_buffer = []
    last_speech_time = time.time()
    silence_threshold = 1.5
    segment_start_time = None
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get('type') == 'audio':
                audio_bytes = base64.b64decode(data['audio'])
                audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                is_speech = pipeline.vad.is_speech(audio_float, threshold=0.5)
                
                if is_speech:
                    if segment_start_time is None:
                        segment_start_time = time.time()
                    last_speech_time = time.time()
                    audio_buffer.append(audio_float)
                else:
                    silence_duration = time.time() - last_speech_time
                    
                    if silence_duration > silence_threshold and len(audio_buffer) > 0 and segment_start_time is not None:
                        segment_end_time = last_speech_time
                        
                        combined_audio = np.concatenate(audio_buffer)
                        final_transcription = pipeline.asr.transcribe_audio(combined_audio)
                        
                        if final_transcription:
                            segment = {
                                'start': segment_start_time,
                                'end': segment_end_time,
                                'text': final_transcription
                            }
                            
                            await websocket.send_json({
                                'type': 'transcription',
                                'segment': segment
                            })
                            
                            tts_audio = pipeline.tts.synthesize(final_transcription)
                            
                            audio_b64 = base64.b64encode(tts_audio.tobytes()).decode('utf-8')
                            await websocket.send_json({
                                'type': 'audio',
                                'audio': audio_b64,
                                'sample_rate': settings.tts.sample_rate
                            })
                        
                        audio_buffer = []
                        segment_start_time = None
    
    except WebSocketDisconnect:
        pass
