from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
import json
import base64
from app.config import settings
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools


# Thread pool for running blocking operations
executor = ThreadPoolExecutor(max_workers=4)


def run_in_executor(func):
    """Decorator to run blocking functions in thread pool"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))
    return wrapper


async def handle_websocket(websocket: WebSocket, pipeline):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio = np.frombuffer(data, dtype=np.float32)
            
            # Run ASR in thread pool (non-blocking)
            transcription = await asyncio.get_event_loop().run_in_executor(
                executor,
                pipeline.process_audio_to_text,
                audio,
                settings.asr.sample_rate
            )
            
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
                
                # Run VAD in thread pool (non-blocking)
                is_speech = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    pipeline.vad.is_speech,
                    audio_float,
                    0.5  # threshold
                )
                
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
                        
                        # Run ASR and TTS in parallel using asyncio.gather
                        asr_task = asyncio.get_event_loop().run_in_executor(
                            executor,
                            pipeline.asr.transcribe_audio,
                            combined_audio
                        )
                        
                        # Wait for ASR to complete
                        final_transcription = await asr_task
                        
                        if final_transcription:
                            segment = {
                                'start': segment_start_time,
                                'end': segment_end_time,
                                'text': final_transcription
                            }
                            
                            # Send transcription immediately (don't wait for TTS)
                            await websocket.send_json({
                                'type': 'transcription',
                                'segment': segment
                            })
                            
                            # Start TTS in background (async)
                            tts_task = asyncio.get_event_loop().run_in_executor(
                                executor,
                                pipeline.tts.synthesize,
                                final_transcription
                            )
                            
                            # Wait for TTS to complete
                            tts_audio = await tts_task
                            
                            # Send TTS audio
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
