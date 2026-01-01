import gradio as gr
import requests
import numpy as np
import base64
from datetime import datetime
from scipy import signal
import threading
import queue


# API endpoint
API_URL = "http://localhost:8000"

# Streaming configuration
SR = 16000  # Target sample rate
CHUNK_SECONDS = 2  # Process every 2 seconds
CHUNK_SAMPLES = SR * CHUNK_SECONDS


class StreamingSession:
    """Manages streaming audio and transcription"""
    
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcripts = []
        self.active = True
        self.last_process_time = datetime.now()
        self.is_speaking = False
        self.silence_chunks = 0
        self.silence_threshold_chunks = 3  # Wait for 3 consecutive silent chunks (~1.5s)
    
    def preprocess_audio(self, audio):
        """Preprocess audio to 16kHz mono float32"""
        sr, y = audio
        
        # Convert to mono
        if y.ndim > 1:
            y = y.mean(axis=1)
        
        # Resample if needed
        if sr != SR:
            y = signal.resample_poly(y, SR, sr)
        
        # Convert to float32 and normalize
        y = y.astype(np.float32)
        if np.abs(y).max() > 0:
            y /= (np.abs(y).max() + 1e-9)
        
        return y
    
    def process_chunk(self, chunk, speaker_id=92):
        """Send chunk to backend and get transcription + TTS"""
        try:
            # Convert to int16 for transmission
            audio_int16 = (chunk * 32768).astype(np.int16)
            audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
            
            # Send to STT-TTS API
            response = requests.post(
                f"{API_URL}/api/stt-tts",
                json={
                    "audio": audio_b64,
                    "sample_rate": SR,
                    "speaker": speaker_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('transcription', '').strip()
                
                if text:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    # Decode TTS audio
                    tts_audio_b64 = result.get('audio', '')
                    tts_audio = None
                    if tts_audio_b64:
                        tts_bytes = base64.b64decode(tts_audio_b64)
                        tts_float = np.frombuffer(tts_bytes, dtype=np.float32)
                        # Convert to int16 to avoid Gradio warning
                        tts_audio = (tts_float * 32767).astype(np.int16)
                        tts_sr = result.get('sample_rate', 44100)
                    
                    return f"[{timestamp}] {text}", (tts_sr, tts_audio) if tts_audio is not None else None
            
        except Exception as e:
            print(f"Processing error: {e}")
        
        return None, None


# Warm up ASR on import (reduces first-call latency)
def _warmup():
    print("Warming up ASR model...")
    try:
        dummy_audio = np.zeros(SR * 2, dtype=np.float32)
        audio_int16 = (dummy_audio * 32768).astype(np.int16)
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
        
        requests.post(
            f"{API_URL}/api/stt-tts",
            json={"audio": audio_b64, "sample_rate": SR, "speaker": 92},
            timeout=30
        )
        print("ASR warm-up complete!")
    except Exception as e:
        print(f"Warm-up failed (backend may not be running yet): {e}")

# Run warmup in background thread
import threading
threading.Thread(target=_warmup, daemon=True).start()


def stream_audio(audio, state: StreamingSession, speaker_id):
    """Process streaming audio input - process every 3 seconds of speech"""
    if audio is None or not state.active:
        return (
            "\n".join(state.transcripts) if state.transcripts else "Ready...",
            None,
            state
        )
    
    # Preprocess incoming audio
    processed = state.preprocess_audio(audio)
    
    # Add to buffer
    state.audio_buffer = np.concatenate([state.audio_buffer, processed])
    
    # Process when we have 3+ seconds of audio
    if len(state.audio_buffer) >= SR * 3:
        # Get 3 seconds
        chunk = state.audio_buffer[:SR * 3]
        state.audio_buffer = state.audio_buffer[SR * 3:]
        
        # Check if chunk has speech (not just noise)
        energy = np.sqrt(np.mean(chunk ** 2))
        print(f"Processing chunk - Energy: {energy:.4f}")
        
        if energy > 0.05:  # Has meaningful content
            # Process the chunk
            text_result, tts_audio = state.process_chunk(chunk, speaker_id)
            
            # Filter out very short transcriptions
            if text_result:
                text_only = text_result.split('] ', 1)[-1] if '] ' in text_result else text_result
                if len(text_only.strip()) > 3:
                    state.transcripts.append(text_result)
                    print(f"Transcribed: {text_only}")
                    
                    # Return with TTS audio
                    output = "\n".join(state.transcripts)
                    return output, tts_audio, state
        else:
            print(f"Skipped - too quiet")
    
    # Return current state
    output = "\n".join(state.transcripts) if state.transcripts else "Listening..."
    return output, None, state


def reset_session():
    """Create new session"""
    return StreamingSession(), "", None


# Create Gradio interface
with gr.Blocks(title="Real-Time STT-TTS") as demo:
    gr.Markdown("""
    # Real-Time Speech-to-Text Pipeline
    
    **Powered by NVIDIA NeMo**
    - ASR: Parakeet-TDT-0.6B-v2
    - Real-time streaming with timestamps
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Microphone Input")
            
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Live Audio Stream"
            )
            
            speaker_dropdown = gr.Dropdown(
                choices=list(range(0, 12800)),
                value=92,
                label="TTS Speaker ID (0-12799)",
                info="Select voice for text-to-speech output"
            )
            
            reset_btn = gr.Button("🔄 Reset", variant="secondary")
        
        with gr.Column():
            gr.Markdown("### Live Transcription")
            
            transcription_output = gr.Textbox(
                label="Transcription with Timestamps",
                placeholder="Click 'Record' and start speaking...",
                lines=12,
                max_lines=15,
                autoscroll=True,
                interactive=False
            )
            
            gr.Markdown("### TTS Output")
            audio_output = gr.Audio(
                label="Generated Speech",
                type="numpy",
                autoplay=True
            )
    
    # Session state
    session_state = gr.State(lambda: StreamingSession())
    speaker_state = gr.State(92)  # Default speaker
    
    # Update speaker when dropdown changes
    speaker_dropdown.change(
        fn=lambda x: x,
        inputs=[speaker_dropdown],
        outputs=[speaker_state]
    )
    
    # Stream audio processing
    audio_input.stream(
        fn=stream_audio,
        inputs=[audio_input, session_state, speaker_state],
        outputs=[transcription_output, audio_output, session_state],
        stream_every=0.5  # Process every 0.5 seconds
    )
    
    # Reset button
    reset_btn.click(
        fn=reset_session,
        inputs=[],
        outputs=[session_state, transcription_output, audio_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7900,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
