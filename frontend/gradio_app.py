import gradio as gr
import requests
import numpy as np
import base64
import uuid
from scipy import signal


# Configuration
API_URL = "http://localhost:8000"
SR = 16000  # Target sample rate


def preprocess_audio(audio):
    """Preprocess audio to 16kHz mono"""
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
        y /= np.abs(y).max() + 1e-9

    return y


def process_audio(audio, session_id, speaker_id):
    """Send audio to backend for processing"""
    if audio is None:
        return "Listening...", None, session_id

    try:
        # Preprocess audio
        processed = preprocess_audio(audio)

        # Convert to int16 for transmission
        audio_int16 = (processed * 32768).astype(np.int16)
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

        # Send to streaming API
        response = requests.post(
            f"{API_URL}/api/stream/process",
            json={
                "session_id": session_id,
                "audio": audio_b64,
                "sample_rate": SR,
                "speaker": speaker_id,
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            transcription = result.get("transcription", "Listening...")

            # Decode TTS audio if present
            tts_audio = None
            tts_b64 = result.get("audio")
            if tts_b64:
                tts_bytes = base64.b64decode(tts_b64)
                tts_float = np.frombuffer(tts_bytes, dtype=np.float32)
                tts_audio = (tts_float * 32767).astype(np.int16)
                tts_sr = result.get("sample_rate", 44100)
                tts_audio = (tts_sr, tts_audio)

            return transcription, tts_audio, session_id

    except Exception as e:
        print(f"Processing error: {e}")

    return "Listening...", None, session_id


def reset_session(session_id):
    """Reset the streaming session"""
    try:
        requests.post(f"{API_URL}/api/stream/reset", params={"session_id": session_id})
    except Exception:
        pass

    # Generate new session ID
    new_session_id = str(uuid.uuid4())
    return "", None, new_session_id


# Warmup now done in backend startup


# Create Gradio interface
with gr.Blocks(title="Real-Time STT-TTS") as demo:
    gr.Markdown(
        """
    # Real-Time Speech-to-Text Pipeline

    **Powered by NVIDIA NeMo**
    - ASR: Parakeet-CTC-1.1B
    - Real-time streaming with timestamps
    - TTS: FastPitch Multispeaker 
    - VoCoder: HiFiGAN 
    """
    )

    # Session state - generate ID at startup
    session_id = gr.State(str(uuid.uuid4()))

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Microphone Input")

            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Live Audio Stream",
            )

            speaker_dropdown = gr.Dropdown(
                choices=list(range(0, 12800)),
                value=92,
                label="TTS Speaker ID (0-12799)",
                info="Select voice for text-to-speech output",
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
                interactive=False,
            )

            gr.Markdown("### TTS Output")
            audio_output = gr.Audio(
                label="Generated Speech", type="numpy", autoplay=True
            )

    # Stream audio processing
    audio_input.stream(
        fn=process_audio,
        inputs=[audio_input, session_id, speaker_dropdown],
        outputs=[transcription_output, audio_output, session_id],
        stream_every=0.25,  # Process every 0.25 seconds for faster response
    )

    # Reset button
    reset_btn.click(
        fn=reset_session,
        inputs=[session_id],
        outputs=[transcription_output, audio_output, session_id],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7900,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
