from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse

import tempfile
import os
from app.services.audio import load_audio, save_audio
from app.config import settings

router = APIRouter()


async def transcribe_audio(pipeline, file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        audio, sr = load_audio(tmp_path, target_sr=settings.asr.sample_rate)
        transcription = pipeline.process_audio_to_text(audio, sr)

        return {"transcription": transcription, "sample_rate": sr}
    finally:
        os.unlink(tmp_path)


async def synthesize_speech(pipeline, text: str, speaker: int = None):
    audio = pipeline.process_text_to_audio(text, speaker)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        save_audio(audio, tmp.name, settings.tts.sample_rate)
        tmp_path = tmp.name

    return FileResponse(tmp_path, media_type="audio/wav", filename="output.wav")


async def full_pipeline_process(pipeline, file: UploadFile, speaker: int = None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        audio, sr = load_audio(tmp_path, target_sr=settings.asr.sample_rate)
        transcription, output_audio = pipeline.process_full_pipeline(audio, sr, speaker)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_tmp:
            save_audio(output_audio, out_tmp.name, settings.tts.sample_rate)
            out_tmp_path = out_tmp.name

        return {
            "transcription": transcription,
            "audio_file": FileResponse(
                out_tmp_path, media_type="audio/wav", filename="output.wav"
            ),
        }
    finally:
        os.unlink(tmp_path)
