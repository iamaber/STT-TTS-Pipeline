import asyncio
from collections import deque
from pathlib import Path
import uuid
from typing import Optional
import soundfile as sf

from app.config import settings


class TTSQueueManager:
    """Manages TTS generation queue and audio playback"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tts_queue = deque(maxlen=settings.queue.max_tts_queue_size)
        self.audio_queue = deque()
        self.is_generating = False
        self.temp_dir = Path(settings.queue.audio_temp_dir)
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        print(f"TTS Queue Manager initialized: {self.temp_dir}")

    async def add_to_tts_queue(self, text: str, speaker_id: int) -> str:
        """
        Add text to TTS generation queue.

        Args:
            text: Text to synthesize
            speaker_id: TTS speaker ID

        Returns:
            Unique ID for this TTS item
        """
        item = {
            "id": str(uuid.uuid4()),
            "text": text,
            "speaker_id": speaker_id,
            "status": "queued",
        }
        self.tts_queue.append(item)

        # Start processing if not already running
        if not self.is_generating:
            asyncio.create_task(self._process_tts_queue())

        return item["id"]

    async def _process_tts_queue(self):
        """Process TTS queue in background"""
        self.is_generating = True

        while self.tts_queue:
            item = self.tts_queue.popleft()

            try:
                # Generate TTS audio in thread pool to avoid blocking
                import asyncio
                loop = asyncio.get_event_loop()
                audio = await loop.run_in_executor(
                    None,  # Use default executor
                    self.pipeline.process_text_to_audio,
                    item["text"],
                    item["speaker_id"]
                )

                # Save to temporary file
                audio_path = self.temp_dir / f"{item['id']}.wav"
                sf.write(audio_path, audio, settings.tts.sample_rate)

                # Add to audio playback queue
                self.audio_queue.append(
                    {
                        "id": item["id"],
                        "path": str(audio_path),
                        "text": item["text"],
                        "status": "ready",
                    }
                )

                print(f"TTS generated: {item['text'][:50]}... ({len(audio)} samples)")

            except Exception as e:
                print(f"TTS generation error for '{item['text']}': {e}")

        self.is_generating = False

    async def get_next_audio(self) -> Optional[dict]:
        """
        Get next audio from playback queue.

        Returns:
            Audio item dict or None
        """
        if self.audio_queue:
            return self.audio_queue.popleft()
        return None

    def cleanup_audio(self, audio_id: str):
        """
        Remove temporary audio file after playback.

        Args:
            audio_id: Audio item ID
        """
        if settings.queue.cleanup_after_play:
            audio_path = self.temp_dir / f"{audio_id}.wav"
            if audio_path.exists():
                try:
                    audio_path.unlink()
                    print(f"Cleaned up audio: {audio_id}")
                except Exception as e:
                    print(f"Cleanup error for {audio_id}: {e}")

    def get_queue_sizes(self) -> dict:
        """Get current queue sizes"""
        return {"tts_queue": len(self.tts_queue), "audio_queue": len(self.audio_queue)}

    def clear_queues(self):
        """Clear all queues"""
        self.tts_queue.clear()
        self.audio_queue.clear()
