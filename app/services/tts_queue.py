import asyncio
from collections import deque
from typing import Optional

from app.config import settings
from app.utils.audio import encode_audio


class TTSQueueManager:
    """Manages TTS generation queue and audio playback"""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.tts_queue = deque(maxlen=settings.queue.max_tts_queue_size)
        self.audio_queue = deque()
        self.is_generating = False

    async def add_to_tts_queue(self, text: str, speaker_id: int) -> str:
        """Add text to TTS generation queue."""
        import uuid

        item = {
            "id": str(uuid.uuid4()),
            "text": text,
            "speaker_id": speaker_id,
        }
        self.tts_queue.append(item)

        print(
            f"[TTS ENQUEUE] q={len(self.tts_queue)} audio_q={len(self.audio_queue)} text='{text[:60]}...'"
        )

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
                # Generate TTS audio with timeout to avoid stuck jobs
                loop = asyncio.get_event_loop()
                audio = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self.pipeline.process_text_to_audio,
                        item["text"],
                        item["speaker_id"],
                    ),
                    timeout=30,
                )

                # Encode audio to base64
                audio_b64 = encode_audio(audio)

                # Add to audio playback queue
                self.audio_queue.append(
                    {
                        "id": item["id"],
                        "audio": audio_b64,
                        "sample_rate": settings.tts.sample_rate,
                        "text": item["text"],
                    }
                )

                duration = len(audio) / settings.tts.sample_rate
                print(
                    f"[TTS QUEUE] '{item['text'][:60]}...' -> {len(audio)} samples ({duration:.2f}s), b64: {len(audio_b64)} chars"
                )

            except asyncio.TimeoutError:
                print(f"TTS timeout for '{item['text'][:60]}...'")
            except Exception as e:
                print(f"TTS error for '{item['text']}': {e}")

        self.is_generating = False

    async def get_next_audio(self) -> Optional[dict]:
        """Get next audio from playback queue."""
        if self.audio_queue:
            return self.audio_queue.popleft()
        return None

    def cleanup_audio(self, audio_id: str):
        """Cleanup audio (no-op for in-memory mode)."""
        pass

    def get_queue_sizes(self) -> dict:
        """Get current queue sizes"""
        return {"tts_queue": len(self.tts_queue), "audio_queue": len(self.audio_queue)}

    def clear_queues(self):
        """Clear all queues"""
        self.tts_queue.clear()
        self.audio_queue.clear()
