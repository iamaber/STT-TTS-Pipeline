from collections import deque
from datetime import datetime
from typing import Optional

from app.config import settings


class ConversationManager:
    def __init__(self):
        self.user_queue = deque(maxlen=settings.queue.max_user_queue_size)
        self.is_processing = False
        self.conversation_history = []
        self.current_response_id = None
        self.current_session_id = None

    async def add_user_input(
        self, text: str, session_id: str, speaker_id: int = None
    ) -> dict:
        """Add user input to queue."""
        input_item = {
            "text": text,
            "session_id": session_id,
            "speaker_id": speaker_id,
            "timestamp": datetime.now().isoformat(),
        }

        if self.is_processing:
            self.user_queue.append(input_item)
            return {"status": "queued", "position": len(self.user_queue)}
        else:
            return {"status": "processing", "position": 0}

    async def cancel_processing(self):
        """Cancel current processing and clear queues"""
        self.is_processing = False
        self.user_queue.clear()
        print("Processing cancelled, queue cleared")

    async def get_next_input(self) -> Optional[dict]:
        """Get next user input from queue."""
        if self.user_queue:
            return self.user_queue.popleft()
        return None

    def add_to_history(self, role: str, content: str):
        pass

    def get_history(self) -> list:
        pass

    def clear_history(self):
        return []
