import asyncio
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

    async def add_user_input(self, text: str, session_id: str, speaker_id: int = None) -> dict:
        """
        Add user input to queue.

        Args:
            text: User's message text
            session_id: Session identifier
            speaker_id: TTS speaker ID (optional, uses default if None)

        Returns:
            Dict with status and position info
        """
        input_item = {
            "text": text,
            "session_id": session_id,
            "speaker_id": speaker_id,  # Store speaker_id for later use
            "timestamp": datetime.now().isoformat(),
            "status": "queued",
        }

        if self.is_processing:
            self.user_queue.append(input_item)
            return {"status": "queued", "position": len(self.user_queue)}
        else:
            return {"status": "processing", "position": 0}

    async def get_next_input(self) -> Optional[dict]:
        """
        Get next user input from queue.

        Returns:
            Next queued input or None
        """
        if self.user_queue:
            return self.user_queue.popleft()
        return None

    def add_to_history(self, role: str, content: str):

        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

        # Keep only last 20 messages to manage context window
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def get_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return len(self.user_queue)
