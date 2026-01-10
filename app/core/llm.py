import re
import httpx
from typing import AsyncIterator

from app.config import settings


class LLMService:
    def __init__(self):
        self.api_url = settings.llm.api_url
        self.session_base_url = (
            settings.llm.api_url.rsplit("/api/", 1)[0] + "/api/session"
        )
        self.client = httpx.AsyncClient(timeout=60.0)
        self.active_sessions = set()  # Track initialized sessions

    async def initialize_session(self, user_id: str, session_id: str) -> bool:
        """Initialize a new session with the LLM API"""
        session_key = f"{user_id}:{session_id}"

        if session_key in self.active_sessions:
            return True  # Already initialized

        try:
            session_url = f"{self.session_base_url}/{user_id}/{session_id}"
            response = await self.client.post(session_url)
            response.raise_for_status()
            self.active_sessions.add(session_key)
            print(f"Initialized session: {session_key}")
            return True
        except httpx.HTTPError as e:
            print(f"Session initialization error (continuing anyway): {e}")
            # Continue anyway - the chat endpoint might auto-create sessions
            self.active_sessions.add(session_key)
            return True

    async def clear_session(self, user_id: str, session_id: str) -> bool:
        """Clear a specific session"""
        session_key = f"{user_id}:{session_id}"

        try:
            session_url = f"{self.session_base_url}/{user_id}/{session_id}"
            response = await self.client.delete(session_url)
            response.raise_for_status()
            self.active_sessions.discard(session_key)
            print(f"Cleared session: {session_key}")
            return True
        except httpx.HTTPError as e:
            print(f"Session clear error: {e}")
            self.active_sessions.discard(session_key)
            return False

    async def clear_memory(self) -> bool:
        """Clear all active sessions"""
        self.active_sessions.clear()
        return True

    async def generate(
        self,
        user_message: str,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> AsyncIterator[str]:
        # Initialize session if not already done
        await self.initialize_session(user_id, session_id)

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "prompt": user_message,
            "max_tokens": settings.llm.max_tokens,
            "temperature": settings.llm.temperature,
            "top_p": settings.llm.top_p,
        }

        sentence_buffer = ""

        try:
            async with self.client.stream(
                "POST", self.api_url, json=payload
            ) as response:
                response.raise_for_status()

                # SSE format - word-by-word streaming
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    raw_chunk = line[6:]
                    chunk = raw_chunk.rstrip("\r\n")
                    marker = chunk.strip()

                    if not marker or marker == "[DONE]":
                        continue

                    # Add chunk to buffer (preserve spacing)
                    sentence_buffer += chunk
                    if not chunk.endswith(" "):
                        sentence_buffer += " "

                    # Extract and yield complete sentences
                    complete_sentences, sentence_buffer = self._extract_sentences(
                        sentence_buffer
                    )
                    for sentence in complete_sentences:
                        yield sentence

            # Yield remaining text
            if sentence_buffer.strip():
                yield sentence_buffer.strip()

        except httpx.HTTPError as e:
            print(f"LLM HTTP error: {e}")
            yield f"Error connecting to LLM: {str(e)}"
        except Exception as e:
            print(f"LLM error: {e}")
            yield f"Error: {str(e)}"

    def _extract_sentences(self, text: str) -> tuple[list[str], str]:
        """
        Extract complete sentences (. ! ? only) from buffer.
        Returns complete sentences and remaining buffer.
        """
        sentences = []

        # Split only on sentence-ending punctuation: . ! ?
        pattern = r"([.!?])\s+"
        parts = re.split(pattern, text)

        # Reconstruct: text + punctuation pairs (preserve internal spacing)
        i = 0
        while i < len(parts) - 1:
            text_part = parts[i]
            if not text_part or len(text_part.strip()) < 2:  # Skip empty/too short
                i += 2
                continue

            punct = parts[i + 1] if i + 1 < len(parts) else ""
            sentence = text_part + punct
            if sentence.strip():
                sentences.append(sentence)
            i += 2

        # Remaining incomplete text (no punctuation yet)
        remaining = parts[-1] if parts else ""

        return sentences, remaining
