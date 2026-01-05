import re
import httpx
from typing import AsyncIterator

from app.config import settings


class LLMService:
    def __init__(self):
        self.api_url = settings.llm.api_url
        self.client = httpx.AsyncClient()
        print(f"LLM Service initialized: {self.api_url}")

    async def clear_memory(self) -> bool:
        """No-op: API handles memory management"""
        return True

    def _should_skip_line(self, text: str) -> bool:
        """Check if line should be skipped (code blocks, markdown, etc)"""
        stripped = text.strip()
        # Skip code blocks
        if "```" in stripped or "```python" in stripped:
            return True
        # Skip system messages/prompts
        if any(
            x in stripped.lower()
            for x in [
                "system:",
                "assistant:",
                "user:",
                "def ",
                "python",
                "error:",
                "**error**",
            ]
        ):
            return True
        # Skip markdown headers
        if stripped.startswith("#"):
            return True
        # Skip empty or very short
        if len(stripped) < 2:
            return True
        return False

    async def generate(self, user_message: str) -> AsyncIterator[str]:
        """
        Generate response from LLM API.
        API handles memory management and yields complete sentences.
        """
        payload = {
            "prompt": user_message,
            "max_tokens": settings.llm.max_tokens,
            "temperature": settings.llm.temperature,
            "top_p": settings.llm.top_p,
            "include_chat_history": settings.llm.include_chat_history,
            "context": settings.llm.context,
        }

        sentence_buffer = ""
        in_code_block = False

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

                    # Track code blocks
                    if "```" in chunk:
                        in_code_block = not in_code_block
                        continue

                    # Skip if in code block
                    if in_code_block:
                        continue

                    # Skip undesirable content
                    if self._should_skip_line(chunk):
                        continue

                    # Preserve spacing from stream; add a space only if missing
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

            # Skip if contains code/system markers
            if self._should_skip_line(text_part):
                i += 2
                continue

            punct = parts[i + 1] if i + 1 < len(parts) else ""
            sentence = text_part + punct
            if sentence.strip():
                sentences.append(sentence)
            i += 2

        # Remaining incomplete text (no punctuation yet)
        remaining = parts[-1] if parts else ""

        # Clean remaining to prevent corruption carryover
        if remaining and self._should_skip_line(remaining):
            remaining = ""

        return sentences, remaining
