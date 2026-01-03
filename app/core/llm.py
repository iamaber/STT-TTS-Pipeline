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

        try:
            async with self.client.stream(
                "POST", self.api_url, json=payload
            ) as response:
                response.raise_for_status()

                # SSE format - word-by-word streaming
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    chunk = line[6:].strip()

                    if chunk == "[DONE]" or not chunk:
                        continue

                    sentence_buffer += chunk + " "

                    # Extract and yield complete sentences
                    complete_sentences, sentence_buffer = self._extract_sentences(
                        sentence_buffer
                    )
                    for sentence in complete_sentences:
                        yield sentence

            print(f"\n\nChecking sentence buffer: {sentence_buffer}\n\n")

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

        # Reconstruct: text + punctuation pairs
        i = 0
        while i < len(parts) - 1:
            text_part = parts[i].strip()
            if text_part:
                punct = parts[i + 1]  # . ! or ?
                sentence = text_part + punct
                sentences.append(sentence)
            i += 2

        # Remaining incomplete text (no punctuation yet)
        remaining = parts[-1].strip() if parts else ""

        return sentences, remaining
