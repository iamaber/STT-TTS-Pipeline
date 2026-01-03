import re
import os
from typing import AsyncIterator
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

from app.config import settings


class LLMService:
    """Gemini LLM service with streaming sentence-by-sentence generation"""

    def __init__(self):
        # Configure Gemini API
        if not settings.llm.api_key:
            raise ValueError(
                "LLM__API_KEY environment variable must be set. "
            )

        # Set environment variable for Pydantic AI (it looks for GEMINI_API_KEY)
        os.environ["GEMINI_API_KEY"] = settings.llm.api_key

        # Initialize Google Model using latest Pydantic AI API
        self.model = GoogleModel(
            model_name=settings.llm.model_name,
            provider='google-gla'  # Use Generative Language API
        )

        # Create agent with system prompt
        self.agent = Agent(
            model=self.model,
            system_prompt=settings.llm.system_prompt
        )

        print(f"LLM Service initialized: {settings.llm.model_name}")

    async def generate_streaming(
        self, user_message: str, conversation_history: list = None
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from LLM.
        Yields complete sentences as they're generated.

        Args:
            user_message: User's input text
            conversation_history: Previous conversation messages

        Yields:
            Complete sentences as they're generated
        """
        sentence_buffer = ""

        try:
            # Run agent with streaming
            async with self.agent.run_stream(user_message) as response:
                async for chunk in response.stream_text():
                    sentence_buffer += chunk

                    # Check for sentence endings
                    sentences = self._extract_complete_sentences(sentence_buffer)

                    for sentence in sentences:
                        yield sentence
                        # Remove yielded sentence from buffer
                        sentence_buffer = sentence_buffer[len(sentence) :].lstrip()

            # Yield any remaining text as final sentence
            if sentence_buffer.strip():
                yield sentence_buffer.strip()

        except Exception as e:
            print(f"LLM generation error: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"

    def _extract_complete_sentences(self, text: str) -> list[str]:
        """
        Extract complete sentences from text buffer.
        Optimized for faster TTS generation by detecting more punctuation.

        Args:
            text: Text buffer to process

        Returns:
            List of complete sentences
        """
        # Enhanced pattern for sentence endings:
        # . ! ? , ; : followed by space or end of string
        # This allows TTS to start earlier on commas and semicolons
        pattern = r"([^.!?,;:]*[.!?,;:])(?:\s+|$)"

        matches = re.findall(pattern, text)
        sentences = []
        
        for match in matches:
            match = match.strip()
            if match:
                # Only yield if it has meaningful content (not just punctuation)
                if len(match) > 2:  # At least 2 chars + punctuation
                    sentences.append(match)
        
        return sentences

    def _is_sentence_complete(self, text: str) -> bool:
        """
        Check if text ends with sentence-ending punctuation.

        Args:
            text: Text to check

        Returns:
            True if sentence is complete
        """
        return text.rstrip().endswith((".", "!", "?"))
