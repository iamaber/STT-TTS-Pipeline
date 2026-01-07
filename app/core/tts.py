import re
from typing import Optional

import numpy as np
import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

from app.config import settings

# CUDA Performance Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class TTSModel:
    def __init__(
        self,
        acoustic_model: str,
        vocoder_model: str,
        device: str,
        verbose: bool = True,
    ):
        self.device = settings.tts.device
        self.verbose = verbose
        self.max_text_length = settings.tts.max_text_length
        self.min_audio_duration = settings.tts.min_audio_duration

        if acoustic_model.endswith(".nemo"):
            self.fastpitch = FastPitchModel.restore_from(restore_path=acoustic_model)
        else:
            self.fastpitch = FastPitchModel.from_pretrained(acoustic_model)

        self.fastpitch = self.fastpitch.to(self.device)
        self.fastpitch.eval()

        if vocoder_model.endswith(".nemo"):
            self.hifigan = HifiGanModel.restore_from(restore_path=vocoder_model)
        else:
            self.hifigan = HifiGanModel.from_pretrained(vocoder_model)

        self.hifigan = self.hifigan.to(self.device)
        self.hifigan.eval()

        self.n_speakers = self.fastpitch.cfg.n_speakers
        self.sample_rate = settings.tts.sample_rate

        if verbose:
            print(f"TTS ready: {self.n_speakers} speakers available")

    def _split_text(self, text: str):
        if len(text) <= self.max_text_length:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_text_length:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text[: self.max_text_length]]

    def _generate_chunk(self, text: str, speaker: Optional[int] = None):
        """Generate audio for a single text chunk"""
        tokens = self.fastpitch.parse(text)
        spec = self.fastpitch.generate_spectrogram(tokens=tokens, speaker=speaker)
        audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()

        return audio

    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None):
        """Synthesize speech from text with hallucination prevention"""
        if speaker is None:
            speaker = settings.tts.default_speaker_id

        # Text sanitization
        text = text.strip()
        text = re.sub(r"[^\w\s.,!?;:\'-]", "", text)  # Keep only safe chars
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.replace("...", ".")  # Normalize ellipsis

        # Validate text
        if not text or len(text.split()) < 1:
            if self.verbose:
                print("TTS: No valid words, skipping")
            return np.array([], dtype=np.float32)

        # Validate speaker ID
        if speaker >= self.n_speakers or speaker < 0:
            speaker = min(0, self.n_speakers - 1)
            if self.verbose:
                print(f"TTS: Invalid speaker, using {speaker}")

        try:
            chunks = self._split_text(text)
            audio_chunks = []

            for chunk in chunks:
                audio = self._generate_chunk(chunk, speaker)
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(audio)

            if not audio_chunks:
                if self.verbose:
                    print("TTS: No audio generated")
                return np.array([], dtype=np.float32)

            combined_audio = np.concatenate(audio_chunks)

            # Log generation
            if self.verbose:
                print(
                    f"TTS generated: {text[:50]}... ({len(combined_audio)} samples, "
                    f"{len(combined_audio) / self.sample_rate:.2f}s)"
                )

            # Add silence padding for very short audio
            min_samples = int(self.min_audio_duration * self.sample_rate)
            if len(combined_audio) < min_samples:
                silence_needed = min_samples - len(combined_audio)
                combined_audio = np.concatenate(
                    [combined_audio, np.zeros(silence_needed, dtype=np.float32)]
                )

            # Clear CUDA cache after full generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return combined_audio

        except Exception as e:
            if self.verbose:
                print(f"TTS synthesis error: {e}")
            return np.array([], dtype=np.float32)

    @torch.no_grad()
    def synthesize_batch(self, texts, speaker: Optional[int] = None):
        """Synthesize a batch of texts"""
        return [self.synthesize(text, speaker=speaker) for text in texts]
