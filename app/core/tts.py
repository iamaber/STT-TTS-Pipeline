from pathlib import Path

import soundfile as sf
import torch
import numpy as np
from typing import Optional
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# CUDA Performance Optimizations
# Note: PYTORCH_CUDA_ALLOC_CONF is set in ASR module or globally
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Monkey-patch torch.load to fix PyTorch 2.6+ weights_only issue with NeMo models
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # Remove weights_only argument and set to False
    kwargs.pop("weights_only", None)
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


class TTS:
    def __init__(
        self,
        acoustic_model: str = "tts_en_fastpitch_multispeaker",
        vocoder_model: str = "tts_en_hifitts_hifigan_ft_fastpitch",
        device: str = "cuda",
        verbose: bool = True,
    ):
        from app.config import settings

        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.max_text_length = settings.tts.max_text_length
        self.min_audio_duration = settings.tts.min_audio_duration

        if verbose:
            print("Loading TTS models...")

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
        self.sample_rate = 44100

        if verbose:
            print(f"TTS ready: {self.n_speakers} speakers available")

    def _split_text(self, text: str):
        import re

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

    def generate(
        self,
        text: str,
        speaker: int = None,
        output: Optional[str] = None,
    ):
        chunks = self._split_text(text)
        audio_chunks = []

        for chunk in chunks:
            with torch.no_grad():
                tokens = self.fastpitch.parse(chunk)
                # FastPitch doesn't support pace parameter directly
                # Duration is controlled via duration_tgt or other parameters
                spec = self.fastpitch.generate_spectrogram(
                    tokens=tokens, speaker=speaker
                )
                audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Ensure audio is valid before adding
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not audio_chunks:
            if self.verbose:
                print("TTS: No audio generated")
            return np.array([], dtype=np.float32)

        combined_audio = np.concatenate(audio_chunks)

        # Debug: Log audio generation
        print(
            f"TTS generated: {text[:50]}... ({len(combined_audio)} samples, {len(combined_audio) / self.sample_rate:.2f}s)"
        )

        # Add silence padding for very short audio (for frontend compatibility)
        min_samples = int(self.min_audio_duration * self.sample_rate)
        if len(combined_audio) < min_samples:
            silence_needed = min_samples - len(combined_audio)
            combined_audio = np.concatenate(
                [combined_audio, np.zeros(silence_needed, dtype=np.float32)]
            )

        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, combined_audio, self.sample_rate)
            if self.verbose:
                print(
                    f"Saved: {output} ({len(combined_audio) / self.sample_rate:.2f}s)"
                )

        return combined_audio

    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None):
        """Synthesize speech from text with hallucination prevention"""
        from app.config import settings
        import re

        if speaker is None:
            speaker = settings.tts.default_speaker_id

        # Text sanitization to prevent hallucination
        text = text.strip()

        # Remove or replace problematic characters
        text = re.sub(r"[^\w\s.,!?;:\'-]", "", text)  # Keep only safe chars
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.replace("...", ".")  # Normalize ellipsis

        # Validate text - check for at least one word
        words = text.split()
        if not text or len(words) < 1:
            if self.verbose:
                print("TTS: No valid words, skipping")
            return np.array([], dtype=np.float32)

        # Note: max_text_length is already handled in _split_text() method
        # No need to truncate here as it will break sentences mid-word

        # Validate speaker ID (use safer range)
        if speaker >= self.n_speakers or speaker < 0:
            speaker = min(50, self.n_speakers - 1)  # Use speaker 50 or max available
            if self.verbose:
                print(f"TTS: Invalid speaker, using {speaker}")

        try:
            audio = self.generate(text, speaker=speaker)

            return audio
        except Exception as e:
            if self.verbose:
                print(f"TTS synthesis error: {e}")
            return np.array([], dtype=np.float32)

    @torch.no_grad()
    def synthesize_batch(self, texts, speaker: Optional[int] = None):
        return [self.generate(text, speaker=speaker or 92) for text in texts]


class TTSModel:
    def __init__(self, acoustic_model: str, vocoder_model: str, device: str = "cuda"):
        self.tts = TTS(
            acoustic_model=acoustic_model,
            vocoder_model=vocoder_model,
            device=device,
            verbose=True,
        )

    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None):
        return self.tts.synthesize(text, speaker)

    @torch.no_grad()
    def synthesize_batch(self, texts, speaker: Optional[int] = None):
        return self.tts.synthesize_batch(texts, speaker)
