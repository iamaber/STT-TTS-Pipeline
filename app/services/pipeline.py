from omegaconf import OmegaConf
from typing import Optional
from app.core.vad import SileroVAD
from app.core.asr import ASRModel
from app.core.tts import TTSModel
from app.config import settings
import numpy as np


class Pipeline:
    def __init__(self):
        self.vad = SileroVAD(
            threshold=settings.vad.threshold, sample_rate=settings.asr.sample_rate
        )

        self.asr = ASRModel(
            model_path=settings.asr.model_name, device=settings.asr.device
        )
        try:
            # Check if using a model that supports validation configuration
            # Some older NeMo models or specific classes might differ

            # Default to greedy batch which is faster
            decoding_cfg = OmegaConf.create(
                {
                    "strategy": "greedy_batch",
                    "preserve_alignments": False,
                    "compute_timestamps": False,
                }
            )

            if hasattr(self.asr, "asr_model") and hasattr(
                self.asr.asr_model, "change_decoding_strategy"
            ):
                self.asr.asr_model.change_decoding_strategy(decoding_cfg)
                print("ASR decoding strategy set to: greedy_batch")
        except Exception as e:
            print(f"Warning: Could not set ASR decoding strategy: {e}")

        # Warmup ASR BEFORE loading TTS to avoid CUDA conflicts
        self.asr.warmup()

        self.tts = TTSModel(
            acoustic_model=settings.tts.acoustic_model,
            vocoder_model=settings.tts.vocoder_model,
            device=settings.tts.device,
        )

    def process_audio_to_text(
        self, audio: np.ndarray, sample_rate: int = settings.streaming.sample_rate
    ) -> str:
        # VAD check is done in streaming.py, so skip here for performance
        transcription = self.asr.transcribe_audio(audio, sample_rate)
        return transcription

    def process_text_to_audio(
        self, text: str, speaker: Optional[int] = None
    ) -> np.ndarray:
        audio = self.tts.synthesize(text, speaker)
        return audio

    def process_full_pipeline(
        self,
        input_audio: np.ndarray,
        input_sr: int = 16000,
        speaker: Optional[int] = None,
    ) -> tuple[str, np.ndarray]:
        transcription = self.process_audio_to_text(input_audio, input_sr)

        if not transcription:
            return "", np.array([])

        output_audio = self.process_text_to_audio(transcription, speaker)

        return transcription, output_audio
