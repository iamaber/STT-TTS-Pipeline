import torch
import numpy as np
import os
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from omegaconf import open_dict


# CRITICAL: Enable expandable segments for CUDA memory allocation
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in alloc_conf:
    if len(alloc_conf) > 0:
        alloc_conf += ",expandable_segments:True"
    else:
        alloc_conf = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf


class BufferedASRModel:
    """GPU-enabled ASR with proper configuration for TDT models"""

    def __init__(self, model_path: str, device: str = "cuda", verbose: bool = True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.verbose = verbose

        if verbose:
            print(f"Loading Parakeet-TDT ASR model from {model_path}...")
            print(f"Device: {self.device.upper()}")

        # Load model
        if model_path.endswith(".nemo"):
            self.model = EncDecRNNTBPEModel.restore_from(restore_path=model_path)
        else:
            self.model = EncDecRNNTBPEModel.from_pretrained(model_path)

        # Configure for streaming (from NeMo example)
        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0

        # Check normalization type
        if self.model.cfg.preprocessor.normalize != "per_feature":
            if verbose:
                print(
                    "Warning: Model should use per_feature normalization for best streaming results"
                )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Configure decoding for TDT (greedy strategy required)
        decoding_cfg = RNNTDecodingConfig(
            strategy="greedy",  # TDT requires greedy, not greedy_batch
            preserve_alignments=False,  # Not needed for simple transcription
            fused_batch_size=-1,
        )

        with open_dict(decoding_cfg):
            decoding_cfg.greedy.loop_labels = True  # Enable label looping for TDT

        self.model.change_decoding_strategy(decoding_cfg)

        if verbose:
            print(f"Parakeet-TDT ready for streaming ({self.device.upper()} mode)")

    @torch.no_grad()
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        try:
            # Ensure audio is float32 and 1D
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Resample if needed
            if sample_rate != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            if len(audio) < 1600:
                return ""

            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Use model's transcribe method directly (simpler than BatchedFrameASRTDT)
            # This works because we configured the model properly for TDT
            transcriptions = self.model.transcribe([audio], batch_size=1)

            if transcriptions and len(transcriptions) > 0:
                result = transcriptions[0]
                if hasattr(result, "text"):
                    return result.text.strip()
                return str(result).strip()
            return ""

        except Exception as e:
            if self.verbose:
                print(f"Transcription error: {e}")
            # Re-raise to trigger fallback
            raise

    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        try:
            # Clear CUDA cache
            if self.device == "cuda":
                torch.cuda.empty_cache()

            transcriptions = self.model.transcribe([audio_path], batch_size=1)

            if transcriptions and len(transcriptions) > 0:
                result = transcriptions[0]
                if hasattr(result, "text"):
                    return result.text.strip()
                return str(result).strip()
            return ""
        except Exception as e:
            if self.verbose:
                print(f"File transcription error: {e}")
            raise
