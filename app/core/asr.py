import torch
import numpy as np
import os
from nemo.collections.asr.models import EncDecCTCModelBPE
from app.config import settings
from scipy.io import wavfile

# CUDA Performance Optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
torch.backends.cudnn.allow_tf32 = True  # Use TF32 for cuDNN
torch.set_float32_matmul_precision("high")  # Faster matmul


class ASRModel:
    def __init__(self, model_path: str, device: str = "cuda", verbose: bool = True):
        self.verbose = verbose
        self._warmed_up = False

        # Use GPU if available
        if torch.cuda.is_available() and device == settings.asr.device:
            self.device = torch.device("cuda:0")

        if verbose:
            print(f"Loading ASR model on {self.device}...")

        # Load CTC model (from pretrained or local file)
        if model_path.endswith(".nemo"):
            self.model = EncDecCTCModelBPE.restore_from(restore_path=model_path)
        else:
            # Load from pretrained name (e.g., "nvidia/parakeet-ctc-1.1b")
            self.model = EncDecCTCModelBPE.from_pretrained(model_path)

        # Configure preprocessor
        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0

        # Move to device
        self.model = self.model.to(self.device)
        self.model.freeze()
        self.model.eval()

        # Set decoding strategy to greedy_batch for better performance
        try:
            self.model.change_decoding_strategy(
                decoder_type="ctc", decoding_cfg={"strategy": "greedy_batch"}
            )
            if verbose:
                print("Using greedy_batch decoding strategy")
        except Exception as e:
            if verbose:
                print(f"Could not set greedy_batch strategy: {e}")

    def warmup(self):
        if self._warmed_up:
            return
        if self.verbose:
            print("Warming up ASR model...")
        try:
            dummy_audio = np.zeros(settings.asr.sample_rate, dtype=np.float32)
            self.transcribe_audio(dummy_audio, settings.asr.sample_rate)
            self._warmed_up = True
            if self.verbose:
                print("ASR warmup complete!")
        except Exception as e:
            if self.verbose:
                print(f"ASR warmup failed: {e}")
            self._warmed_up = True

    def transcribe_audio(
        self, audio: np.ndarray, sample_rate: int = settings.streaming.sample_rate
    ) -> str:
        # Prepare audio
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Only resample if sample_rate is not already 16000 (allow small tolerance)
        if abs(sample_rate - 16000) > 10:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        if len(audio) < 1600:
            return ""

        # Use temp file in RAM disk (/dev/shm) for fastest I/O
        # NeMo doesn't support BytesIO, so this is the optimal approach
        import tempfile
        from pathlib import Path

        shm_dir = Path("/dev/shm")
        tmp_dir = shm_dir if shm_dir.exists() else None

        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=tmp_dir
        ) as tmp:
            tmp_path = tmp.name
            wavfile.write(tmp_path, 16000, (audio * 32767).astype(np.int16))

        try:
            with torch.inference_mode():
                transcriptions = self.model.transcribe([tmp_path], batch_size=1)
            if transcriptions and len(transcriptions) > 0:
                result = transcriptions[0]
                if hasattr(result, "text"):
                    return result.text.strip()
                return str(result).strip()
            return ""
        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
