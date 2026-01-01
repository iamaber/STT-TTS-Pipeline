import torch
import numpy as np
import os
from nemo.collections.asr.models import EncDecRNNTBPEModel


# CRITICAL: Enable expandable segments for CUDA memory allocation
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in alloc_conf:
    if len(alloc_conf) > 0:
        alloc_conf += ",expandable_segments:True"
    else:
        alloc_conf = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf


class ASRModel:
    """Wrapper for ASR model - uses buffered streaming on GPU"""

    def __init__(self, model_path: str, device: str = "cuda", verbose: bool = True):
        # Try to use buffered streaming on GPU
        try:
            from app.models.asr_buffered import BufferedASRModel

            self.model = BufferedASRModel(model_path, device, verbose)
            self.use_buffered = True
            if verbose:
                print("Using GPU buffered streaming mode")
        except Exception as e:
            # Fallback to CPU simple mode
            if verbose:
                print(f"Buffered streaming failed ({e}), using CPU fallback")
            self._init_cpu_fallback(model_path, verbose)
            self.use_buffered = False

    def _init_cpu_fallback(self, model_path: str, verbose: bool):
        """Fallback to simple CPU mode"""
        self.device = "cpu"
        self.verbose = verbose

        if verbose:
            print("Loading model in CPU fallback mode...")

        if model_path.endswith(".nemo"):
            self.cpu_model = EncDecRNNTBPEModel.restore_from(restore_path=model_path)
        else:
            self.cpu_model = EncDecRNNTBPEModel.from_pretrained(model_path)

        self.cpu_model.preprocessor.featurizer.dither = 0.0
        self.cpu_model.preprocessor.featurizer.pad_to = 0
        self.cpu_model = self.cpu_model.to("cpu")
        self.cpu_model.eval()

        if verbose:
            print("CPU fallback ready")

    @torch.no_grad()
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self.use_buffered:
            return self.model.transcribe_audio(audio, sample_rate)
        else:
            return self._cpu_transcribe(audio, sample_rate)

    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        if self.use_buffered:
            return self.model.transcribe_file(audio_path)
        else:
            try:
                transcriptions = self.cpu_model.transcribe([audio_path])
                if transcriptions and len(transcriptions) > 0:
                    result = transcriptions[0]
                    if hasattr(result, "text"):
                        return result.text.strip()
                    return str(result).strip()
                return ""
            except Exception as e:
                if self.verbose:
                    print(f"File transcription error: {e}")
                return ""

    def _cpu_transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Simple CPU transcription fallback"""
        try:
            # Prepare audio
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            if sample_rate != 16000:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            if len(audio) < 1600:
                return ""

            # Save to temp file and transcribe
            import tempfile
            from pathlib import Path
            from scipy.io import wavfile

            shm_dir = Path("/dev/shm")
            tmp_dir = shm_dir if (shm_dir.exists() and shm_dir.is_dir()) else None

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=tmp_dir
            ) as tmp:
                tmp_path = tmp.name
                wavfile.write(tmp_path, 16000, (audio * 32767).astype(np.int16))

            try:
                transcriptions = self.cpu_model.transcribe([tmp_path], batch_size=1)
                if transcriptions and len(transcriptions) > 0:
                    result = transcriptions[0]
                    if hasattr(result, "text"):
                        return result.text.strip()
                    return str(result).strip()
                return ""
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        except Exception as e:
            if self.verbose:
                print(f"CPU transcription error: {e}")
            return ""
