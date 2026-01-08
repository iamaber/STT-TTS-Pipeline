import torch
import numpy as np
import os
from nemo.collections.asr.models import ASRModel as NeMoASRModel
from app.config import settings
from scipy.io import wavfile

# CUDA Performance Optimizations for TDT model
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"


class ASRModel:
    def __init__(self, model_path: str, device: str = "cuda", verbose: bool = True):
        self.verbose = verbose
        self._warmed_up = False

        # Use GPU for TDT model
        if torch.cuda.is_available() and device == settings.asr.device:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model = NeMoASRModel.from_pretrained(model_name=model_path)

        # Configure preprocessor
        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set decoding strategy to greedy for better performance
        try:
            self.model.change_decoding_strategy(decoding_cfg={"strategy": "greedy"})
        except Exception as e:
            print(f"Could not set greedy strategy: {e}")

    def transcribe_audio(
        self, audio: np.ndarray, sample_rate: int = settings.streaming.sample_rate
    ) -> str:
        try:
            # Prepare audio
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Only resample if sample_rate is not already 16000 (allow small tolerance)
            if abs(sample_rate - settings.asr.sample_rate) > 5:
                import librosa

                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=settings.asr.sample_rate
                )
                sample_rate = settings.asr.sample_rate

            # Use temp file in RAM disk (/dev/shm) for fastest I/O
            import tempfile
            from pathlib import Path

            shm_dir = Path("/dev/shm")
            tmp_dir = shm_dir if shm_dir.exists() else None

            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, dir=tmp_dir
            ) as tmp:
                tmp_path = tmp.name
                wavfile.write(
                    tmp_path, settings.asr.sample_rate, (audio * 32767).astype(np.int16)
                )

            try:
                with torch.no_grad():
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                        transcriptions = self.model.transcribe([tmp_path], batch_size=1)
                        torch.cuda.empty_cache()
                    else:
                        transcriptions = self.model.transcribe([tmp_path], batch_size=1)

                # Process results and move to CPU if needed
                if transcriptions and len(transcriptions) > 0:
                    result = transcriptions[0]
                    if hasattr(result, "text"):
                        text_result = result.text.strip()
                    else:
                        text_result = str(result).strip()

                    # Ensure result is on CPU for further processing
                    if isinstance(text_result, torch.Tensor):
                        text_result = text_result.cpu()

                    return text_result

            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    if self.verbose:
                        print(f"CUDA memory error: {e}")

                    # Aggressive cache clearing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                    # Retry transcription
                    try:
                        with torch.inference_mode():
                            if self.device.type == "cuda":
                                with torch.autocast(
                                    device_type="cuda", dtype=torch.float16
                                ):
                                    transcriptions = self.model.transcribe(
                                        [tmp_path], batch_size=256
                                    )
                            else:
                                transcriptions = self.model.transcribe(
                                    [tmp_path], batch_size=256
                                )

                        if transcriptions and len(transcriptions) > 0:
                            result = transcriptions[0]
                            if hasattr(result, "text"):
                                return result.text.strip()
                            return str(result).strip()
                    except Exception as retry_error:
                        if self.verbose:
                            print(f"Retry failed: {retry_error}")
                    return ""
                else:
                    raise
            finally:
                # Clean up temp file and GPU memory
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            if self.verbose:
                print(f"Transcription error: {e}")
            return ""
