import torch
import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
from scipy.io import wavfile
import tempfile
from pathlib import Path
import os


# CRITICAL: Enable expandable segments for CUDA memory allocation
# This prevents memory corruption in RNNT models (from NeMo official example)
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in alloc_conf:
    if len(alloc_conf) > 0:
        alloc_conf += ",expandable_segments:True"
    else:
        alloc_conf = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf


class ASRModel:
    def __init__(self, model_path: str, device: str = 'cuda', verbose: bool = True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        # Pre-allocate reusable buffer for audio processing
        self._audio_buffer = np.zeros(16000 * 30, dtype=np.float32)  # 30 seconds max
        
        if verbose:
            print(f'Loading Parakeet-TDT ASR model from {model_path}...')
            print(f'Device: {self.device.upper()}')
            if self.device == 'cuda':
                print('Using expandable_segments for CUDA memory (prevents corruption)')
        
        if model_path.endswith('.nemo'):
            self.model = EncDecRNNTBPEModel.restore_from(restore_path=model_path)
        else:
            self.model = EncDecRNNTBPEModel.from_pretrained(model_path)
        
        # Configure for streaming inference (from NeMo example)
        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if verbose:
            print(f'Parakeet-TDT ready for real-time streaming ({self.device.upper()} mode)')
    
    def _prepare_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Optimize audio preparation with pre-allocated buffers"""
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        return audio
    
    @torch.no_grad()
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        try:
            # Prepare audio with optimized processing
            audio = self._prepare_audio(audio, sample_rate)
            
            # Ensure audio is not too short
            if len(audio) < 1600:  # At least 0.1 seconds
                return ''
            
            # Clear CUDA cache before transcription
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Use RAM disk if available for faster I/O
            shm_dir = Path('/dev/shm')
            tmp_dir = shm_dir if (shm_dir.exists() and shm_dir.is_dir()) else None
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=tmp_dir) as tmp:
                tmp_path = tmp.name
                # Write directly without intermediate conversion
                wavfile.write(tmp_path, 16000, (audio * 32767).astype(np.int16))
            
            try:
                # Transcribe from file
                transcriptions = self.model.transcribe([tmp_path], batch_size=1)
                
                if transcriptions and len(transcriptions) > 0:
                    result = transcriptions[0]
                    if hasattr(result, 'text'):
                        return result.text.strip()
                    return str(result).strip()
                return ''
                    
            finally:
                # Clean up temp file
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except:
                    pass
            
        except Exception as e:
            if self.verbose:
                print(f'Transcription error: {e}')
            return ''
    
    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        try:
            # Clear CUDA cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            transcriptions = self.model.transcribe([audio_path])
            
            if transcriptions and len(transcriptions) > 0:
                result = transcriptions[0]
                if hasattr(result, 'text'):
                    return result.text.strip()
                return str(result).strip()
            return ''
        except Exception as e:
            if self.verbose:
                print(f'File transcription error: {e}')
            return ''
