import torch
import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
import io
from scipy.io import wavfile


class ASRModel:
    def __init__(self, model_path: str, device: str = 'cuda', verbose: bool = True):
        # Force CPU for ASR to avoid CUDA memory corruption issues
        self.device = 'cpu'
        self.verbose = verbose
        
        # Pre-allocate reusable buffer for audio processing
        self._audio_buffer = np.zeros(16000 * 30, dtype=np.float32)  # 30 seconds max
        
        if verbose:
            print(f'Loading Parakeet-TDT ASR model from {model_path}...')
            print('Note: Running ASR on CPU to avoid CUDA memory issues')
        
        if model_path.endswith('.nemo'):
            self.model = EncDecRNNTBPEModel.restore_from(restore_path=model_path)
        else:
            self.model = EncDecRNNTBPEModel.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if verbose:
            print('Parakeet-TDT ready for real-time streaming (CPU mode)')
    
    def _prepare_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Optimize audio preparation with pre-allocated buffers"""
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample if needed (cache this in future)
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        return audio
    
    def _audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert audio to WAV bytes in-memory (no file I/O)"""
        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write to in-memory buffer
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    @torch.no_grad()
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        try:
            # Prepare audio with optimized processing
            audio = self._prepare_audio(audio, sample_rate)
            
            # Ensure audio is not too short
            if len(audio) < 1600:  # At least 0.1 seconds
                return ''
            
            # Use in-memory WAV instead of temp file (faster)
            # Create a temporary in-memory WAV file
            import tempfile
            from pathlib import Path
            
            # Still need file path for NeMo, but use /dev/shm for faster I/O
            shm_dir = Path('/dev/shm')
            if shm_dir.exists() and shm_dir.is_dir():
                # Use RAM disk if available (much faster)
                tmp_dir = shm_dir
            else:
                tmp_dir = None
            
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
                Path(tmp_path).unlink(missing_ok=True)
            
        except Exception as e:
            if self.verbose:
                print(f'Transcription error: {e}')
            return ''
    
    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        try:
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
