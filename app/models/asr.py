import torch
import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
import tempfile
import soundfile as sf
from pathlib import Path


class ASRModel:
    def __init__(self, model_path: str, device: str = 'cuda', verbose: bool = True):
        # Force CPU for ASR to avoid CUDA memory corruption issues
        self.device = 'cpu'
        self.verbose = verbose
        
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
    
    @torch.no_grad()
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        try:
            # Ensure audio is float32 and 1D
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            audio = audio.astype(np.float32)
            
            # Resample if needed
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Ensure audio is not too short
            if len(audio) < 1600:  # At least 0.1 seconds
                return ''
            
            # Save to temp file for more stable transcription
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio, 16000)
            
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
            print(f'File transcription error: {e}')
            return ''
