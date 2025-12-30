import torch
import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel


class ASRModel:
    def __init__(self, model_path: str, device: str = 'cuda', verbose: bool = True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        if verbose:
            print(f'Loading Parakeet-TDT ASR model from {model_path}...')
        
        if model_path.endswith('.nemo'):
            self.model = EncDecRNNTBPEModel.restore_from(restore_path=model_path)
        else:
            self.model = EncDecRNNTBPEModel.from_pretrained(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if verbose:
            print('Parakeet-TDT ready for real-time streaming')
    
    @torch.no_grad()
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        transcriptions = self.model.transcribe([audio], batch_size=1)
        
        if transcriptions and len(transcriptions) > 0:
            result = transcriptions[0]
            if hasattr(result, 'text'):
                return result.text
            return str(result)
        return ''
    
    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        transcriptions = self.model.transcribe([audio_path])
        
        if transcriptions and len(transcriptions) > 0:
            result = transcriptions[0]
            if hasattr(result, 'text'):
                return result.text
            return str(result)
        return ''
