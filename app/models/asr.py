import nemo.collections.asr as nemo_asr
import torch
import numpy as np
from typing import Optional, List


class ASRModel:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def transcribe_file(self, audio_path: str) -> str:
        transcription = self.model.transcribe([audio_path])[0]
        return transcription
    
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model.forward(
                input_signal=audio_tensor.unsqueeze(0),
                input_signal_length=torch.tensor([len(audio)]).to(self.device)
            )
            
        transcription = self.model.decoding.ctc_decoder_predictions_tensor(
            logits[0], 
            decoder_lengths=None
        )[0][0]
        
        return transcription
    
    @torch.no_grad()
    def transcribe_streaming(self, audio_chunks: List[np.ndarray], sample_rate: int = 16000) -> List[str]:
        transcriptions = []
        
        for chunk in audio_chunks:
            audio_tensor = torch.from_numpy(chunk).float().to(self.device)
            logits = self.model.forward(
                input_signal=audio_tensor.unsqueeze(0),
                input_signal_length=torch.tensor([len(chunk)]).to(self.device)
            )
            
            transcription = self.model.decoding.ctc_decoder_predictions_tensor(
                logits[0],
                decoder_lengths=None
            )[0][0]
            
            transcriptions.append(transcription)
            
        return transcriptions
