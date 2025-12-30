import nemo.collections.tts as nemo_tts
import torch
import numpy as np
from typing import Optional


class TTSModel:
    def __init__(self, acoustic_model: str, vocoder_model: str, device: str = 'cuda'):
        self.device = device
        
        self.spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
            acoustic_model
        ).to(device)
        self.spec_generator.eval()
        
        self.vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
            vocoder_model
        ).to(device)
        self.vocoder.eval()
        
    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None) -> np.ndarray:
        parsed = self.spec_generator.parse(text)
        
        if speaker is not None and hasattr(self.spec_generator, 'speaker'):
            spectrogram = self.spec_generator.generate_spectrogram(
                tokens=parsed, 
                speaker=speaker
            )
        else:
            spectrogram = self.spec_generator.generate_spectrogram(tokens=parsed)
        
        audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
        
        audio_np = audio.squeeze().cpu().numpy()
        return audio_np
    
    @torch.no_grad()
    def synthesize_batch(self, texts: list[str], speaker: Optional[int] = None) -> list[np.ndarray]:
        audios = []
        
        for text in texts:
            audio = self.synthesize(text, speaker)
            audios.append(audio)
            
        return audios
