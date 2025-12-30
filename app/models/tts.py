from pathlib import Path
import soundfile as sf
import torch
import numpy as np
from typing import Optional
from nemo.collections.tts.models import FastPitchModel, HifiGanModel


class TTS:
    def __init__(self, acoustic_model: str = 'tts_en_fastpitch_multispeaker',
                 vocoder_model: str = 'tts_en_hifitts_hifigan_ft_fastpitch',
                 device: str = 'cuda', verbose: bool = True):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        if verbose:
            print('Loading TTS models...')
        
        if acoustic_model.endswith('.nemo'):
            self.fastpitch = FastPitchModel.restore_from(restore_path=acoustic_model)
        else:
            self.fastpitch = FastPitchModel.from_pretrained(acoustic_model)
        
        self.fastpitch = self.fastpitch.to(self.device)
        self.fastpitch.eval()
        
        if vocoder_model.endswith('.nemo'):
            self.hifigan = HifiGanModel.restore_from(restore_path=vocoder_model)
        else:
            self.hifigan = HifiGanModel.from_pretrained(vocoder_model)
        
        self.hifigan = self.hifigan.to(self.device)
        self.hifigan.eval()
        
        self.n_speakers = self.fastpitch.cfg.n_speakers
        self.sample_rate = 44100
        
        if verbose:
            print(f'TTS ready: {self.n_speakers} speakers available')
    
    def generate(self, text: str, speaker: int = 92, pace: float = 1.0, 
                 output: Optional[str] = None) -> np.ndarray:
        pace = max(0.1, min(2.0, pace))
        if speaker >= self.n_speakers:
            raise ValueError(f'Speaker {speaker} out of range (0-{self.n_speakers-1})')
        
        with torch.no_grad():
            tokens = self.fastpitch.parse(text)
            spec = self.fastpitch.generate_spectrogram(tokens=tokens, speaker=speaker, pace=pace)
            audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)
        
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, audio, self.sample_rate)
            if self.verbose:
                print(f'Saved: {output} ({len(audio)/self.sample_rate:.2f}s)')
        
        return audio
    
    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None) -> np.ndarray:
        return self.generate(text, speaker=speaker or 92)
    
    @torch.no_grad()
    def synthesize_batch(self, texts: list[str], speaker: Optional[int] = None) -> list[np.ndarray]:
        return [self.generate(text, speaker=speaker or 92) for text in texts]


class TTSModel:
    def __init__(self, acoustic_model: str, vocoder_model: str, device: str = 'cuda'):
        self.tts = TTS(acoustic_model=acoustic_model, vocoder_model=vocoder_model, 
                      device=device, verbose=True)
        
    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None) -> np.ndarray:
        return self.tts.synthesize(text, speaker)
    
    @torch.no_grad()
    def synthesize_batch(self, texts: list[str], speaker: Optional[int] = None) -> list[np.ndarray]:
        return self.tts.synthesize_batch(texts, speaker)
