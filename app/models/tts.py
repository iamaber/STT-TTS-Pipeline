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
        self.max_text_length = 300
        
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
    
    def _split_text(self, text: str):
        import re
        
        if len(text) <= self.max_text_length:
            return [text]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ''
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_text_length:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text[:self.max_text_length]]
    
    def generate(self, text: str, speaker: int = None, pace: float = 1.0, 
                 output: Optional[str] = None):
        # Use config default if speaker not specified
        from app.config import settings
        if speaker is None:
            speaker = settings.tts.default_speaker_id
        
        pace = max(0.1, min(2.0, pace))
        if speaker >= self.n_speakers:
            raise ValueError(f'Speaker {speaker} out of range (0-{self.n_speakers-1})')
        
        chunks = self._split_text(text)
        audio_chunks = []
        
        for chunk in chunks:
            with torch.no_grad():
                tokens = self.fastpitch.parse(chunk)
                spec = self.fastpitch.generate_spectrogram(tokens=tokens, speaker=speaker, pace=pace)
                audio = self.hifigan.convert_spectrogram_to_audio(spec=spec)
            
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()
            
            audio_chunks.append(audio)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        combined_audio = np.concatenate(audio_chunks)
        
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, combined_audio, self.sample_rate)
            if self.verbose:
                print(f'Saved: {output} ({len(combined_audio)/self.sample_rate:.2f}s)')
        
        return combined_audio
    
    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None):
        return self.generate(text, speaker=speaker or 92)
    
    @torch.no_grad()
    def synthesize_batch(self, texts, speaker: Optional[int] = None):
        return [self.generate(text, speaker=speaker or 92) for text in texts]


class TTSModel:
    def __init__(self, acoustic_model: str, vocoder_model: str, device: str = 'cuda'):
        self.tts = TTS(acoustic_model=acoustic_model, vocoder_model=vocoder_model, 
                      device=device, verbose=True)
        
    @torch.no_grad()
    def synthesize(self, text: str, speaker: Optional[int] = None):
        return self.tts.synthesize(text, speaker)
    
    @torch.no_grad()
    def synthesize_batch(self, texts, speaker: Optional[int] = None):
        return self.tts.synthesize_batch(texts, speaker)
