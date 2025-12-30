import torch
import numpy as np
from typing import Optional
import logging
import copy
from omegaconf import OmegaConf, open_dict

logging.getLogger('nemo_logging').setLevel(logging.ERROR)


class SimpleFastConformerStreamer:
    def __init__(self, model_name: str = 'nvidia/stt_en_fastconformer_hybrid_large_streaming_multi', 
                 device: str = 'cuda'):
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
            self.EncDecCTCModelBPE = EncDecCTCModelBPE
        except ImportError:
            raise ImportError('NeMo toolkit not installed. Install with: pip install nemo-toolkit[asr]')
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 16000
        self.model_name = model_name
        self.lookahead_size = 480
        self.encoder_step_length = 80
        chunk_size_ms = self.lookahead_size + self.encoder_step_length
        self.chunk_size = int(self.sample_rate * chunk_size_ms / 1000)
        
        print(f'Loading FastConformer on {self.device}...')
        
        if model_name.endswith('.nemo'):
            self.asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=model_name)
        else:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        
        self.asr_model.change_decoding_strategy(decoder_type='rnnt')
        
        left_context = self.asr_model.encoder.att_context_size[0]
        right_context = self.lookahead_size // self.encoder_step_length
        self.asr_model.encoder.set_default_att_context_size([left_context, right_context])
        
        decoding_cfg = self.asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = 'greedy'
            decoding_cfg.preserve_alignments = False
            if hasattr(self.asr_model, 'joint'):
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
            self.asr_model.change_decoding_strategy(decoding_cfg)
        
        self.asr_model.eval()
        self.asr_model = self.asr_model.to(self.device)
        
        cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = 'None'
        
        self.preprocessor = self.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self.preprocessor.to(self.asr_model.device)
        
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros((1, num_channels, self.pre_encode_cache_size), 
                                            device=self.asr_model.device)
        (self.cache_last_channel,
         self.cache_last_time,
         self.cache_last_channel_len) = self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.audio_buffer = np.array([], dtype=np.float32)
        
        print(f'FastConformer ready (lookahead: {self.lookahead_size}ms, chunk: {chunk_size_ms}ms)')
    
    def feed_audio(self, audio_data: bytes):
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
    
    def should_transcribe(self) -> bool:
        return len(self.audio_buffer) >= self.chunk_size
    
    def transcribe(self) -> Optional[str]:
        if not self.should_transcribe():
            return None
        
        try:
            chunk = self.audio_buffer[:self.chunk_size]
            audio_data = np.array(chunk, dtype=np.float32)
            
            audio_signal = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            audio_signal_len = torch.Tensor([audio_data.shape[0]]).to(self.device)
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=audio_signal, length=audio_signal_len
            )
            
            processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
            processed_signal_length += self.cache_pre_encode.shape[2]
            self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]
            
            with torch.no_grad():
                (self.pred_out_stream,
                 transcribed_texts,
                 self.cache_last_channel,
                 self.cache_last_time,
                 self.cache_last_channel_len,
                 self.previous_hypotheses) = self.asr_model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.pred_out_stream,
                    drop_extra_pre_encoded=None,
                    return_transcription=True,
                )
            
            text = ''
            if transcribed_texts and len(transcribed_texts) > 0:
                hyp = transcribed_texts[0]
                text = hyp.text if hasattr(hyp, 'text') else str(hyp)
            
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            if text and text.strip():
                return text.strip()
            return None
        
        except Exception as e:
            print(f'FastConformer error: {e}')
            self.reset()
            return None
    
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if sample_rate != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        self.reset()
        audio_bytes = (audio * 32768.0).astype(np.int16).tobytes()
        self.feed_audio(audio_bytes)
        
        results = []
        while self.should_transcribe():
            text = self.transcribe()
            if text:
                results.append(text)
        
        return ' '.join(results)
    
    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        (self.cache_last_channel,
         self.cache_last_time,
         self.cache_last_channel_len) = self.asr_model.encoder.get_initial_cache_state(batch_size=1)
        self.previous_hypotheses = None
        self.pred_out_stream = None
        num_channels = self.asr_model.cfg.preprocessor.features
        self.cache_pre_encode = torch.zeros((1, num_channels, self.pre_encode_cache_size),
                                            device=self.asr_model.device)


class ASRModel:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.streamer = SimpleFastConformerStreamer(model_name=model_name, device=device)
        
    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return self.streamer.transcribe_audio(audio, sample_rate)
    
    def feed_audio(self, audio_data: bytes):
        self.streamer.feed_audio(audio_data)
    
    def should_transcribe(self) -> bool:
        return self.streamer.should_transcribe()
    
    def transcribe(self) -> Optional[str]:
        return self.streamer.transcribe()
    
    def reset(self):
        self.streamer.reset()
