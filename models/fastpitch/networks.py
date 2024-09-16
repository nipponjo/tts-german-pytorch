from typing import List, Union

import text
import torch
import torch.nn as nn

from utils import get_basic_config
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser

from .fastpitch.model import FastPitch as _FastPitch

emotion_to_id = {'neutral': 0, 'amused': 1, 'angry': 2,
                 'disgusted': 3, 'drunk': 4, 'sleepy': 5,
                 'surprised': 6, 'whisper': 7}

def text_collate_fn(batch: List[torch.Tensor]):
    """
    Args:
        batch: List[text_ids]
    Returns:
        text_ids_pad
        input_lens_sorted
        reverse_ids 
    """
    input_lens_sorted, input_sort_ids = torch.sort(
        torch.LongTensor([len(x) for x in batch]), descending=True)
    max_input_len = input_lens_sorted[0]

    text_ids_pad = torch.LongTensor(len(batch), max_input_len)
    text_ids_pad.zero_()
    for i in range(len(input_sort_ids)):
        text_ids = batch[input_sort_ids[i]]
        text_ids_pad[i, :text_ids.size(0)] = text_ids

    return text_ids_pad, input_lens_sorted, input_sort_ids.argsort()


class FastPitch(_FastPitch):
    def __init__(self,
                 checkpoint: str = None,                
                 **kwargs):
        from models.fastpitch import net_config
        sds = torch.load(checkpoint, map_location='cpu')
        
        if 'config' in sds:
            net_config = sds['config']
        if 'net_config' in sds:
            net_config = sds['net_config']
        super().__init__(**net_config)
        #self.n_eos = len(EOS_TOKENS)

        # if checkpoint is not None:
        self.load_state_dict(sds['model'])

        self.eval()
        
    @property
    def device(self):
        return next(self.parameters()).device

    def _tokenize(self, utterance: str):
        return text.text_to_tokens(utterance)  # , append_space=False)

    @torch.inference_mode()
    def ttmel_single(self,
                     utterance: str,
                     speed: float = 1,
                     speaker_id: int = 0,
                     emotion_id: Union[int, str] = 0,
                     ipa: bool = False,
                     ):
        
        if isinstance(emotion_id, str):
            emotion_id = emotion_to_id[emotion_id]

        
        tokens = utterance if ipa else self._tokenize(utterance)

        token_ids = text.tokens_to_ids(tokens)
        ids_batch = torch.LongTensor(token_ids).unsqueeze(0).to(self.device)
        # sid = torch.LongTensor([speaker_id]).to(self.device)

        # Infer spectrogram and wave
        (mel_spec, *_) = self.infer(ids_batch, pace=speed,
                                    speaker=speaker_id,
                                    emotion=emotion_id)

        mel_spec = mel_spec[0]

        return mel_spec  # [F, T]

    @torch.inference_mode()
    def ttmel_batch(self,
                    batch: List[str],
                    speed: float = 1,
                    speaker_id: int = 0,
                    emotion_id: Union[int, str] = 0,
                    ipa: bool = False,
                    ):
        
        if isinstance(emotion_id, str):
            emotion_id = emotion_to_id[emotion_id]

        batch_tokens = [line if ipa else self._tokenize(line) for line in batch]

        batch_ids = [torch.LongTensor(text.tokens_to_ids(tokens))
                     for tokens in batch_tokens]

        batch = text_collate_fn(batch_ids)
        (batch_ids_padded, batch_lens_sorted,
         reverse_sort_ids) = batch

        batch_ids_padded = batch_ids_padded.to(self.device)
        batch_lens_sorted = batch_lens_sorted.to(self.device)

        batch_sids = batch_lens_sorted*0 + speaker_id

        y_pred = self.infer(batch_ids_padded, pace=speed,
                            speaker=speaker_id,
                            emotion=emotion_id)
        mel_outputs, mel_specgram_lengths, *_ = y_pred

        mel_list = []
        for i, id in enumerate(reverse_sort_ids):
            mel = mel_outputs[id, :, :mel_specgram_lengths[id]]
            mel_list.append(mel)

        return mel_list

    def ttmel(self,
              text: Union[str, List[str]],
              speed: float = 1,
              speaker_id: int = 0,
              emotion_id: Union[int, str] = 0,
              ipa: bool = False,
              batch_size: int = 1,
              ):

        # input: string
        if isinstance(text, str):
            return self.ttmel_single(text, speed=speed,
                                     speaker_id=speaker_id,
                                     emotion_id=emotion_id,
                                     ipa=ipa)

        # input: list
        assert isinstance(text, list)
        batch = text
        mel_list = []

        if batch_size == 1:
            for sample in batch:
                mel = self.ttmel_single(sample, speed=speed,
                                        speaker_id=speaker_id,
                                        emotion_id=emotion_id,
                                        ipa=ipa)
                mel_list.append(mel)
            return mel_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.ttmel_batch(batch, speed=speed,
                                    speaker_id=speaker_id,
                                    emotion_id=emotion_id,
                                    ipa=ipa)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            mels = self.ttmel_batch(batch, speed=speed,
                                    speaker_id=speaker_id,
                                    emotion_id=emotion_id,
                                    ipa=ipa)
            mel_list += mels

        return mel_list


class FastPitch2Wave(nn.Module):
    def __init__(self,
                 model_sd_path,
                 vocoder_sd=None,
                 vocoder_config=None,                
                 ):

        super().__init__()

        # from models.fastpitch import net_config
        state_dicts = torch.load(model_sd_path, map_location='cpu')
        # if 'config' in state_dicts:
        #     net_config = state_dicts['config']

        model = FastPitch(model_sd_path)
        model.load_state_dict(state_dicts['model'])
        self.model = model

        if vocoder_sd is None or vocoder_config is None:
            config = get_basic_config()
            vocoder_sd = config.vocoder_state_path
            vocoder_config = config.vocoder_config_path

        vocoder = load_hifigan(vocoder_sd, vocoder_config)
        self.vocoder = vocoder
        self.denoiser = Denoiser(vocoder)

        self.eval()
        
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        return x

    @torch.inference_mode()
    def tts_single(self,
                   text_buckw: str,
                   speed: float = 1,
                   speaker_id: int = 0,
                   emotion_id: Union[int, str] = 0,
                   denoise: float = 0,
                   ipa: bool = False,
                   return_mel=False):

        mel_spec = self.model.ttmel_single(text_buckw, speed, 
                                           speaker_id, emotion_id, 
                                           ipa)

        wave = self.vocoder(mel_spec)

        if denoise > 0:
            wave = self.denoiser(wave, denoise)

        if return_mel:
            return wave[0].cpu(), mel_spec

        return wave[0].cpu()

    @torch.inference_mode()
    def tts_batch(self,
                  batch: List[str],
                  speed: float = 1,
                  speaker_id: int = 0,
                  emotion_id: Union[int, str] = 0,                 
                  denoise: float = 0,
                  ipa: bool = False,
                  return_mel=False):

        mel_list = self.model.ttmel_batch(batch, speed, 
                                          speaker_id, emotion_id, 
                                          ipa)

        wav_list = []
        for mel in mel_list:
            wav_inferred = self.vocoder(mel)
            if denoise > 0:
                wav_inferred = self.denoiser(wav_inferred, denoise)

            wav_list.append(wav_inferred[0].cpu())

        if return_mel:
            wav_list, mel_list

        return wav_list

    def tts(self,
            text_buckw: Union[str, List[str]],
            speed: float = 1,
            denoise: float = 0.003,
            speaker_id: int = 0,
            emotion_id: Union[int, str] = 0,
            ipa: bool = False,
            batch_size: int = 2,
            return_mel: bool = False):

        # input: string
        if isinstance(text_buckw, str):
            return self.tts_single(text_buckw, speed=speed, 
                                   speaker_id=speaker_id,
                                   emotion_id=emotion_id,
                                   denoise=denoise, ipa=ipa,
                                   return_mel=return_mel)

        # input: list
        assert isinstance(text_buckw, list)
        batch = text_buckw
        wav_list = []

        if batch_size == 1:
            for sample in batch:
                wav = self.tts_single(sample, speed=speed, 
                                      speaker_id=speaker_id,
                                      emotion_id=emotion_id,
                                      denoise=denoise, ipa=ipa,
                                      return_mel=return_mel)
                wav_list.append(wav)
            return wav_list

        # infer one batch
        if len(batch) <= batch_size:
            return self.tts_batch(batch, speed=speed, 
                                  speaker_id=speaker_id,
                                  emotion_id=emotion_id,
                                  denoise=denoise, ipa=ipa,
                                  return_mel=return_mel)

        # batched inference
        batches = [batch[k:k+batch_size]
                   for k in range(0, len(batch), batch_size)]

        for batch in batches:
            wavs = self.tts_batch(batch, speed=speed, 
                                  speaker_id=speaker_id,
                                  emotion_id=emotion_id,
                                  denoise=denoise, ipa=ipa,
                                  return_mel=return_mel)
            wav_list += wavs

        return wav_list
