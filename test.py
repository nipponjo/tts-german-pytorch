# %%
import matplotlib.pyplot as plt
import torch

from models.fastpitch import FastPitch, FastPitch2Wave
from vocoder import load_hifigan
from vocoder.hifigan.denoiser import Denoiser

from IPython.display import Audio

# %%

text0 = "Hallo Welt! Wie geht's?"

sd_path = './pretrained/fastpitch_de.pth'

# %%

mel_model = FastPitch(sd_path).cuda()

vocoder = load_hifigan('./pretrained/hifigan-thor-v1/hifigan-thor.pth', 
                       './pretrained/hifigan-thor-v1/config.json')
vocoder = vocoder.cuda()
denoiser = Denoiser(vocoder)

# %%

mel_spec = mel_model.ttmel(text0, emotion_id='amused')
with torch.inference_mode():
    wave_gen = vocoder(mel_spec)[0]
wave_enhan = denoiser(wave_gen, strength=0.003)[0]

Audio(0.5*wave_enhan.cpu(), rate=22050)

# %%

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.imshow(mel_spec.cpu(), origin='lower', aspect='auto')
ax2.plot(wave_enhan.cpu())

# %%

wave_model = FastPitch2Wave(sd_path).cuda()

# %%

wave_gen = wave_model.tts(text0)
Audio(0.5*wave_gen, rate=22050)

# %%

fig, ax = plt.subplots()
ax.plot(wave_gen.cpu())

# %%

phonemes_ipa = """ bɛɾlˈiːn ɪst diː hˈaʊptʃtat ʊnt aɪn lˈant dɛɾ bˈʊndəsrˌeːpuːblˌɪk dˈɔøtʃlant."""

wave_gen = wave_model.tts(phonemes_ipa, ipa=True)
Audio(0.5*wave_gen, rate=22050)

# %%
