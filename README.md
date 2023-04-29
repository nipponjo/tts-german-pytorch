# tts-german-pytorch

FastPitch ([arXiv](https://arxiv.org/abs/2006.06873)) trained on [Thorsten Müller](https://github.com/thorstenMueller)'s [Thorsten–2022.10](https://doi.org/10.5281/zenodo.7265581) and [Thorsten-21.06-emotional](https://doi.org/10.5281/zenodo.5525023) datasets.

<div align="center">
  <img src="https://user-images.githubusercontent.com/28433296/234650185-9a841968-00a9-4169-8bdb-16bfb44277e1.png" width="90%"></img>
</div>

## Audio Samples

You can listen to some audio samples [here](https://nipponjo.github.io/tts-german-samples).

## Quick Setup
Required packages:
`torch torchaudio pyyaml phonemizer`

 Please refer to [here](https://bootphon.github.io/phonemizer/install.html) to install `phonemizer` and the `espeak-ng` backend.
 
~ for training: `librosa matplotlib tensorboard`

~ for the demo app: `fastapi "uvicorn[standard]"`

Download the pretrained weights for the FastPitch model [link](https://drive.google.com/u/1/uc?id=1AqgObECvPp2SrYtklmNWuaDLmY1CU95Y&export=download).

Download the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) weights ([link](https://drive.google.com/u/0/uc?id=1HZJ2kRMysldjxr3MjAndyD-jA2mVucHT&export=download)). Either put them into `pretrained/hifigan-thor-v1` or edit the following lines in `configs/basic.yaml`.

```yaml
# vocoder
vocoder_state_path: pretrained/hifigan-thor-v1/hifigan-thor.pth
vocoder_config_path: pretrained/hifigan-thor-v1/config.json
```

## Using the models

The `FastPitch` from `models.fastpitch` is a wrapper that simplifies text-to-mel inference. The `FastPitch2Wave` model includes the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan) for direct text-to-speech inference.

## Inferring the Mel spectrogram

```python
from models.fastpitch import FastPitch
model = FastPitch('pretrained/fastpitch_de.pth')
model = model.cuda()
mel_spec = model.ttmel("Hallo Welt!")
```

## End-to-end Text-to-Speech

```python
from models.fastpitch import FastPitch2Wave
model = FastPitch2Wave('pretrained/fastpitch_de.pth')
model = model.cuda()
wave = model.tts("Hallo Welt!")

wave_list = model.tts(["null", "eins", "zwei", "drei", "vier", "fünf"])
```

## Web app

The web app uses the FastAPI library. To run the app you need the following packages:

fastapi: for the backend api | uvicorn: for serving the app

Install with: `pip install fastapi "uvicorn[standard]"`

Run with: `python app.py`

Preview:

<div align="center">
  <img src="https://user-images.githubusercontent.com/28433296/235007435-242e1bf8-5edc-466f-97f9-a993eda26935.png" width="66%"></img>
</div>


## Acknowledgements

Thanks to [Thorsten Müller](https://github.com/thorstenMueller) for the high-quality datasets.

The FastPitch files stem from NVIDIA's [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/)


