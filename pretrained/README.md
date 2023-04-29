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