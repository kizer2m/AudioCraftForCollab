# 🎵 AudioCraft for Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kizer2m/AudioCraftForCollab/blob/main/AudioCraft_Colab.ipynb)

> ⚡ **Optimized for Google Colab!**  
> This repository is a Google Colab-optimized fork of Meta AI's [AudioCraft](https://github.com/facebookresearch/audiocraft) library. It includes a turnkey Colab Notebook, a unified Gradio Web UI, non-breaking dependency setups, and automatic VRAM memory management for T4, V100, and A100 GPU runtimes.

---

## 🚀 Google Colab Quickstart

1. Click the **[Open in Colab](https://colab.research.google.com/github/kizer2m/AudioCraftForCollab/blob/main/AudioCraft_Colab.ipynb)** badge above.
2. Enable GPU Acceleration in Colab:  
   `Runtime` -> `Change runtime type` -> select **T4 GPU** (or V100/A100).
3. Execute the notebook cells in order:
   - **Step 1**: Check GPU environment (`!nvidia-smi`)
   - **Step 2**: Install FFmpeg and Colab-compatible dependencies (`!pip install -r requirements_colab.txt`)
   - **Step 3**: Launch the Web UI (`!python colab_app.py --share`)
4. Open the generated **`https://xxxx.gradio.live`** public link to start generating audio!

---

## ✨ Features & Colab Optimizations

- 🎛️ **Unified Multi-Model Web UI (`colab_app.py`)**: Generate music, sound effects, and audio samples using **MusicGen**, **AudioGen**, and **MAGNeT** models from a single interactive interface.
- 🧹 **VRAM Memory Hygiene**: Automatically calls `gc.collect()` and `torch.cuda.empty_cache()` between generation calls to prevent Out-Of-Memory (OOM) errors on Google Colab free T4 GPUs (~15GB VRAM).
- ⚙️ **PyTorch 2.x & CUDA Compatibility**: Resolved strict version lockings (`torch==2.1.0`, `xformers<0.0.23`, `av==11.0.0`) to run seamlessly on Google Colab's default Python 3.10/3.11 environment.
- 📓 **Interactive Notebook (`AudioCraft_Colab.ipynb`)**: Pre-configured cells for environment check, repository setup, Web UI launching, and direct Python API code execution with inline audio playback.

---

## 💻 Local & Manual Installation

To install and run AudioCraft locally or on a custom server:

```bash
# Install FFmpeg (required for audio processing)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Clone the repository
git clone https://github.com/kizer2m/AudioCraftForCollab.git
cd AudioCraftForCollab

# Install dependencies and package
pip install -r requirements_colab.txt
pip install -e .
```

---

## 🎼 Models Supported

AudioCraft contains inference and training code for Meta AI's state-of-the-art generative audio models:

* **[MusicGen](./docs/MUSICGEN.md)**: Controllable text-to-music and melody-conditioned music generation.
* **[AudioGen](./docs/AUDIOGEN.md)**: High-fidelity text-to-sound-effect generation.
* **[MAGNeT](./docs/MAGNET.md)**: Fast non-autoregressive text-to-music and text-to-sound model.
* **[EnCodec](./docs/ENCODEC.md)**: High-fidelity neural audio codec.
* **[MultiBand Diffusion](./docs/MBD.md)**: EnCodec-compatible diffusion decoder for enhanced audio quality.
* **[JASCO](./docs/JASCO.md)**: Text-to-music generation conditioned on chords, melodies, and drum tracks.
* **[AudioSeal](./docs/WATERMARKING.md)**: Audio watermarking for AI-generated sound.

---

## 🐍 Python API Usage Example

You can also use AudioCraft directly in Python code or inside Colab notebook cells:

```python
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from IPython.display import Audio, display

# 1. Load pretrained MusicGen model
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10) # 10 seconds generation

# 2. Generate music from text prompt
descriptions = ['80s synthwave track with heavy bassline and punchy drums']
print("Generating music...")
wav = model.generate(descriptions) # [batch, channels, time]

# 3. Save to WAV audio file
audio_data = wav[0].cpu()
audio_write('generated_track', audio_data, model.sample_rate, strategy="loudness", add_suffix=False)

# 4. Play audio inline (in Jupyter/Colab)
display(Audio('generated_track.wav'))
```

---

## ❓ FAQ

#### Where are pretrained model weights stored?
Hugging Face stores downloaded models in `~/.cache/huggingface/hub`. You can override the AudioCraft model cache location by setting the `AUDIOCRAFT_CACHE_DIR` environment variable.

#### Is training code provided?
Yes, training pipelines for EnCodec, MusicGen, AudioGen, MultiBand Diffusion, and JASCO are included in the repository. Refer to the [Training Documentation](./docs/TRAINING.md) for details.

---

## 📜 License
- **Code**: Released under the [MIT License](LICENSE).
- **Model Weights**: Released under the [CC-BY-NC 4.0 License](LICENSE_weights).

---

## 📑 Citation

If you use AudioCraft or MusicGen in your research, please cite:

```bibtex
@inproceedings{copet2023simple,
    title={Simple and Controllable Music Generation},
    author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```
