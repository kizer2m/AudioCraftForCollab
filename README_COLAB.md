# 🚀 AudioCraft for Google Colab Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kizer2m/AudioCraftForCollab/blob/main/AudioCraft_Colab.ipynb)

This repository contains an optimized version of Meta's **AudioCraft** library specifically tuned for stable execution in **Google Colab** (including free T4 GPU instances).

---

## 📖 Step-by-Step Google Colab Guide

### 1. Launching via Google Colab Notebook
1. Click the **[Open in Colab](https://colab.research.google.com/github/kizer2m/AudioCraftForCollab/blob/main/AudioCraft_Colab.ipynb)** badge above.
2. Ensure GPU acceleration is enabled:  
   `Runtime` -> `Change runtime type` -> select **T4 GPU** (or `V100`/`A100`).
3. Run Step 1 (`!nvidia-smi`) to check your GPU environment.
4. Run Step 2 cell to install system dependencies (`ffmpeg`, `pkg-config`, `libavformat-dev`) and Colab-compatible Python packages.
5. Run Step 3 cell to launch the Web UI:
   ```bash
   !python colab_app.py --share
   ```
6. Click the generated public Gradio link **`https://xxxx.gradio.live`**.

---

### 2. Web UI Features (`colab_app.py`)
- **🎼 MusicGen**: Text-to-Music & Melody-conditioned generation (`facebook/musicgen-small`, `medium`, `melody`, `stereo`).
- **🔊 AudioGen**: Sound effect generation (`facebook/audiogen-medium`).
- **⚡ MAGNeT**: Fast non-autoregressive audio generation (`facebook/magnet-small-10secs`).
- **Memory Cleanup**: Automatic VRAM cache clearing (`torch.cuda.empty_cache()` & `gc.collect()`) after generation to prevent Out-Of-Memory (OOM) errors.

---

### 3. Python API Code Example
You can also run AudioCraft directly via Python in notebook cells:

```python
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from IPython.display import Audio, display

# Load MusicGen model (small or medium recommended for T4 GPU)
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10) # 10 seconds duration

# Generate audio
wav = model.generate(['synthwave track with heavy bass and energetic drums'])

# Save & Play audio inline
audio_data = wav[0].cpu()
audio_write('music_sample', audio_data, model.sample_rate, strategy="loudness", add_suffix=False)
display(Audio('music_sample.wav'))
```

---
*Repository: [AudioCraftForCollab](https://github.com/kizer2m/AudioCraftForCollab)*
