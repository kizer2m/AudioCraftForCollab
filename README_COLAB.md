# 🚀 AudioCraft for Google Colab Guide / Инструкция по запуску AudioCraft в Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kizer2m/AudioCraftForCollab/blob/main/AudioCraft_Colab.ipynb)

Данный репозиторий содержит оптимизированную версию библиотеки **Meta AudioCraft** для комфортного и стабильного запуска в **Google Colab** (включая бесплатные GPU T4).

---

## 🇷🇺 Инструкция на русском языке

### 1. Быстрый запуск через Google Colab Notebook
1. Нажмите на кнопку **[Open in Colab](https://colab.research.google.com/github/kizer2m/AudioCraftForCollab/blob/main/AudioCraft_Colab.ipynb)** выше.
2. В верхнем меню Colab убедитесь, что включен GPU ускоритель:  
   `Среда выполнения` -> `Сменить тип среды выполнения` -> выберите `T4 GPU` (или `V100`/`A100`).
3. Запустите первую ячейку с `!nvidia-smi` для проверки подключения видеокарты.
4. Запустите ячейку установки зависимостей (установит `ffmpeg`, клонирует репозиторий и установит `requirements_colab.txt`).
5. Запустите ячейку Web UI:
   ```bash
   !python colab_app.py --share
   ```
6. Перейдите по сгенерированной публичной ссылке вида **`https://xxxx.gradio.live`**.

### 2. Возможности Web UI (`colab_app.py`)
- **🎼 MusicGen**: Генерация музыки по текстовому описанию и образцу мелодии (`facebook/musicgen-small`, `medium`, `melody`, `stereo`).
- **🔊 AudioGen**: Генерация звуковых эффектов (`facebook/audiogen-medium`).
- **⚡ MAGNeT**: Быстрая неавторегрессионная генерация аудио (`facebook/magnet-small-10secs`).
- **Авто-очистка VRAM**: Автоматическое освобождение оперативной памяти видеокарты (`torch.cuda.empty_cache()`) после каждой генерации для предотвращения ошибок Out of Memory (OOM).

### 3. Запуск через Python API в ячейках Colab
Вы также можете использовать AudioCraft прямо в Python коде:
```python
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from IPython.display import Audio, display

# Загрузка модели (рекомендуется small или medium для T4 GPU)
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10) # Длительность 10 сек

# Генерация
wav = model.generate(['synthwave track with heavy bass and energetic drums'])

# Сохранение и воспроизведение
audio_data = wav[0].cpu()
audio_write('music_sample', audio_data, model.sample_rate, strategy="loudness", add_suffix=False)
display(Audio('music_sample.wav'))
```

---

## 🇬🇧 English Instructions

### 1. Quick Start with Colab Notebook
1. Click the **[Open in Colab](https://colab.research.google.com/github/kizer2m/AudioCraftForCollab/blob/main/AudioCraft_Colab.ipynb)** badge at the top.
2. Ensure GPU accelerator is enabled:  
   `Runtime` -> `Change runtime type` -> select `T4 GPU` (or `V100`/`A100`).
3. Run Step 1 (`!nvidia-smi`) to check your GPU environment.
4. Run Step 2 cell to install `ffmpeg` and Colab-compatible requirements.
5. Run Step 3 cell to launch the Web UI:
   ```bash
   !python colab_app.py --share
   ```
6. Click the generated public Gradio link **`https://xxxx.gradio.live`**.

### 2. Key Optimizations for Colab
- Non-breaking requirements compatible with Colab's pre-installed PyTorch & CUDA.
- Integrated Gradio Web UI supporting MusicGen, AudioGen, and MAGNeT.
- Automatic PyTorch CUDA memory cache clearing (`gc.collect()` & `torch.cuda.empty_cache()`) to prevent T4 VRAM Out-Of-Memory errors.

---
*Repository: [AudioCraftForCollab](https://github.com/kizer2m/AudioCraftForCollab)*
