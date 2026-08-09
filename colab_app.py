# Copyright (c) Meta Platforms, Inc. and affiliates.
# AudioCraft Colab Optimized Launcher App
# Multi-model Web UI for Google Colab (MusicGen, AudioGen, MAGNeT)

import argparse
import gc
import os
import sys
import time
import typing as tp
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen, AudioGen, MAGNeT


CURRENT_MODEL = None
CURRENT_MODEL_NAME = None


def clear_gpu_memory():
    """Clean up memory and empty PyTorch CUDA cache."""
    global CURRENT_MODEL, CURRENT_MODEL_NAME
    if CURRENT_MODEL is not None:
        del CURRENT_MODEL
        CURRENT_MODEL = None
        CURRENT_MODEL_NAME = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_musicgen_model(model_name: str):
    global CURRENT_MODEL, CURRENT_MODEL_NAME
    if CURRENT_MODEL_NAME != model_name or CURRENT_MODEL is None:
        print(f"Loading MusicGen model: {model_name}")
        clear_gpu_memory()
        CURRENT_MODEL = MusicGen.get_pretrained(model_name)
        CURRENT_MODEL_NAME = model_name
    return CURRENT_MODEL


def load_audiogen_model(model_name: str = "facebook/audiogen-medium"):
    global CURRENT_MODEL, CURRENT_MODEL_NAME
    if CURRENT_MODEL_NAME != model_name or CURRENT_MODEL is None:
        print(f"Loading AudioGen model: {model_name}")
        clear_gpu_memory()
        CURRENT_MODEL = AudioGen.get_pretrained(model_name)
        CURRENT_MODEL_NAME = model_name
    return CURRENT_MODEL


def load_magnet_model(model_name: str = "facebook/magnet-small-10secs"):
    global CURRENT_MODEL, CURRENT_MODEL_NAME
    if CURRENT_MODEL_NAME != model_name or CURRENT_MODEL is None:
        print(f"Loading MAGNeT model: {model_name}")
        clear_gpu_memory()
        CURRENT_MODEL = MAGNeT.get_pretrained(model_name)
        CURRENT_MODEL_NAME = model_name
    return CURRENT_MODEL


def generate_music(
    text: str,
    melody_audio: tp.Optional[tp.Tuple[int, torch.Tensor]],
    model_name: str,
    duration: int,
    top_k: int,
    top_p: float,
    temperature: float,
    cfg_coef: float,
    progress=gr.Progress(track_tqdm=True)
):
    if not text and melody_audio is None:
        raise gr.Error("Please provide either a text description or a melody audio file.")

    progress(0, desc="Loading MusicGen Model...")
    model = load_musicgen_model(model_name)
    model.set_generation_params(
        duration=duration,
        top_k=int(top_k),
        top_p=float(top_p),
        temperature=float(temperature),
        cfg_coef=float(cfg_coef)
    )

    progress(0.3, desc="Generating Music...")
    be = time.time()
    
    if melody_audio is not None and "melody" in model_name:
        sr, melody_np = melody_audio
        melody_tensor = torch.from_numpy(melody_np).to(model.device).float()
        if melody_tensor.dim() == 1:
            melody_tensor = melody_tensor[None]
        if melody_tensor.dim() == 2 and melody_tensor.shape[0] > 2:
            melody_tensor = melody_tensor.t()
        melody_tensor = melody_tensor[..., :int(sr * duration)]
        melody_tensor = convert_audio(melody_tensor, sr, 32000, 1)
        output = model.generate_with_chroma([text], melody_tensor[None], 32000)
    else:
        output = model.generate([text])

    output = output.detach().cpu().float()[0]
    
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as tmp_file:
        audio_write(
            tmp_file.name,
            output,
            model.sample_rate,
            strategy="loudness",
            loudness_headroom_db=16,
            loudness_compressor=True,
            add_suffix=False
        )
        wav_path = tmp_file.name

    print(f"Generation took {time.time() - be:.2f} seconds.")
    clear_gpu_memory()
    return wav_path


def generate_audio_effects(
    text: str,
    model_name: str,
    duration: int,
    top_k: int,
    top_p: float,
    temperature: float,
    cfg_coef: float,
    progress=gr.Progress(track_tqdm=True)
):
    if not text:
        raise gr.Error("Please provide a text description for AudioGen.")

    progress(0, desc="Loading AudioGen Model...")
    model = load_audiogen_model(model_name)
    model.set_generation_params(
        duration=duration,
        top_k=int(top_k),
        top_p=float(top_p),
        temperature=float(temperature),
        cfg_coef=float(cfg_coef)
    )

    progress(0.3, desc="Generating Sound Effect...")
    be = time.time()
    output = model.generate([text])
    output = output.detach().cpu().float()[0]

    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as tmp_file:
        audio_write(
            tmp_file.name,
            output,
            model.sample_rate,
            strategy="loudness",
            loudness_headroom_db=16,
            loudness_compressor=True,
            add_suffix=False
        )
        wav_path = tmp_file.name

    print(f"AudioGen generation took {time.time() - be:.2f} seconds.")
    clear_gpu_memory()
    return wav_path


def generate_magnet_audio(
    text: str,
    model_name: str,
    temperature: float,
    max_cfg_coef: float,
    min_cfg_coef: float,
    progress=gr.Progress(track_tqdm=True)
):
    if not text:
        raise gr.Error("Please provide a text description for MAGNeT.")

    progress(0, desc="Loading MAGNeT Model...")
    model = load_magnet_model(model_name)
    model.set_generation_params(
        temperature=float(temperature),
        max_cfg_coef=float(max_cfg_coef),
        min_cfg_coef=float(min_cfg_coef)
    )

    progress(0.3, desc="Generating Audio with MAGNeT...")
    be = time.time()
    output = model.generate([text])
    output = output.detach().cpu().float()[0]

    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as tmp_file:
        audio_write(
            tmp_file.name,
            output,
            model.sample_rate,
            strategy="loudness",
            loudness_headroom_db=16,
            loudness_compressor=True,
            add_suffix=False
        )
        wav_path = tmp_file.name

    print(f"MAGNeT generation took {time.time() - be:.2f} seconds.")
    clear_gpu_memory()
    return wav_path


def build_colab_ui():
    with gr.Blocks(title="AudioCraft Google Colab WebUI") as demo:
        gr.Markdown(
            """
            # 🎵 AudioCraft Web UI (Google Colab Optimized)
            Generate music, sound effects, and audio samples using Meta's **MusicGen**, **AudioGen**, and **MAGNeT** models.
            """
        )
        with gr.Tabs():
            # Tab 1: MusicGen
            with gr.TabItem("🎼 MusicGen"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Prompt / Music Description",
                            placeholder="An 80s synthwave track with heavy bass and energetic drums",
                            lines=3
                        )
                        melody_input = gr.Audio(
                            label="Conditioning Melody Audio (Optional)",
                            type="numpy"
                        )
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "facebook/musicgen-small",
                                "facebook/musicgen-medium",
                                "facebook/musicgen-melody",
                                "facebook/musicgen-large",
                                "facebook/musicgen-stereo-small",
                                "facebook/musicgen-stereo-medium",
                                "facebook/musicgen-stereo-melody"
                            ],
                            value="facebook/musicgen-small",
                            label="Model Preset (Use small or medium for faster generation on T4)"
                        )
                        duration_slider = gr.Slider(
                            minimum=1, maximum=60, value=10, step=1, label="Duration (seconds)"
                        )
                        with gr.Accordion("Advanced Settings", open=False):
                            top_k = gr.Number(label="Top-k", value=250)
                            top_p = gr.Number(label="Top-p", value=0.0)
                            temperature = gr.Number(label="Temperature", value=1.0)
                            cfg_coef = gr.Number(label="Classifier Free Guidance (CFG)", value=3.0)
                        
                        generate_btn = gr.Button("🚀 Generate Music", variant="primary")

                    with gr.Column():
                        audio_output = gr.Audio(label="Generated Music (WAV)", type="filepath")

                generate_btn.click(
                    fn=generate_music,
                    inputs=[
                        text_input, melody_input, model_dropdown,
                        duration_slider, top_k, top_p, temperature, cfg_coef
                    ],
                    outputs=[audio_output]
                )

            # Tab 2: AudioGen
            with gr.TabItem("🔊 AudioGen (Sound Effects)"):
                with gr.Row():
                    with gr.Column():
                        ag_text = gr.Textbox(
                            label="Sound Effect Description",
                            placeholder="Dog barking in a spacious park with wind blowing",
                            lines=3
                        )
                        ag_model = gr.Dropdown(
                            choices=["facebook/audiogen-medium"],
                            value="facebook/audiogen-medium",
                            label="AudioGen Model"
                        )
                        ag_duration = gr.Slider(
                            minimum=1, maximum=30, value=5, step=1, label="Duration (seconds)"
                        )
                        with gr.Accordion("Advanced Settings", open=False):
                            ag_top_k = gr.Number(label="Top-k", value=250)
                            ag_top_p = gr.Number(label="Top-p", value=0.0)
                            ag_temperature = gr.Number(label="Temperature", value=1.0)
                            ag_cfg = gr.Number(label="CFG Coef", value=3.0)

                        ag_btn = gr.Button("🚀 Generate Sound Effect", variant="primary")

                    with gr.Column():
                        ag_output = gr.Audio(label="Generated Sound Effect (WAV)", type="filepath")

                ag_btn.click(
                    fn=generate_audio_effects,
                    inputs=[
                        ag_text, ag_model, ag_duration,
                        ag_top_k, ag_top_p, ag_temperature, ag_cfg
                    ],
                    outputs=[ag_output]
                )

            # Tab 3: MAGNeT
            with gr.TabItem("⚡ MAGNeT (Fast Non-Autoregressive)"):
                with gr.Row():
                    with gr.Column():
                        mag_text = gr.Textbox(
                            label="Prompt",
                            placeholder="Disco beat with electric piano and funky synth line",
                            lines=3
                        )
                        mag_model = gr.Dropdown(
                            choices=[
                                "facebook/magnet-small-10secs",
                                "facebook/magnet-medium-10secs"
                            ],
                            value="facebook/magnet-small-10secs",
                            label="MAGNeT Model"
                        )
                        with gr.Accordion("Advanced Settings", open=False):
                            mag_temp = gr.Number(label="Temperature", value=3.0)
                            mag_max_cfg = gr.Number(label="Max CFG", value=10.0)
                            mag_min_cfg = gr.Number(label="Min CFG", value=1.0)

                        mag_btn = gr.Button("🚀 Generate with MAGNeT", variant="primary")

                    with gr.Column():
                        mag_output = gr.Audio(label="Generated Audio (WAV)", type="filepath")

                mag_btn.click(
                    fn=generate_magnet_audio,
                    inputs=[mag_text, mag_model, mag_temp, mag_max_cfg, mag_min_cfg],
                    outputs=[mag_output]
                )

        gr.Markdown(
            """
            ---
            *Optimized for Google Colab GPU environment (T4 / V100 / A100).*
            *Repository: [AudioCraftForCollab](https://github.com/kizer2m/AudioCraftForCollab)*
            """
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AudioCraft Colab WebUI Launcher")
    parser.add_argument("--share", action="store_true", default=True, help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app")
    args = parser.parse_args()

    ui = build_colab_ui()
    ui.queue().launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
