# Neoclassical Style Transfer

This repository contains the active technical development line for my EMI 2026 final project.

The project explores how to transform a source image or camera frame into a neoclassical oil-painting style while preserving the original content, composition, and geometry. The work is developed as a sequence of prototypes, starting from traditional image processing and moving toward Stable Diffusion, ControlNet, IP-Adapter reference-image guidance, LoRA style training, and InstantStyle-style block-wise control.

If this repository is cloned onto another machine and needs to be continued with Codex, start here:

```text
docs/CODEX_SERVER_HANDOFF_zh.md
```

## Current Prototype

The first prototype is stored in a file with a historical name:

```text
src/neoclassical_street_demo.py
```

It should now be understood as a **real-time neoclassical style-transfer baseline**, not a street-only project. The script uses real-time image processing to create a neoclassical oil-painting look:

1. edge-aware smoothing
2. muted colour palette
3. classical light and shadow shaping
4. warm or marble-like split toning
5. canvas texture
6. vignette
7. optional side-by-side comparison

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the camera demo:

```powershell
python src/neoclassical_street_demo.py --compare
```

If the default camera does not open:

```powershell
python src/neoclassical_street_demo.py --camera 1 --compare
```

Run a non-camera self-test:

```powershell
python src/neoclassical_street_demo.py --self-test
```

## Controls

While the camera window is open:

1. `1`: David warm oil
2. `2`: Ingres soft glaze
3. `3`: Marble study
4. `+`: increase style strength
5. `-`: decrease style strength
6. `c`: toggle comparison view
7. `s`: save screenshot
8. `q` or `Esc`: quit

## Project Direction

The project is organized around six technical versions:

| Version | Method | Purpose |
| --- | --- | --- |
| V1 | OpenCV visual baseline | Fast handcrafted style-transfer baseline without Stable Diffusion |
| V2 | Stable Diffusion img2img + style prompt | Use the source image as the content input and a text prompt as style guidance |
| V3 | Stable Diffusion img2img + ControlNet Canny | Preserve source structure more strongly through edge control |
| V4 | Stable Diffusion img2img + ControlNet + IP-Adapter | Use a neoclassical painting reference image to guide the style while keeping layout |
| V5 | Stable Diffusion img2img + ControlNet + IP-Adapter + light LoRA | Tighten neoclassical style consistency without letting LoRA replace content control |
| V6 | Stable Diffusion img2img + ControlNet + InstantStyle-style IP-Adapter blocks | Restrict image-prompt injection to style-focused blocks for stronger style control with less layout drift |

The current design goal is style transfer, not scene generation: the source image should keep its layout, perspective, objects, and geometry, while the rendering style changes toward restrained neoclassical oil painting.

All diffusion-based versions now use the same SD1.5 model family by default:

```text
base model: runwayml/stable-diffusion-v1-5
ControlNet: lllyasviel/sd-controlnet-canny
IP-Adapter: h94/IP-Adapter, models/ip-adapter_sd15.bin
LoRA target: SD1.5 UNet LoRA
```

Version 2 adds a Stable Diffusion image-to-image prototype:

```text
src/diffusion_neoclassical_demo.py
```

This version takes a camera frame or still image and uses a diffusion model to translate the source image toward a neoclassical oil-painting style. It is not intended to be high-FPS realtime video. In camera mode, the user points the camera, presses `g`, and the current frame is sent to the diffusion model.

Each saved generation keeps the evidence pair:

1. the original input image
2. the generated neoclassical image
3. a side-by-side comparison image

Install the extra diffusion dependencies:

```powershell
pip install -r requirements_diffusion.txt
```

Run the diffusion camera prototype:

```powershell
python src/diffusion_neoclassical_demo.py
```

Run it on one image:

```powershell
python src/diffusion_neoclassical_demo.py --image path\to\source_photo.jpg
```

When `--lora` is used in image mode, the script saves both the baseline output and the LoRA output:

```powershell
python src/diffusion_neoclassical_demo.py --image path\to\source_photo.jpg --lora lora_outputs\neoclassical_style_lora_sd15
```

Version 2 does not include training yet. The reason is practical: diffusion inference can be tested first, then a later version can add a LoRA training workflow once the dataset and aesthetic target are clearer.

Supporting workflow: LoRA training for later V5 experiments. It adds a neoclassical style dataset structure and helper scripts:

```text
data/neoclassical_lora/
scripts/download_neoclassical_references.py
scripts/prepare_lora_dataset.py
scripts/export_lora_dataset.ps1
scripts/run_lora_training.ps1
src/lora_comparison_demo.py
docs/version_3_lora_training_plan_zh.md
docs/version_3_gpu_training_runbook_zh.md
docs/version_3_local_cpu_training_log_zh.md
```

The Version 2 prompt has also been adjusted toward style transfer only: preserve the source image content and change the visual rendering style, instead of asking the model to invent a new scene.

Download public-domain/open-access starter references:

```powershell
python scripts\download_neoclassical_references.py --limit 30
```

Prepare local LoRA metadata:

```powershell
python scripts\prepare_lora_dataset.py --write-missing-captions
```

Export the local training dataset for Colab or another GPU machine:

```powershell
.\scripts\export_lora_dataset.ps1
```

After a LoRA has been trained, compare baseline and LoRA outputs:

```powershell
python src\lora_comparison_demo.py --image path\to\source.jpg --lora lora_outputs\neoclassical_style_lora_sd15
```

Version 3 adds a ControlNet Canny prototype for stronger structure preservation:

```text
src/controlnet_neoclassical_demo.py
docs/version_4_controlnet_canny_zh.md
```

Run a ControlNet test:

```powershell
python src\controlnet_neoclassical_demo.py --image path\to\source.jpg --size 512 --steps 28 --strength 0.32 --control-scale 0.7 --guidance 5.0 --canny-low 100 --canny-high 200 --no-window
```

Version 4 adds an IP-Adapter prototype on top of ControlNet for reference-guided style transfer:

```text
src/ip_adapter_style_transfer_demo.py
docs/version_5_ip_adapter_zh.md
```

IP-Adapter uses a separate style reference image. In this project, that reference should be a neoclassical painting. It uses the same SD1.5 model family as the other diffusion branches, so the comparison is easier to interpret.

Two implementation details matter a lot here:

1. the style reference is square-preprocessed before CLIP encoding because the official IP-Adapter README notes square references work better
2. the default `--ip-adapter-scale` is now `0.5`, matching Tencent IP-Adapter guidance for balanced multimodal prompting

Run a ControlNet + IP-Adapter test:

```powershell
python src\ip_adapter_style_transfer_demo.py --image path\to\source.jpg --style-image path\to\neoclassical_reference.jpg --size 512 --steps 28 --strength 0.32 --control-scale 0.7 --ip-adapter-scale 0.5 --guidance 5.0 --style-square-mode squash --no-window
```

Version 5 keeps the V4 pipeline and adds a light LoRA:

```powershell
python src\ip_adapter_style_transfer_demo.py --image path\to\source.jpg --style-image path\to\neoclassical_reference.jpg --lora lora_outputs\neoclassical_style_lora_sd15 --size 512 --steps 28 --strength 0.32 --control-scale 0.7 --ip-adapter-scale 0.5 --guidance 5.0 --style-square-mode squash --no-window
```

Version 6 adds an InstantStyle-style branch by using layer-wise IP-Adapter scales instead of activating IP-Adapter everywhere:

```text
src/instantstyle_neoclassical_demo.py
docs/version_6_instantstyle_zh.md
```

This branch follows the official diffusers and InstantStyle guidance: use style-focused attention blocks only by default, and optionally enable style+layout blocks when needed.

Run a V6 InstantStyle test:

```powershell
python src\instantstyle_neoclassical_demo.py --image path\to\source.jpg --style-image path\to\neoclassical_reference.jpg --size 512 --steps 30 --strength 0.35 --control-scale 0.7 --ip-adapter-scale 0.8 --instantstyle-mode style --guidance 5.0 --style-square-mode squash --no-window
```

Generate a one-image comparison grid. The grid saves the original image plus six method outputs:

```powershell
python src\method_comparison_demo.py --image path\to\source.jpg --style-image path\to\neoclassical_reference.jpg --size 512 --steps 28 --strength 0.32 --control-scale 0.7 --guidance 5.0 --ip-adapter-scale 0.5 --instantstyle-scale 0.8 --style-square-mode squash
```

The comparison script saves:

1. original source image
2. V1 OpenCV output
3. V2 Stable Diffusion prompt-only output
4. V3 ControlNet output
5. V4 ControlNet + IP-Adapter output
6. V5 ControlNet + IP-Adapter + LoRA output
7. V6 InstantStyle output

## Narrative Note

Some internal filenames still keep earlier labels such as `neoclassical_street_demo.py`. Those names are now compatibility artifacts for scripts and packaging rather than the definition of the project.

The project narrative is unified as:

**Neoclassical Style Transfer**

## AI and Third-party Resources

This is a working draft. The final submitted README and weblog should be rewritten in my own words and should clearly state which parts were assisted by AI tools, which libraries were used, and what my own contribution was.
