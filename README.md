# Real-time Neoclassical Street Translation

This repository contains the first prototype for my EMI 2026 final project.

The current version is an OpenCV-based real-time camera demo. It reads a webcam feed and applies a neoclassical-inspired visual treatment to the live image. The goal is to test the basic creative direction before moving into a larger machine-learning version using diffusion models and LoRA fine-tuning.

## Current Prototype

The first prototype is:

```text
src/neoclassical_street_demo.py
```

It uses real-time image processing to create a neoclassical oil-painting look:

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

Version 1 is not yet the final machine-learning system. It is a fast visual prototype used to test the interaction and aesthetic direction.

Version 2 adds a Stable Diffusion image-to-image prototype:

```text
src/diffusion_neoclassical_demo.py
```

This version takes a camera frame or still image and uses a diffusion model to translate it toward a neoclassical oil-painting style. It is not intended to be high-FPS realtime video. In camera mode, the user points the camera, presses `g`, and the current frame is sent to the diffusion model.

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
python src/diffusion_neoclassical_demo.py --image path\to\street_photo.jpg
```

Version 2 does not include training yet. The reason is practical: diffusion inference can be tested first, then a later version can add a LoRA training workflow once the dataset and aesthetic target are clearer.

Version 3 starts the LoRA training workflow. It adds a neoclassical style dataset structure and helper scripts:

```text
data/neoclassical_lora/
scripts/download_neoclassical_references.py
scripts/prepare_lora_dataset.py
scripts/run_lora_training.ps1
src/lora_comparison_demo.py
docs/version_3_lora_training_plan_zh.md
```

The Version 2 prompt has also been adjusted toward style transfer only: preserve the original street layout and change the visual rendering style, instead of asking the model to invent a new scene.

Download public-domain/open-access starter references:

```powershell
python scripts\download_neoclassical_references.py --limit 30
```

Prepare local LoRA metadata:

```powershell
python scripts\prepare_lora_dataset.py --write-missing-captions
```

After a LoRA has been trained, compare baseline and LoRA outputs:

```powershell
python src\lora_comparison_demo.py --image path\to\street.jpg --lora lora_outputs\neoclassical_style_lora
```

The planned next stage is to add training:

1. use a large image-to-image model for neoclassical street translation
2. collect a small neoclassical painting reference dataset
3. train or fine-tune a LoRA style adapter
4. compare the OpenCV prototype with the diffusion/LoRA version
5. evaluate speed, stability, visual quality, and ethical issues around street cameras

## AI and Third-party Resources

This is a working draft. The final submitted README and weblog should be rewritten in my own words and should clearly state which parts were assisted by AI tools, which libraries were used, and what my own contribution was.
