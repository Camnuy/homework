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

This first version is not yet the final machine-learning system. It is a fast visual prototype used to test the interaction and aesthetic direction.

The planned next stage is to build a diffusion-based version:

1. use a large image-to-image model for neoclassical street translation
2. collect a small neoclassical painting reference dataset
3. train or fine-tune a LoRA style adapter
4. compare the OpenCV prototype with the diffusion/LoRA version
5. evaluate speed, stability, visual quality, and ethical issues around street cameras

## AI and Third-party Resources

This is a working draft. The final submitted README and weblog should be rewritten in my own words and should clearly state which parts were assisted by AI tools, which libraries were used, and what my own contribution was.
