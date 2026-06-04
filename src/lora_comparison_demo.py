from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image

from diffusion_neoclassical_demo import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    generate_image,
    load_pipeline,
    resize_for_diffusion,
)


def make_triptych(original: Image.Image, baseline: Image.Image, lora: Image.Image) -> Image.Image:
    original = original.convert("RGB").resize(baseline.size, Image.Resampling.LANCZOS)
    baseline = baseline.convert("RGB")
    lora = lora.convert("RGB").resize(baseline.size, Image.Resampling.LANCZOS)

    width, height = baseline.size
    canvas = Image.new("RGB", (width * 3, height), (20, 18, 16))
    canvas.paste(original, (0, 0))
    canvas.paste(baseline, (width, 0))
    canvas.paste(lora, (width * 2, 0))
    return canvas


def save_lora_comparison(
    original: Image.Image,
    baseline: Image.Image,
    lora: Image.Image,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "original": output_dir / f"lora_compare_original_{timestamp}.png",
        "baseline": output_dir / f"lora_compare_baseline_{timestamp}.png",
        "lora": output_dir / f"lora_compare_lora_{timestamp}.png",
        "triptych": output_dir / f"lora_compare_triptych_{timestamp}.png",
    }
    original.convert("RGB").save(paths["original"])
    baseline.convert("RGB").save(paths["baseline"])
    lora.convert("RGB").save(paths["lora"])
    make_triptych(original, baseline, lora).save(paths["triptych"])
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline diffusion output with a trained LoRA output.")
    parser.add_argument("--image", required=True, help="Input source image.")
    parser.add_argument("--lora", required=True, help="Trained LoRA directory or safetensors file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SD1.5 base model id.")
    parser.add_argument("--size", type=int, default=512, help="Longest side passed into diffusion model.")
    parser.add_argument("--steps", type=int, default=12, help="Inference steps.")
    parser.add_argument("--strength", type=float, default=0.45, help="Image-to-image strength.")
    parser.add_argument("--guidance", type=float, default=6.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable comparison.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output images.")
    parser.add_argument("--quiet", action="store_true", help="Hide diffusion progress bars.")
    parser.add_argument("--disable-safety-checker", action="store_true", help="Accepted for CLI compatibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Image.open(args.image)
    init_image = resize_for_diffusion(source, args.size)

    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.lora = None
    pipe, _ = load_pipeline(baseline_args)

    print("Generating baseline image without LoRA...")
    baseline = generate_image(pipe, init_image, baseline_args)

    print(f"Loading LoRA for comparison: {args.lora}")
    pipe.load_lora_weights(args.lora)
    print("Generating image with LoRA...")
    lora_output = generate_image(pipe, init_image, args)

    paths = save_lora_comparison(init_image, baseline, lora_output, Path(args.output_dir))
    print(f"Saved original: {paths['original']}")
    print(f"Saved baseline: {paths['baseline']}")
    print(f"Saved LoRA: {paths['lora']}")
    print(f"Saved triptych: {paths['triptych']}")


if __name__ == "__main__":
    main()
