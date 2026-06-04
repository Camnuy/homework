from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from diffusion_neoclassical_demo import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    choose_device,
    make_comparison_image,
    make_lora_comparison_image,
    pil_to_cv,
    resize_for_diffusion,
)


WINDOW_NAME = "ControlNet Neoclassical Street Translation"


def import_controlnet_stack():
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing ControlNet dependencies.\n"
            "Install diffusion requirements first:\n"
            r"& C:\Users\23913\.conda\envs\homework2\python.exe -m pip install -r requirements_diffusion.txt"
        ) from exc
    return torch, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline


def make_canny_image(image: Image.Image, low: int, high: int) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = np.repeat(edges[:, :, None], 3, axis=2)
    return Image.fromarray(edges_rgb)


def generator_for(pipe, seed: int | None):
    if seed is None:
        return None
    import torch

    device = getattr(pipe, "_execution_device", getattr(pipe, "device", "cpu"))
    generator_device = "cpu" if str(device).startswith("mps") else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def load_pipeline(args: argparse.Namespace):
    torch, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline = import_controlnet_stack()
    device, dtype = choose_device(torch)
    print(f"Loading base model: {args.model}")
    print(f"Loading ControlNet: {args.controlnet}")
    print(f"Device: {device} | dtype: {dtype}")
    if device == "cpu":
        print("Warning: ControlNet on CPU is slow. Use low --size/--steps for tests.")

    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    load_kwargs = {}
    if args.disable_safety_checker:
        load_kwargs["safety_checker"] = None

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model,
        controlnet=controlnet,
        torch_dtype=dtype,
        **load_kwargs,
    )
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=args.quiet)

    return pipe


def generate_controlnet_image(
    pipe,
    init_image: Image.Image,
    control_image: Image.Image,
    args: argparse.Namespace,
) -> Image.Image:
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        control_image=control_image,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        controlnet_conditioning_scale=args.control_scale,
        generator=generator_for(pipe, args.seed),
    )
    return result.images[0]


def save_controlnet_outputs(
    original: Image.Image,
    control: Image.Image,
    generated: Image.Image,
    prefix: str = "controlnet_neoclassical",
) -> dict[str, Path]:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "original": out_dir / f"{prefix}_original_{timestamp}.png",
        "control": out_dir / f"{prefix}_canny_{timestamp}.png",
        "generated": out_dir / f"{prefix}_generated_{timestamp}.png",
        "comparison": out_dir / f"{prefix}_comparison_{timestamp}.png",
    }
    original.convert("RGB").save(paths["original"])
    control.convert("RGB").save(paths["control"])
    generated.convert("RGB").save(paths["generated"])
    make_comparison_image(original, generated).save(paths["comparison"])
    return paths


def save_controlnet_lora_outputs(
    original: Image.Image,
    control: Image.Image,
    baseline: Image.Image,
    lora: Image.Image,
    prefix: str = "controlnet_lora_compare",
) -> dict[str, Path]:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "original": out_dir / f"{prefix}_original_{timestamp}.png",
        "control": out_dir / f"{prefix}_canny_{timestamp}.png",
        "baseline": out_dir / f"{prefix}_baseline_{timestamp}.png",
        "lora": out_dir / f"{prefix}_lora_{timestamp}.png",
        "triptych": out_dir / f"{prefix}_triptych_{timestamp}.png",
    }
    original.convert("RGB").save(paths["original"])
    control.convert("RGB").save(paths["control"])
    baseline.convert("RGB").save(paths["baseline"])
    lora.convert("RGB").save(paths["lora"])
    make_lora_comparison_image(original, baseline, lora).save(paths["triptych"])
    return paths


def run_image(args: argparse.Namespace) -> None:
    source = Image.open(args.image)
    init_image = resize_for_diffusion(source, args.size)
    control_image = make_canny_image(init_image, args.canny_low, args.canny_high)
    pipe = load_pipeline(args)

    if args.lora:
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.lora = None
        print("Generating ControlNet baseline without LoRA...")
        baseline = generate_controlnet_image(pipe, init_image, control_image, baseline_args)

        print(f"Loading LoRA: {args.lora}")
        pipe.load_lora_weights(args.lora)
        print("Generating ControlNet result with LoRA...")
        lora_output = generate_controlnet_image(pipe, init_image, control_image, args)

        paths = save_controlnet_lora_outputs(init_image, control_image, baseline, lora_output)
        print(f"Saved original: {paths['original']}")
        print(f"Saved canny: {paths['control']}")
        print(f"Saved baseline: {paths['baseline']}")
        print(f"Saved LoRA: {paths['lora']}")
        print(f"Saved triptych: {paths['triptych']}")

        if not args.no_window:
            cv2.imshow(WINDOW_NAME, pil_to_cv(make_lora_comparison_image(init_image, baseline, lora_output)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    output = generate_controlnet_image(pipe, init_image, control_image, args)
    paths = save_controlnet_outputs(init_image, control_image, output)
    print(f"Saved original: {paths['original']}")
    print(f"Saved canny: {paths['control']}")
    print(f"Saved generated: {paths['generated']}")
    print(f"Saved comparison: {paths['comparison']}")

    if not args.no_window:
        cv2.imshow(WINDOW_NAME, pil_to_cv(make_comparison_image(init_image, output)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ControlNet Canny neoclassical street style transfer.")
    parser.add_argument("--image", required=True, help="Input street image.")
    parser.add_argument("--model", default="stabilityai/sd-turbo", help="SD 2.x-compatible base model.")
    parser.add_argument("--controlnet", default="thibaud/controlnet-sd21-canny-diffusers", help="SD 2.1 Canny ControlNet model.")
    parser.add_argument("--lora", help="Optional trained LoRA directory or safetensors file.")
    parser.add_argument("--size", type=int, default=512, help="Longest side passed into the model.")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps. SD-Turbo usually uses 2-4.")
    parser.add_argument("--strength", type=float, default=0.42, help="Img2img strength. Lower preserves more structure.")
    parser.add_argument("--guidance", type=float, default=0.0, help="Guidance scale. SD-Turbo usually uses 0.")
    parser.add_argument("--control-scale", type=float, default=1.0, help="How strongly ControlNet follows the Canny edges.")
    parser.add_argument("--canny-low", type=int, default=80, help="Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=180, help="Canny high threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable comparisons.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--quiet", action="store_true", help="Hide diffusion progress bars.")
    parser.add_argument("--no-window", action="store_true", help="Do not open preview window.")
    parser.add_argument("--disable-safety-checker", action="store_true", help="Accepted for local testing compatibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.steps * args.strength) < 1:
        raise SystemExit("Increase --steps or --strength: image-to-image needs at least one effective denoising step.")
    run_image(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
