from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from controlnet_neoclassical_demo import generator_for, make_canny_image
from diffusion_neoclassical_demo import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_MODEL,
    choose_device,
    pil_to_cv,
    resize_for_diffusion,
)


DEFAULT_CONTROLNET = "lllyasviel/sd-controlnet-canny"
DEFAULT_IP_ADAPTER_REPO = "h94/IP-Adapter"
DEFAULT_IP_ADAPTER_SUBFOLDER = "models"
DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"
DEFAULT_IMAGE_ENCODER_FOLDER = "models/image_encoder"
DEFAULT_STYLE_DATASET_DIR = Path("data/neoclassical_lora")
WINDOW_NAME = "IP-Adapter Neoclassical Style Transfer"


def import_ip_adapter_stack():
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing IP-Adapter dependencies.\n"
            "Install diffusion requirements first:\n"
            r"& C:\Users\23913\.conda\envs\homework2\python.exe -m pip install -r requirements_diffusion.txt"
        ) from exc
    return torch, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline


def find_style_image(style_image: str | None, dataset_dir: str | Path) -> Path | None:
    if style_image:
        path = Path(style_image)
        return path if path.exists() else None

    root = Path(dataset_dir)
    if not root.exists():
        return None

    candidates = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        candidates.extend(root.rglob(pattern))
    return sorted(candidates)[0] if candidates else None


def load_pipeline(args: argparse.Namespace):
    torch, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline = import_ip_adapter_stack()
    device, dtype = choose_device(torch)
    print(f"Loading IP-Adapter base model: {args.model}")
    print(f"Loading ControlNet: {args.controlnet}")
    print(f"Loading IP-Adapter: {args.ip_adapter_repo}/{args.ip_adapter_subfolder}/{args.ip_adapter_weight}")
    print(f"Device: {device} | dtype: {dtype}")
    if device == "cpu":
        print("Warning: SD1.5 + ControlNet + IP-Adapter on CPU is slow. Use low --size/--steps for tests.")

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
    pipe.load_ip_adapter(
        args.ip_adapter_repo,
        subfolder=args.ip_adapter_subfolder,
        weight_name=args.ip_adapter_weight,
        image_encoder_folder=args.image_encoder_folder,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=args.quiet)

    return pipe


def generate_ip_adapter_image(
    pipe,
    source_image: Image.Image,
    control_image: Image.Image,
    style_image: Image.Image,
    args: argparse.Namespace,
) -> Image.Image:
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=source_image,
        control_image=control_image,
        ip_adapter_image=style_image,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        controlnet_conditioning_scale=args.control_scale,
        generator=generator_for(pipe, args.seed),
    )
    return result.images[0]


def label_image(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    label_height = 34
    result = Image.new("RGB", (image.width, image.height + label_height), (20, 18, 16))
    result.paste(image, (0, label_height))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    text_width = draw.textlength(label, font=font)
    draw.text(((image.width - text_width) / 2, 11), label, fill=(242, 232, 214), font=font)
    return result


def make_reference_comparison(
    source_image: Image.Image,
    style_image: Image.Image,
    generated_image: Image.Image,
) -> Image.Image:
    width, height = generated_image.size
    panels = [
        ("Source", source_image.resize((width, height), Image.Resampling.LANCZOS)),
        ("Style Reference", style_image.resize((width, height), Image.Resampling.LANCZOS)),
        ("IP-Adapter Result", generated_image),
    ]
    labelled = [label_image(image, label) for label, image in panels]
    comparison = Image.new("RGB", (width * len(labelled), labelled[0].height), (20, 18, 16))
    for index, panel in enumerate(labelled):
        comparison.paste(panel, (index * width, 0))
    return comparison


def save_outputs(
    source_image: Image.Image,
    style_image: Image.Image,
    control_image: Image.Image,
    generated_image: Image.Image,
    output_dir: str | Path,
    prefix: str = "ip_adapter_neoclassical",
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "source": out_dir / f"{prefix}_source_{timestamp}.png",
        "style": out_dir / f"{prefix}_style_reference_{timestamp}.png",
        "control": out_dir / f"{prefix}_canny_{timestamp}.png",
        "generated": out_dir / f"{prefix}_generated_{timestamp}.png",
        "comparison": out_dir / f"{prefix}_comparison_{timestamp}.png",
    }
    source_image.convert("RGB").save(paths["source"])
    style_image.convert("RGB").save(paths["style"])
    control_image.convert("RGB").save(paths["control"])
    generated_image.convert("RGB").save(paths["generated"])
    make_reference_comparison(source_image, style_image, generated_image).save(paths["comparison"])
    return paths


def run_image(args: argparse.Namespace) -> dict[str, Path]:
    style_path = find_style_image(args.style_image, args.style_dataset_dir)
    if not style_path:
        raise SystemExit(
            "No style reference image found. Pass --style-image path\\to\\neoclassical_reference.jpg "
            "or download references into data\\neoclassical_lora first."
        )

    source_image = resize_for_diffusion(Image.open(args.image), args.size)
    style_image = resize_for_diffusion(Image.open(style_path), args.style_size)
    control_image = make_canny_image(source_image, args.canny_low, args.canny_high)

    print(f"Using style reference: {style_path}")
    pipe = load_pipeline(args)
    print("Generating IP-Adapter style-transfer result...")
    generated_image = generate_ip_adapter_image(pipe, source_image, control_image, style_image, args)

    paths = save_outputs(source_image, style_image, control_image, generated_image, args.output_dir)
    if not args.no_window:
        cv2.imshow(WINDOW_NAME, pil_to_cv(make_reference_comparison(source_image, style_image, generated_image)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-guided neoclassical style transfer with Stable Diffusion, ControlNet, and IP-Adapter."
    )
    parser.add_argument("--image", required=True, help="Source image to stylize.")
    parser.add_argument("--style-image", help="Neoclassical painting reference image for IP-Adapter.")
    parser.add_argument("--style-dataset-dir", default=str(DEFAULT_STYLE_DATASET_DIR), help="Fallback directory for style references.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SD1.5-compatible base model.")
    parser.add_argument("--controlnet", default=DEFAULT_CONTROLNET, help="SD1.5 Canny ControlNet model.")
    parser.add_argument("--ip-adapter-repo", default=DEFAULT_IP_ADAPTER_REPO, help="IP-Adapter repository.")
    parser.add_argument("--ip-adapter-subfolder", default=DEFAULT_IP_ADAPTER_SUBFOLDER, help="IP-Adapter weight subfolder.")
    parser.add_argument("--ip-adapter-weight", default=DEFAULT_IP_ADAPTER_WEIGHT, help="IP-Adapter weight file.")
    parser.add_argument("--image-encoder-folder", default=DEFAULT_IMAGE_ENCODER_FOLDER, help="IP-Adapter image encoder folder.")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.75, help="Reference image influence strength.")
    parser.add_argument("--size", type=int, default=384, help="Longest side for the source image.")
    parser.add_argument("--style-size", type=int, default=384, help="Longest side for the style reference image.")
    parser.add_argument("--steps", type=int, default=12, help="Inference steps. Increase for quality, decrease for CPU tests.")
    parser.add_argument("--strength", type=float, default=0.45, help="Img2img strength. Lower preserves more source content.")
    parser.add_argument("--guidance", type=float, default=6.5, help="Classifier-free guidance scale.")
    parser.add_argument("--control-scale", type=float, default=0.9, help="How strongly ControlNet follows source edges.")
    parser.add_argument("--canny-low", type=int, default=80, help="Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=180, help="Canny high threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable comparisons.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive style-transfer prompt.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--quiet", action="store_true", help="Hide diffusion progress bars.")
    parser.add_argument("--no-window", action="store_true", help="Do not open preview window.")
    parser.add_argument("--disable-safety-checker", action="store_true", help="Disable the local safety checker.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.steps * args.strength) < 1:
        raise SystemExit("Increase --steps or --strength: image-to-image needs at least one effective denoising step.")

    paths = run_image(args)
    print(f"Saved source: {paths['source']}")
    print(f"Saved style reference: {paths['style']}")
    print(f"Saved Canny control image: {paths['control']}")
    print(f"Saved generated: {paths['generated']}")
    print(f"Saved comparison: {paths['comparison']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
