from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from controlnet_neoclassical_demo import (
    DEFAULT_CONTROLNET,
    generate_controlnet_image,
    load_pipeline as load_controlnet_pipeline,
    make_canny_image,
)
from diffusion_neoclassical_demo import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    generate_image,
    load_pipeline as load_img2img_pipeline,
    pil_to_cv,
    resolve_lora_path,
    resize_for_diffusion,
)
from ip_adapter_style_transfer_demo import (
    DEFAULT_INSTANTSTYLE_MODE,
    DEFAULT_IMAGE_ENCODER_FOLDER,
    DEFAULT_IP_ADAPTER_REPO,
    DEFAULT_IP_ADAPTER_SUBFOLDER,
    DEFAULT_IP_ADAPTER_WEIGHT,
    DEFAULT_STYLE_DATASET_DIR,
    apply_ip_adapter_scale,
    find_style_image,
    generate_ip_adapter_image,
    load_pipeline as load_ip_adapter_pipeline,
    prepare_style_reference,
)
from neoclassical_street_demo import apply_neoclassical, variant_by_key


DEFAULT_LORA = Path(__file__).resolve().parent.parent / "lora_outputs" / "neoclassical_style_lora_sd15"
WINDOW_NAME = "Neoclassical Method Comparison"


def make_args_copy(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_opencv_version(init_image: Image.Image, style: str, strength: float) -> Image.Image:
    frame = pil_to_cv(init_image)
    variant = variant_by_key(style)
    output = apply_neoclassical(frame, variant, strength)
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


def placeholder_panel(size: tuple[int, int], title: str, detail: str) -> Image.Image:
    image = Image.new("RGB", size, (36, 33, 29))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    lines = [title, detail]
    y = max(12, size[1] // 2 - 18)
    for line in lines:
        text_width = draw.textlength(line, font=font)
        draw.text(((size[0] - text_width) / 2, y), line, fill=(230, 220, 204), font=font)
        y += 18
    return image


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


def make_comparison_grid(items: list[tuple[str, Image.Image]]) -> Image.Image:
    base_width, base_height = items[0][1].size
    labelled = [
        label_image(image.resize((base_width, base_height), Image.Resampling.LANCZOS), label)
        for label, image in items
    ]
    grid = Image.new("RGB", (base_width * len(labelled), labelled[0].height), (20, 18, 16))
    for index, panel in enumerate(labelled):
        grid.paste(panel, (index * base_width, 0))
    return grid


def save_outputs(
    output_dir: Path,
    init_image: Image.Image,
    opencv_image: Image.Image,
    sd_image: Image.Image,
    canny_image: Image.Image,
    controlnet_image: Image.Image,
    ip_adapter_image: Image.Image,
    lora_image: Image.Image,
    instantstyle_image: Image.Image,
    comparison: Image.Image,
    style_image: Image.Image | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "original": output_dir / f"method_compare_original_{timestamp}.png",
        "opencv": output_dir / f"method_compare_v1_opencv_{timestamp}.png",
        "sd_prompt": output_dir / f"method_compare_v2_sd_prompt_{timestamp}.png",
        "canny": output_dir / f"method_compare_canny_{timestamp}.png",
        "controlnet": output_dir / f"method_compare_v3_controlnet_{timestamp}.png",
        "ip_adapter": output_dir / f"method_compare_v4_controlnet_ip_adapter_{timestamp}.png",
        "controlnet_lora": output_dir / f"method_compare_v5_controlnet_ip_lora_{timestamp}.png",
        "instantstyle": output_dir / f"method_compare_v6_instantstyle_{timestamp}.png",
        "comparison": output_dir / f"method_compare_grid_{timestamp}.png",
    }
    if style_image is not None:
        paths["style_reference"] = output_dir / f"method_compare_style_reference_{timestamp}.png"
    init_image.convert("RGB").save(paths["original"])
    opencv_image.convert("RGB").save(paths["opencv"])
    sd_image.convert("RGB").save(paths["sd_prompt"])
    canny_image.convert("RGB").save(paths["canny"])
    controlnet_image.convert("RGB").save(paths["controlnet"])
    ip_adapter_image.convert("RGB").save(paths["ip_adapter"])
    lora_image.convert("RGB").save(paths["controlnet_lora"])
    instantstyle_image.convert("RGB").save(paths["instantstyle"])
    if style_image is not None:
        style_image.convert("RGB").save(paths["style_reference"])
    comparison.convert("RGB").save(paths["comparison"])
    return paths


def run_comparison(args: argparse.Namespace) -> dict[str, Path]:
    source = Image.open(args.image)
    init_image = resize_for_diffusion(source, args.size)
    lora_path = None if args.skip_lora else resolve_lora_path(args.lora)

    print("Generating V1 OpenCV style-transfer baseline...")
    opencv_image = run_opencv_version(init_image, args.opencv_style, args.opencv_strength)

    print("Loading Stable Diffusion img2img pipeline...")
    sd_args = make_args_copy(args, lora=None)
    sd_pipe, _ = load_img2img_pipeline(sd_args)
    print("Generating V2 Stable Diffusion prompt-only result...")
    sd_image = generate_image(sd_pipe, init_image, sd_args)
    del sd_pipe
    cleanup_memory()

    canny_image = make_canny_image(init_image, args.canny_low, args.canny_high)

    print("Loading Stable Diffusion + ControlNet pipeline...")
    control_args = make_args_copy(args, lora=None)
    control_pipe = load_controlnet_pipeline(control_args)
    print("Generating V3 ControlNet structure-preserving result...")
    controlnet_image = generate_controlnet_image(control_pipe, init_image, canny_image, control_args)

    del control_pipe
    cleanup_memory()

    style_image = None
    style_path = find_style_image(args.style_image, args.style_dataset_dir)
    if args.skip_ip_adapter:
        print("Skipping ControlNet + IP-Adapter panel.")
        ip_adapter_image = placeholder_panel(init_image.size, "ControlNet + IP-Adapter", "Skipped")
        lora_image = placeholder_panel(init_image.size, "V5 + LoRA", "Needs V4 pipeline")
        instantstyle_image = placeholder_panel(init_image.size, "V6 InstantStyle", "Needs V4 pipeline")
    elif not style_path:
        print("Skipping ControlNet + IP-Adapter panel: no style reference image found.")
        ip_adapter_image = placeholder_panel(init_image.size, "ControlNet + IP-Adapter", "No style image")
        lora_image = placeholder_panel(init_image.size, "V5 + LoRA", "No style image")
        instantstyle_image = placeholder_panel(init_image.size, "V6 InstantStyle", "No style image")
    else:
        print(f"Using IP-Adapter style reference: {style_path}")
        style_image = prepare_style_reference(Image.open(style_path), args.style_size, args.style_square_mode)
        ip_args = make_args_copy(
            args,
            lora=None,
            model=args.ip_model,
            controlnet=args.ip_controlnet,
            ip_adapter_repo=args.ip_adapter_repo,
            ip_adapter_subfolder=args.ip_adapter_subfolder,
            ip_adapter_weight=args.ip_adapter_weight,
            image_encoder_folder=args.image_encoder_folder,
            ip_adapter_scale=args.ip_adapter_scale,
            instantstyle_mode=DEFAULT_INSTANTSTYLE_MODE,
            steps=args.ip_steps,
            strength=args.ip_strength,
            guidance=args.ip_guidance,
            control_scale=args.ip_control_scale,
        )
        print("Loading SD1.5 + ControlNet + IP-Adapter pipeline...")
        ip_pipe = load_ip_adapter_pipeline(ip_args)
        print("Generating V4 ControlNet + IP-Adapter result...")
        ip_adapter_image = generate_ip_adapter_image(ip_pipe, init_image, canny_image, style_image, ip_args)

        if args.skip_instantstyle:
            print("Skipping V6 InstantStyle panel.")
            instantstyle_image = placeholder_panel(init_image.size, "V6 InstantStyle", "Skipped")
        else:
            print("Generating V6 InstantStyle result...")
            apply_ip_adapter_scale(ip_pipe, args.instantstyle_scale, args.instantstyle_mode)
            instantstyle_args = make_args_copy(
                ip_args,
                instantstyle_mode=args.instantstyle_mode,
                ip_adapter_scale=args.instantstyle_scale,
                steps=args.instantstyle_steps,
                strength=args.instantstyle_strength,
                guidance=args.instantstyle_guidance,
                control_scale=args.instantstyle_control_scale,
            )
            instantstyle_image = generate_ip_adapter_image(ip_pipe, init_image, canny_image, style_image, instantstyle_args)

        if args.skip_lora:
            print("Skipping V5 LoRA panel.")
            lora_image = placeholder_panel(init_image.size, "V5 + LoRA", "Skipped")
        elif lora_path:
            apply_ip_adapter_scale(ip_pipe, args.ip_adapter_scale, DEFAULT_INSTANTSTYLE_MODE)
            print(f"Loading LoRA: {lora_path}")
            ip_pipe.load_lora_weights(lora_path)
            print("Generating V5 ControlNet + IP-Adapter + LoRA result...")
            lora_args = make_args_copy(ip_args, lora=lora_path)
            lora_image = generate_ip_adapter_image(ip_pipe, init_image, canny_image, style_image, lora_args)
        else:
            detail = "LoRA not found" if args.lora else "LoRA skipped"
            print(f"Skipping V5 LoRA panel: {detail}.")
            lora_image = placeholder_panel(init_image.size, "V5 + LoRA", detail)

        del ip_pipe
        cleanup_memory()

    comparison_items = [
        ("Original", init_image),
        ("V1 OpenCV", opencv_image),
        ("V2 SD Prompt", sd_image),
        ("V3 ControlNet", controlnet_image),
    ]
    if not args.skip_ip_adapter:
        comparison_items.append(("V4 ControlNet + IP-Adapter", ip_adapter_image))
    if not args.skip_lora and not args.skip_ip_adapter:
        comparison_items.append(("V5 + LoRA", lora_image))
    if not args.skip_instantstyle and not args.skip_ip_adapter:
        comparison_items.append(("V6 InstantStyle", instantstyle_image))

    comparison = make_comparison_grid(comparison_items)
    paths = save_outputs(
        Path(args.output_dir),
        init_image,
        opencv_image,
        sd_image,
        canny_image,
        controlnet_image,
        ip_adapter_image,
        lora_image,
        instantstyle_image,
        comparison,
        style_image,
    )

    if args.show:
        cv2.imshow(WINDOW_NAME, pil_to_cv(comparison))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a one-image comparison grid across the current neoclassical style-transfer methods."
    )
    parser.add_argument("--image", required=True, help="Input image to stylize.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved comparison images.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SD1.5 image-to-image model id.")
    parser.add_argument("--controlnet", default=DEFAULT_CONTROLNET, help="SD1.5 Canny ControlNet model id.")
    parser.add_argument("--lora", default=str(DEFAULT_LORA), help="Optional SD1.5 LoRA directory or safetensors file.")
    parser.add_argument("--skip-lora", action="store_true", help="Generate the grid without loading LoRA.")
    parser.add_argument("--style-image", help="Neoclassical painting reference image for IP-Adapter.")
    parser.add_argument("--style-dataset-dir", default=str(DEFAULT_STYLE_DATASET_DIR), help="Fallback directory for style references.")
    parser.add_argument("--skip-ip-adapter", action="store_true", help="Generate the grid without loading IP-Adapter.")
    parser.add_argument("--size", type=int, default=512, help="Longest side passed into diffusion models.")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps. Increase for quality, decrease for CPU tests.")
    parser.add_argument("--strength", type=float, default=0.32, help="Img2img strength. Lower preserves more source content.")
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--control-scale", type=float, default=0.7, help="How strongly ControlNet follows source edges.")
    parser.add_argument("--canny-low", type=int, default=100, help="Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=200, help="Canny high threshold.")
    parser.add_argument("--opencv-style", default="david", help="OpenCV style: david, ingres, or marble.")
    parser.add_argument("--opencv-strength", type=float, default=0.9, help="OpenCV style strength from 0 to 1.")
    parser.add_argument("--ip-model", default=DEFAULT_MODEL, help="SD1.5-compatible base model for IP-Adapter.")
    parser.add_argument("--ip-controlnet", default=DEFAULT_CONTROLNET, help="SD1.5 Canny ControlNet model for IP-Adapter.")
    parser.add_argument("--ip-adapter-repo", default=DEFAULT_IP_ADAPTER_REPO, help="IP-Adapter repository.")
    parser.add_argument("--ip-adapter-subfolder", default=DEFAULT_IP_ADAPTER_SUBFOLDER, help="IP-Adapter weight subfolder.")
    parser.add_argument("--ip-adapter-weight", default=DEFAULT_IP_ADAPTER_WEIGHT, help="IP-Adapter weight file.")
    parser.add_argument("--image-encoder-folder", default=DEFAULT_IMAGE_ENCODER_FOLDER, help="IP-Adapter image encoder folder.")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.5, help="Reference image influence strength for V4/V5.")
    parser.add_argument("--style-size", type=int, default=512, help="Target size for the style reference image.")
    parser.add_argument("--style-square-mode", choices=("squash", "crop", "pad", "none"), default="squash", help="Prepare the style reference as a square before CLIP encoding.")
    parser.add_argument("--ip-steps", type=int, default=28, help="Inference steps for the IP-Adapter SD1.5 branch.")
    parser.add_argument("--ip-strength", type=float, default=0.32, help="Img2img strength for the IP-Adapter branch.")
    parser.add_argument("--ip-guidance", type=float, default=5.0, help="Guidance scale for the IP-Adapter branch.")
    parser.add_argument("--ip-control-scale", type=float, default=0.7, help="ControlNet scale for the IP-Adapter branch.")
    parser.add_argument("--skip-instantstyle", action="store_true", help="Generate the grid without the V6 InstantStyle panel.")
    parser.add_argument("--instantstyle-mode", choices=("style", "style_layout"), default="style", help="InstantStyle block preset for V6.")
    parser.add_argument("--instantstyle-scale", type=float, default=0.8, help="IP-Adapter scale for V6 InstantStyle.")
    parser.add_argument("--instantstyle-steps", type=int, default=30, help="Inference steps for V6 InstantStyle.")
    parser.add_argument("--instantstyle-strength", type=float, default=0.35, help="Img2img strength for V6 InstantStyle.")
    parser.add_argument("--instantstyle-guidance", type=float, default=5.0, help="Guidance scale for V6 InstantStyle.")
    parser.add_argument("--instantstyle-control-scale", type=float, default=0.7, help="ControlNet scale for V6 InstantStyle.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable comparisons.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive style-transfer prompt.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--quiet", action="store_true", help="Hide diffusion progress bars.")
    parser.add_argument("--show", action="store_true", help="Open the generated comparison window.")
    parser.add_argument("--disable-safety-checker", action="store_true", help="Accepted for local testing compatibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.steps * args.strength) < 1:
        raise SystemExit("Increase --steps or --strength: image-to-image needs at least one effective denoising step.")

    paths = run_comparison(args)
    print(f"Saved original: {paths['original']}")
    print(f"Saved OpenCV: {paths['opencv']}")
    print(f"Saved SD prompt-only: {paths['sd_prompt']}")
    print(f"Saved Canny control image: {paths['canny']}")
    print(f"Saved ControlNet: {paths['controlnet']}")
    if "style_reference" in paths:
        print(f"Saved style reference: {paths['style_reference']}")
    print(f"Saved V4 ControlNet + IP-Adapter: {paths['ip_adapter']}")
    print(f"Saved V5 ControlNet + IP-Adapter + LoRA: {paths['controlnet_lora']}")
    print(f"Saved V6 InstantStyle: {paths['instantstyle']}")
    print(f"Saved comparison grid: {paths['comparison']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
