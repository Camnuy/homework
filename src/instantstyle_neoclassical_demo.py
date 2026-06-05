from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image

from controlnet_neoclassical_demo import make_canny_image
from diffusion_neoclassical_demo import (
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    pil_to_cv,
    resize_for_diffusion,
)
from ip_adapter_style_transfer_demo import (
    DEFAULT_IMAGE_ENCODER_FOLDER,
    DEFAULT_IP_ADAPTER_REPO,
    DEFAULT_IP_ADAPTER_SUBFOLDER,
    DEFAULT_IP_ADAPTER_WEIGHT,
    DEFAULT_STYLE_DATASET_DIR,
    find_style_image,
    generate_ip_adapter_image,
    load_pipeline,
    make_reference_comparison,
    prepare_style_reference,
    save_outputs,
)


WINDOW_NAME = "InstantStyle Neoclassical Style Transfer"


def run_image(args: argparse.Namespace):
    style_path = find_style_image(args.style_image, args.style_dataset_dir)
    if not style_path:
        raise SystemExit(
            "No style reference image found. Pass --style-image path\\to\\neoclassical_reference.jpg "
            "or download references into data\\neoclassical_lora first."
        )

    source_image = resize_for_diffusion(Image.open(args.image), args.size)
    style_image = prepare_style_reference(Image.open(style_path), args.style_size, args.style_square_mode)
    control_image = make_canny_image(source_image, args.canny_low, args.canny_high)

    print(f"Using style reference: {style_path}")
    pipe = load_pipeline(args)
    print("Generating ControlNet + InstantStyle result...")
    generated_image = generate_ip_adapter_image(pipe, source_image, control_image, style_image, args)

    result_label = "ControlNet + InstantStyle Result" if args.instantstyle_mode == "style" else "ControlNet + InstantStyle+Layout"
    paths = save_outputs(
        source_image,
        style_image,
        control_image,
        generated_image,
        args.output_dir,
        prefix="instantstyle_neoclassical",
        result_label=result_label,
    )
    if not args.no_window:
        cv2.imshow(WINDOW_NAME, pil_to_cv(make_reference_comparison(source_image, style_image, generated_image, result_label)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neoclassical style transfer with ControlNet and InstantStyle-style block-wise IP-Adapter control."
    )
    parser.add_argument("--image", required=True, help="Source image to stylize.")
    parser.add_argument("--style-image", help="Neoclassical painting reference image for InstantStyle.")
    parser.add_argument("--style-dataset-dir", default=str(DEFAULT_STYLE_DATASET_DIR), help="Fallback directory for style references.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SD1.5-compatible base model.")
    parser.add_argument("--controlnet", default="lllyasviel/sd-controlnet-canny", help="SD1.5 ControlNet model.")
    parser.add_argument("--ip-adapter-repo", default=DEFAULT_IP_ADAPTER_REPO, help="IP-Adapter repository.")
    parser.add_argument("--ip-adapter-subfolder", default=DEFAULT_IP_ADAPTER_SUBFOLDER, help="IP-Adapter weight subfolder.")
    parser.add_argument("--ip-adapter-weight", default=DEFAULT_IP_ADAPTER_WEIGHT, help="IP-Adapter weight file.")
    parser.add_argument("--image-encoder-folder", default=DEFAULT_IMAGE_ENCODER_FOLDER, help="IP-Adapter image encoder folder.")
    parser.add_argument("--instantstyle-mode", choices=("style", "style_layout"), default="style", help="InstantStyle block preset.")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.8, help="Reference image influence strength for InstantStyle.")
    parser.add_argument("--size", type=int, default=512, help="Longest side for the source image.")
    parser.add_argument("--style-size", type=int, default=512, help="Target size for the style reference image.")
    parser.add_argument("--style-square-mode", choices=("squash", "crop", "pad", "none"), default="squash", help="Prepare the style reference as a square before CLIP encoding.")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps.")
    parser.add_argument("--strength", type=float, default=0.35, help="Img2img strength.")
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--control-scale", type=float, default=0.7, help="How strongly ControlNet follows source edges.")
    parser.add_argument("--canny-low", type=int, default=100, help="Canny low threshold.")
    parser.add_argument("--canny-high", type=int, default=200, help="Canny high threshold.")
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
