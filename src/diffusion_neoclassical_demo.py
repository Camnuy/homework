from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "EMI-2026-main").exists():
        return here
    if (here.parent / "README.md").exists():
        return here.parent
    return here


PROJECT_ROOT = _project_root()
HF_CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


DEFAULT_PROMPT = (
    "style transfer only, preserve the exact source image content, composition, perspective, "
    "camera angle, object positions, edges, and geometry, change only the visual rendering style, "
    "restrained neoclassical oil painting, academic realism, smooth classical brushwork, "
    "muted warm earth palette, soft chiaroscuro, marble light, museum canvas texture"
)

DEFAULT_NEGATIVE_PROMPT = (
    "new scene, changed content, changed layout, different camera angle, altered perspective, "
    "added objects, removed objects, warped geometry, distorted faces, extra limbs, cartoon, "
    "anime, candy colors, mosaic filter, pop art, glitch, cyberpunk, fantasy, abstract, "
    "low quality, blurry, text, watermark"
)

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
WINDOW_NAME = "Diffusion Neoclassical Style Transfer"


def import_diffusion_stack():
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
    except ImportError as exc:
        raise SystemExit(
            "Missing diffusion dependencies.\n"
            "Install them first:\n"
            r"& C:\Users\23913\.conda\envs\homework2\python.exe -m pip install -r requirements_diffusion.txt"
        ) from exc
    return torch, AutoPipelineForImage2Image


def choose_device(torch_module):
    if torch_module.cuda.is_available():
        return "cuda", torch_module.float16
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps", torch_module.float32
    return "cpu", torch_module.float32


def resize_for_diffusion(image: Image.Image, size: int) -> Image.Image:
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    scale = size / max(width, height)
    new_width = max(8, int(width * scale) // 8 * 8)
    new_height = max(8, int(height * scale) // 8 * 8)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def cv_to_pil(frame: np.ndarray, size: int) -> Image.Image:
    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return resize_for_diffusion(Image.fromarray(rgb), size)


def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resolve_cached_model_source(model_id_or_path: str | Path) -> str:
    path = Path(model_id_or_path)
    if path.exists():
        return str(path)

    cache_roots = [HF_CACHE_ROOT / "hub"]
    env_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env_cache:
        env_cache_path = Path(env_cache)
        if env_cache_path not in cache_roots:
            cache_roots.append(env_cache_path)

    for cache_root in cache_roots:
        cache_dir = cache_root / f"models--{str(model_id_or_path).replace('/', '--')}"
        if not cache_dir.exists():
            continue

        ref_file = cache_dir / "refs" / "main"
        if ref_file.exists():
            revision = ref_file.read_text(encoding="utf-8").strip()
            snapshot = cache_dir / "snapshots" / revision
            if snapshot.exists():
                return str(snapshot)

        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(
                (candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0])

    return str(model_id_or_path)


def safety_checker_load_kwargs(model_source: str | Path, disable_requested: bool) -> dict[str, object]:
    if disable_requested:
        return {"safety_checker": None, "requires_safety_checker": False}

    path = Path(model_source)
    if path.exists():
        safety_dir = path / "safety_checker"
        if safety_dir.exists():
            weight_files = (
                safety_dir / "model.safetensors",
                safety_dir / "pytorch_model.bin",
                safety_dir / "diffusion_pytorch_model.safetensors",
                safety_dir / "diffusion_pytorch_model.bin",
            )
            if not any(candidate.exists() for candidate in weight_files):
                print("Local snapshot is missing safety-checker weights; disabling the local safety checker.")
                return {"safety_checker": None, "requires_safety_checker": False}

    return {}


def resolve_lora_path(path_like: str | Path | None) -> str | None:
    if not path_like:
        return None

    path = Path(path_like)
    if not path.exists():
        return None

    if path.is_file():
        return str(path.parent)

    if (path / "pytorch_lora_weights.safetensors").exists():
        return str(path)

    checkpoints = []
    for candidate in path.glob("checkpoint-*"):
        if candidate.is_dir():
            try:
                step = int(candidate.name.split("-", 1)[1])
            except (IndexError, ValueError):
                step = -1
            checkpoints.append((step, candidate))

    for _, checkpoint in sorted(checkpoints, reverse=True):
        if (checkpoint / "pytorch_lora_weights.safetensors").exists():
            return str(checkpoint)

    nested = sorted(path.rglob("pytorch_lora_weights.safetensors"))
    if nested:
        return str(nested[-1].parent)

    return str(path)


def load_pipeline(args: argparse.Namespace):
    torch, AutoPipelineForImage2Image = import_diffusion_stack()
    device, dtype = choose_device(torch)
    model_source = resolve_cached_model_source(args.model)
    print(f"Loading model: {args.model}")
    if model_source != args.model:
        print(f"Resolved local model cache: {model_source}")
    print(f"Device: {device} | dtype: {dtype}")
    if device == "cpu":
        print("Warning: CPU diffusion will be slow. For live use and LoRA training, use an NVIDIA GPU.")

    load_kwargs = safety_checker_load_kwargs(model_source, args.disable_safety_checker)

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_source,
        torch_dtype=dtype,
        **load_kwargs,
    )
    pipe = pipe.to(device)

    lora_path = resolve_lora_path(getattr(args, "lora", None))
    if getattr(args, "lora", None) and not lora_path:
        raise SystemExit(f"LoRA path not found: {args.lora}")
    if lora_path:
        print(f"Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=args.quiet)

    return pipe, device


def generate_image(pipe, init_image: Image.Image, args: argparse.Namespace) -> Image.Image:
    call_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "image": init_image,
        "strength": args.strength,
        "guidance_scale": args.guidance,
        "num_inference_steps": args.steps,
    }
    if args.seed is not None:
        import torch

        device = getattr(pipe, "_execution_device", getattr(pipe, "device", "cpu"))
        generator_device = "cpu" if str(device).startswith("mps") else device
        call_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(args.seed)

    result = pipe(**call_kwargs)
    return result.images[0]


def make_comparison_image(original: Image.Image, generated: Image.Image) -> Image.Image:
    from PIL import Image

    generated = generated.convert("RGB")
    original = original.convert("RGB").resize(generated.size, Image.Resampling.LANCZOS)
    comparison = Image.new("RGB", (generated.width * 2, generated.height), (20, 18, 16))
    comparison.paste(original, (0, 0))
    comparison.paste(generated, (generated.width, 0))
    return comparison


def save_generation_pair(original: Image.Image, generated: Image.Image, prefix: str) -> dict[str, Path]:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "original": out_dir / f"{prefix}_original_{timestamp}.png",
        "generated": out_dir / f"{prefix}_neoclassical_{timestamp}.png",
        "comparison": out_dir / f"{prefix}_comparison_{timestamp}.png",
    }
    original.convert("RGB").save(paths["original"])
    generated.convert("RGB").save(paths["generated"])
    make_comparison_image(original, generated).save(paths["comparison"])
    return paths


def make_lora_comparison_image(original: Image.Image, baseline: Image.Image, lora: Image.Image) -> Image.Image:
    from PIL import Image

    baseline = baseline.convert("RGB")
    original = original.convert("RGB").resize(baseline.size, Image.Resampling.LANCZOS)
    lora = lora.convert("RGB").resize(baseline.size, Image.Resampling.LANCZOS)
    comparison = Image.new("RGB", (baseline.width * 3, baseline.height), (20, 18, 16))
    comparison.paste(original, (0, 0))
    comparison.paste(baseline, (baseline.width, 0))
    comparison.paste(lora, (baseline.width * 2, 0))
    return comparison


def save_lora_comparison(
    original: Image.Image,
    baseline: Image.Image,
    lora: Image.Image,
    prefix: str = "diffusion_lora_compare",
) -> dict[str, Path]:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    paths = {
        "original": out_dir / f"{prefix}_original_{timestamp}.png",
        "baseline": out_dir / f"{prefix}_baseline_{timestamp}.png",
        "lora": out_dir / f"{prefix}_lora_{timestamp}.png",
        "triptych": out_dir / f"{prefix}_triptych_{timestamp}.png",
    }
    original.convert("RGB").save(paths["original"])
    baseline.convert("RGB").save(paths["baseline"])
    lora.convert("RGB").save(paths["lora"])
    make_lora_comparison_image(original, baseline, lora).save(paths["triptych"])
    return paths


def run_image(args: argparse.Namespace) -> None:
    from PIL import Image

    source = Image.open(args.image)
    init_image = resize_for_diffusion(source, args.size)

    if args.lora:
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.lora = None
        pipe, _ = load_pipeline(baseline_args)

        print("Generating baseline image without LoRA...")
        baseline = generate_image(pipe, init_image, baseline_args)

        lora_path = resolve_lora_path(args.lora)
        if not lora_path:
            raise SystemExit(f"LoRA path not found: {args.lora}")
        print(f"Loading LoRA for comparison: {lora_path}")
        pipe.load_lora_weights(lora_path)
        print("Generating image with LoRA...")
        output = generate_image(pipe, init_image, args)

        paths = save_lora_comparison(init_image, baseline, output)
        print(f"Saved original: {paths['original']}")
        print(f"Saved baseline: {paths['baseline']}")
        print(f"Saved LoRA: {paths['lora']}")
        print(f"Saved triptych: {paths['triptych']}")

        if not args.no_window:
            cv2.imshow(WINDOW_NAME, pil_to_cv(make_lora_comparison_image(init_image, baseline, output)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    pipe, _ = load_pipeline(args)
    output = generate_image(pipe, init_image, args)
    paths = save_generation_pair(init_image, output, "diffusion_neoclassical")
    print(f"Saved original: {paths['original']}")
    print(f"Saved generated: {paths['generated']}")
    print(f"Saved comparison: {paths['comparison']}")

    if not args.no_window:
        preview = np.hstack([
            cv2.resize(pil_to_cv(init_image), (pil_to_cv(output).shape[1], pil_to_cv(output).shape[0])),
            pil_to_cv(output),
        ])
        cv2.imshow(WINDOW_NAME, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_help(frame: np.ndarray, last_output: np.ndarray | None, status: str) -> np.ndarray:
    display = frame.copy()
    if last_output is not None:
        preview = cv2.resize(last_output, (frame.shape[1], frame.shape[0]))
        display = np.hstack([frame, preview])

    lines = [
        "Diffusion Neoclassical Demo | g/space generate | s save pair | q quit",
        status,
    ]
    pad = 10
    line_height = 24
    cv2.rectangle(display, (0, 0), (display.shape[1], pad * 2 + line_height * len(lines)), (20, 18, 16), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            display,
            text,
            (12, pad + 17 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 238, 222),
            1,
            cv2.LINE_AA,
        )
    return display


def run_camera(args: argparse.Namespace) -> None:
    pipe, _ = load_pipeline(args)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}. Try --camera 1.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Camera mode running.")
    print("This is not high-FPS realtime. Press g/space to generate the current frame.")
    print("Keys: g/space generate, s save original+generated pair, q/Esc quit.")

    last_output_cv: np.ndarray | None = None
    last_input_pil: Image.Image | None = None
    last_output_pil: Image.Image | None = None
    status = "Ready. Point camera at a source scene, then press g."

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        cv2.imshow(WINDOW_NAME, draw_help(frame, last_output_cv, status))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("g"), ord(" ")):
            status = "Generating with diffusion model..."
            cv2.imshow(WINDOW_NAME, draw_help(frame, last_output_cv, status))
            cv2.waitKey(1)

            start = time.perf_counter()
            init_image = cv_to_pil(frame, args.size)
            last_input_pil = init_image
            last_output_pil = generate_image(pipe, init_image, args)
            last_output_cv = pil_to_cv(last_output_pil)
            elapsed = time.perf_counter() - start
            status = f"Generated in {elapsed:.1f}s. Press g again or s to save pair."

        if key == ord("s") and last_input_pil is not None and last_output_pil is not None:
            paths = save_generation_pair(last_input_pil, last_output_pil, "diffusion_neoclassical_camera")
            status = f"Saved pair: {paths['original'].name} + {paths['generated'].name}"
            print(status)
            print(f"Original: {paths['original']}")
            print(f"Generated: {paths['generated']}")
            print(f"Comparison: {paths['comparison']}")

    capture.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion image-to-image neoclassical style transfer.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SD1.5 image-to-image model id.")
    parser.add_argument("--lora", help="Optional trained LoRA directory or safetensors file.")
    parser.add_argument("--image", help="Run on a single image instead of camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Camera width.")
    parser.add_argument("--height", type=int, default=480, help="Camera height.")
    parser.add_argument("--size", type=int, default=512, help="Longest side passed into diffusion model.")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps. Increase for quality, decrease for CPU tests.")
    parser.add_argument("--strength", type=float, default=0.3, help="Image-to-image strength. Lower values preserve more of the source image.")
    parser.add_argument("--guidance", type=float, default=5.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, help="Optional random seed for repeatable comparisons.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--mirror", action="store_true", help="Mirror camera preview.")
    parser.add_argument("--quiet", action="store_true", help="Hide diffusion progress bars.")
    parser.add_argument("--no-window", action="store_true", help="Do not open preview window in image mode.")
    parser.add_argument("--disable-safety-checker", action="store_true", help="Accepted for CLI compatibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.steps * args.strength) < 1:
        raise SystemExit("Increase --steps or --strength: image-to-image needs at least one effective denoising step.")
    if args.image:
        run_image(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
