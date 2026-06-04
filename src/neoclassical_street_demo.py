from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Neoclassical Street Camera"


@dataclass(frozen=True)
class NeoclassicalVariant:
    key: str
    label: str
    saturation: float
    contrast: float
    gamma: float
    smooth: float
    texture: float
    vignette: float
    shadow_bgr: tuple[float, float, float]
    highlight_bgr: tuple[float, float, float]


VARIANTS: list[NeoclassicalVariant] = [
    NeoclassicalVariant(
        key="david",
        label="Neoclassical / David warm oil",
        saturation=0.58,
        contrast=1.18,
        gamma=0.92,
        smooth=0.78,
        texture=0.080,
        vignette=0.34,
        shadow_bgr=(26.0, 8.0, -7.0),
        highlight_bgr=(-5.0, 12.0, 30.0),
    ),
    NeoclassicalVariant(
        key="ingres",
        label="Neoclassical / Ingres soft glaze",
        saturation=0.48,
        contrast=1.08,
        gamma=0.98,
        smooth=0.86,
        texture=0.050,
        vignette=0.24,
        shadow_bgr=(16.0, 8.0, 0.0),
        highlight_bgr=(3.0, 11.0, 22.0),
    ),
    NeoclassicalVariant(
        key="marble",
        label="Neoclassical / Marble study",
        saturation=0.32,
        contrast=1.14,
        gamma=1.02,
        smooth=0.82,
        texture=0.060,
        vignette=0.28,
        shadow_bgr=(21.0, 9.0, -6.0),
        highlight_bgr=(7.0, 8.0, 15.0),
    ),
]


def variant_by_key(key: str) -> NeoclassicalVariant:
    for variant in VARIANTS:
        if variant.key == key:
            return variant
    valid = ", ".join(variant.key for variant in VARIANTS)
    raise ValueError(f"Unknown style '{key}'. Valid styles: {valid}")


def edge_aware_smooth(frame: np.ndarray) -> np.ndarray:
    try:
        return cv2.edgePreservingFilter(frame, flags=1, sigma_s=72, sigma_r=0.28)
    except cv2.error:
        return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)


def adjust_luminance_contrast(frame: np.ndarray, contrast: float, gamma: float) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[:, :, 0] / 255.0
    luminance = np.clip((luminance - 0.5) * contrast + 0.5, 0.0, 1.0)
    luminance = np.power(luminance, gamma)
    lab[:, :, 0] = luminance * 255.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def reduce_saturation(frame: np.ndarray, saturation: float) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def split_tone(frame: np.ndarray, variant: NeoclassicalVariant, strength: float) -> np.ndarray:
    image = frame.astype(np.float32)
    luma = (
        0.114 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.299 * image[:, :, 2]
    ) / 255.0
    shadow = np.power(1.0 - luma, 1.45)[:, :, None]
    highlight = np.power(luma, 1.55)[:, :, None]
    shadow_color = np.array(variant.shadow_bgr, dtype=np.float32)[None, None, :]
    highlight_color = np.array(variant.highlight_bgr, dtype=np.float32)[None, None, :]
    toned = image + strength * (shadow * shadow_color + highlight * highlight_color)
    return np.clip(toned, 0, 255).astype(np.uint8)


@lru_cache(maxsize=12)
def canvas_texture(height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(1784 + height * 31 + width * 17)
    grain = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.7)

    vertical = rng.normal(0.0, 1.0, (height, max(1, width // 18))).astype(np.float32)
    vertical = cv2.resize(vertical, (width, height), interpolation=cv2.INTER_CUBIC)
    horizontal = rng.normal(0.0, 1.0, (max(1, height // 18), width)).astype(np.float32)
    horizontal = cv2.resize(horizontal, (width, height), interpolation=cv2.INTER_CUBIC)

    texture = grain * 0.55 + vertical * 0.25 + horizontal * 0.20
    texture -= texture.min()
    texture /= max(texture.max(), 1e-6)
    return texture


@lru_cache(maxsize=12)
def vignette_mask(height: int, width: int) -> np.ndarray:
    y, x = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt(((x - cx) / max(cx, 1.0)) ** 2 + ((y - cy) / max(cy, 1.0)) ** 2)
    mask = np.clip(1.0 - radius**2, 0.0, 1.0)
    return mask.astype(np.float32)


def apply_canvas_and_vignette(
    frame: np.ndarray,
    texture_strength: float,
    vignette_strength: float,
    user_strength: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    image = frame.astype(np.float32)

    texture = canvas_texture(height, width)[:, :, None]
    image *= 1.0 + (texture - 0.5) * texture_strength * user_strength

    mask = vignette_mask(height, width)[:, :, None]
    image *= 1.0 - (1.0 - mask) * vignette_strength * user_strength
    return np.clip(image, 0, 255).astype(np.uint8)


def add_painterly_edges(frame: np.ndarray, strength: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    edges = cv2.GaussianBlur(edges, (0, 0), sigmaX=0.8)
    edge_mask = (edges.astype(np.float32) / 255.0)[:, :, None]
    image = frame.astype(np.float32)
    image *= 1.0 - edge_mask * 0.16 * strength
    return np.clip(image, 0, 255).astype(np.uint8)


def apply_neoclassical(frame: np.ndarray, variant: NeoclassicalVariant, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))

    smooth = edge_aware_smooth(frame)
    blended = cv2.addWeighted(smooth, variant.smooth, frame, 1.0 - variant.smooth, 0.0)
    image = adjust_luminance_contrast(blended, variant.contrast, variant.gamma)
    image = reduce_saturation(image, variant.saturation)
    image = split_tone(image, variant, strength)
    image = add_painterly_edges(image, strength)
    image = apply_canvas_and_vignette(image, variant.texture, variant.vignette, strength)

    return cv2.addWeighted(image, strength, frame, 1.0 - strength, 0.0)


def overlay_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    result = image.copy()
    pad = 10
    line_height = 24
    box_height = pad * 2 + line_height * len(lines)
    cv2.rectangle(result, (0, 0), (result.shape[1], box_height), (22, 20, 18), -1)

    for index, text in enumerate(lines):
        y = pad + 17 + index * line_height
        cv2.putText(
            result,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (246, 240, 226),
            1,
            cv2.LINE_AA,
        )
    return result


def make_display(
    frame: np.ndarray,
    styled: np.ndarray,
    strength: float,
    fps: float,
    variant: NeoclassicalVariant,
    side_by_side: bool,
) -> np.ndarray:
    if side_by_side:
        display = np.hstack([frame, styled])
    else:
        display = styled

    lines = [
        f"Style: {variant.label} | FPS: {fps:.1f} | Strength: {strength:.2f}",
        "Keys: 1-3 neoclassical variants | +/- strength | c compare | s save | q quit",
    ]
    return overlay_text(display, lines)


def save_frame(image: np.ndarray) -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"neoclassical_street_{timestamp}.jpg"
    cv2.imwrite(str(path), image)
    return path


def run_camera(args: argparse.Namespace) -> None:
    variant = variant_by_key(args.style)
    strength = args.strength
    side_by_side = args.compare

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open camera {args.camera}. Try --camera 1, or check Windows camera permissions."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Neoclassical street camera demo running.")
    print("Keys: 1-3 variants, +/- strength, c compare, s save, q quit")

    last_time = time.perf_counter()
    fps = 0.0
    latest_display: np.ndarray | None = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        if args.process_width and frame.shape[1] > args.process_width:
            scale = args.process_width / float(frame.shape[1])
            frame = cv2.resize(frame, (args.process_width, int(frame.shape[0] * scale)))

        styled = apply_neoclassical(frame, variant, strength)
        now = time.perf_counter()
        elapsed = max(now - last_time, 1e-6)
        last_time = now
        fps = fps * 0.85 + (1.0 / elapsed) * 0.15

        latest_display = make_display(frame, styled, strength, fps, variant, side_by_side)
        cv2.imshow(WINDOW_NAME, latest_display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            strength = min(1.0, strength + 0.05)
        if key in (ord("-"), ord("_")):
            strength = max(0.0, strength - 0.05)
        if key == ord("c"):
            side_by_side = not side_by_side
        if key == ord("s") and latest_display is not None:
            print(f"Saved {save_frame(latest_display)}")
        if key in (ord("1"), ord("2"), ord("3")):
            variant = VARIANTS[int(chr(key)) - 1]

    capture.release()
    cv2.destroyAllWindows()


def run_image(args: argparse.Namespace) -> None:
    variant = variant_by_key(args.style)
    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    styled = apply_neoclassical(frame, variant, args.strength)
    display = make_display(frame, styled, args.strength, 0.0, variant, side_by_side=True)
    path = save_frame(display)
    print(f"Saved styled preview: {path}")
    cv2.imshow(WINDOW_NAME, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def list_styles() -> None:
    print("Available neoclassical variants:")
    for index, variant in enumerate(VARIANTS, start=1):
        print(f"  {index}. {variant.key:8s} {variant.label}")


def self_test() -> None:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    image[:, :80] = (70, 90, 150)
    image[:, 80:160] = (130, 130, 105)
    image[:, 160:] = (160, 185, 210)
    print("Running self-test...")
    for variant in VARIANTS:
        out = apply_neoclassical(image, variant, 0.95)
        print(f"  {variant.key:8s} ok | shape={out.shape} | mean={int(out.mean())}")
    print("Self-test complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time neoclassical street-camera demo.")
    parser.add_argument("--style", default="david", help="Initial style: david, ingres, or marble.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument("--process-width", type=int, default=640, help="Resize live frame to this width before styling.")
    parser.add_argument("--strength", type=float, default=0.92, help="Style strength from 0 to 1.")
    parser.add_argument("--compare", action="store_true", help="Start in side-by-side compare mode.")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam preview.")
    parser.add_argument("--image", help="Style a single image instead of opening the camera.")
    parser.add_argument("--list-styles", action="store_true", help="List variants and exit.")
    parser.add_argument("--self-test", action="store_true", help="Run a non-GUI processing test and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_styles:
        list_styles()
        return
    if args.self_test:
        self_test()
        return
    if args.image:
        run_image(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
