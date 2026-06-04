from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_CAPTION = (
    "neoclassical oil painting style, restrained academic realism, "
    "smooth classical brushwork, muted warm earth palette, soft chiaroscuro, "
    "marble-like stone light, museum canvas texture, elegant classical atmosphere"
)


def normalize_caption(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_caption(image_path: Path, captions_dir: Path, default_caption: str) -> str:
    caption_path = captions_dir / f"{image_path.stem}.txt"
    if caption_path.exists():
        return normalize_caption(caption_path.read_text(encoding="utf-8"))
    return normalize_caption(default_caption)


def write_missing_caption(image_path: Path, captions_dir: Path, default_caption: str) -> None:
    caption_path = captions_dir / f"{image_path.stem}.txt"
    if not caption_path.exists():
        caption_path.write_text(normalize_caption(default_caption) + "\n", encoding="utf-8")


def build_metadata(
    dataset_dir: Path,
    caption_column: str,
    default_caption: str,
    write_missing_captions: bool,
) -> Path:
    images_dir = dataset_dir / "images"
    captions_dir = dataset_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    metadata_path = dataset_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as file:
        for image_path in images:
            if write_missing_captions:
                write_missing_caption(image_path, captions_dir, default_caption)
            caption = read_caption(image_path, captions_dir, default_caption)
            row = {
                "file_name": f"images/{image_path.name}",
                caption_column: caption,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an imagefolder metadata.jsonl file for neoclassical LoRA training."
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/neoclassical_lora",
        help="Dataset directory containing images/ and captions/ folders.",
    )
    parser.add_argument(
        "--caption-column",
        default="text",
        help="Caption column name expected by the Diffusers training script.",
    )
    parser.add_argument(
        "--default-caption",
        default=DEFAULT_CAPTION,
        help="Caption used when an image does not have a matching captions/<name>.txt file.",
    )
    parser.add_argument(
        "--write-missing-captions",
        action="store_true",
        help="Create missing caption txt files from the default caption.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    metadata_path = build_metadata(
        dataset_dir=dataset_dir,
        caption_column=args.caption_column,
        default_caption=args.default_caption,
        write_missing_captions=args.write_missing_captions,
    )
    image_count = len(list_images(dataset_dir / "images"))
    print(f"Prepared {image_count} training image records.")
    print(f"Metadata: {metadata_path}")
    if image_count == 0:
        print("Add reference images to data/neoclassical_lora/images before training.")


if __name__ == "__main__":
    main()
