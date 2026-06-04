from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from prepare_lora_dataset import DEFAULT_CAPTION, build_metadata


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


MET_SEARCH_URL = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
AIC_SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"

DEFAULT_QUERIES = [
    "Jacques Louis David",
    "Jean Auguste Dominique Ingres",
    "Anne Louis Girodet",
    "Francois Gerard",
    "Pierre-Paul Prud'hon",
    "Angelica Kauffman",
    "Benjamin West",
    "neoclassical painting",
    "classical painting",
]

USER_AGENT = "EMI-neoclassical-lora-dataset/0.1 (educational student project)"


def fetch_json(url: str, params: dict[str, object] | None = None, retries: int = 3) -> dict:
    if params:
        url = f"{url}?{urlencode(params, doseq=True)}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "AIC-User-Agent": USER_AGENT})
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, IncompleteRead) as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Could not fetch JSON from {url}: {last_error}")


def download_file(url: str, path: Path, retries: int = 3) -> bool:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "AIC-User-Agent": USER_AGENT})
            with urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
            return True
        except (HTTPError, URLError, TimeoutError, IncompleteRead) as exc:
            last_error = exc
            time.sleep(1 + attempt)
    print(f"Skipped download: {url} ({last_error})")
    return False


def safe_slug(text: str, max_length: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:max_length].strip("_") or "artwork"


def caption_for_artwork() -> str:
    return DEFAULT_CAPTION


def met_candidates(query: str, max_objects: int) -> list[dict]:
    search = fetch_json(
        MET_SEARCH_URL,
        {
            "q": query,
            "hasImages": "true",
            "medium": "Paintings",
            "dateBegin": 1750,
            "dateEnd": 1900,
        },
    )
    object_ids = search.get("objectIDs") or []
    candidates: list[dict] = []
    for object_id in object_ids[:max_objects]:
        try:
            record = fetch_json(MET_OBJECT_URL.format(object_id=object_id))
        except RuntimeError as exc:
            print(f"Skipped Met object {object_id}: {exc}")
            continue
        image_url = record.get("primaryImageSmall") or record.get("primaryImage")
        if not record.get("isPublicDomain") or not image_url:
            continue
        candidates.append(
            {
                "source": "The Metropolitan Museum of Art",
                "source_id": str(record.get("objectID")),
                "title": record.get("title") or "Untitled",
                "artist": record.get("artistDisplayName") or "Unknown artist",
                "date": record.get("objectDate") or "",
                "license": "Public Domain / Open Access",
                "object_url": record.get("objectURL") or "",
                "image_url": image_url,
                "query": query,
            }
        )
    return candidates


def aic_candidates(query: str, limit: int) -> list[dict]:
    search = fetch_json(
        AIC_SEARCH_URL,
        {
            "q": query,
            "query[term][is_public_domain]": "true",
            "limit": limit,
                "fields": "id,title,artist_display,date_display,image_id,is_public_domain,api_link,web_url,classification_title,medium_display,artwork_type_title",
        },
    )
    iiif_url = (search.get("config") or {}).get("iiif_url") or "https://www.artic.edu/iiif/2"
    candidates: list[dict] = []
    for record in search.get("data") or []:
        image_id = record.get("image_id")
        if not record.get("is_public_domain") or not image_id:
            continue
        type_text = " ".join(
            str(record.get(key) or "")
            for key in ("classification_title", "medium_display", "artwork_type_title")
        ).lower()
        if "painting" not in type_text and "oil" not in type_text:
            continue
        candidates.append(
            {
                "source": "Art Institute of Chicago",
                "source_id": str(record.get("id")),
                "title": record.get("title") or "Untitled",
                "artist": record.get("artist_display") or "Unknown artist",
                "date": record.get("date_display") or "",
                "license": "Public Domain / CC0 where available",
                "object_url": record.get("web_url") or "",
                "image_url": f"{iiif_url}/{image_id}/full/843,/0/default.jpg",
                "query": query,
            }
        )
    return candidates


def unique_candidates(candidates: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for candidate in candidates:
        key = (candidate["source"], candidate["source_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def download_dataset(args: argparse.Namespace) -> list[dict]:
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    captions_dir = dataset_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    for query in args.queries:
        print(f"Searching museum APIs: {query}", flush=True)
        if "met" in args.sources:
            candidates.extend(met_candidates(query, args.per_query))
            time.sleep(args.delay)
        if "aic" in args.sources:
            candidates.extend(aic_candidates(query, args.per_query))
            time.sleep(args.delay)

    downloaded: list[dict] = []
    for index, candidate in enumerate(unique_candidates(candidates), start=1):
        if len(downloaded) >= args.limit:
            break
        stem = f"{len(downloaded) + 1:04d}_{safe_slug(candidate['source'])}_{candidate['source_id']}"
        image_path = images_dir / f"{stem}.jpg"
        caption_path = captions_dir / f"{stem}.txt"

        if image_path.exists() and caption_path.exists() and not args.overwrite:
            print(f"Already exists: {image_path.name}", flush=True)
        else:
            print(
                f"Downloading {len(downloaded) + 1}/{args.limit}: "
                f"{candidate['artist']} - {candidate['title']}",
                flush=True,
            )
            if not download_file(candidate["image_url"], image_path):
                continue
            caption_path.write_text(caption_for_artwork() + "\n", encoding="utf-8")
            time.sleep(args.delay)

        candidate["local_image"] = str(image_path.as_posix())
        candidate["local_caption"] = str(caption_path.as_posix())
        downloaded.append(candidate)

    manifest_path = dataset_dir / "source_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file:
        for item in downloaded:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    build_metadata(
        dataset_dir=dataset_dir,
        caption_column="text",
        default_caption=DEFAULT_CAPTION,
        write_missing_captions=True,
    )
    print(f"Downloaded references: {len(downloaded)}")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {dataset_dir / 'metadata.jsonl'}")
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public-domain/open-access neoclassical reference images for LoRA training."
    )
    parser.add_argument("--dataset-dir", default="data/neoclassical_lora", help="Output dataset directory.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum number of images to download.")
    parser.add_argument("--per-query", type=int, default=12, help="Maximum candidates checked per source query.")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between API/download requests.")
    parser.add_argument("--sources", nargs="+", default=["met", "aic"], choices=["met", "aic"], help="Sources to use.")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES, help="Museum search queries.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloaded images and captions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dataset(args)


if __name__ == "__main__":
    main()
