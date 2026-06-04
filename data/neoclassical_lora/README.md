# Neoclassical LoRA Dataset

This folder is for Version 3 training data.

Place reference images here:

```text
data/neoclassical_lora/images/
```

Write one caption file per image here:

```text
data/neoclassical_lora/captions/
```

Example:

```text
images/0001.jpg
captions/0001.txt
```

The caption should describe the style, not a new scene. A useful starting caption is:

```text
neoclassical oil painting style, restrained academic realism, smooth classical brushwork, muted warm earth palette, soft chiaroscuro, marble-like stone light, museum canvas texture, elegant classical atmosphere
```

Run this before training:

```powershell
python scripts\prepare_lora_dataset.py --write-missing-captions
```

This creates `metadata.jsonl`, which is used by the Diffusers LoRA training script.

You can also download a starter set from museum open-access APIs:

```powershell
python scripts\download_neoclassical_references.py --limit 30
```

The downloader writes:

```text
images/
captions/
metadata.jsonl
source_manifest.jsonl
```

Review the downloaded images before training and delete anything that does not match the intended neoclassical style.
