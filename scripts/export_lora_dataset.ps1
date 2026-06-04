param(
    [string]$Python = "C:\Users\23913\.conda\envs\homework2\python.exe",
    [string]$DatasetDir = "data\neoclassical_lora",
    [string]$OutputZip = "release\neoclassical_lora_dataset.zip"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

& $Python scripts\prepare_lora_dataset.py `
    --dataset-dir $DatasetDir `
    --write-missing-captions

$zipPath = Resolve-Path -LiteralPath (Split-Path $OutputZip -Parent) -ErrorAction SilentlyContinue
if (!$zipPath) {
    New-Item -ItemType Directory -Path (Split-Path $OutputZip -Parent) | Out-Null
}

if (Test-Path $OutputZip) {
    Remove-Item -LiteralPath $OutputZip
}

$items = @(
    (Join-Path $DatasetDir "images")
    (Join-Path $DatasetDir "captions")
    (Join-Path $DatasetDir "metadata.jsonl")
    (Join-Path $DatasetDir "source_manifest.jsonl")
    (Join-Path $DatasetDir "README.md")
)

Compress-Archive -Path $items -DestinationPath $OutputZip
Write-Output "Dataset archive written to $OutputZip"
