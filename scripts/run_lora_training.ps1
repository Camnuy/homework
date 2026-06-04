param(
    [string]$Python = "C:\Users\23913\.conda\envs\homework2\python.exe",
    [string]$DatasetDir = "data\neoclassical_lora",
    [string]$OutputDir = "lora_outputs\neoclassical_style_lora_sd15",
    [string]$DiffusersRepo = "external\diffusers",
    [string]$Model = "runwayml/stable-diffusion-v1-5",
    [string]$MixedPrecision = "no",
    [int]$Resolution = 512,
    [int]$MaxTrainSteps = 800,
    [int]$GradientAccumulationSteps = 4,
    [double]$LearningRate = 0.0001,
    [int]$Rank = 8,
    [int]$Seed = 42,
    [string]$ValidationPrompt = "a source image transformed only in visual style into a restrained neoclassical oil painting",
    [switch]$NoValidation,
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$projectRoot = (Resolve-Path ".").Path
$env:HF_HOME = Join-Path $projectRoot ".cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
$env:HF_HUB_ETAG_TIMEOUT = "30"

if (!(Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

$cudaAvailable = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($cudaAvailable.Trim() -ne "1" -and -not $AllowCpu) {
    throw "CUDA GPU is not available. LoRA training on CPU will be extremely slow. Run on a GPU machine, or pass -AllowCpu only for a tiny smoke test."
}

if (!(Test-Path $DiffusersRepo)) {
    git clone https://github.com/huggingface/diffusers $DiffusersRepo
}

$trainScript = Join-Path $DiffusersRepo "examples\text_to_image\train_text_to_image_lora.py"
if (!(Test-Path $trainScript)) {
    throw "Diffusers LoRA training script not found: $trainScript"
}

$diffusersSrc = Join-Path $DiffusersRepo "src"
if (!(Test-Path $diffusersSrc)) {
    throw "Diffusers source directory not found: $diffusersSrc"
}
$env:PYTHONPATH = "$diffusersSrc;$env:PYTHONPATH"

& $Python scripts\prepare_lora_dataset.py `
    --dataset-dir $DatasetDir `
    --write-missing-captions

$trainArgs = @(
    $trainScript,
    "--pretrained_model_name_or_path=$Model",
    "--train_data_dir=$DatasetDir",
    "--caption_column=text",
    "--resolution=$Resolution",
    "--center_crop",
    "--random_flip",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=$GradientAccumulationSteps",
    "--max_train_steps=$MaxTrainSteps",
    "--learning_rate=$LearningRate",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--rank=$Rank",
    "--output_dir=$OutputDir",
    "--checkpointing_steps=100",
    "--seed=$Seed"
)

if (-not $NoValidation) {
    $trainArgs += "--validation_prompt=$ValidationPrompt"
}

& $Python -m accelerate.commands.launch --mixed_precision=$MixedPrecision @trainArgs
