param(
    [string]$Python = "C:\Users\23913\.conda\envs\homework2\python.exe",
    [string]$DatasetDir = "data\neoclassical_lora",
    [string]$OutputDir = "lora_outputs\neoclassical_style_lora",
    [string]$DiffusersRepo = "external\diffusers",
    [string]$Model = "stabilityai/sd-turbo",
    [string]$MixedPrecision = "no",
    [int]$Resolution = 512,
    [int]$MaxTrainSteps = 800,
    [double]$LearningRate = 0.0001,
    [int]$Rank = 8,
    [int]$Seed = 42,
    [string]$ValidationPrompt = "a street photograph transformed only in visual style into a restrained neoclassical oil painting",
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

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
    "--gradient_accumulation_steps=4",
    "--max_train_steps=$MaxTrainSteps",
    "--learning_rate=$LearningRate",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--rank=$Rank",
    "--output_dir=$OutputDir",
    "--checkpointing_steps=100",
    "--validation_prompt=$ValidationPrompt",
    "--seed=$Seed"
)

& $Python -m accelerate.commands.launch --mixed_precision=$MixedPrecision @trainArgs
