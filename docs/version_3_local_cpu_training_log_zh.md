# Version 3: 本地 CPU LoRA 训练记录

## 1. 为什么先在本地跑

虽然当前电脑没有 NVIDIA CUDA GPU，但为了验证第三版训练链路，先在本地 CPU 上跑一个小规模训练 trial。

这个 trial 的目的不是追求最终画质，而是确认：

1. 训练数据能被 Diffusers 正确读取
2. LoRA 训练脚本能启动
3. 训练完成后能保存 `.safetensors` 权重
4. 第二版图生图程序能加载这个 LoRA
5. 项目路线从 prompt-only 推进到 trainable style adapter

## 2. 环境结果

```text
torch 2.12.0+cpu
cuda False
diffusers 0.38.0
datasets 4.8.5
peft 0.19.1
accelerate 1.13.0
```

因为没有 CUDA，所以训练速度明显慢于 GPU。

## 3. 本地训练数据

当前训练数据：

```text
20 images
20 captions
20 metadata rows
```

图片主要来自 The Metropolitan Museum of Art 和 Art Institute of Chicago 的开放馆藏 API，并人工删除了一些不适合新古典主义风格训练的图片。

## 4. 遇到的问题和修复

### 4.1 Diffusers examples 和本地 diffusers 版本不一致

最新版官方训练脚本要求 source version 的 Diffusers。本地环境安装的是 `diffusers 0.38.0`。

修复方式是在 `scripts/run_lora_training.ps1` 中把：

```text
external/diffusers/src
```

加入 `PYTHONPATH`。

### 4.2 Hugging Face cache 默认写到 C 盘

训练时模型权重一开始下载到：

```text
C:/Users/23913/.cache/huggingface
```

C 盘空间不足，所以改成项目内 D 盘 cache：

```text
D:/homework_yibai/EMI-2026-main/.cache/huggingface
```

### 4.3 CPU validation 太慢

官方训练脚本如果设置 `validation_prompt`，会在训练后生成验证图。CPU 上这一步很慢。

修复方式是在 `run_lora_training.ps1` 中加入：

```powershell
-NoValidation
```

本地 CPU trial 训练时关闭 validation。

### 4.4 图生图 steps 太少会失败

测试时发现：

```powershell
--steps 1 --strength 0.5
```

会导致 img2img 实际 denoising step 变成 0。后续应至少使用：

```powershell
--steps 2 --strength 0.5
```

代码中已经加入参数检查，避免再次踩这个坑。

## 5. 成功运行的 smoke training

命令：

```powershell
.\scripts\run_lora_training.ps1 -AllowCpu -NoValidation -Resolution 128 -MaxTrainSteps 1 -Rank 4 -OutputDir lora_outputs\neoclassical_style_lora_cpu_smoke -MixedPrecision no
```

结果：

```text
Model weights saved in lora_outputs/neoclassical_style_lora_cpu_smoke/pytorch_lora_weights.safetensors
```

## 6. 成功运行的本地 CPU trial

命令：

```powershell
.\scripts\run_lora_training.ps1 -AllowCpu -NoValidation -Resolution 128 -MaxTrainSteps 50 -GradientAccumulationSteps 1 -Rank 4 -OutputDir lora_outputs\neoclassical_style_lora_cpu_trial -MixedPrecision no
```

训练结果：

```text
50 steps
resolution 128
rank 4
training time about 9 minutes 15 seconds
output: lora_outputs/neoclassical_style_lora_cpu_trial/pytorch_lora_weights.safetensors
```

## 7. LoRA 加载测试

命令：

```powershell
python src\diffusion_neoclassical_demo.py --image outputs\test_street_input.png --lora lora_outputs\neoclassical_style_lora_cpu_trial --size 128 --steps 2 --strength 0.5 --guidance 0 --seed 42 --no-window --quiet --disable-safety-checker
```

结果：程序成功加载 LoRA，并保存原图、生成图和对比图。

后续代码已经改成：图片模式只要传入 `--lora`，就会自动保存四张图：

1. 原图
2. 无 LoRA 的 baseline 结果
3. 有 LoRA 的结果
4. 三栏对比图

## 8. 当前结论

本地 CPU 可以跑通第三版训练闭环，但只适合小规模验证。

如果要追求更好的风格效果，下一步建议：

1. 把分辨率提高到 256 或 512
2. 把训练步数提高到 400-800
3. 使用 NVIDIA GPU
4. 扩充数据集到 30-60 张
5. 用真实街景照片做无 LoRA / 有 LoRA 对比
