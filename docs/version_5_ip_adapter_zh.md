# Version 5：IP-Adapter 参考图风格迁移

## 目标

前面的版本主要依赖文字 prompt、ControlNet 或 LoRA 来接近“新古典主义风格”。Version 5 加入 IP-Adapter，让系统可以额外输入一张新古典主义绘画作为风格参考图。

本版本的目标不是生成新的街景，而是做图像风格迁移：

```text
源图像：提供内容、构图、透视和物体位置
ControlNet Canny：约束源图像结构
IP-Adapter style image：提供新古典主义参考风格
Prompt：只做辅助风格说明
```

## 统一模型路线

为了让方法对比更公平，扩散模型版本统一使用 SD1.5 模型族：

```text
base model: runwayml/stable-diffusion-v1-5
ControlNet: lllyasviel/sd-controlnet-canny
IP-Adapter repo: h94/IP-Adapter
IP-Adapter weight: models/ip-adapter_sd15.bin
image encoder: models/image_encoder
LoRA target: SD1.5 UNet LoRA
```

之前基于 `stabilityai/sd-turbo` 跑过的 LoRA 只能作为早期实验记录。正式路线中，LoRA 需要重新基于 SD1.5 训练，默认输出目录是：

```text
lora_outputs/neoclassical_style_lora_sd15/
```

## 新增文件

```text
src/ip_adapter_style_transfer_demo.py
src/method_comparison_demo.py
```

## 单独运行 IP-Adapter

```powershell
python src\ip_adapter_style_transfer_demo.py --image path\to\source.jpg --style-image path\to\neoclassical_reference.jpg --size 384 --steps 12 --strength 0.45 --ip-adapter-scale 0.75 --no-window
```

如果不传 `--style-image`，脚本会尝试从本地 `data/neoclassical_lora` 目录自动寻找一张参考图。

## 生成整体对比

```powershell
python src\method_comparison_demo.py --image path\to\source.jpg --style-image path\to\neoclassical_reference.jpg --size 384 --steps 12 --strength 0.45 --control-scale 0.9 --guidance 6.5
```

整体对比图包含：

1. Original
2. V1 OpenCV
3. V2 Stable Diffusion prompt-only
4. V3 ControlNet
5. V4 IP-Adapter
6. V5 ControlNet + LoRA

输出保存在 `outputs/`。其中 `method_compare_grid_时间戳.png` 是最适合放进报告或 weblog 的总对比图。
