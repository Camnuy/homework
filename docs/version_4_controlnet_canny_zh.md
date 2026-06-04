# Version 4: ControlNet Canny 结构约束版本

## 1. 为什么要加 ControlNet

第三版的 LoRA 可以让风格有变化，但普通 Stable Diffusion img2img 有一个明显问题：`strength` 一高，模型就会重新生成街景，原图的街道结构、建筑轮廓和透视关系容易丢失。

客户要的不是“重新画一条街”，而是：

```text
保留原始街景结构
只改变视觉风格、光影、笔触和油画质感
```

所以第四版加入 ControlNet Canny。Canny 边缘图会把原图的轮廓交给 ControlNet，让模型在生成时更重视原图结构。

## 2. 技术路线

```text
输入街景图
→ 缩放到模型尺寸
→ OpenCV Canny 生成边缘图
→ Stable Diffusion + ControlNet Canny 图生图
→ 可选加载 LoRA
→ 保存原图、Canny 图、无 LoRA 结果、有 LoRA 结果和对比图
```

新增脚本：

```text
src/controlnet_neoclassical_demo.py
```

## 3. 推荐测试命令

CPU 上先用较低参数：

```powershell
python src\controlnet_neoclassical_demo.py --image C:\Users\23913\Desktop\source.jpg --size 384 --steps 12 --strength 0.45 --control-scale 0.9 --guidance 6.5 --no-window
```

如果要同时比较无 LoRA 和有 LoRA：

```powershell
python src\controlnet_neoclassical_demo.py --image C:\Users\23913\Desktop\source.jpg --lora lora_outputs\neoclassical_style_lora_sd15 --size 384 --steps 12 --strength 0.45 --control-scale 0.9 --guidance 6.5 --no-window
```

GPU 或更有耐心时可以试：

```powershell
python src\controlnet_neoclassical_demo.py --image C:\Users\23913\Desktop\source.jpg --lora lora_outputs\neoclassical_style_lora_sd15 --size 512 --steps 12 --strength 0.45 --control-scale 0.9 --guidance 6.5 --no-window
```

## 4. 输出文件

没有 LoRA 时保存：

```text
controlnet_neoclassical_original_时间.png
controlnet_neoclassical_canny_时间.png
controlnet_neoclassical_generated_时间.png
controlnet_neoclassical_comparison_时间.png
```

有 LoRA 时保存：

```text
controlnet_lora_compare_original_时间.png
controlnet_lora_compare_canny_时间.png
controlnet_lora_compare_baseline_时间.png
controlnet_lora_compare_lora_时间.png
controlnet_lora_compare_triptych_时间.png
```

其中 `triptych` 是：

```text
原图 / 无 LoRA ControlNet / 有 LoRA ControlNet
```

## 5. 参数怎么调

### strength

控制改动幅度：

```text
0.30-0.38: 更保留原图，风格较弱
0.40-0.50: 折中
0.55 以上: 可能开始重绘
```

### control-scale

控制 ControlNet 对边缘的约束强度：

```text
0.7-0.9: 更自由，风格可能更明显
1.0: 默认平衡
1.2-1.5: 更贴原结构，但画面可能僵硬
```

### Canny 阈值

默认：

```text
--canny-low 80 --canny-high 180
```

如果边缘太少，可以降低到：

```text
--canny-low 50 --canny-high 150
```

如果边缘太乱，可以提高到：

```text
--canny-low 100 --canny-high 220
```

## 6. 当前限制

1. 这个版本默认使用 `runwayml/stable-diffusion-v1-5` 和 SD1.5 Canny ControlNet：`lllyasviel/sd-controlnet-canny`。
2. 第一次运行需要下载 ControlNet model。
3. CPU 推理会很慢，建议先用 `--size 384 --steps 12` 测试；如果只是确认流程，可以临时降低到 `--size 320 --steps 8`。
4. 当前 LoRA 是本地 CPU trial，风格强度有限。第四版主要先验证结构控制。
5. 更好的最终路线是：用同一个 base model 继续训练更强 LoRA，或者换到 SDXL + SDXL ControlNet + SDXL LoRA。
