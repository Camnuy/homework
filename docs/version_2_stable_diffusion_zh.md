# Version 2: Stable Diffusion 图生图原型

## 1. 这一版做什么

第二版加入 Stable Diffusion 图生图原型，用大模型把街景图像转译成新古典主义油画感。

这一版先不训练，重点是先验证：

1. 摄像头画面能不能作为输入
2. Stable Diffusion 图生图能不能把街景推向新古典主义风格
3. prompt、strength、steps 等参数对结果有什么影响
4. 大模型方案和第一版 OpenCV 原型相比有什么优势和问题

训练会放到第三版做。这样项目迭代更清楚：第一版是实时视觉原型，第二版是大模型推理，第三版再做 LoRA 微调。

## 2. 主要代码

```text
src/diffusion_neoclassical_demo.py
```

这个脚本支持两种模式：

1. 图片模式：输入一张街景图片，输出新古典主义风格图像
2. 摄像头模式：打开摄像头，按 `g` 或空格键，把当前帧送进 Stable Diffusion 生成

摄像头模式不是每一帧连续生成，因为 Stable Diffusion 速度较慢。这个版本更适合“按下按钮生成当前画面”的交互方式。

## 3. 依赖

第二版需要额外安装：

```powershell
pip install -r requirements_diffusion.txt
```

主要依赖包括：

1. `torch`
2. `diffusers`
3. `transformers`
4. `accelerate`
5. `opencv-python`
6. `pillow`

## 4. 运行图片模式

```powershell
python src/diffusion_neoclassical_demo.py --image path\to\street_photo.jpg
```

输出图片会保存在：

```text
outputs/
```

## 5. 运行摄像头模式

```powershell
python src/diffusion_neoclassical_demo.py
```

按键：

1. `g` 或空格：生成当前摄像头画面
2. `s`：保存上一次生成结果
3. `q` 或 `Esc`：退出

## 6. 为什么第二版先不训练

Stable Diffusion 的训练或微调需要更明确的数据集和 GPU 资源。当前电脑没有检测到 NVIDIA GPU，所以直接训练会非常慢。

更合理的顺序是：

1. 第二版先测试大模型图生图效果
2. 观察 prompt 和参数能达到什么程度
3. 收集失败案例
4. 再决定第三版 LoRA 需要学习什么

## 7. 第三版计划

第三版可以加入 LoRA 微调：

1. 收集新古典主义绘画参考图
2. 为每张图写 caption
3. 使用 `diffusers` 的 LoRA 训练脚本
4. 得到一个新古典主义风格 LoRA
5. 在第二版图生图 demo 中加载这个 LoRA
6. 对比没有 LoRA 和有 LoRA 的效果

这样项目就能同时包含大模型使用和训练过程，更符合 EMI 最终项目要求。
