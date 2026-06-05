# Version 3: LoRA 新古典主义风格训练计划

## 1. 第三版目标

第三版的目标不是重新做一个程序，而是在第二版 Stable Diffusion 图生图原型上加入“风格训练”。

第二版现在主要依赖 prompt 描述，例如“neoclassical oil painting”。这种方式的问题是模型只是根据文字猜风格，结果不稳定。第三版要训练一个 LoRA 风格适配器，让模型更稳定地学到新古典主义绘画特征。

预期效果：

1. 源图像结构大体保留
2. 画面更像古典油画，而不是普通滤镜
3. 色彩更克制，有暖色光影、学院派构图和古典建筑质感
4. 人物、建筑、物体不会完全丢失原图关系
5. 可以对比“无 LoRA”和“有 LoRA”的生成差异

## 2. 技术路线

第三版路线：

```text
收集新古典主义参考图
→ 整理训练图片和 caption
→ 用 Stable Diffusion 基础模型训练 LoRA
→ 得到 LoRA 权重文件
→ 在第二版程序里用 --lora 加载
→ 保存原图、无 LoRA 结果、有 LoRA 结果和对比图
```

当前第二版代码已经有 LoRA 入口：

```powershell
python src\diffusion_neoclassical_demo.py --image path\to\source.jpg --lora path\to\lora
```

所以第三版主要新增的是训练数据、训练配置、训练说明和效果对比。

本项目已经新增两个第三版辅助脚本：

```text
scripts/download_neoclassical_references.py
scripts/prepare_lora_dataset.py
scripts/export_lora_dataset.ps1
scripts/run_lora_training.ps1
src/lora_comparison_demo.py
```

第一个脚本负责从博物馆开放 API 下载候选参考图，并记录来源。第二个脚本负责生成 Diffusers 训练常用的 `metadata.jsonl`。第三个脚本负责把本地数据集打包，方便上传到 Colab 或 GPU 机器。第四个脚本负责拉取 Hugging Face Diffusers 官方 examples，并调用其中的 `train_text_to_image_lora.py` 训练脚本。第五个脚本用于在训练完成后比较“无 LoRA”和“有 LoRA”的效果。

如果要把数据转到 Colab 或 GPU 机器，可以运行：

```powershell
.\scripts\export_lora_dataset.ps1
```

它会生成：

```text
release/neoclassical_lora_dataset.zip
```

## 3. 数据集准备

建议先准备 20 到 50 张新古典主义绘画参考图，用于小规模风格 LoRA。

参考方向：

1. Jacques-Louis David 风格：英雄式构图、强烈明暗、古典题材
2. Jean-Auguste-Dominique Ingres 风格：平滑笔触、精致轮廓、学院派人物
3. 古典建筑、柱廊、雕塑、石材光感
4. 暖色、低饱和、博物馆油画质感

数据目录建议：

```text
data/
  neoclassical_lora/
    images/
      0001.jpg
      0002.jpg
    captions/
      0001.txt
      0002.txt
```

caption 示例：

```text
neoclassical oil painting, academic composition, warm chiaroscuro, refined realism, classical architecture, smooth brushwork
```

运行数据准备脚本：

```powershell
python scripts\prepare_lora_dataset.py --write-missing-captions
```

也可以先用开放馆藏下载脚本生成一批候选图片：

```powershell
python scripts\download_neoclassical_references.py --limit 30
```

下载后一定要人工筛选，把不符合新古典主义风格的图片从 `images/` 和 `captions/` 中删掉，再重新运行 `prepare_lora_dataset.py`。

注意：尽量使用公版作品或允许学习使用的图片，不要把不确定版权的大量图片直接提交到 GitHub。可以在 README 或 weblog 中说明图片来源和用途。

## 4. 本机训练风险

当前电脑没有检测到 NVIDIA GPU，本机 CPU 可以做推理测试，但不适合认真训练 LoRA。

推荐方案：

1. 本机负责整理数据、写 caption、测试推理程序
2. 训练放到有 NVIDIA GPU 的环境，比如实验室机器、云服务器或 Colab
3. 训练完成后只把 LoRA 权重和使用说明接回本项目

如果必须本机尝试，可以做极小规模 proof-of-concept，但质量可能很差，而且会非常慢。

如果之后换到有 NVIDIA GPU 的电脑，可以运行：

```powershell
.\scripts\run_lora_training.ps1 -MixedPrecision fp16 -MaxTrainSteps 800
```

当前电脑没有 NVIDIA GPU 时，可以先只运行数据准备脚本，不建议直接训练。

训练完成后，可以用同一张测试图对比第二版和第三版：

```powershell
python src\lora_comparison_demo.py --image path\to\source.jpg --lora lora_outputs\neoclassical_style_lora_sd15
```

这个脚本会保存四张图：原图、无 LoRA 结果、有 LoRA 结果、三栏对比图。

## 5. 第三版代码任务

第三版可以拆成几个小任务，方便形成课程要求中的代码迭代记录：

1. 新增数据集目录规范和 caption 模板
2. 新增 LoRA 训练配置说明
3. 新增训练启动脚本或命令说明
4. 修改第二版程序，让它可以一次输出“无 LoRA”和“有 LoRA”的对比结果
5. 新增结果评估文档，记录哪些参数有效、哪些失败

## 6. 和第二版的关系

第二版已经完成：

1. 支持图片输入和摄像头抓帧
2. 使用 Stable Diffusion 图生图
3. 保存原图、生成图和对比图
4. 支持 `--lora` 参数

第三版不需要推翻第二版，只需要让 `--lora` 真正有训练出来的权重可以加载。

## 7. 推荐第三版验收标准

第三版完成时，至少应该能展示：

1. 一张原始输入图
2. 第二版无训练结果
3. 第三版 LoRA 结果
4. 三者对比图
5. 一段中文说明，解释 LoRA 是否改善了新古典主义风格

如果 LoRA 效果仍然不好，也可以作为项目反思的一部分：说明数据量、训练步数、基础模型和硬件限制如何影响结果。

## 8. 下一步

下一步建议先做数据和训练准备，而不是马上调更多 prompt。

具体顺序：

1. 确定第三版使用的基础模型
2. 收集 20 到 50 张新古典主义参考图
3. 为每张图写简短 caption
4. 建立训练目录
5. 编写训练命令或训练脚本
6. 跑一次小规模训练
7. 把 LoRA 加载进现有 demo，对比结果

参考资料：

1. Hugging Face Diffusers LoRA training guide: https://huggingface.co/docs/diffusers/training/lora
2. Hugging Face Diffusers dataset preparation guide: https://huggingface.co/docs/diffusers/training/create_dataset
