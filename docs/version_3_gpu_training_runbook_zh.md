# Version 3: LoRA GPU 训练运行手册

## 1. 训练数据够不够

当前数据集有 20 张新古典主义/学院派绘画参考图，配套 20 个 caption 和 20 行 `metadata.jsonl`。

这个规模足够做课程项目里的 LoRA 风格训练原型。它可以证明项目已经从“只用 prompt”推进到“准备并使用自定义风格数据训练”。

但它仍然偏小，风险是：

1. LoRA 可能过拟合少数肖像画
2. 风格可能更像 Ingres/David 肖像，而不是完整的新古典主义图像转译
3. 生成效果可能稳定性有限

如果后面有时间，建议扩充到 30 到 60 张，并继续删除 Renoir、Courbet、Cezanne、风景画、现代感过强或构图差异太大的图片。

## 2. 为什么不能直接在当前电脑正式训练

当前环境检测结果：

```text
torch 2.12.0+cpu
cuda False
cuda_count 0
```

也就是说当前电脑没有可用 NVIDIA CUDA GPU。Stable Diffusion LoRA 训练在 CPU 上会非常慢，不适合作为正式训练环境。

因此当前电脑适合做：

1. 数据下载
2. 数据筛选
3. caption 准备
4. metadata 生成
5. 推理测试
6. 训练脚本准备

正式训练建议放到 Colab、实验室 GPU 电脑或云 GPU。

如果一定要在本地 CPU 上先验证训练链路，可以运行低分辨率 trial：

```powershell
.\scripts\run_lora_training.ps1 -AllowCpu -NoValidation -Resolution 128 -MaxTrainSteps 50 -GradientAccumulationSteps 1 -Rank 4 -OutputDir lora_outputs\neoclassical_style_lora_sd15_smoke -MixedPrecision no
```

这个命令只适合证明训练流程能跑通，不代表最终画质。

## 3. 本地已经准备好的文件

数据集目录：

```text
data/neoclassical_lora/
```

数据集压缩包：

```text
release/neoclassical_lora_dataset.zip
```

训练启动脚本：

```text
scripts/run_lora_training.ps1
```

数据集导出脚本：

```text
scripts/export_lora_dataset.ps1
```

LoRA 对比脚本：

```text
src/lora_comparison_demo.py
```

## 4. 在有 NVIDIA GPU 的 Windows 电脑训练

先安装依赖：

```powershell
pip install -r requirements_diffusion.txt
```

然后运行：

```powershell
.\scripts\run_lora_training.ps1 -MixedPrecision fp16 -MaxTrainSteps 800
```

如果显存比较小，可以降低参数：

```powershell
.\scripts\run_lora_training.ps1 -MixedPrecision fp16 -Resolution 384 -MaxTrainSteps 500 -Rank 4
```

训练输出默认保存到：

```text
lora_outputs/neoclassical_style_lora_sd15/
```

## 5. 在 Colab 训练的基本流程

先把本地数据集打包：

```powershell
.\scripts\export_lora_dataset.ps1
```

然后把下面这个文件上传到 Colab：

```text
release/neoclassical_lora_dataset.zip
```

Colab 中可以执行类似命令：

```python
!git clone https://github.com/Camnuy/homework.git
%cd homework
!pip install -r requirements_diffusion.txt
!mkdir -p data/neoclassical_lora
```

上传 zip 后解压：

```python
!unzip -q /content/neoclassical_lora_dataset.zip -d data/neoclassical_lora
!python scripts/prepare_lora_dataset.py --write-missing-captions
```

训练：

```python
!git clone https://github.com/huggingface/diffusers external/diffusers
!accelerate launch --mixed_precision=fp16 external/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
  --train_data_dir=data/neoclassical_lora \
  --caption_column=text \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=800 \
  --learning_rate=0.0001 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --rank=8 \
  --output_dir=lora_outputs/neoclassical_style_lora_sd15 \
  --checkpointing_steps=100 \
  --validation_prompt="a source image transformed only in visual style into a restrained neoclassical oil painting" \
  --seed=42
```

训练完成后打包 LoRA：

```python
!zip -r neoclassical_style_lora_sd15.zip lora_outputs/neoclassical_style_lora_sd15
```

## 6. 训练完成后怎么测试

把训练好的 LoRA 放回本项目，比如：

```text
lora_outputs/neoclassical_style_lora_sd15/
```

运行单图生成：

```powershell
python src\diffusion_neoclassical_demo.py --image path\to\source.jpg --lora lora_outputs\neoclassical_style_lora_sd15 --size 512 --steps 12 --strength 0.45 --guidance 6.5 --seed 42
```

运行无 LoRA / 有 LoRA 对比：

```powershell
python src\lora_comparison_demo.py --image path\to\source.jpg --lora lora_outputs\neoclassical_style_lora_sd15 --size 512 --steps 12 --strength 0.45 --guidance 6.5 --seed 42
```

输出会保存到：

```text
outputs/
```

## 7. 推荐第一轮训练参数

第一轮建议：

```text
model: runwayml/stable-diffusion-v1-5
resolution: 512
rank: 8
learning_rate: 0.0001
max_train_steps: 800
batch_size: 1
gradient_accumulation_steps: 4
mixed_precision: fp16
```

如果效果太弱，可以增加到 1200 steps。  
如果画面过拟合、源图像结构被污染，可以降低到 400-600 steps，或者扩充数据集。

## 8. 项目记录怎么写

第三版 weblog 可以写：

1. 第二版 prompt-only 风格迁移效果不稳定
2. 第三版开始收集开放馆藏新古典主义参考图
3. 使用 The Met 和 Art Institute of Chicago 开放 API 下载候选数据
4. 人工筛掉不合适图片
5. 生成 caption 和 `metadata.jsonl`
6. 准备 LoRA 训练脚本
7. 当前本机没有 CUDA，所以正式训练需要转移到 GPU 环境
