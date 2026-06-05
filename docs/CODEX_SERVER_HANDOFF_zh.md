# Codex / Server Handoff

这份文档的目标不是课程展示，而是保证这个仓库在 GitHub 上被重新 clone 之后，下一位使用 Codex 的人可以快速继续工作。

## 1. 当前仓库是什么

这个仓库当前的主线项目是：

**Neoclassical Style Transfer**

技术路线已经整理成：

1. `V1` OpenCV baseline
2. `V2` Stable Diffusion img2img + prompt-only
3. `V3` ControlNet
4. `V4` ControlNet + IP-Adapter
5. `V5` V4 + light LoRA
6. `V6` InstantStyle

当前最值得继续优化的部分是：

1. `V4/V5/V6` 的结果质量
2. style reference 的选择
3. 更强的 GPU 路线，例如 SDXL / IP-Adapter Plus / InstantStyle-Plus

## 2. clone 后先看哪些文件

优先阅读：

1. `README.md`
2. `docs/version_5_ip_adapter_zh.md`
3. `docs/version_6_instantstyle_zh.md`
4. `src/method_comparison_demo.py`
5. `src/ip_adapter_style_transfer_demo.py`
6. `src/instantstyle_neoclassical_demo.py`

如果要看整个项目计划，再看：

1. 根目录外层的中文计划文档不在这个 repo 内
2. 但 repo 内的 README 和 docs 已经足够接着开发

## 3. 推荐环境

当前本地开发环境名字是：

```text
homework2
```

当前使用的 Python 版本是：

```text
Python 3.11
```

仓库里已经附带：

1. `requirements.txt`
2. `requirements_diffusion.txt`
3. `environment.homework2.yaml`
4. `scripts/bootstrap_homework2.sh`

## 4. 在服务器上如何起步

### 4.1 最简单的方法

```bash
git clone https://github.com/Camnuy/homework.git
cd homework
chmod +x scripts/bootstrap_homework2.sh
./scripts/bootstrap_homework2.sh
```

默认会创建：

```text
conda env: homework2
```

### 4.2 手动方式

```bash
conda env create -f environment.homework2.yaml
conda activate homework2
python -m pip install -r requirements.txt
python -m pip install -r requirements_diffusion.txt
```

## 5. 服务器上让 Codex 接着搞

建议下一位 Codex 从仓库根目录启动，然后先看：

1. `README.md`
2. `docs/CODEX_SERVER_HANDOFF_zh.md`
3. `src/method_comparison_demo.py`

推荐的第一句上下文可以直接写成：

```text
这是一个 Neoclassical Style Transfer 项目。当前主线是 V1 OpenCV 到 V6 InstantStyle。请先阅读 README、docs/version_5_ip_adapter_zh.md、docs/version_6_instantstyle_zh.md 和 src/method_comparison_demo.py，然后继续优化 V4-V6 的结果质量。
```

## 6. 当前方法入口

### V3

```bash
python src/controlnet_neoclassical_demo.py --image /path/to/source.jpg --no-window
```

### V4

```bash
python src/ip_adapter_style_transfer_demo.py --image /path/to/source.jpg --style-image /path/to/style.jpg --no-window
```

### V5

```bash
python src/ip_adapter_style_transfer_demo.py --image /path/to/source.jpg --style-image /path/to/style.jpg --lora lora_outputs/neoclassical_style_lora_sd15 --no-window
```

### V6

```bash
python src/instantstyle_neoclassical_demo.py --image /path/to/source.jpg --style-image /path/to/style.jpg --no-window
```

### 一次性出总对比图

```bash
python src/method_comparison_demo.py --image /path/to/source.jpg --style-image /path/to/style.jpg
```

## 7. 当前已知限制

1. 当前默认底模还是 `SD1.5`
2. 当前本地机器是 CPU-only，所以高质量出图效率有限
3. `V4/V5/V6` 虽然已经接通，但默认参数偏保守，结果差异可能不够大
4. 如果要追求更接近公开 demo 的质量，建议后续转向 GPU + SDXL + IP-Adapter Plus + InstantStyle

## 8. GitHub 里应该保留什么

建议保留：

1. `src/`
2. `docs/`
3. `scripts/`
4. `requirements.txt`
5. `requirements_diffusion.txt`
6. `environment.homework2.yaml`
7. `.gitignore`
8. `README.md`

不建议提交：

1. `outputs/`
2. `.cache/`
3. 大模型权重
4. 训练图片原始大批量素材

## 9. 课程要求里和交接最相关的点

根据本地课程要求整理，重点不是“现场实地演示”，而是：

1. 仓库必须 public
2. 需要 `README`
3. 需要 `weblog`
4. 需要至少 5 次 commit
5. 需要一个 `3-5` 分钟的演示视频链接
6. 老师会通过仓库看代码、README、weblog 和视频

所以这门作业更像：

**公开仓库 + 项目文档 + 结果展示 + 演示视频**

而不是：

**现场到老师面前实时跑 demo 才算完成**

## 10. 重要提醒

课程要求特别强调：

1. `README` 和 `weblog` 最终正文要用学生自己的话写
2. 不能让 AI 代写最终提交版 README / weblog 正文
3. 但可以用 AI 帮忙整理结构、梳理路线、检查技术一致性

所以这个 handoff 文档可以进 repo，但最终提交时，课程 README 和 weblog 仍然需要人工重写和确认。
