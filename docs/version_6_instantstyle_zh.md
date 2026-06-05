# Version 6：InstantStyle 式 block-wise 风格控制

## 目标

Version 6 不是再换一个底模名字，而是在 **ControlNet + IP-Adapter** 的基础上，把 IP-Adapter 的作用范围限制到更偏风格的 attention blocks 上。

这条线的核心问题是：

1. V4 / V5 里 IP-Adapter 默认会作用到更多层
2. 风格更强时，容易带来布局漂移、内容泄漏或局部扭曲
3. InstantStyle 的思路是把风格注入限制到更合适的层，让“给风格”和“保结构”更容易分开

在这个项目里，V6 的定义是：

```text
ControlNet: 继续负责结构和布局
IP-Adapter: 继续负责参考图风格
InstantStyle-style block selection: 只在特定 blocks 上激活 IP-Adapter
```

## 当前实现

当前可运行脚本：

```text
src/instantstyle_neoclassical_demo.py
```

默认模式：

```text
--instantstyle-mode style
```

这表示优先把 IP-Adapter 限制在更偏风格的 blocks 上，而不是像原始 V4 那样全层生效。

如果要更强一点的布局参与，也可以切到：

```text
--instantstyle-mode style_layout
```

## 为什么这版值得加

它不是推翻 V4 / V5，而是把前面的逻辑继续做细：

1. V4：有参考图，但 IP-Adapter 还是比较“全局”
2. V5：在 V4 上再叠一层 light LoRA，收紧风格一致性
3. V6：进一步限制风格注入位置，减少风格泄漏和结构扭曲

所以 V6 的价值主要在于：

1. 让风格更集中
2. 让结构控制更稳定
3. 给后续参数调优一个更合理的上限

## 运行方式

```powershell
python src\instantstyle_neoclassical_demo.py --image path\to\source.jpg --style-image path\to\neoclassical_reference.jpg --size 512 --steps 30 --strength 0.35 --control-scale 0.7 --ip-adapter-scale 0.8 --instantstyle-mode style --guidance 5.0 --style-square-mode squash --no-window
```

## 和前面几版的关系

建议现在把路线统一成：

1. V1 OpenCV baseline
2. V2 prompt-only diffusion
3. V3 ControlNet
4. V4 ControlNet + IP-Adapter
5. V5 V4 + light LoRA
6. V6 InstantStyle

其中 V6 不是替代前面所有版本，而是作为更精细的风格控制版本，帮助你解释为什么同样是“参考图引导”，不同层级的风格注入会带来不同结果。
