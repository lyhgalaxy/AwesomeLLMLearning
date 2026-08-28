# 第 3 章：ViT 与视觉 Transformer

> ViT（Vision Transformer）不是“能看图聊天的模型”，而是一种视觉编码器架构。本章讲清它怎样把二维图片变成序列、怎样训练，以及为什么它成为现代 VLM 的常见视觉骨干。

## 1. 知识地图

```text
图片 B×C×H×W
  ↓ 切成 P×P Patch
N=(H/P)×(W/P) 个 Patch
  ↓ 展平 + 线性映射
Patch Embedding
  ↓ 加 CLS 与位置编码
Token 序列
  ↓ L 层 Transformer Encoder
上下文化视觉 Token
  ├─ CLS/Pooling → 分类头
  └─ Patch Tokens → VLM 连接器 → LLM
```

## 2. 为什么从 CNN 走向 ViT

CNN 具有局部连接和权重共享，样本效率高，但远距离关系要经过多层传播。Transformer 的注意力可以让任意 Token 直接交互。ViT 论文证明：把图片视为 Patch 序列、在大规模数据上预训练，纯 Transformer 也能取得很强的图像识别效果。

这不表示 CNN 被彻底淘汰：CNN 的局部先验仍然有效，ConvNeXt、Swin 等模型也说明两类设计可以相互借鉴。

## 3. Patch Embedding

输入 `x ∈ R^(H×W×C)`，Patch 大小 `P×P`。Patch 数：

```text
N = (H/P) × (W/P)
```

每个 Patch 展平后长度为 `P²C`，再乘可学习矩阵 `E ∈ R^(P²C×D)`：

```text
z_i = flatten(patch_i) E
```

得到 `N` 个 D 维 Token。也可以用 Kernel Size=P、Stride=P 的卷积一次完成。

示例：`224×224×3`、Patch 16：

```text
14×14 = 196 Patch
每个 Patch 原始长度 16×16×3 = 768
若隐藏维 D=768，则序列形状为 196×768
```

## 4. CLS Token

原始 ViT 在 Patch 序列前加入一个可学习的 `[CLS]`：

```text
[CLS], patch_1, patch_2, ..., patch_N
```

经过多层注意力后，用 CLS 输出做整图分类。它不是某个真实图块，而是一个聚合信息的位置。

在 VLM 中通常更关心全部 Patch Token，因为只保留一个全局 CLS 容易丢失小物体、文字和位置。模型可能丢弃 CLS、保留 Patch，或同时使用多层特征。

## 5. 位置编码

Self-Attention 本身不知道 Token 在二维图像中的位置。ViT 给每个 Token 加位置向量：

```text
输入 = Patch Embedding + Position Embedding
```

原始 ViT 使用可学习一维位置编码，Patch 按固定顺序展平。虽然序列是一维的，位置表对应二维网格。

### 5.1 改变分辨率怎么办

预训练网格 `14×14`，微调改成 `24×24` 时，位置数量不匹配，常见做法是把位置向量恢复成二维网格后插值。

风险：插值只是数值适配，不保证模型自动掌握更高分辨率细节。因此很多 VLM 还需要高分辨率继续训练、动态分辨率或二维位置编码。

## 6. ViT Encoder Block

典型 Pre-Norm Block：

```text
x
├─ LayerNorm → Multi-Head Self-Attention → + ─┐
└──────────────────────────────────────────────┘
                           ↓
├─ LayerNorm → MLP/FFN → + ───────────────────┐
└─────────────────────────────────────────────┘
```

简化公式：

```text
x' = x + MSA(LN(x))
y  = x' + MLP(LN(x'))
```

### 6.1 注意力在图片中的含义

```text
Q = XW_Q, K = XW_K, V = XW_V
Attention(X)=softmax(QKᵀ/√d_k)V
```

一个 Patch 可以根据内容关注另一个远处 Patch，例如车轮区域与车身区域建立关系。注意力权重是信息混合系数，不应直接等同于严格可解释的“模型看这里”。

### 6.2 MLP

注意力负责 Token 间交流，MLP 对每个 Token 独立做通道变换，通常先升维、激活、再降维。

## 7. 尺寸数据流示例

以 `B=2, H=W=224, P=16, D=768` 为例：

```text
图片                [2, 3, 224, 224]
Patch               [2, 196, 768]（展平前的每块长度）
线性映射            [2, 196, 768]
加入 CLS            [2, 197, 768]
12 层 Encoder 后    [2, 197, 768]
分类                取 [:,0,:]
VLM                 常取 [:,1:,:] 或多层 Patch 特征
```

## 8. ViT-B/L/H 名称怎样理解

名称通常反映层数、隐藏维和注意力头数，而不是输入分辨率。经典配置示意：

| 模型 | 层数 | 隐藏维 | 头数 | 典型参数规模 |
|---|---:|---:|---:|---:|
| ViT-Base | 12 | 768 | 12 | 约 86M |
| ViT-Large | 24 | 1024 | 16 | 约 307M |
| ViT-Huge | 32 | 1280 | 16 | 约 632M |

具体实现可能不同，必须看模型配置。“ViT-L/14”中的 `/14` 表示 Patch Size 14。

## 9. ViT 怎样训练

### 9.1 有监督分类

数据：图片和类别标签。流程：

```text
图片 → ViT → CLS → Linear → 类别 Logits
                          ↓
                    Cross-Entropy
```

### 9.2 图文监督

CLIP/SigLIP 用图片—文本对训练 ViT 图像塔，使视觉表示靠近匹配文本。损失详见第 4 章。

### 9.3 自监督训练

DINO、MAE 等不依赖人工类别：

- DINO：不同视图之间的教师—学生一致性；
- MAE：遮住大量 Patch，让模型重建；
- DINOv2/v3：扩展数据治理和自监督配方。

### 9.4 在 VLM 中训练

常见选择：

| 策略 | 优点 | 风险 |
|---|---|---|
| 冻结 ViT | 省显存、稳定 | 领域适配有限 |
| 解冻最后几层 | 折中 | 需小学习率 |
| ViT LoRA | 参数高效 | 框架支持有差异 |
| 全量解冻 | 适应性最强 | 显存大、可能遗忘原视觉能力 |

## 10. 数据量需要多少

没有固定答案。监督 ViT 从头训练通常比 CNN 更依赖数据与正则化；若使用已有 CLIP/SigLIP/DINO 权重，领域 VLM 不必重新训练整个视觉骨干。

评估数据需求时至少看：模型尺寸、是否从头训练、图片多样性、标签噪声、分辨率、增强、预训练任务和领域差距。小规模领域数据更适合冻结或参数高效微调，而不是从头训练大 ViT。

## 11. 高分辨率成本

Patch 固定时 `N∝HW`，全局注意力矩阵 `N×N`。

```text
224² / Patch14 → 256 Tokens
448² / Patch14 → 1,024 Tokens
896² / Patch14 → 4,096 Tokens
```

相对 224：448 的注意力矩阵约 16 倍；896 约 256 倍。这推动了窗口注意力、Patch Merge、切块和 Token 压缩。

## 12. ViT 输出怎样给 VLM

```text
ViT 最后一层 Patch Tokens
  ↓ 可选：去掉 CLS
  ↓ 可选：选择倒数第二层
  ↓ 可选：拼接多层 / 2×2 Merge / Pool
  ↓ Projector / Resampler
  ↓ LLM 隐藏维视觉 Token
```

选择倒数第二层有时是为了保留更通用的视觉细节；最后一层可能更贴近原预训练目标。多层特征可兼顾浅层细节和深层语义，但成本更高。

## 13. ViT、Swin、ConvNeXt

| 模型 | 核心方式 | 特点 |
|---|---|---|
| ViT | 固定 Patch + 全局注意力 | 结构简洁，长序列成本高 |
| Swin | 层级特征 + 移位窗口注意力 | 高分辨率和密集任务友好 |
| ConvNeXt | 现代化纯卷积网络 | 保留卷积高效实现与局部先验 |

VLM 视觉骨干并非只能用 ViT，但 ViT Token 序列与 Transformer LLM 的接口自然，因此非常常见。

## 14. 小型形状实验

```python
import torch
from torch import nn

B, C, H, W = 2, 3, 224, 224
P, D = 16, 768
x = torch.randn(B, C, H, W)

patch_embed = nn.Conv2d(C, D, kernel_size=P, stride=P)
y = patch_embed(x)             # [B, D, 14, 14]
tokens = y.flatten(2).transpose(1, 2)  # [B, 196, D]

print(y.shape)
print(tokens.shape)
```

## 15. 常见误区

1. ViT 本身就是 VLM：ViT 通常只是视觉编码器。
2. Patch Token 天生代表完整物体：初始 Patch 只对应局部像素块，语义由训练形成。
3. CLS 是图片缩略图：它是可学习聚合 Token。
4. 注意力图就是可靠解释：注意力只展示一种内部混合关系。
5. 改输入尺寸只需插值位置编码：还存在分布和细节学习问题。
6. Patch 越小一定越好：Token 和计算会显著增加。
7. ViT 不包含任何卷积：Patch Embedding 可用卷积实现，变体也可能混用。
8. VLM 必须使用最后层 CLS：多数生成式 VLM使用多个 Patch Token。

## 16. 本章总结

```text
ViT = 把图片切成 Patch 序列，再用 Transformer Encoder 编码。
Patch Embedding 解决输入形式；位置编码补充空间顺序；
Attention 建立 Patch 关系；MLP 变换通道信息。

ViT 可用类别、图文对或无标签图片训练。
进入 VLM 时通常保留局部 Patch Tokens，
再经连接器映射到 LLM 隐藏空间。
```

## 17. 复习题

1. ViT 为什么要切 Patch？
2. 写出 Patch 数公式。
3. CLS Token 的作用是什么？
4. 为什么需要位置编码？
5. 位置编码插值解决了什么、没有解决什么？
6. Attention 与 MLP 分别负责什么？
7. ViT-B/16 中 B 和 16 各表示什么？
8. ViT 可以通过哪些目标预训练？
9. 冻结和解冻视觉编码器各有什么代价？
10. 为什么 VLM 常保留 Patch 而非只用 CLS？
11. 分辨率提高为何昂贵？
12. ViT、Swin、ConvNeXt 有什么差异？

## 18. 参考资料

1. ViT：<https://arxiv.org/abs/2010.11929>
2. DeiT：<https://arxiv.org/abs/2012.12877>
3. Swin Transformer：<https://arxiv.org/abs/2103.14030>
4. ConvNeXt：<https://arxiv.org/abs/2201.03545>
5. MAE：<https://arxiv.org/abs/2111.06377>
6. CLIP：<https://arxiv.org/abs/2103.00020>

> 信息核对日期：2026-08-28。
