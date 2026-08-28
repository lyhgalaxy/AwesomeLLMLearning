# 第 6 章：VLM 的连接与融合架构

> 本章解决最核心的问题：视觉编码器输出的特征，究竟怎样变成 LLM 能理解的上下文。

## 1. 总图

```text
图片/视频
  ↓ Image Processor
Vision Encoder → N_v×D_v 视觉特征
  ↓
┌──────────────连接/压缩方案──────────────┐
│ Linear/MLP Projector                   │
│ Q-Former                               │
│ Perceiver Resampler                    │
│ Pool/Merge/Token Pruning               │
└────────────────────────────────────────┘
  ↓ N'_v×D_l
┌────────────────融合方式────────────────┐
│ 与文本 Token 拼接后做 Self-Attention    │
│ LLM 层中插入 Cross-Attention            │
│ 视觉专家/深层多级特征融合                │
└────────────────────────────────────────┘
  ↓
LLM → 文本/坐标/工具调用
```

## 2. 模态鸿沟

视觉特征 `V∈R^(N_v×D_v)` 与 LLM Token Embedding `T∈R^(N_t×D_l)` 存在：

- 维度不同；
- 训练目标不同；
- 数值分布不同；
- 序列长度不同；
- 空间位置与文字顺序含义不同。

连接器需要解决维度和语义适配；融合机制决定视觉信息在何处、以何种方式影响语言计算。

## 3. Linear Projector

```text
Z = VW + b
W∈R^(D_v×D_l)
```

它对每个视觉 Token 独立做同一个线性映射，不改变 Token 数。优点是参数少、训练快、容易验证；缺点是跨 Token 关系主要依赖 ViT 和后续 LLM。

LLaVA 证明简单线性连接器配合视觉指令数据也能形成通用对话能力。

## 4. MLP Projector

```text
Z = W_2 activation(W_1V+b_1)+b_2
```

相比线性层增加非线性容量。LLaVA-1.5 使用两层 MLP 等改进建立了强基线。容量更强不保证无限提升：连接器过大可能过拟合，且不能替代视觉数据质量。

## 5. Q-Former

BLIP-2 使用 Querying Transformer：一组可学习 Query 从冻结图像编码器特征中提取对语言最有用的信息。

```text
Learned Queries ──Q─────────────┐
                                ▼
Image Features ──K,V→ Cross-Attention
                                ↓
固定数量 Query Outputs → LLM
```

如果图像产生 257 个甚至更多特征，Q-Former 可输出固定数量 Query Token。BLIP-2 分两阶段训练：

1. 冻结图像编码器，做视觉—语言表征学习；
2. 冻结 LLM，做视觉到语言的生成学习。

优点：压缩视觉序列、冻结大模型也能训练；代价：固定 Query 数可能成为细节瓶颈。

## 6. Perceiver Resampler

Flamingo 用一组 Latent Query 通过 Cross-Attention 把可变数量视觉特征压成固定数量：

```text
可变长度图片/视频特征 → 固定 K 个视觉 Latents
```

这有利于多图和视频控制序列长度。它与 Q-Former 都使用可学习 Query，但训练目标、层结构和连接位置需按具体模型区分，不能只因“都压缩 Token”就视为同一模块。

## 7. Token Pooling、Merge 与 Pruning

### 7.1 Pooling/Merge

例如把相邻 `2×2` Patch 合并：

```text
32×32=1024 Tokens → 16×16=256 Tokens
```

Token 数变四分之一。可以平均、拼接后映射或学习式聚合。

### 7.2 Token Pruning

按重要性保留部分 Token。优点是动态省算力；风险是问题相关的小物体在看到问题前就被删掉。

### 7.3 固定 Query 压缩

无论原始分辨率多少，输出 K 个 Token，成本稳定；但高分辨率新增细节未必能穿过固定瓶颈。

## 8. 拼接式融合

把映射后的视觉 Token 插在文本序列中：

```text
[SYSTEM] [USER] [IMG_START] v1 ... vN [IMG_END]
请描述图片 [ASSISTANT] 答案...
```

之后 LLM 使用 Self-Attention 联合建模。实现简单，可复用 Decoder-only LLM 和 Next-Token Loss。

注意：文本里的 `<image>` 常只是模板占位符，处理器会把对应位置替换/扩展成真实视觉 Embedding。

## 9. Cross-Attention 融合

文本流保留原序列，LLM 某些层额外让文本 Query 关注视觉 K/V：

```text
Text Hidden ─→ Q ─┐
                   ├→ Cross-Attention → 注入视觉信息
Visual Latents → K,V┘
```

Flamingo 在冻结语言模型层间插入 Gated Cross-Attention，新模块初始门控可接近零，以减少破坏原语言能力。

## 10. Early、Middle、Late Fusion

| 方式 | 位置 | 特点 |
|---|---|---|
| Early | 输入阶段就合并 Token | 交互充分、序列长 |
| Middle | 网络中间层交叉注意力/专家融合 | 灵活但结构复杂 |
| Late | 各自编码后在输出附近融合 | 高效，细粒度交互有限 |

这些名称在不同论文中边界不完全一致，应结合数据流说明。

## 11. 动态分辨率

固定缩放会丢小字、扭曲长图。动态分辨率根据长宽比和像素预算选择网格：

```text
原图 H×W
 → 调整到 Patch Size 的整数倍
 → 产生可变长度 Patch Tokens
 → 记录二维位置
 → Merge/Project → LLM
```

优点：保留原始长宽比和细节。代价：Batch 形状、负载、显存和服务延迟波动。

## 12. AnyRes 与切块

LLaVA-NeXT 等路线会保留全局图，同时把高分辨率图片切成局部块：

```text
全局缩略图：理解整体
局部 Tiles：读取小字与细节
```

必须定义块顺序、边界、重叠和位置；否则 LLM 不知道局部块来自图片哪里。块数也是视觉 Token 预算的核心变量。

## 13. 多图与视频组织

### 13.1 多图

```text
[IMG1_START] ... [IMG1_END]
[IMG2_START] ... [IMG2_END]
问题：两张图有什么不同？
```

需要图像 ID、顺序和边界 Token，防止特征混淆。

### 13.2 视频

```text
frame_1 tokens + time_1
frame_2 tokens + time_2
...
```

可逐帧编码、时空 Patch、时间池化或抽帧。时间位置必须表达顺序和真实时间，否则难以做事件定位。

## 14. 位置编码

- 视觉编码器内部：二维 Patch 位置；
- 图块：Tile 行列位置；
- LLM 内部：视觉与文本在统一序列中的位置；
- 视频：时间/帧位置；
- Grounding：坐标与原图尺寸的映射。

若预处理缩放、裁剪或 Pad，输出坐标必须逆变换回原图。

## 15. Deep Fusion 与多层视觉特征

只使用 ViT 最后一层可能偏重高层语义。可以抽取浅、中、深多层特征：

```text
浅层：边缘、纹理、细节
中层：部件和形状
深层：对象与语义
```

Qwen3-VL 的 DeepStack 属于把多级 ViT 特征更深地注入语言模型的代表思路。收益是细粒度视觉信息，代价是实现和计算更复杂。

## 16. 冻结策略

| Vision | Connector | LLM | 场景 |
|---|---|---|---|
| 冻结 | 训练 | 冻结 | 初始空间对齐、最低成本 |
| 冻结 | 训练 | LoRA/训练 | 常见指令微调 |
| 部分解冻 | 训练 | LoRA | 领域视觉适配 |
| 全解冻 | 训练 | 全解冻 | 大规模联合训练 |

视觉编码器学习率常小于新连接器；突然全解冻可能造成梯度冲击和视觉遗忘。

## 17. 损失怎样传回视觉模块

生成答案的 Cross-Entropy 可以沿计算图反传：

```text
答案 Loss → LLM → Connector → Vision Encoder
```

若 Vision 冻结，梯度到此停止；若解冻，它会根据语言答案调整视觉特征。这可能改善任务，也可能让视觉骨干过拟合指令数据。

## 18. Token 与成本手算

`448×448, Patch14 = 1024` 个视觉 Token：

- 不压缩：LLM 接收 1024；
- `2×2 Merge`：接收 256；
- 32 Query Resampler：接收 32。

压缩率分别为 `1×、4×、32×`。但 32 个 Query 是否能保留小字，必须通过 OCR/Grounding 评估，而不能只看速度。

## 19. 如何选择连接方式

```text
通用对话、实现简单 → MLP Projector + 拼接
大量图片/视频、需固定成本 → Resampler/Q-Former
需保留 LLM 原结构和语言能力 → 门控 Cross-Attention
OCR/文档/高分辨率 → 动态分辨率 + 适度 Merge + 多层特征
边缘部署 → 强压缩，但必须业务回归测试
```

不存在所有任务都最优的连接器。

## 20. 常见误区

1. Projector 只改维度，不需训练：维度与语义都要适配。
2. Q-Former 就是固定平均池化：它用可学习 Query 和注意力取信息。
3. Token 越少一定越好：细节可能被压掉。
4. 动态分辨率没有代价：服务负载和显存更不稳定。
5. `<image>` 文本 Token 就是全部图像：通常只是占位和边界标记。
6. 多图直接拼起来即可：还需要身份、顺序、位置和边界。
7. Cross-Attention 一定优于拼接：效果依数据、结构和任务。
8. 解冻越多能力越强：可能遗忘、过拟合并大幅增加显存。

## 21. 总结与复习题

```text
连接器：把视觉特征变成 LLM 可用的维度与语义；
融合：决定视觉在何处影响语言计算；
压缩：控制视觉 Token 成本，但可能损失细节；
位置：保证模型知道 Patch、图片和帧来自哪里。
```

复习：Linear 与 MLP 有何区别？Q-Former Query 做什么？Resampler 为什么适合视频？拼接与 Cross-Attention 如何不同？动态分辨率带来什么工程问题？`2×2 Merge` 怎样改变 Token 数？

## 22. 参考资料

1. LLaVA：<https://arxiv.org/abs/2304.08485>
2. LLaVA-1.5：<https://arxiv.org/abs/2310.03744>
3. BLIP-2 / Q-Former：<https://arxiv.org/abs/2301.12597>
4. Flamingo / Perceiver Resampler：<https://arxiv.org/abs/2204.14198>
5. Perceiver：<https://arxiv.org/abs/2103.03206>
6. Qwen2-VL 动态分辨率：<https://arxiv.org/abs/2409.12191>
7. Qwen3-VL 官方仓库：<https://github.com/QwenLM/Qwen3-VL>

> 信息核对日期：2026-08-28。
