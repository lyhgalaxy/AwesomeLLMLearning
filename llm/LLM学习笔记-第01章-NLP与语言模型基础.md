# LLM 学习笔记：第 1 章——NLP 与语言模型基础

> 本章目标：看懂一条文本如何变成模型可计算的数字，模型如何预测下一个 Token，以及“训练模型”究竟改变了什么。
>
> 阅读前提：会阅读简单 Python 代码即可，不要求微积分或深度学习基础。

---

## 1. 先用一句话理解 NLP 和语言模型

**自然语言处理（Natural Language Processing，NLP）**，就是让计算机处理人类语言的一整类技术。

常见任务包括：

- 文本分类：判断评论是好评还是差评；
- 信息抽取：从合同里提取甲方、金额和日期；
- 机器翻译：把中文翻译成英文；
- 问答与对话：根据问题生成回答；
- 摘要、改写、纠错和代码生成。

**语言模型（Language Model，LM）**并不等于 NLP 的全部。它更像 NLP 中的一种通用发动机：给一段已经出现的文本，为后续可能出现的 Token 分配概率。

例如输入：

```text
今天天气很
```

模型可能给出：

| 候选 Token | 概率（示意） |
|---|---:|
| 好 | 0.46 |
| 冷 | 0.21 |
| 热 | 0.15 |
| 奇怪 | 0.05 |
| 其他所有 Token | 0.13 |

它不是直接“想出一句话”，而是在每一步预测下一个 Token，再把选出的 Token 接回输入中继续预测：

```text
今天天气很 → 好 → ， → 我们 → 去 → 公园 → 吧
```

这就是自回归生成（autoregressive generation）的核心。

---

## 2. 一条文本经过模型的完整旅程

```text
原始文本
  │
  ▼
清洗与 Tokenization（切分）
  │
  ▼
Token IDs（整数编号）
  │
  ▼
Embedding（稠密向量）
  │
  ▼
神经网络处理上下文
  │
  ▼
Logits（每个候选 Token 的原始分数）
  │
  ▼ Softmax
概率分布
  │
  ├──训练：与正确答案比较 → Loss → 反向传播 → 更新参数
  │
  └──推理：按某种解码策略选出下一个 Token → 继续生成
```

先牢牢记住两点：

1. 模型看到的不是汉字或单词，而是一串整数 ID；
2. 模型最后输出的也不是文字，而是对整个词表中每个 Token 的分数。

---

## 3. Token：模型处理文本的基本单位

### 3.1 为什么不能直接把“词”作为单位？

如果每个完整词都是一个 Token，会出现两个问题：

- 词表巨大：人名、地名、专业术语和新词几乎没有尽头；
- 未登录词：训练时没见过的词只能变成 `<unk>`，信息直接丢失。

如果每个字符甚至每个字节都是一个 Token，词表会很小，但序列会变长，模型需要更多计算才能理解一个词。

现代 LLM 通常采用**子词或字节级切分**，在词表大小与序列长度之间折中。

```text
unbelievable

按词：     [unbelievable]
按子词：   [un] [believ] [able]      ← 示意，实际结果取决于词表
按字符：   [u] [n] [b] [e] ...
```

对中文而言，一个汉字可能是一个 Token，也可能与相邻汉字合并；生僻字还可能被拆为多个字节级 Token。因此，**一个 Token 不等于一个字，也不等于一个英文单词**。

### 3.2 常见 Tokenizer 算法

| 方法 | 直观理解 | 常见特点 |
|---|---|---|
| Word-level | 一个完整词一个 Token | 简单，但词表大、未登录词多 |
| BPE | 从小单位开始，反复合并高频相邻对 | 常见、确定性强 |
| WordPiece | 倾向选择能较好解释语料的子词 | BERT 系列常见 |
| Unigram | 从较大候选词表中逐步删减，选择高概率切分 | 可保留多种切分可能 |
| Byte-level BPE | 以字节为底层单位再做 BPE | 几乎不会出现无法编码的字符 |

工程上的 Tokenization 通常不只有“切分”一步，而是一条流水线：

```text
Normalization → Pre-tokenization → 子词模型 → 添加特殊 Token → IDs
```

例如特殊 Token 可能包括：

- `<bos>`：序列开始；
- `<eos>`：序列结束；
- `<pad>`：批量训练时补齐长度；
- `<unk>`：未知单元；
- BERT 中的 `[CLS]`、`[SEP]`、`[MASK]`。

### 3.3 Tokenizer 为什么会影响模型？

Tokenizer 决定了：

- 同一段文本会占多少上下文长度；
- 中文、英文、代码等语言的编码效率；
- 模型需要学习多长的依赖关系；
- 输入和输出层的词表规模，从而影响参数量和计算量。

例如，一个模型宣称支持 32K 上下文，含义是最多约 32K 个 Token，而不是 32K 个汉字或单词。

### 3.4 Tokenizer 到底需不需要“模型”？

这个问题要先区分“模型”一词的两种含义。

**第一种：神经网络模型。**经典 BPE、WordPiece、Unigram Tokenizer 通常不需要运行 Transformer、RNN 之类的神经网络。切分时也不做大规模矩阵计算。

**第二种：由数据训练出来的切分模型或规则文件。**主流 Tokenizer 通常需要这个“模型”。它是从语料统计得到的，主要保存：

- Token 词表及其整数 ID；
- BPE 的合并规则，或者 Unigram 中各候选子词的概率/分数；
- 文本规范化规则；
- 特殊 Token 及其 ID；
- 预切分、空格和字节回退等配置。

因此，较准确的结论是：

> 常规 Tokenizer 通常不是神经网络，但它往往包含一个需要从文本语料中训练出来的统计切分模型。

这里的“训练 Tokenizer”和“训练 LLM”差别很大：

| 对比项 | 训练 Tokenizer | 训练语言模型 |
|---|---|---|
| 学到什么 | 词表、合并规则或子词分数 | 数十亿个神经网络权重 |
| 主要计算 | 字符串统计、计数、排序或概率估计 | 大规模矩阵乘法与反向传播 |
| 是否一般需要 GPU | 通常不需要 | 通常需要 |
| 是否使用交叉熵反向传播 | BPE 等通常不使用 | 通常使用 |
| 训练结果 | tokenizer 配置/词表文件 | 模型权重文件 |
| 训练完成后是否应随意修改 | 不应修改 | 可按训练方案继续更新权重 |

也存在例外，例如学习式分词器、神经 Tokenizer 和“无 Token/字节级”模型，但它们不是目前最常见的 LLM 文本切分方案。

### 3.5 Tokenizer 模型是怎样训练出来的？

完整流程通常是：

```text
确定模型将处理的语言与领域
  ↓
从预训练语料中抽取有代表性的文本样本
  ↓
清洗、去重并确定 Unicode/空格规范化规则
  ↓
选择 BPE、Unigram、WordPiece 或字节级方案
  ↓
确定词表大小和特殊 Token
  ↓
统计语料并学习词表/规则
  ↓
在保留集上评估编码效率、覆盖率和公平性
  ↓
冻结 Tokenizer，再用它编码 LLM 训练数据
```

关键点是：**Tokenizer 通常先于语言模型训练，并在预训练开始前冻结。**如果中途改变 Token 与 ID 的对应关系，原 Embedding 第 100 行所代表的 Token 可能改变，已训练权重就会发生语义错位。

#### BPE 是怎样学出来的？

用一个极小语料示意：

```text
low low lower lowest
```

初始时从字符或字节开始：

```text
l o w
l o w
l o w e r
l o w e s t
```

然后重复执行：

1. 统计所有相邻单元对的频次；
2. 找到最高频的一对，例如 `l + o`；
3. 合并为新单元 `lo`；
4. 重新统计，下一轮可能把 `lo + w` 合成 `low`；
5. 直到词表达到目标大小或完成指定次数的合并。

训练产物包含按顺序排列的合并规则。编码新文本时按照已学习规则合并，而不是重新统计，也不会更新规则。BPE 最初是数据压缩方法；在 NLP 中，它让固定词表能够用子词组合表示罕见词和新词。

#### WordPiece 是怎样学出来的？

WordPiece 与 BPE 外观很像，都会不断构造更长的子词。核心差异在于选择候选合并时，不只是简单看相邻对出现了多少次，还倾向选择能较好改善语料表示/似然的合并。不同实现的精确训练细节可能不同。

编码阶段常见做法是 **longest-match-first**：从当前位置优先找词表中最长的可匹配子词。例如 BERT 风格结果可能是：

```text
playing → play + ##ing
```

`##` 表示该子词接在一个词的内部，并不是原文真的含有两个井号。

#### Unigram 是怎样学出来的？

Unigram 的方向几乎与 BPE 相反：

1. 先建立一个较大的候选子词集合；
2. 为每个子词估计概率；
3. 计算一段文本的各种切分方式及其概率；
4. 逐轮删除对整体语料似然贡献较小的候选；
5. 反复估计概率和裁剪，直到达到目标词表大小。

同一句话可能存在多种切分。编码时通常选择概率较高的切分，也可以在训练语言模型时随机采样合理切分，使模型不依赖唯一边界。

#### SentencePiece 是一种算法吗？

严格来说，SentencePiece 是一套可以直接从原始句子训练、编码和解码的工具/框架，支持 BPE、Unigram 等模型。它会把空格视为普通符号处理，常用 `▁` 表示空格边界，因此适合中文、日文以及没有可靠空格分词的语言。

```text
BPE / Unigram       → 子词模型算法
SentencePiece       → 可训练和运行这些模型的工具/实现体系
Tokenizer 文件      → 最终保存的词表、规则和配置
```

### 3.6 训练 Tokenizer 需要多少数据？

没有一个由词表大小直接决定的固定答案。所需数据取决于语言数量、领域、语料均衡程度、目标词表大小、重复程度和希望覆盖多低频的子串。

实用上不必把全部预训练数据都交给 Tokenizer。若预训练语料有数万亿 Token，通常可以从中抽取一个**有代表性的子集**，因为高频子串的排序会逐渐稳定。重要的是覆盖和配比，而不是一味增加重复文本。

下面是规划实验的经验范围，不是学术定律：

| 用途 | 可从何种规模起步 | 目的 |
|---|---:|---|
| 教学/原型 | 数 MB～数十 MB | 验证代码和词表大小 |
| 单语言、单一领域实验 | 数百 MB 左右起，并做收敛评估 | 覆盖主要词形和术语 |
| 多语言或通用生产模型 | GB 级到数十 GB 的平衡抽样较常见 | 覆盖语言、代码和领域长尾 |

不能只说“使用了 10 GB”就认为充分。应逐步增加样本量，并观察：

1. **覆盖率/字节回退率**：是否仍大量出现 `<unk>` 或字节碎片；
2. **fertility（膨胀率）**：每个词、字符或字节平均被切成多少 Token；
3. **压缩率**：每个 Token 平均承载多少原始字节或字符；
4. **跨语言公平性**：同样信息量的不同语言，Token 数是否悬殊；
5. **领域术语切分**：代码、公式和专业词是否被过度打碎；
6. **保留集稳定性**：扩大数据时，高频词表与上述指标是否基本稳定。

更可靠的停止标准是：扩大代表性样本后，保留集上的词表重合度、压缩率和各语言 fertility 已基本稳定，继续加数据的边际收益很小。

### 3.7 词表大小如何选择？

词表不是越大越好：

```text
词表小 → 输入/输出层较小，但序列更长
词表大 → 常见词所需 Token 更少，但词表矩阵更大，低频 Token 可能学不充分
```

隐藏维度为 `d`、词表为 `V` 时，输入 Embedding 参数约为 `V × d`；若输出层不共享权重，还会再有一个相近规模的矩阵。因此应结合语言覆盖、计算预算和编码效率做实验。

### 3.8 一个可复现的 Tokenizer 训练示例

下面用 Hugging Face `tokenizers` 从文本文件训练 Byte-level BPE：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=32_000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    initial_alphabet=ByteLevel.alphabet(),  # 保留全部 256 种字节的表示能力
)

tokenizer.train(["corpus.txt"], trainer)
tokenizer.save("tokenizer.json")

encoded = tokenizer.encode("我正在学习 Tokenizer。")
print(encoded.tokens)
print(encoded.ids)
print(tokenizer.decode(encoded.ids))
```

真正用于 LLM 前还要测试：规范化是否可逆、特殊 Token 是否冲突、各种语言的编码效率、长数字和代码的切分，以及 encode/decode 往返是否符合设计。

---

## 4. Token ID 与词表

Tokenizer 会维护一个词表，把 Token 映射为整数：

```text
词表（示意）
0: <pad>
1: <bos>
2: <eos>
3: 我
4: 喜欢
5: 学习
6: 大模型
```

句子：

```text
我 喜欢 学习 大模型
```

编码后可能是：

```text
[1, 3, 4, 5, 6, 2]
```

ID 本身只有“索引”意义。`6` 并不比 `3` 更大、更强或更相似。语义关系要由下一步的 Embedding 学出来。

---

## 5. Embedding：把离散编号变成可学习的向量

假设词表大小为 50,000，Embedding 维度为 4,096，模型会维护一个形状如下的矩阵：

```text
Embedding 矩阵 E: [50,000, 4,096]
```

输入 Token ID 为 `6`，本质上就是取矩阵第 6 行：

```text
E[6] = [0.12, -0.43, 0.08, ..., 0.71]
```

这个向量不是人工规定的。训练开始时通常近似随机，随后通过大量预测任务逐渐调整。在相似上下文中出现的 Token，向量往往会形成某些相似结构。

Embedding 参数量为：

```text
词表大小 × 隐藏维度
```

上例中是：

```text
50,000 × 4,096 = 204,800,000 ≈ 2.05 亿参数
```

有些语言模型会让输入 Embedding 与输出词表投影共享权重（weight tying），减少参数并利用二者天然的对应关系。

---

## 6. 什么是“概率语言模型”？

### 6.1 一句话的概率如何分解？

假设一句话由 `x₁, x₂, …, xₙ` 组成。概率链式法则告诉我们：

```text
P(x₁, x₂, …, xₙ)
= P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × … × P(xₙ|x₁,…,xₙ₋₁)
```

翻译成白话：

> 一句话出现的概率，等于第一个 Token 的概率，乘以看到第一个后出现第二个的概率，再乘以看到前两个后出现第三个的概率……

例如：

```text
P(我 喜欢 学习)
= P(我) × P(喜欢 | 我) × P(学习 | 我, 喜欢)
```

GPT 类模型直接学习这些“给定前文，预测下一个 Token”的条件概率。

### 6.2 训练样本是如何自动构造的？

原始序列：

```text
<bos> 我 喜欢 学习 大模型 <eos>
```

只需错开一位，就得到输入和标签：

```text
输入：<bos>  我    喜欢  学习    大模型
标签：我     喜欢  学习  大模型  <eos>
```

因此，一段包含 `n` 个有效 Token 的文本可以同时提供接近 `n` 个预测位置，而不必为每个位置手工标注答案。这是 LLM 能利用海量无标注文本预训练的重要原因。

### 6.3 语言模型不只有“预测下一个 Token”一种训练方式

这里要区分三个概念：

- **网络架构**：Encoder-only、Decoder-only、Encoder–Decoder；
- **预训练目标**：模型在训练数据上要完成什么预测任务；
- **生成/解码方式**：推理时怎样得到最终文本。

三者高度相关，但不是同一个概念。

#### 方式一：自回归语言建模（Causal / Next-token LM）

```text
输入：  <bos> 我   喜欢  学习
标签：  我    喜欢 学习  <eos>
可见：  每个位置只能看自己左侧的 Token
```

目标为：

```text
P(x₁,…,xₙ) = ∏ₜ P(xₜ | x₁,…,xₜ₋₁)
```

注意力使用因果掩码，禁止当前位置偷看未来答案。GPT、Llama、Qwen 等生成式 Decoder-only 模型主要采用这种目标。

推理时预测一个 Token，把它追加到序列，再继续预测。这是现代聊天式 LLM 最常见的自由文本生成方式。

#### 方式二：掩码语言建模（Masked Language Modeling，MLM）

```text
原文：  我 喜欢 学习 大模型
输入：  我 [MASK] 学习 大模型
标签：     喜欢
可见：  被遮住位置左右两侧的上下文
```

BERT 的经典做法是选择约 15% 的 Token 作为预测目标；原论文中，对这些被选位置，80% 替换为 `[MASK]`，10% 替换为随机 Token，10% 保持原样。损失主要计算在被选中的位置上。

优点是能同时利用左右上下文，适合学习文本理解表征。局限是 `[MASK]` 通常不会出现在真实下游输入中，而且它并不天然定义从左到右生成任意长文本的过程。

> MLM 是一种预训练目标。BERT 能填空，但经典 BERT 不是像 GPT 那样不断追加下一个 Token 的通用生成模型。

如果要用掩码模型生成，可以反复放置、预测和替换 `[MASK]`，但顺序、停止条件和联合一致性需要额外设计。近年来也有专门的掩码生成或离散扩散语言模型，这与经典 BERT 的简单填空仍应区分。

#### 方式三：去噪/跨度腐化（Denoising / Span Corruption）

不是只遮一个 Token，而是破坏一段输入，让模型生成被删除的内容。例如 T5 风格：

```text
原文：  我喜欢学习大语言模型，因为它很有趣
输入：  我喜欢 <X>，因为 <Y>
目标：  <X> 学习大语言模型 <Y> 它很有趣
```

Encoder 读取被破坏的输入，Decoder 自回归生成被删除的跨度。这兼顾了双向理解和生成，常见于 Encoder–Decoder 模型。去噪还可以包括删除、打乱或替换文本。

#### 方式四：Prefix LM / 混合注意力目标

将序列分为前缀和生成区：

```text
[可以双向理解的前缀] → [只能从左向右生成的目标]
```

前缀内部可以互相看见；目标区中的每个位置只能看见前缀和更早的目标 Token。它适合“给定输入，生成输出”的任务，可以用单一 Transformer 堆叠，通过注意力掩码改变信息流。

#### 方式五：Seq2Seq 条件生成

```text
Encoder 输入：把这句话翻译成英文：今天天气很好
Decoder 输出：The weather is nice today.
```

它学习 `P(输出 | 输入)`。Decoder 仍然逐 Token 预测，但每一步还可以读取 Encoder 对完整输入的表示。翻译、摘要和传统问答经常使用这种方式。T5 把许多 NLP 任务统一写成 text-to-text 格式。

### 6.4 这些训练方式与“解码策略”不是一回事

| 层次 | 典型选项 | 回答的问题 |
|---|---|---|
| 预训练目标 | Next-token、MLM、Span Corruption | 模型训练时预测什么？ |
| 信息可见性 | 双向、因果、Prefix | 每个位置能看见哪些 Token？ |
| 模型架构 | Encoder、Decoder、Encoder–Decoder | 用什么网络组织输入输出？ |
| 解码策略 | Greedy、Beam Search、Top-k、Top-p | 已有概率后怎样选择输出？ |

例如“GPT 使用 next-token 目标，并在推理时用 top-p 采样”是一句完整描述；把“掩码预测”和“top-p”列为同一级的两种生成方法则不准确。

### 6.5 训练目标横向对比

| 目标 | 能否看右侧上下文 | 是否天然适合自由续写 | 常见架构/代表 |
|---|---|---|---|
| Causal LM | 否 | 是 | Decoder-only；GPT 类 |
| MLM | 是 | 否 | Encoder-only；BERT |
| Span Corruption | Encoder 可双向看输入 | 是，Decoder 生成缺失跨度 | Encoder–Decoder；T5 |
| Prefix LM | 前缀双向，输出因果 | 是 | UniLM 类混合方案 |
| Seq2Seq 条件生成 | Encoder 双向，Decoder 因果 | 是，条件于输入 | Transformer、T5、BART 类 |

### 6.6 用同一句话彻底看懂四种语言模型目标

先固定原始文本：

```text
我 喜欢 学习 大语言模型
```

所谓“Causal LM、MLM、Span Corruption、Prefix LM”，核心是在改变三件事：

1. 给模型看什么输入；
2. 哪些位置需要模型预测；
3. 一个位置能不能看见右侧内容。

#### 6.6.1 Causal LM：遮住未来，逐个预测下一个 Token

全称是 **Causal Language Modeling**，可译为“因果语言建模”或“自回归语言建模”。这里的“因果”不是说模型真正理解了现实世界的因果关系，而是说信息流只能从过去流向未来。

训练样本：

```text
输入位置： <bos>   我      喜欢    学习
预测标签： 我      喜欢    学习    大语言模型
```

一次前向传播就能并行计算所有位置的损失：

```text
L = -[log P(我|<bos>)
    + log P(喜欢|<bos>,我)
    + log P(学习|<bos>,我,喜欢)
    + log P(大语言模型|<bos>,我,喜欢,学习)] / 4
```

虽然训练时整句话已经放在显存中，但因果注意力掩码会阻止每个位置偷看右边答案。用 `✓` 表示可以看，`×` 表示不能看：

```text
正在计算的位置 \ 被查看位置
                 我   喜欢  学习  大语言模型
我               ✓    ×     ×      ×
喜欢             ✓    ✓     ×      ×
学习             ✓    ✓     ✓      ×
大语言模型       ✓    ✓     ✓      ✓
```

训练时可以并行，是因为正确的历史 Token 已经给出，这称为 teacher forcing。推理时没有后续正确答案，所以必须循环：

```text
提示词 → 预测 Token 1 → 拼回输入 → 预测 Token 2 → …… → <eos>
```

优点：

- 训练数据构造简单，任意连续文本都能使用；
- 训练目标和自由生成过程一致；
- 每个有效位置几乎都贡献一次损失，数据利用直接；
- 容易把提示、回答、代码和工具调用统一成一个 Token 序列。

局限：

- 当前 Token 不能利用右侧上下文，做双向理解时不如 MLM 自然；
- 推理必须逐 Token 进行，后一个 Token 依赖前一个，难以完全并行；
- 早期生成错误会进入后续上下文并继续影响结果。

代表模型：GPT 系列、Llama、Qwen、DeepSeek 等 Decoder-only LLM。

#### 6.6.2 MLM：在原句中挖空，再结合左右文填空

全称是 **Masked Language Modeling**，即“掩码语言建模”。

```text
原文：我 喜欢 学习 大语言模型
输入：我 [MASK] 学习 大语言模型
标签：   喜欢
损失：   只在选中的掩码位置计算
```

预测 `[MASK]` 时，模型既能看左侧的“我”，也能看右侧的“学习 大语言模型”：

```text
我  ←→  [MASK]  ←→  学习  ←→  大语言模型
          │
          └── 预测“喜欢”
```

BERT 原始方案随机选择约 15% 的位置作为预测目标。为减轻训练时总能看到 `[MASK]`、实际使用时却看不到它的差异，被选位置并非全都替换成 `[MASK]`：

```text
选中的 15% 位置中：
80% → 替换为 [MASK]
10% → 替换为随机 Token
10% → 保持原 Token 不变
```

不管输入最终如何处理，模型仍需预测被选位置原来的 Token。

优点：

- 每个目标位置可以同时利用左右上下文；
- 很适合分类、实体识别、句子匹配和抽取式问答等理解任务；
- 整段输入可一次并行处理。

局限：

- 一次只对被选中的一小部分位置计算主要训练损失；
- `[MASK]` 造成预训练与真实输入之间的差异；
- 没有天然规定一篇文章应从哪里开始、按什么顺序、何时结束生成。

代表模型：BERT、RoBERTa 等 Encoder-only 模型。

重要结论：**MLM 会“预测词”，但它不是经典自回归自由生成。**把一句话挖空再填回去，和从空白开始连续写一段回答，是两个不同问题。

#### 6.6.3 Span Corruption：不是遮一个词，而是让模型恢复一整段

Span Corruption 可译为“连续片段破坏”或“跨度腐化”，属于去噪语言建模的一种。

原文：

```text
我 喜欢 学习 大语言模型 因为 它 很有趣
```

随机选中两个连续片段后，可构造为：

```text
Encoder 输入：我 喜欢 <extra_id_0> 因为 <extra_id_1>
Decoder 标签：<extra_id_0> 学习 大语言模型 <extra_id_1> 它 很有趣 <eos>
```

其中 `<extra_id_0>`、`<extra_id_1>` 是哨兵 Token，分别标记缺失片段的位置。信息流如下：

```text
被破坏的输入
      │
      ▼
Encoder 双向理解整段输入
      │
      ▼
Decoder 从左到右生成所有缺失片段
```

损失通常计算在 Decoder 要生成的目标 Token 上。Decoder 每生成一个 Token，都能读取完整的 Encoder 表示和自己已经生成的左侧内容。

优点：

- 学会根据完整上下文恢复多 Token 内容；
- 比逐个 `[MASK]` 更接近摘要、改写和条件生成；
- Encoder 负责理解，Decoder 负责生成，职责清晰。

局限：

- 需要 Encoder 和 Decoder 两套网络模块，部署结构比纯 Decoder 更复杂；
- 腐化比例、片段长度和哨兵格式都会影响训练；
- 自由续写并不是其预训练任务唯一、直接的形式。

代表模型：T5；BART 也采用更广义的去噪式预训练，但具体破坏策略不同。

#### 6.6.4 Prefix LM：前缀充分理解，输出部分从左向右生成

Prefix 是“前缀”的意思。把一个序列分成两个区域：

```text
[输入前缀：请把下句翻译成英文：今天天气很好]
[输出区域：The weather is nice today.]
```

可见范围：

- 前缀中的 Token 可以互相看见，适合双向理解输入；
- 输出区域可以看见整个前缀；
- 输出区域只能看见自己左侧已经出现的输出，不能偷看未来答案。

简化的注意力图：

```text
                         被查看区域
正在计算的区域       输入前缀    已生成输出    未来输出
输入前缀                ✓           ×            ×
当前输出 Token          ✓           ✓            ×
```

训练时通常只关心输出区域的预测损失，也可以按具体实现对其他位置设置目标。推理时给定前缀，再自回归生成输出。

优点：

- 输入部分可双向理解，输出部分又天然支持生成；
- 可以在同一个 Transformer 序列中表达“输入 → 输出”；
- 适合翻译、摘要、问答等条件生成任务。

局限：

- 需要明确输入与输出边界；
- 注意力掩码和数据格式比纯 Causal LM 更复杂；
- 不同论文对“Prefix LM”的具体实现并不完全相同。

代表思路：UniLM 的统一注意力掩码、部分 Encoder–Decoder 替代方案。不要把 Prefix LM 与“Prefix Tuning”混淆：前者是注意力可见性/训练目标，后者是一种参数高效微调技术。

### 6.7 四种方式放在一张表里

| 对比项 | Causal LM | MLM | Span Corruption | Prefix LM |
|---|---|---|---|---|
| 中文直觉 | 接着写 | 完形填空 | 恢复被删段落 | 读完题目再作答 |
| 输入是否被破坏 | 否 | 遮若干 Token | 删除连续片段并放哨兵 | 通常不破坏前缀 |
| 目标位置能否看右侧 | 不能 | 能 | Encoder 能；Decoder 不能看未来输出 | 前缀能；输出不能看未来输出 |
| 损失算在哪里 | 几乎每个下一个 Token | 被选中的掩码位置 | Decoder 生成的缺失片段 | 通常是输出区域 |
| 能否直接自由续写 | 能 | 不天然适合 | 能条件生成 | 能条件生成 |
| 常见架构 | Decoder-only | Encoder-only | Encoder–Decoder | 单栈混合掩码或相关变体 |
| 代表模型 | GPT、Llama、Qwen | BERT、RoBERTa | T5 | UniLM 类方法 |

### 6.8 为什么 GPT 类通用 LLM 选择 Causal LM？

不是因为 Causal LM 在所有任务上都绝对最好，而是它特别适合把多种能力统一为“根据已有序列继续写”：

```text
文章续写：文章前文 → 后文
问答：问题 + 分隔符 → 答案
翻译：翻译要求 + 原文 → 译文
代码：需求 + 已有代码 → 新代码
对话：系统消息 + 历史对话 → 助手回复
工具调用：任务描述 → 结构化调用参数
```

它还有四个工程优势：

1. **数据构造统一**：普通文本不需要额外标注或人为制造 `[MASK]`；
2. **目标和使用方式一致**：训练时预测下一个 Token，聊天时也预测下一个 Token；
3. **架构统一**：只使用 Decoder 堆叠，容易随参数、数据和算力扩展；
4. **上下文学习自然**：示例、指令和问题都可放在前文，模型继续生成答案。

代价是推理串行且双向理解不是原生形式。不过随着规模、数据和指令训练增强，Decoder-only 模型也能把分类、抽取等理解任务改写成文本生成任务来完成。

### 6.9 自测：不要混淆以下四句话

1. “BERT 随机遮住一部分词再预测”——说的是 **MLM 训练目标**；
2. “GPT 只能让当前位置看见左侧”——说的是 **因果注意力可见性**；
3. “T5 有 Encoder 和 Decoder”——说的是 **网络架构**；
4. “生成时使用 Top-p=0.9”——说的是 **推理解码策略**。

如果能解释这四句话为什么不在同一个层次，就真正理解了这一节。

---

## 7. Logits、Softmax 与概率

模型在每个位置为词表里的每个 Token 输出一个原始分数，称为 **logit**。

假设候选只有三个：

```text
好: 2.0
冷: 1.0
猫: -1.0
```

Logit 可以是任何实数，不是概率。Softmax 把它们转成总和为 1 的概率：

```text
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
```

大致得到：

```text
好: 0.705
冷: 0.259
猫: 0.035
```

Softmax 的作用可以理解为：保留“谁的分数更高”，同时把所有候选正规化成一个概率分布。

注意：模型的输出概率是它根据训练数据和参数形成的预测置信度，不等于现实世界中事件的真实概率，更不保证事实正确。

---

## 8. 交叉熵：模型如何知道自己错了多少？

假设正确答案是“好”，模型给它的概率为 `p`。这个位置常用的损失是负对数似然：

```text
Loss = -log(p)
```

| 正确 Token 的预测概率 p | Loss = -log(p) | 含义 |
|---:|---:|---|
| 0.90 | 0.105 | 很有把握且答对，惩罚小 |
| 0.50 | 0.693 | 不够确定 |
| 0.10 | 2.303 | 正确答案只获很低概率，惩罚大 |
| 0.01 | 4.605 | 非常自信地忽视正确答案，惩罚很大 |

对多个位置取平均，就得到一批文本的交叉熵损失。

### 为什么常用交叉熵？

因为训练目标本来就是让正确序列的概率尽可能大。最大化：

```text
P(正确文本)
```

等价于最大化其对数：

```text
log P(正确文本)
```

优化器习惯最小化目标，因此加一个负号，得到负对数似然；当标签是 one-hot 分布时，它就是常见的交叉熵形式。

### 一个容易混淆的细节

在 PyTorch 中，`CrossEntropyLoss` 接收的是**原始 logits**，内部已经完成 `log-softmax + NLLLoss`。训练时不要先手工 Softmax 再传进去，否则数值稳定性和接口语义都会有问题。

---

## 9. 困惑度 Perplexity 是什么？

语言模型常见指标困惑度（PPL）与平均交叉熵的关系是：

```text
PPL = exp(平均交叉熵)
```

直观上可以把它粗略理解成：模型在每一步“有效地纠结于多少个候选”。

- PPL 越低，说明模型给真实文本分配的概率通常越高；
- PPL = 1 表示每一步都以概率 1 预测正确，是理想极限；
- 不同 Tokenizer、词表、数据集上的 PPL 往往不能直接横向比较；
- PPL 更低不自动等于事实性、推理能力或对话体验更好。

例如平均损失为 2：

```text
PPL = exp(2) ≈ 7.39
```

---

## 10. 训练究竟做了什么？

训练循环可以压缩成五步：

```text
1. 前向传播：输入 Token IDs，算出 logits
2. 计算损失：比较 logits 与正确的下一个 Token
3. 反向传播：计算每个参数对损失的影响方向
4. 优化器更新：把参数向降低损失的方向移动一点
5. 对下一批数据重复以上过程
```

用符号表示一次最基础的梯度下降：

```text
θ ← θ - η ∇θL
```

- `θ`：模型所有可训练参数；
- `L`：损失；
- `∇θL`：各参数应该如何变化才能最快增大损失，因此前面加负号；
- `η`：学习率，控制每一步走多远。

所谓“模型学到了知识”，从实现层面看，就是大量浮点数参数经过许多次微小更新后，形成了可以压缩和复现训练数据统计规律的数值结构。

### Epoch、Step、Batch

- **Sample**：一条训练样本或一个训练序列；
- **Batch**：一次并行计算的一组样本；
- **Step**：处理一个 Batch 并更新一次参数；
- **Epoch**：把整个训练集大致遍历一遍。

大规模 LLM 训练更常用“训练了多少 Token”描述规模，因为数据可能只遍历一次，或者来自持续混合的数据流。

---

## 11. 训练与推理的区别

| 对比项 | 训练 | 推理/生成 |
|---|---|---|
| 是否有正确标签 | 有，可由文本错位得到 | 没有 |
| 是否计算损失 | 是 | 通常否 |
| 是否反向传播 | 是 | 否 |
| 是否更新参数 | 是 | 否 |
| 输出用途 | 改进参数 | 选择下一个 Token |
| 常见资源重点 | 显存、吞吐、稳定性 | 延迟、吞吐、KV Cache |

模型完成训练后，默认情况下聊天不会继续修改它的权重。聊天上下文只是本次推理的输入，不等于模型把对话永久学进参数。

---

## 12. 生成文本时怎样选择下一个 Token？

模型给出概率后，还需要解码策略：

### Greedy decoding

每次选概率最高者。稳定、简单，但容易重复或显得刻板。

### Temperature

在 Softmax 前用温度 `T` 缩放 logits：

```text
softmax(logits / T)
```

- `T < 1`：分布更尖锐，回答更稳定；
- `T > 1`：分布更平坦，随机性更强；
- 温度不会为模型增加新知识。

### Top-k

只保留概率最高的 `k` 个候选，再重新归一化抽样。

### Top-p / nucleus sampling

按概率从高到低选择一个最小候选集合，使累计概率达到 `p`，再从中抽样。候选数会随上下文动态变化。

### Beam Search

同时保留累计得分最高的若干条候选序列，每一步继续扩展，再裁剪回固定数量。它试图寻找整体概率较高的序列，常用于翻译、摘要等条件生成；在开放式聊天中可能偏保守、重复，并不一定比采样自然。

同一模型、同一提示可能得到不同回答，通常不是参数发生了变化，而是采样过程带有随机性。

---

## 13. 最小 PyTorch 实验：训练一个二元语言模型

下面不是完整 LLM，而是一个最小可运行实验：模型看到当前 Token，预测下一个 Token。它没有 Transformer，也看不到更长上下文，但完整展示了 Token ID、Embedding、logits、交叉熵和参数更新。

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# 小语料。模型会反复观察其中相邻字符的转移规律。
text = "我爱学习。我爱大模型。学习使我快乐。"

vocab = sorted(set(text))
stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for token, i in stoi.items()}

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
x = data[:-1]   # 当前 Token
y = data[1:]    # 下一个 Token


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # 每一行既可看作当前 Token 的向量，
        # 也直接给出对所有下一个 Token 的 logits。
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids, targets=None):
        logits = self.table(token_ids)  # [sequence, vocab_size]
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss


model = BigramLanguageModel(len(vocab))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

for step in range(500):
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step={step:3d}, loss={loss.item():.4f}")


@torch.no_grad()
def generate(start: str, max_new_tokens: int = 20) -> str:
    current_id = torch.tensor([stoi[start]])
    result = [start]

    for _ in range(max_new_tokens):
        logits, _ = model(current_id)
        probs = torch.softmax(logits[-1], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        result.append(itos[next_id.item()])
        current_id = next_id

    return "".join(result)


print(generate("我"))
```

### 观察重点

1. `x` 与 `y` 只是同一段文本错开一位；
2. 模型输出的最后一维大小等于词表大小；
3. `cross_entropy` 直接接收 logits；
4. `backward()` 计算梯度，`optimizer.step()` 更新参数；
5. 生成时没有标签，只能从模型概率中选下一个字符。

### 这个模型为什么很弱？

它只根据当前一个字符预测下一个字符：

```text
P(xₜ | xₜ₋₁)
```

而现代自回归 LLM 希望利用此前整个有效上下文：

```text
P(xₜ | x₁, x₂, …, xₜ₋₁)
```

后续章节中的 RNN、LSTM 和 Transformer，本质上都在解决“如何更好地利用上下文”这个问题。

---

## 14. 常见误区

### 误区 1：Token 就是词

不一定。它可能是字、子词、标点、空格的一部分或字节。

### 误区 2：模型一次生成整段回答

自回归模型通常逐 Token 生成，只是推理速度很快，看起来像连续输出。

### 误区 3：概率最高就一定是事实

高概率只代表该输出在模型所学分布中更合适，不代表它经过事实数据库验证。

### 误区 4：训练损失越低，所有能力都越强

损失是重要信号，但可能受数据泄漏、过拟合、Tokenizer 和评测语料影响；事实性、安全性、推理和指令遵循还需单独评估。

### 误区 5：聊天会实时训练基础模型

普通推理只使用参数，不更新参数。上下文记忆、检索系统、产品级长期记忆与权重训练是不同机制。

---

## 15. 本章知识地图

```text
人类文字
└── Tokenizer
    ├── Token
    ├── Vocabulary
    └── Token ID
        └── Embedding
            └── 上下文建模网络
                └── Logits
                    ├── Softmax → 概率 → 解码 → 新 Token
                    └── Cross Entropy → 梯度 → 参数更新
```

如果这张图中的每条箭头都能用自己的话解释，本章目标就已经达成。

---

## 16. 复习题

### 概念题

1. 为什么现代语言模型通常不使用“完整词”作为唯一切分单位？
2. Token ID 为 100 的词是否比 ID 为 10 的词更重要？为什么？
3. Embedding 矩阵的两维分别代表什么？
4. Logit 和概率有什么区别？Softmax 做了什么？
5. 为什么下一 Token 预测不需要人工为每个位置单独标注？
6. 正确 Token 的预测概率从 0.8 降到 0.01 时，交叉熵会怎样变化？
7. 为什么不同 Tokenizer 对应的困惑度不适合直接比较？
8. 训练和推理阶段对模型参数的处理有何不同？

### 动手题

1. 修改实验语料，观察词表和生成结果如何变化；
2. 将采样改为 `torch.argmax`，比较结果的随机性；
3. 在生成函数中加入温度参数；
4. 计算初始损失和最终损失对应的困惑度；
5. 解释为什么二元模型无法区分“我爱学习”和“他不爱学习”中的长距离语义。

---

## 17. 参考资料

以下优先列出教材、论文和官方文档，而非二手博客：

1. Daniel Jurafsky, James H. Martin, *Speech and Language Processing (3rd ed. draft)*：<https://web.stanford.edu/~jurafsky/slp3/>
2. Hugging Face Tokenizers 官方文档——Tokenizer 流水线：<https://huggingface.co/docs/tokenizers/main/pipeline>
3. Hugging Face Tokenizers 官方文档——组件与 BPE、WordPiece、Unigram：<https://huggingface.co/docs/tokenizers/main/components>
4. PyTorch 官方文档——`CrossEntropyLoss`：<https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
5. Vaswani et al., *Attention Is All You Need*：<https://arxiv.org/abs/1706.03762>
6. Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*（NLP 中的 BPE）：<https://arxiv.org/abs/1508.07909>
7. Kudo, Richardson, *SentencePiece: A simple and language independent subword tokenizer and detokenizer*：<https://aclanthology.org/D18-2012/>
8. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*：<https://arxiv.org/abs/1810.04805>
9. Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*（T5）：<https://www.jmlr.org/papers/v21/20-074.html>

> 资料核对日期：2026-08-27。软件文档会更新；后续实操应以当时所用版本的官方文档为准。

---

## 18. 下一章预告

下一章将沿时间线解释 NLP 模型为什么从 N-gram 走向神经网络：

```text
N-gram → Word2Vec → TextCNN → RNN → LSTM/GRU → Seq2Seq → Attention
```

重点不是背年份，而是追踪同一个问题：**旧模型遇到了什么瓶颈，新模型具体改了哪里，又付出了什么代价？**
