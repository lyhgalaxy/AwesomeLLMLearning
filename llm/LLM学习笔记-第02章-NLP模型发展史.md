# LLM 学习笔记：第 2 章——从 N-gram 到 Transformer

> 主线：每一代模型都在解决上一代的一个瓶颈。本章不背模型名，而是追踪“上下文如何被表示”。

## 1. 一张时间线先建立全局认识

```text
1948 信息论/语言概率思想
  │
1980s～2000s 统计 NLP：N-gram、HMM、CRF
  │  瓶颈：稀疏、无法共享相似词知识
  ▼
2003～2013 神经语言模型、Word2Vec
  │  收益：词变成可学习的稠密向量
  │  瓶颈：固定窗口或缺少序列记忆
  ▼
2013～2016 RNN / LSTM / GRU / Seq2Seq
  │  收益：按顺序读文本，形成状态记忆
  │  瓶颈：长距离遗忘、训练难并行、固定向量压缩
  ▼
2014～2017 Attention
  │  收益：生成每个词时动态查看输入各位置
  ▼
2017 Transformer
  │  收益：用自注意力直接连接任意位置，并行训练
  ├── 2018 BERT：Encoder-only，双向理解
  ├── 2018 GPT：Decoder-only，自回归生成
  └── 2019 T5：Encoder–Decoder，统一 text-to-text
       │
       ▼
2020～至今 Scaling、指令微调、RLHF、MoE、长上下文、推理强化学习
```

## 2. N-gram：第一个容易理解的语言模型

N-gram 假设下一个词只依赖前面有限的 `N-1` 个词。

```text
Bigram： P(学习 | 喜欢)
Trigram：P(学习 | 我, 喜欢)
```

一句话概率近似为：

```text
P(x₁...xₙ) ≈ ∏ₜ P(xₜ | xₜ₋ₙ₊₁...xₜ₋₁)
```

训练不需要神经网络，只需统计计数：

```text
Count(我 喜欢 学习)
P(学习 | 我 喜欢) = ─────────────────
                      Count(我 喜欢)
```

### 数据、损失和参数

- 数据：普通文本即可，不需人工标签；
- “参数”：每种 N-gram 的计数或概率表；
- 训练：计数并平滑，不是反向传播；
- 评价：对数似然或困惑度；
- 生成：根据最近 `N-1` 个词对应的概率表抽样。

### 为什么需要平滑？

没在训练集出现的 N-gram 会得到概率 0。一句话只要包含一个未知组合，整句概率就变为 0。加一平滑、Kneser–Ney 等方法把一部分概率质量分给未见组合。

### 根本瓶颈：稀疏

```text
“我喜欢学习”出现很多次
“我热爱学习”出现很少

统计模型不会天然知道“喜欢”和“热爱”相似。
```

窗口越大，可能组合呈指数式增长；数据永远覆盖不完。这推动了分布式表示。

## 3. 神经概率语言模型：让相似词共享统计力量

2003 年 Bengio 等人的神经概率语言模型用 Embedding 表示词，再用前馈网络预测下一个词：

```text
[我] [喜欢] [深度]
 │      │      │
 ▼      ▼      ▼
Embedding 向量拼接
        │
        ▼
     MLP 隐藏层
        │
        ▼
  词表概率：学习/睡觉/……
```

若“喜欢”和“热爱”的向量相近，网络在一个上下文学到的规律可迁移到另一个上下文。

仍然存在固定窗口：看 5 个词的模型无法利用 50 个词前的信息。

## 4. Word2Vec：高效学习词向量

Word2Vec 不是完整的长文本生成模型，而是学习静态词向量的方法。

### CBOW

根据周围词预测中间词：

```text
我 [喜欢] 学习 大模型
↑             ↑
上下文 ──────→ 预测“喜欢”
```

### Skip-gram

根据中心词预测周围词：

```text
中心词“学习” → 预测附近的“喜欢”“大模型”
```

负采样避免每次对整个巨大词表做 Softmax：让真实词对得高分，让抽取的错误词对得低分。

### 局限

一个词只有一个向量：“苹果手机”和“吃苹果”中的“苹果”共享表示，无法根据上下文动态改变含义。

## 5. TextCNN：用卷积捕捉局部模式

卷积核像一个滑动窗口，在文本中寻找局部短语模式：

```text
我  很  喜欢  这个  产品
└──────┘             3-gram 卷积窗口
    └────────┘
        └──────────┘
              │
              ▼
          最大池化
              │
              ▼
         正面 / 负面
```

优点：并行、高效，善于文本分类；局限：标准卷积感受野有限，生成和长距离依赖不自然。

## 6. RNN：把过去压进一个隐藏状态

RNN 按时间顺序读取 Token：

```text
x₁ ─→ [RNN] ─h₁→ [RNN] ─h₂→ [RNN] ─h₃→
       ↑             ↑             ↑
      “我”          “喜欢”        “学习”
```

更新形式：

```text
hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)
```

同一组参数在所有时间步复用，因此理论上可处理任意长度。

### 如何训练？

通过时间反向传播（BPTT）把误差沿展开的时间轴传回去：

```text
Loss₃ ← Loss₂ ← Loss₁
  │       │       │
 h₃  ←   h₂  ←   h₁
```

长链相乘会造成梯度消失或爆炸：早期 Token 对后期输出的影响难以稳定传回。

## 7. LSTM 与 GRU：给记忆加“门”

LSTM 引入单独的细胞状态 `cₜ`：

```text
旧记忆 cₜ₋₁ ───────────────→ 新记忆 cₜ
             × 遗忘门          ↑
                              + 新信息 × 输入门

新记忆 cₜ ─× 输出门→ 当前隐藏状态 hₜ
```

- 遗忘门：旧信息保留多少；
- 输入门：新信息写入多少；
- 输出门：当前暴露多少记忆。

GRU 将门结构简化为更新门和重置门，参数更少。二者缓解而非彻底消除长距离问题，且时间步仍需串行。

## 8. Seq2Seq：输入序列映射到输出序列

经典机器翻译结构：

```text
Encoder LSTM                         Decoder LSTM
我 → 喜欢 → 学习 → [上下文向量] → I → like → studying
```

训练时使用 teacher forcing：Decoder 每一步读取正确的前一个目标词。推理时只能读取自己刚生成的词，因此会出现 exposure bias：训练从未学过如何从自己的错误中恢复。

最大的结构瓶颈是：无论输入多长，都被压进一个固定长度向量。

## 9. Attention：不再强迫模型只看一个压缩向量

生成每个目标词时，Decoder 都计算它与所有输入位置的相关性：

```text
输入：   我      喜欢      学习
权重：  0.05     0.10      0.85
                         ╲
输出当前词：             studying
```

步骤：

```text
Decoder 当前状态 + 每个 Encoder 状态
              │
              ▼
          对齐分数 scores
              │ Softmax
              ▼
          注意力权重 α
              │ 加权求和
              ▼
          当前上下文向量
```

Bahdanau Attention 解决了固定向量瓶颈，但 RNN 主体仍然串行。

## 10. Transformer：把 Attention 变成主角

2017 年的 Transformer 去掉循环，以自注意力直接建模序列内任意位置关系：

```text
RNN 路径：Token 1 → 2 → 3 → 4 → 5（远距离要走很多步）
Attention：Token 1 ─────────→ Token 5（直接建立联系）
```

训练时所有位置可并行计算。原论文在机器翻译任务中展示了更好的质量和更高的并行性。具体结构在下一章展开。

## 11. BERT、GPT、T5 三条路线

```text
Transformer
├── Encoder-only：BERT
│   双向注意力 + MLM → 强理解表征
├── Decoder-only：GPT
│   因果注意力 + Next-token → 自由生成
└── Encoder–Decoder：T5
    双向理解输入 + 自回归生成输出 → 条件生成
```

### BERT

- BERT Base：12 层、隐藏维度 768、12 头，约 110M 参数；
- BERT Large：24 层、隐藏维度 1024、16 头，约 340M 参数；
- 原始预训练数据：BooksCorpus（约 800M 词）与英文 Wikipedia（约 2.5B 词）；
- 目标：Masked LM 与原论文中的 Next Sentence Prediction。

收益是一个模型经微调可适配多个理解任务。它不是天然的长文本自回归生成器。

### GPT 路线

GPT-1 展示生成式预训练再微调；GPT-2 强化零样本任务迁移；GPT-3 扩到 175B 参数并展示 in-context learning：不给参数做梯度更新，只在提示中提供说明和示例。

```text
预训练知识 + 提示中的任务描述 + 示例
                    │
                    ▼
             直接继续生成答案
```

### T5

把所有任务转成 text-to-text：

```text
translate English to German: That is good. → Das ist gut.
summarize: [长文]                         → [摘要]
cola sentence: The course is jumping.    → unacceptable
```

## 12. 为什么 Transformer 最终占据主流？

| 维度 | RNN/LSTM | Transformer |
|---|---|---|
| 训练并行 | 时间步串行 | 序列位置可并行 |
| 长距离路径 | 随距离增长 | 一层注意力即可直接连接 |
| 归纳偏置 | 强顺序递归 | 弱顺序偏置，需位置编码 |
| 长序列成本 | 计算线性但难并行 | 标准注意力为平方成本 |
| 扩展到大规模 | 困难 | 对 GPU/TPU 矩阵计算友好 |

它不是没有缺点，而是最适合用现代硬件和海量数据稳定扩展。

## 13. 统一比较：每一代究竟改了什么？

| 模型 | 上下文表示 | 主要目标 | 解决的问题 | 新代价 |
|---|---|---|---|---|
| N-gram | 离散计数窗口 | 最大似然 | 简单可解释 | 数据稀疏 |
| 神经 LM | 固定窗口向量 | Next-token | 相似词共享 | 窗口仍有限 |
| Word2Vec | 静态词向量 | 上下文预测 | 稠密语义 | 一词一义 |
| TextCNN | 局部卷积特征 | 分类等 | 高效局部模式 | 长依赖弱 |
| RNN | 递归隐藏状态 | 序列预测 | 可变长度 | 梯度、串行 |
| LSTM/GRU | 门控记忆 | 序列预测 | 缓解遗忘 | 仍串行 |
| Seq2Seq | 编码器状态 | 条件生成 | 端到端序列映射 | 固定向量瓶颈 |
| Attention | 动态加权输入 | 条件生成 | 软对齐 | RNN 仍在 |
| Transformer | 全局自注意力 | 多种 | 并行与长距离 | 平方注意力成本 |

## 14. 复习题

1. N-gram 的稀疏问题为什么不能只靠无限增大 N 解决？
2. Word2Vec 和语言生成模型有什么区别？
3. RNN 的隐藏状态为何既是优点又是信息瓶颈？
4. LSTM 的三个门分别控制什么？
5. Seq2Seq 固定上下文向量为何不利于长句翻译？
6. Attention 如何缓解固定向量瓶颈？
7. Transformer 相比 RNN 更适合现代硬件的关键原因是什么？
8. BERT、GPT、T5 的信息可见范围分别是什么？

## 15. 参考资料

1. Bengio et al., *A Neural Probabilistic Language Model*：<https://www.jmlr.org/papers/v3/bengio03a.html>
2. Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*：<https://arxiv.org/abs/1301.3781>
3. Hochreiter, Schmidhuber, *Long Short-Term Memory*：<https://www.bioinf.jku.at/publications/older/2604.pdf>
4. Sutskever et al., *Sequence to Sequence Learning with Neural Networks*：<https://arxiv.org/abs/1409.3215>
5. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*：<https://arxiv.org/abs/1409.0473>
6. Vaswani et al., *Attention Is All You Need*：<https://arxiv.org/abs/1706.03762>
7. Devlin et al., *BERT*：<https://arxiv.org/abs/1810.04805>
8. Brown et al., *Language Models are Few-Shot Learners*：<https://arxiv.org/abs/2005.14165>
