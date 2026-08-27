# LLM 学习笔记：第 3 章——深入理解 Transformer

> 学习目标：从张量流和信息流理解 Transformer，而不是只会背 `QKV`。

## 1. Transformer 总图

```text
Token IDs
   │
   ▼
Token Embedding + Position Information
   │
   ▼
┌──────────────── Transformer Block × N ────────────────┐
│  Norm → Self-Attention → 残差相加                      │
│    │                                                   │
│    └→ Norm → FFN/MLP → 残差相加                        │
└────────────────────────────────────────────────────────┘
   │
   ▼
Final Norm → Linear Vocabulary Head → Logits
```

一个 Block 有两类信息处理：

- Attention：不同 Token 之间交换信息；
- FFN：每个 Token 位置独立进行非线性变换。

## 2. Self-Attention 的直觉

句子：

```text
小明把书给了小王，因为他已经看完了。
```

理解“他”时，模型应结合“小明”“小王”“看完”等位置。Self-Attention 允许“他”向所有 Token 发出查询，并按相关程度汇总信息。

```text
             小明   小王   书   看完
“他”的关注    0.45  0.25  0.05  0.25   （示意）
                 ╲    │          ╱
                   加权汇总
```

## 3. Q、K、V 分别是什么？

每个输入向量 `x` 通过三个可学习矩阵得到：

```text
Q = xWQ   Query：我在找什么？
K = xWK   Key：我能用什么特征被匹配？
V = xWV   Value：如果关注我，应取走什么信息？
```

图书馆类比：

```text
读者的检索条件 Q ──与── 书目关键词 K 计算相似度
                              │
                              ▼
                      按相似度取书的内容 V
```

注意：Q/K/V 都是从当前隐藏状态线性投影得到，不是人工定义的语法标签。

## 4. 缩放点积注意力逐步计算

公式：

```text
Attention(Q,K,V) = softmax(QKᵀ / √dₖ + Mask)V
```

拆成五步：

```text
QKᵀ              每个 Query 与每个 Key 的点积相似度
  ↓
/ √dₖ            防止维度大时点积过大、Softmax 过尖
  ↓
+ Mask            屏蔽未来 Token 或 Padding
  ↓
Softmax           每行变为总和为 1 的权重
  ↓
乘 V              按权重汇总信息
```

形状示例：序列长度 `S=4`、头维度 `D=8`：

```text
Q [4,8] × Kᵀ [8,4] = Scores [4,4]
Scores [4,4] × V [4,8] = Output [4,8]
```

`Scores` 是 4×4，因为每个 Token 都要与 4 个 Token 建立关系。

## 5. 为什么除以 `√dₖ`？

若 Q、K 各维近似均值 0、方差 1，`dₖ` 个乘积求和后的方差会随 `dₖ` 增大。大分数进入 Softmax 后容易接近 one-hot，梯度很小。除以 `√dₖ` 将分数尺度拉回较稳定范围。

## 6. 因果掩码长什么样？

生成“我 喜欢 学习”时：

```text
           被查看位置
当前位置   我   喜欢  学习
我         ✓    ×     ×
喜欢       ✓    ✓     ×
学习       ✓    ✓     ✓
```

实现上把不允许的位置加上负无穷：

```text
[s11, -∞,  -∞]
[s21, s22, -∞]  → Softmax 后 -∞ 位置权重为 0
[s31, s32, s33]
```

Padding Mask 则用于忽略批量补齐的 `<pad>`。

## 7. 为什么使用多头注意力 MHA？

单头只有一套相似性空间，多头把隐藏维度分成多组独立投影：

```text
输入 X
 ├→ Head 1：可能学习局部搭配 ─┐
 ├→ Head 2：可能学习指代关系 ─┼→ Concat → 输出投影 WO
 ├→ Head 3：可能学习句法边界 ─┤
 └→ Head h：其他关系         ─┘
```

每个头不保证对应可命名的语言关系，但多组子空间提高了并行捕获不同关系的能力。

若总隐藏维度 `d_model` 固定，通常每头维度 `d_head=d_model/h`，所以增加头数并不简单等于按头数倍增主要投影参数。

## 8. MHA、MQA、GQA 为什么依次出现？

自回归推理会缓存历史 Token 的 K、V，称为 KV Cache。

```text
生成第 t 个 Token：
新 Q ──查询── [过去所有 K Cache]
              [过去所有 V Cache]
```

### MHA

每个 Query 头都有独立 K/V 头：

```text
Q: 32 heads
K: 32 heads
V: 32 heads
```

表达力强，但 KV Cache 大。

### MQA

所有 Query 头共享一组 K/V：

```text
Q: 32 heads
K: 1 head
V: 1 head
```

显著降低缓存与内存带宽，但共享过强可能影响质量。

### GQA

多个 Query 头组成一组，共享一组 K/V：

```text
Q: 32 heads
K: 8 heads
V: 8 heads
每 4 个 Q 头共享一组 KV
```

它是 MHA 与 MQA 之间的折中，现代 LLM 很常见。

KV Cache 元素量近似：

```text
2 × layers × sequence_length × kv_heads × head_dim
↑K和V
```

因此减少 `kv_heads` 对长上下文推理很重要。

## 9. 为什么需要位置信息？

不加位置时，Self-Attention 对输入排列近似“置换等变”：同样的 Token 换顺序，模型缺少明确的先后线索。

```text
狗咬人 ≠ 人咬狗
Token 集合相同，位置关系不同
```

### 绝对正弦位置编码

原始 Transformer 使用不同频率的正弦/余弦：

```text
PE(pos,2i)   = sin(pos / 10000^(2i/d))
PE(pos,2i+1) = cos(pos / 10000^(2i/d))
```

然后与 Token Embedding 相加。

### 学习式绝对位置

为每个位置训练一个向量。简单，但超过训练最大位置时难外推。

### 相对位置与 RoPE

相对方法更关心 Token 之间距离。RoPE 对 Q/K 的二维分量施加随位置变化的旋转，使点积自然携带相对位置信息：

```text
Q(pos i) 旋转角度 iθ
K(pos j) 旋转角度 jθ
Qᵢ·Kⱼ 关系中出现相对距离 (i-j)
```

RoPE 是现代 Decoder-only LLM 的常见选择；长上下文外推仍需频率缩放等专门策略。

## 10. FFN：每个 Token 自己“思考”

Attention 混合 Token 间信息，FFN 对每个位置使用同一组 MLP：

```text
x → Linear(d_model → d_ff) → 激活 → Linear(d_ff → d_model)
```

现代模型常用 SwiGLU：

```text
SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)
```

FFN 的中间维度通常大于隐藏维度，参数量在整个模型中占很大比例。

## 11. 残差连接与归一化

残差：

```text
y = x + Sublayer(x)
```

为梯度提供短路径，也让子层只需学习对当前表示的“修正量”。

### Post-Norm 与 Pre-Norm

```text
Post-Norm：Norm(x + Sublayer(x))
Pre-Norm： x + Sublayer(Norm(x))
```

深层 LLM 常用 Pre-Norm，因为优化通常更稳定。RMSNorm 只按均方根缩放，不减均值，计算更简洁，现代 LLM 常见。

## 12. 一个 Decoder Block 的数据流

```text
x
├─────────────────────────────────────────────┐
│                                             │
└→ RMSNorm → QKV → RoPE → Causal Attention ──┤ + → h
                                              │
h ────────────────────────────────────────────┐
                                              │
└→ RMSNorm → SwiGLU FFN ──────────────────────┤ + → output
```

重复 N 层后，每个位置的表示已融合其可见上下文，再投影到词表得到下一个 Token logits。

## 13. 参数量粗算

忽略 bias、Norm 和 Embedding，一个标准 Dense Block：

```text
Attention 投影：WQ,WK,WV,WO ≈ 4d²
普通 FFN：d→4d→d          ≈ 8d²
每层合计                   ≈ 12d²
N 层                       ≈ 12Nd²
```

再加词表 Embedding `Vd`。这是估算，不适用于所有 GQA、SwiGLU、MoE 配置，但能建立数量级直觉。

## 14. 计算与显存瓶颈

标准注意力分数矩阵大小是 `S×S`：

```text
序列长度翻倍：注意力分数元素约变 4 倍
```

训练常见优化：FlashAttention 减少显存读写，不必完整物化巨大中间矩阵；长上下文还可使用滑动窗口、稀疏或线性注意力等策略，但可能牺牲全局连接。

推理阶段分为：

```text
Prefill：一次处理整个提示，计算密集，可并行
Decode：每次生成一个 Token，频繁读取模型权重和 KV Cache，常受内存带宽限制
```

## 15. 容易误解的地方

1. Attention 权重高不等于严格的可解释因果关系；
2. 多头不保证每个头都有清晰人工语义；
3. RoPE 提供位置信息，不自动保证无限长度外推；
4. FlashAttention 主要改变计算实现，不改变标准 Attention 的数学结果；
5. GQA 主要为推理效率折中，不是简单的“更先进所以必然更准”。

## 16. 复习题

1. Q、K、V 为什么不能只用同一个原始向量？
2. 为什么要除以 `√dₖ`？
3. Causal Mask 与 Padding Mask 的目的分别是什么？
4. MHA、MQA、GQA 的 KV 头数有何区别？
5. KV Cache 为什么随序列长度增长？
6. 没有位置编码时，“狗咬人”和“人咬狗”会遇到什么问题？
7. Attention 和 FFN 在 Block 中分别承担什么角色？
8. Pre-Norm 为什么有利于深层训练？

## 17. 参考资料

1. Vaswani et al., *Attention Is All You Need*：<https://arxiv.org/abs/1706.03762>
2. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*（MQA）：<https://arxiv.org/abs/1911.02150>
3. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*：<https://arxiv.org/abs/2305.13245>
4. Su et al., *RoFormer*（RoPE）：<https://arxiv.org/abs/2104.09864>
5. Dao et al., *FlashAttention*：<https://arxiv.org/abs/2205.14135>
