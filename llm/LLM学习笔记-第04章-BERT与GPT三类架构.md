# LLM 学习笔记：第 4 章——Encoder、Decoder 与 Encoder–Decoder

> 核心问题：三种架构不是简单的强弱关系，而是信息可见范围与输入输出形式不同。

## 1. 三种架构总图

```text
Encoder-only                  Decoder-only
完整输入双向互看              只能看左侧
[文章全文]                    [提示 + 已生成内容]
    │                              │
    ▼                              ▼
[上下文表示]                  逐 Token 生成
分类/抽取/匹配                 对话/续写/代码

Encoder–Decoder
[完整输入] → Encoder 表示 → Decoder 逐 Token 生成 [输出]
翻译/摘要/结构化转换
```

## 2. Encoder-only：重点是“理解已有文本”

每个 Token 可以查看整段输入：

```text
           被查看 Token
当前位置   A   B   C   D
A          ✓   ✓   ✓   ✓
B          ✓   ✓   ✓   ✓
C          ✓   ✓   ✓   ✓
D          ✓   ✓   ✓   ✓
```

经过多层后，每个位置都形成融合左右上下文的表示。

### BERT 如何工作？

输入：

```text
[CLS] 句子 A [SEP] 句子 B [SEP]
```

- `[CLS]` 表示可用于整段分类；
- 每个 Token 表示可用于命名实体识别；
- 问答头可为每个位置预测答案起止点。

预训练主要通过 MLM 恢复被遮 Token。微调时加一个很小的任务头，全部或部分更新 BERT 参数。

### 适合任务

```text
文本 → Encoder → 分类标签
文本 → Encoder → 每个 Token 的实体标签
问题+文章 → Encoder → 答案起止位置
两句话 → Encoder → 是否相似/是否蕴含
```

### 不擅长什么？

经典 Encoder-only 没有天然的无限续写循环。它能做填空，但不等于自由生成长回答。

## 3. Decoder-only：把一切任务变成续写

因果注意力只允许看左侧：

```text
用户：2+3等于多少？<assistant> 5
                                ↑
模型根据左侧所有内容预测这里
```

训练时整个序列右移一位作为标签；推理时把每次输出追加回上下文。

### 为什么问答也能变成续写？

```text
问题：法国首都是哪里？
答案：
```

若训练数据中大量出现“问题后跟答案”的模式，最大化下一个 Token 概率就会推动模型生成答案。指令微调进一步明确角色和格式。

### 优点

- 普通文本、对话、代码和工具调用都能统一为序列；
- 训练目标与推理一致；
- 架构单一，扩展和部署成熟；
- 提示中可放任务说明与示例，实现 in-context learning。

### 局限

- 推理串行；
- 输入中的每个位置不能原生看到右侧；
- 若输入很长但输出很短，仍需以生成方式表达分类结果；
- 错误 Token 会进入后续上下文。

## 4. Encoder–Decoder：先读完输入，再生成输出

```text
中文输入
  │
  ▼
Encoder（双向 Self-Attention）
  │ 产生整段表示 K,V
  ▼
Decoder
  ├── Causal Self-Attention：看已生成目标
  └── Cross-Attention：查看 Encoder 全部输入
  │
  ▼
英文输出
```

Decoder 中有两种注意力：

- Self-Attention：目标端不能偷看未来；
- Cross-Attention：当前目标词可查看整个源文本。

### 适合任务

- 翻译：源语言 → 目标语言；
- 摘要：长文 → 短文；
- 语法纠错：错误句 → 正确句；
- 信息抽取生成：文章 → JSON；
- 多模态：图像 Encoder → 文本 Decoder。

### 代价

需要维护 Encoder 和 Decoder；推理部署、缓存与并行策略更复杂。对纯续写任务，单一 Decoder 往往更直接。

## 5. 同一个任务放到三种架构中

任务：判断“这部电影很好看”的情感。

```text
Encoder-only
[CLS] 这部电影很好看 [SEP] → 分类头 → 正面

Decoder-only
文本：这部电影很好看
情感：→ 生成“正面”

Encoder–Decoder
Encoder 输入：classify sentiment: 这部电影很好看
Decoder 输出：正面
```

都能完成，但计算方式与输出约束不同。

## 6. 训练目标与架构的常见搭配

| 架构 | 信息可见性 | 常见目标 | 代表 |
|---|---|---|---|
| Encoder-only | 双向 | MLM | BERT、RoBERTa |
| Decoder-only | 因果 | Causal LM | GPT、Llama、Qwen |
| Encoder–Decoder | 输入双向、输出因果 | Span corruption/Seq2Seq | T5、BART |

这是“常见搭配”，不是数学上的唯一组合。UniLM 等工作可通过不同 Mask 在一个网络中模拟多种目标。

## 7. 为什么 GPT 选择 Decoder-only？

### 统一接口

```text
任意任务 = 上下文 → 后续 Token
```

无须为每个任务设计分类头或输出模块。

### 预训练与使用一致

```text
预训练：互联网前文 → 预测后文
聊天：系统+用户消息 → 预测助手消息
代码：已有代码 → 预测后续代码
```

### 易扩展

只堆叠一种 Block，训练系统和推理缓存统一。规模扩大后，模型逐渐表现出零样本、少样本和上下文学习能力。

### 但不是“Decoder 天生更聪明”

GPT 的能力增长来自架构、数据、算力、优化、指令训练、偏好对齐和工具系统共同作用。不能把全部收益归因于 Decoder-only。

## 8. In-context learning 与参数学习

```text
参数学习（训练/微调）
数据 → 反向传播 → 权重改变 → 跨会话保留

上下文学习（提示示例）
示例 → 放进上下文 → 本次输出改变 → 权重不变
```

GPT-3 的关键展示之一是，模型规模扩大后，少样本示例可只通过文本条件影响任务行为，无须梯度更新。

## 9. Chat 模型是怎么从 Base 模型来的？

```text
Base Decoder-only
  │ 大规模 next-token 预训练
  ▼
会续写，但不一定听指令
  │ SFT：指令—回答数据
  ▼
Instruction Model
  │ 偏好优化/RLHF + 安全训练
  ▼
Chat Model
```

Chat Template 把角色编码为特殊格式：

```text
<system>你是助手</system>
<user>解释注意力</user>
<assistant>
```

模板必须与训练格式匹配；格式错了可能明显降低表现。

## 10. 模型参数主要在哪里？

```text
Embedding：Token ID → 隐藏向量
N × Transformer Block：绝大部分推理与表征能力
LM Head：隐藏向量 → 词表 logits
```

Dense Decoder 模型中，Block 的 Attention 与 FFN 权重通常占主体；大词表 Embedding 也可能很大。MoE 模型则把 FFN 换为许多专家，单 Token 只激活其中少数。

## 11. 如何选架构？

```text
主要输出固定标签、抽取位置、向量检索？
  └→ Encoder-only 往往高效

主要做开放式对话、代码、续写、Agent？
  └→ Decoder-only 生态最成熟

输入和输出边界清晰的翻译、摘要、转换？
  └→ Encoder–Decoder 仍很自然
```

实际选择还受模型可用性、推理框架、许可证、延迟和数据影响。

## 12. 复习题

1. Encoder 双向注意力为何适合理解任务？
2. BERT 的填空与 GPT 的自由生成有什么本质区别？
3. Decoder-only 怎样把分类转为生成？
4. Cross-Attention 的 Q 来自哪里，K/V 来自哪里？
5. 为什么 Encoder–Decoder 适合翻译？
6. In-context learning 是否修改参数？
7. Base 模型为什么不一定服从指令？
8. Chat Template 为什么会影响模型表现？

## 13. 参考资料

1. Devlin et al., *BERT*：<https://arxiv.org/abs/1810.04805>
2. Radford et al., *Improving Language Understanding by Generative Pre-Training*：<https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>
3. Brown et al., *Language Models are Few-Shot Learners*：<https://arxiv.org/abs/2005.14165>
4. Raffel et al., *T5*：<https://www.jmlr.org/papers/v21/20-074.html>
5. Dong et al., *Unified Language Model Pre-training*：<https://arxiv.org/abs/1905.03197>
