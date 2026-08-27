# LLM 学习笔记：第 10 章——从零训练小模型到 LoRA、DPO、GRPO

> 目标：用可控的小实验验证理论。不要把“成功启动训练”误当作“训练出可靠模型”。

## 1. 实验路线总图

```text
实验 1 Tokenizer
   ↓
实验 2 Bigram/N-gram 基线
   ↓
实验 3 单层 Self-Attention
   ↓
实验 4 小型 GPT 预训练
   ↓
实验 5 SFT + Loss Mask
   ↓
实验 6 LoRA/QLoRA
   ↓
实验 7 DPO 偏好优化
   ↓
实验 8 简化 GRPO + 可验证数学奖励
   ↓
统一评测、消融和实验报告
```

## 2. 先固定实验规范

每次实验保存：

```text
configs/       超参数与随机种子
data_card.md   数据来源、许可、清洗、划分
tokenizer/     固定的词表与配置
checkpoints/   权重与 optimizer state
logs/          Loss、学习率、梯度、吞吐
eval/          Prompt、原始输出、评分脚本
report.md      结论、失败、下一步
```

训练集、验证集、测试集在数据处理前就固定；测试集不能用于调参。

## 3. 实验 1：训练和评估 Tokenizer

### 变量

```text
算法：BPE vs Unigram
词表：8K / 16K / 32K
数据：10MB / 100MB / 1GB 代表性样本
```

### 指标

```text
字符数 / Token 数
UTF-8 字节数 / Token 数
每种语言 fertility
字节回退比例
领域术语平均切分数
Encode→Decode 往返正确率
```

图表应画：数据量增加时压缩率是否收敛；不同语言每千字符 Token 数；词表大小对序列长度的影响。

## 4. 实验 2：建立非神经和 Bigram 基线

训练字符 N-gram 与第 1 章的 Bigram 神经模型，记录验证 PPL。基线的意义是证明复杂模型是否真的带来收益。

```text
Unigram → Bigram → Trigram → Neural Bigram
   │          │          │          │
   └──────────验证 PPL 与生成样本─────┘
```

## 5. 实验 3：可视化 Self-Attention

实现单头注意力：

```python
scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
weights = scores.softmax(dim=-1)
output = weights @ v
```

将 `weights` 画成热力图：

```text
           我  喜欢 学习
我         █   ·   ·
喜欢       ▒   █   ·
学习       ░   ▓   █
```

检查未来位置权重是否严格为 0；删除 `1/√d` 比较 Softmax 熵和梯度。

## 6. 实验 4：训练小型 GPT

建议教学配置而非性能配置：

```text
层数 4～8
隐藏维度 256～512
头数 4～8
上下文 256～1024
参数约千万级
```

训练流程：

```text
文本 → Tokenize → 固定长度块
  ↓
Embedding + Position
  ↓
Transformer Blocks
  ↓
LM Head → Cross Entropy
  ↓
验证 PPL + 固定 Prompt 生成
```

必须做过拟合测试：先让模型在一个很小 Batch 上把 Loss 降得很低，确认实现能学习，再扩大数据。若小 Batch 都无法过拟合，先修代码而不是加 GPU。

## 7. 训练稳定性检查

```text
Loss 突然 NaN？
├─ 学习率过大
├─ FP16 溢出
├─ 输入/标签越界
├─ 全部 Label 被 mask
└─ Attention 全行被 -inf

Loss 不下降？
├─ 标签未右移或错移
├─ 参数未进入 Optimizer
├─ 忘记 backward/step
├─ 数据几乎全 padding
└─ 梯度被错误 detach
```

监控训练/验证 Loss、梯度范数、学习率、Token/s、显存、样本长度和 NaN 数。

## 8. 实验 5：SFT

准备 1K～10K 条小型指令数据，使用统一 Chat Template。

```text
Tokens： <system> ... <user> ... <assistant> 回答 ...
Labels：    -100       -100                  回答 Token IDs
```

PyTorch CrossEntropy 中常用 `-100` 作为 ignore index。对比：

- 全参数 SFT；
- 只训练 Assistant Loss；
- 错误地对全序列算 Loss（负面对照）；
- 不同学习率与 epoch。

评测不能只看训练 Loss：检查未见指令、格式遵循、知识回归和回答长度。

## 9. 实验 6：LoRA/QLoRA

消融矩阵：

| 实验 | Rank | Target Modules | 量化 | 记录 |
|---|---:|---|---|---|
| A | 8 | q,v | 无 | 质量/显存 |
| B | 16 | q,k,v,o | 无 | 质量/显存 |
| C | 16 | Attention+FFN | 4-bit Base | 质量/显存 |

记录 Trainable Params，而不是只记录总参数。验证 Adapter 单独加载与合并后输出是否近似一致。

## 10. 实验 7：DPO

构造偏好对：

```text
Prompt
├── chosen：正确、简洁、有依据
└── rejected：错误或不符合要求
```

流程：

```text
先有 SFT Policy + 冻结 Reference
  ↓
分别计算 chosen/rejected 的序列 log-prob
  ↓
计算相对于 Reference 的偏好差
  ↓
DPO Loss → 更新 Policy
```

监控 chosen/rejected reward margin、KL、回答长度和保留集胜率。避免让 chosen 永远更长，否则模型可能只学到长度偏好。

## 11. 实验 8：简化 GRPO

选择答案可验证的小学算术：

```text
每个 Prompt 采样 G=4～8 个回答
  ↓
解析最终答案
  ↓
Reward：正确=1，错误/无法解析=0
  ↓
组内标准化 Advantage
  ↓
Clipped Policy Loss + KL
```

先统计：全对组、全错组、有方差组各占多少。如果绝大多数全错，应降低题目难度或先 SFT；如果全对，应增加难度。

## 12. 评测金字塔

```text
              人工盲评
           真实业务 A/B 测试
        任务保留集与安全测试
     标准基准（注意污染与版本）
  验证 Loss / PPL / 单元测试
```

底层便宜、频繁；顶层昂贵但更贴近真实价值。不要用单个 Benchmark 决定部署。

## 13. 资源估算

### 权重显存粗算

```text
FP32：参数 × 4 bytes
BF16/FP16：参数 × 2 bytes
INT8：参数 × 1 byte（另有 scale 等开销）
4-bit：参数 × 0.5 byte（另有量化元数据）
```

7B BF16 权重理论下限约 14GB，但推理还有 KV Cache、运行缓冲；训练还有梯度、优化器和激活，远高于 14GB。

### 不要这样估算 MoE

激活 37B 不代表只需存 37B 权重。单 Token 计算激活较少专家，但全部专家权重仍需分布在内存/设备中。

## 14. 最终实验报告模板

```text
1. 假设：为什么这项改动可能有效？
2. 数据：来源、许可、数量、划分、污染检查
3. 模型：准确 checkpoint、架构、参数和版本
4. 方法：配置、随机种子、计算预算
5. 结果：平均值、波动、原始样本、失败案例
6. 消融：究竟是哪部分带来收益？
7. 成本：训练时长、GPU hours、推理延迟/吞吐
8. 风险：偏差、安全、隐私、已知不适用范围
9. 可复现：代码提交、环境锁定、权重和日志
```

## 15. 推荐学习顺序

```text
第 1 周：Tokenizer + N-gram + Bigram
第 2 周：Attention + 小 GPT 前向
第 3 周：小 GPT 预训练与评测
第 4 周：SFT + LoRA
第 5 周：DPO
第 6 周：GRPO + 综合报告
```

不要跳过小模型直接跑大模型训练脚本。小模型让错误更便宜、曲线更快出现，也更容易真正理解每一行代码。

## 16. 参考资料

1. Karpathy, nanoGPT：<https://github.com/karpathy/nanoGPT>
2. Hugging Face Transformers：<https://huggingface.co/docs/transformers/>
3. Hugging Face PEFT：<https://huggingface.co/docs/peft/>
4. Hugging Face TRL：<https://huggingface.co/docs/trl/>
5. vLLM 文档：<https://docs.vllm.ai/>
6. verl：<https://github.com/volcengine/verl>
