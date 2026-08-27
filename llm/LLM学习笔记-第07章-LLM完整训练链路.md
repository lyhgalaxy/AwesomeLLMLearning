# LLM 学习笔记：第 7 章——Pre-train、CPT、SFT、LoRA 与 RLHF

## 1. 一张总流程图

```text
随机初始化
   │ Pre-training：海量通用文本，next-token
   ▼
Base Model（会续写、具备知识与模式）
   │ CPT：领域/语言/长上下文继续预训练（可选）
   ▼
Domain Base Model
   │ SFT：高质量指令—回答
   ▼
Instruction Model（会按格式回答）
   │ 偏好数据 + RM/PPO 或 DPO 等
   ▼
Aligned Chat Model（更符合人类偏好和安全要求）
   │ 部署优化：量化、并行、推理引擎
   ▼
在线服务与持续评测
```

不是所有项目都需要每一步。已有合适 Base 模型时，通常从 CPT 或 SFT 开始。

## 2. Pre-training：从随机参数学通用规律

### 输入输出

```text
输入：x₁ x₂ ... xₜ₋₁
标签：x₂ x₃ ... xₜ
目标：最小化所有有效位置的交叉熵
```

### 训练循环

```text
数据分片 → Tokenize/Packing → Batch
   ↓
前向传播 → Loss → 反向传播
   ↓
梯度同步/裁剪 → Optimizer Step → 学习率调度
   ↓
保存 Checkpoint → 验证 Loss/任务评测
```

### 常见工程组件

- AdamW 优化器、warmup + 衰减学习率；
- BF16/FP16 混合精度；
- 数据并行、张量并行、流水线并行；
- ZeRO/FSDP 参数、梯度和优化器状态切分；
- Gradient Accumulation 增大有效 Batch；
- Checkpoint 与故障恢复；
- 吞吐、MFU、梯度范数、Loss spike 监控。

### 显存为何远大于权重？

```text
训练显存 ≈ 权重 + 梯度 + 优化器状态 + 激活 + 临时缓冲
```

Adam 常为每个参数维护一阶、二阶矩；混合精度还可能保留 FP32 master weights。Activation Checkpointing 以额外计算换激活显存。

## 3. CPT：继续预训练

CPT（Continued Pre-Training）从已有 Base 权重出发，继续使用语言建模目标。

### 常见目的

```text
领域 CPT：通用模型 + 医疗/法律/金融语料
语言 CPT：增加中文或低资源语言
长上下文 CPT：使用更长序列继续训练
时间 CPT：加入近期知识
```

### 为什么不能只做 SFT？

SFT 适合教行为和格式，但若模型几乎没见过某领域的大量术语与文本分布，CPT 更适合先学习领域语言规律。

### 风险：灾难性遗忘

```text
只训练领域数据 → 领域能力 ↑，通用能力可能 ↓
```

缓解方式：混入一定比例通用数据、降低学习率、缩短训练、分领域评测。CPT 数据需要多少没有固定值，应根据领域验证损失与任务曲线决定。

## 4. SFT：教模型“如何回答”

SFT（Supervised Fine-Tuning）仍主要使用交叉熵，不是强化学习。

```text
<system>你是助手</system>
<user>什么是注意力？</user>
<assistant>注意力是一种……</assistant>
```

常用 Loss Mask：

```text
System Token：   mask=0
User Token：     mask=0
Assistant Token：mask=1  ← 只在这里计算 Loss
```

### 数据量

没有越多越好。少量高质量样本能显著改变行为，但覆盖广泛任务、语言、格式和安全边界通常需要更多数据。应画学习曲线，而不是引用某个固定条数：

```text
样本数：1K → 5K → 20K → 100K
观察：任务覆盖、过拟合、通用能力、安全回归
```

### 常见失败

- 回答模板化、过长；
- Chat Template 错误；
- 多轮对话角色串位；
- 数据冲突；
- 对 User Token 误算 Loss；
- 训练过强导致 Base 能力遗忘。

## 5. Full Fine-tuning 与 PEFT

```text
Full FT：更新全部参数
优点：适应能力强
代价：显存、存储和多任务版本成本高

PEFT：冻结大部分参数，只训练少量新增/选中参数
优点：便宜、每个任务只保存小适配器
代价：极大分布迁移时可能不及 Full FT
```

## 6. LoRA：用两个小矩阵表示权重改变量

不直接更新原矩阵 `W`，学习低秩增量：

```text
原输出：y = Wx
LoRA：  y = Wx + (α/r)BAx

W: [d_out, d_in] 冻结
A: [r, d_in]      训练
B: [d_out, r]     训练
r ≪ d_in,d_out
```

结构图：

```text
x ─────────→ W（冻结）──────────┐
│                               ├→ 相加 → y
└→ A（降维）→ B（升维）× α/r ──┘
```

可加在 Q/K/V/O 投影和 FFN 等模块。训练后可单独保存 Adapter，也可合并回 W，通常不增加合并后的推理延迟。

## 7. QLoRA

```text
Base 权重：4-bit 量化并冻结
LoRA 参数：BF16/FP16 训练
计算时：按需反量化参与矩阵计算
```

它进一步减少 Base 权重显存，使单机微调更大模型成为可能。量化节省不等于训练中所有张量都只有 4-bit；激活和 LoRA 梯度仍需较高精度。

## 8. 为什么需要偏好对齐？

SFT 只告诉模型“这个回答可模仿”，但很多问题没有唯一标准答案。人类更容易比较两个回答哪个更好：

```text
Prompt → 回答 A ┐
                 ├→ 标注者选择 A > B
Prompt → 回答 B ┘
```

偏好可能包含帮助性、真实性、安全性、简洁性和风格，但这些目标会冲突，必须明确标注规范。

## 9. 经典 RLHF 三阶段

```text
① SFT 模型
   │ 对 Prompt 采样多个回答，人类排序
   ▼
② Reward Model
   输入 Prompt+回答 → 输出标量奖励
   chosen 分数应高于 rejected
   │
   ▼
③ PPO 优化策略模型
   最大化奖励，同时用 KL 约束不要偏离参考模型太远
```

奖励模型的成对损失可写为：

```text
L_RM = -log σ(r_chosen - r_rejected)
```

策略目标直觉：

```text
总收益 = Reward Model 分数 - β × 与参考模型的 KL 偏离
```

KL 约束防止策略钻奖励模型漏洞、语言退化或偏离 SFT 分布过远。

## 10. DPO：不用显式训练在线 RL 循环

DPO 直接利用 chosen/rejected 对，推动策略相对参考模型提高 chosen 的对数概率、降低 rejected 的相对概率。

```text
Prompt
├── chosen  → 策略应相对参考模型更加偏好
└── rejected→ 策略应相对参考模型减少偏好
```

优点：实现稳定、无需单独在线采样和 Critic；局限：高度依赖离线偏好数据覆盖，仍有参考模型和超参数选择，且不能简单等同所有 RLHF 场景。

## 11. RLAIF 与可验证奖励

- RLAIF：由 AI 根据规则或宪法原则生成/评估偏好，人类监督标准；
- 可验证奖励：数学答案、代码测试、格式检查器直接给出奖励；
- 混合奖励：正确性 + 格式 + 安全 + 长度惩罚。

```text
奖励设计错误 → 模型找到捷径 → 指标高但实际差
```

必须检查 reward hacking、长度偏置、格式作弊和训练—评测泄漏。

## 12. 阶段选择决策树

```text
模型缺少领域语言/知识？──是→ CPT
          │否
          ▼
模型不会遵循任务格式？──是→ SFT（Full FT 或 LoRA）
          │否
          ▼
多个合理答案需偏好排序？──是→ DPO/RLHF
          │否
          ▼
答案可自动验证且需强化推理？──是→ GRPO/PPO 等在线 RL
```

## 13. 每阶段应评测什么？

| 阶段 | 核心指标 | 回归风险 |
|---|---|---|
| Pre-train | 验证 Loss、知识/代码/推理基准 | 数据污染、训练不稳 |
| CPT | 领域 PPL 与任务指标 | 通用能力遗忘 |
| SFT | 指令遵循、格式、多轮 | 模板化、过拟合 |
| 偏好优化 | 人工胜率、安全、真实性 | 奖励作弊、过拒答 |
| 部署 | 延迟、吞吐、显存、成本 | 量化精度下降 |

## 14. 参考资料

1. Ouyang et al., *Training language models to follow instructions with human feedback*：<https://arxiv.org/abs/2203.02155>
2. Hu et al., *LoRA*：<https://arxiv.org/abs/2106.09685>
3. Dettmers et al., *QLoRA*：<https://arxiv.org/abs/2305.14314>
4. Rafailov et al., *Direct Preference Optimization*：<https://arxiv.org/abs/2305.18290>
5. Touvron et al., *Llama 2*：<https://arxiv.org/abs/2307.09288>
