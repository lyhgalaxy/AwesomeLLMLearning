# LLM 学习笔记：第 8 章——PPO、DPO、GRPO、DAPO 与 GSPO

> 原计划中的 GAPO 暂按 DAPO 处理。不同论文实现细节可能变化，本章先建立共同坐标系。

## 1. 把语言生成看作强化学习

```text
状态 sₜ：Prompt + 已生成 Token
动作 aₜ：选择下一个 Token
策略 πθ：语言模型给出的 Token 概率
轨迹 τ：完整回答
奖励 R：人类偏好模型、规则或答案验证器的分数
```

```text
Prompt → Token₁ → Token₂ → ... → <eos>
                               │
                               ▼
                       Reward / Verifier
```

SFT 告诉模型“模仿这个答案”；RL 告诉模型“自己生成答案，结果好就提高其概率，差就降低”。

## 2. Policy Gradient 的核心

希望最大化期望奖励：

```text
J(θ) = E_{y~πθ(.|x)}[R(x,y)]
```

REINFORCE 形式直觉：

```text
∇J ≈ R × ∇ log πθ(生成的动作)
```

- 奖励为正：提高本次动作概率；
- 奖励为负：降低概率；
- 奖励有噪声时，梯度方差很大。

## 3. Baseline、Value 与 Advantage

仅看奖励 8 分无法判断表现：若这道题平均只有 2 分，8 很好；若平均 9 分，8 反而差。

```text
Advantage A = 实际回报 - 基准预期
```

Actor–Critic：

```text
Actor πθ：决定生成什么
Critic Vφ：估计当前状态未来能得多少分
Advantage：实际结果相对 Critic 预测好多少
```

Critic 降低方差，但需要额外模型、显存和训练稳定性。

## 4. 为什么要限制策略更新？

如果一次把高奖励回答概率推得太猛，模型可能语言退化或钻奖励漏洞。

重要性比率：

```text
rₜ(θ) = πθ(aₜ|sₜ) / πold(aₜ|sₜ)
```

`r=1` 表示新旧策略概率相同；偏离 1 越远，更新越大。

## 5. PPO：裁剪过大的策略更新

PPO 的 clipped objective：

```text
L = E[min(rₜAₜ, clip(rₜ,1-ε,1+ε)Aₜ)]
```

直觉图：

```text
更新收益
  │       ┌────────  超过范围不再继续获益
  │      /
  │_____/
  └────┬────┬────→ ratio
      1-ε   1+ε
```

LLM RLHF 常见系统：

```text
Policy/Actor：正在训练
Reference：冻结，提供 KL 约束
Reward Model：冻结，评估完整回答
Value/Critic：训练，估计价值
```

优点：在线从当前策略采样，能探索新回答；框架成熟。代价：四类模型带来显存与系统复杂度，超参数敏感，奖励模型可能被利用。

## 6. DPO：直接从偏好对优化

数据：

```text
(Prompt, chosen, rejected)
```

DPO 不显式训练 Reward Model，也不运行 PPO/Critic，而是比较策略模型相对于 Reference 对 chosen/rejected 的偏好变化。

```text
                 chosen 概率相对提高
Policy vs Ref  ─┤
                 rejected 概率相对降低
```

优点：像监督学习一样稳定、实现简单、资源较低。局限：通常是离线数据，探索能力有限；偏好数据若来自旧策略，可能覆盖不到新策略的问题；相对偏好不保证 chosen 的绝对概率一定按直觉变化。

严格说 DPO 是偏好优化方法，常与 RLHF 放在一起讨论，但没有经典在线 RL 交互循环。

## 7. GRPO：用同一问题的一组回答互相做基准

GRPO（Group Relative Policy Optimization）由 DeepSeekMath 引入。对同一个 Prompt 采样一组回答：

```text
Prompt x
├→ y₁ → reward 1
├→ y₂ → reward 0
├→ y₃ → reward 1
└→ y₄ → reward 0
        │
        ▼
组内标准化：(Rᵢ - mean(R)) / std(R)
        │
        ▼
作为相对 Advantage 更新策略
```

它不需要单独训练 Value/Critic，节省资源。常保留 PPO 式 ratio clipping 和 KL 正则。

### 适合场景

数学、代码、逻辑题等可自动验证答案，同一 Prompt 可采样多条轨迹。正确答案相对错误答案形成清晰组内信号。

### 局限

- 每题需要多次采样，生成成本高；
- 一组奖励全相同，组内 Advantage 接近 0，无法学习；
- 奖励稀疏，太难或太容易的问题贡献少；
- 长度归一化与 Token 级更新可能引入长度偏差；
- 无 Critic 不代表没有方差或训练不稳定。

## 8. DAPO：针对大规模 GRPO 训练的工程化改进

DAPO 全称 **Decoupled Clip and Dynamic sAmpling Policy Optimization**。论文基于 Qwen2.5-32B Base 和可验证数学奖励，开放了数据与训练系统。

四个关键思路可用一张图理解：

```text
GRPO 基线
├→ Clip-Higher：正向与负向裁剪边界解耦，给低概率好样本更多上升空间
├→ Dynamic Sampling：过滤整组全对/全错、无学习信号的问题并补采样
├→ Token-level Loss：调整长短回答在损失中的权重方式
└→ Overlong Reward Shaping：对过长回答给予平滑惩罚而非突然截断
```

DAPO 不是“完全不同于 GRPO 的新世界”，而是一组为稳定、有效扩展推理 RL 设计的算法与系统改进。

## 9. GSPO：从 Token ratio 转向序列 ratio

GSPO 全称 **Group Sequence Policy Optimization**。其核心动机是：奖励通常评价整条回答，策略更新也可在序列层面衡量新旧策略变化。

```text
GRPO/PPO 常见：每个 Token 一个 importance ratio
r₁, r₂, r₃, ...

GSPO：把整条序列的似然变化汇总为 sequence-level ratio
r_sequence → 序列级裁剪与优化
```

论文报告其对训练效率、稳定性以及 MoE RL 训练有益。由于它是较新的方法，使用时应以具体实现版本为准，不把论文结果无条件外推到所有模型和任务。

## 10. 五种方法统一比较

| 方法 | 数据来源 | 是否在线采样 | Reward Model | Critic | 基准/Advantage | 主要特点 |
|---|---|---:|---:|---:|---|---|
| PPO | 当前策略轨迹 | 是 | 常需要 | 需要 | Value/GAE | 通用但系统复杂 |
| DPO | chosen/rejected | 否（经典形式） | 不需要 | 不需要 | Reference 隐式偏好 | 简单稳定 |
| GRPO | 每题多条回答 | 是 | 可用验证器 | 不需要 | 组内奖励 | 适合可验证推理 |
| DAPO | 动态筛选的组回答 | 是 | 可用验证器 | 不需要 | 改进组内目标 | 强调稳定和有效样本 |
| GSPO | 每题多条序列 | 是 | 可用验证器 | 不需要 | 序列级相对优势 | 序列级 ratio/clip |

## 11. PPO、DPO、GRPO 怎么选？

```text
已有固定 chosen/rejected，预算有限？
  └→ DPO 起步

需要在线探索，奖励来自人类偏好模型？
  └→ PPO 或其他在线 RL

数学/代码答案可自动验证，可每题采样多次？
  └→ GRPO/DAPO/GSPO 类
```

算法名不是最先决定成败的因素。Base 模型、Prompt 难度分布、采样多样性、验证器可靠性和训练监控往往同样重要。

## 12. 奖励设计与作弊

```text
奖励只检查最终数字
→ 模型可能碰巧猜中，无可靠过程

奖励偏爱长回答
→ 模型无限拉长推理

代码测试不完整
→ 模型写 hard-code 通过样例
```

应设置保留测试、长度与格式监控、人工抽检、训练外验证器，并分别记录 reward、真实准确率和平均长度。

## 13. 一条 RL 训练监控面板

```text
任务准确率        应提高
平均 Reward       应提高，但需和真实指标一致
KL(policy||ref)   不应失控
Clip fraction     反映多少更新被裁剪
响应长度          防止奖励诱导膨胀
Entropy           过低可能探索坍缩
每组奖励方差      过低表示无学习信号
训练/保留集差距   检查过拟合与污染
```

## 14. 参考资料

1. Schulman et al., *Proximal Policy Optimization Algorithms*：<https://arxiv.org/abs/1707.06347>
2. Ouyang et al., *InstructGPT/RLHF*：<https://arxiv.org/abs/2203.02155>
3. Rafailov et al., *DPO*：<https://arxiv.org/abs/2305.18290>
4. Shao et al., *DeepSeekMath/GRPO*：<https://arxiv.org/abs/2402.03300>
5. Yu et al., *DAPO*：<https://arxiv.org/abs/2503.14476>
6. Zheng et al., *GSPO*：<https://arxiv.org/abs/2507.18071>
7. DeepSeek-AI, *DeepSeek-R1*：<https://arxiv.org/abs/2501.12948>
