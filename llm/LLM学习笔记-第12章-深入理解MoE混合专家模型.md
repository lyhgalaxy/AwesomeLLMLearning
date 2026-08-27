# LLM 学习笔记：第 12 章——深入理解 MoE 混合专家模型

> 本章只解决一个主题：Mixture of Experts（MoE，混合专家）。
>
> 学习目标：能够解释 Router、Expert、Top-k、总参数、激活参数、负载均衡与 Expert Parallel；能够说明为什么 MoE 计算省了，但部署并不简单。

---

## 1. 先用一句话理解 MoE

普通 Dense 模型让每个 Token 都经过同一套参数：

```text
“猫” ─→ 同一个 FFN
“代码” ─→ 同一个 FFN
“微积分” ─→ 同一个 FFN
```

MoE 准备许多套 FFN，由 Router 为每个 Token 选择其中少数几个：

```text
                  ┌→ Expert 1
Token → Router ───┼→ Expert 2
                  ├→ Expert 3
                  └→ ... Expert N

每个 Token 只进入 Top-k 个 Expert
```

所以 MoE 的核心是：

> 模型拥有很大的总参数容量，但每处理一个 Token，只激活其中一小部分参数。

这叫 **条件计算（Conditional Computation）**：不同输入走不同计算路径。

---

## 2. MoE 通常替换 Transformer 的哪一部分？

标准 Transformer Block：

```text
x
├→ Attention → 残差相加 → h
└───────────────────────

h
├→ FFN → 残差相加 → output
└──────────────────
```

MoE 通常不是把整个 Transformer 复制几十份，而是把某些层中的**单个 FFN**换成多个 Expert FFN：

```text
Dense Block
Attention → 一个 FFN → Output

MoE Block
Attention → Router → Top-k Expert FFN → 加权合并 → Output
```

因此在典型 MoE 中：

- Attention 通常仍是所有 Token 共享的 Dense 参数；
- Embedding、Norm、Attention、LM Head 等通常也共享；
- 主要稀疏的是 FFN/MLP 部分；
- 并非模型的全部参数都只激活 Top-k。

这是理解“激活参数量”的第一个关键。

---

## 3. Expert 到底是什么？

一个 Expert 通常就是一套独立的 FFN 参数。

普通 SwiGLU FFN 可简写为：

```text
Expert_i(x) = W_down_i [SiLU(W_gate_i x) ⊙ (W_up_i x)]
```

每个 Expert 都有自己的：

```text
W_gate_i
W_up_i
W_down_i
```

但结构通常相同，只是权重数值不同。

假设有 8 个 Expert：

```text
Expert 1：[一套 FFN 权重]
Expert 2：[另一套 FFN 权重]
...
Expert 8：[第八套 FFN 权重]
```

“专家”是一个形象名称。训练前没有人规定 Expert 1 必须学数学、Expert 2 必须学中文；它们从随机参数开始，在路由和梯度共同作用下形成不同统计偏好。

---

## 4. Router 是什么？

Router 又叫 Gate，是一个很小的可训练网络。最常见形式是一层线性映射：

```text
router_logits = x W_router
```

若隐藏维度为 `d`、专家数为 `N`：

```text
x：            [d]
W_router：     [d, N]
router_logits：[N]
```

Softmax 后得到当前 Token 对各 Expert 的路由分数：

```text
p = softmax(router_logits)
```

示例：

| Expert | Router 概率 |
|---|---:|
| E1 | 0.05 |
| E2 | 0.60 |
| E3 | 0.10 |
| E4 | 0.25 |

若 `Top-k=2`，选择 E2 和 E4。

```text
x → Router → [E1 .05, E2 .60, E3 .10, E4 .25]
                       │                    │
                       └──── Top-2 ─────────┘
```

Router 的参数相对 Expert FFN 很少，但它决定 Token 的计算路径，训练稳定性非常重要。

---

## 5. Top-k 路由完整计算过程

假设有 4 个 Expert，`Top-k=2`。

### 步骤 1：计算路由分数

```text
x → W_router → logits = [1.0, 3.0, 0.5, 2.0]
```

### 步骤 2：Softmax

```text
probabilities ≈ [0.08, 0.63, 0.05, 0.23]
```

### 步骤 3：选 Top-2

```text
选择 Expert 2 和 Expert 4
```

### 步骤 4：分别计算 Expert 输出

```text
y₂ = Expert₂(x)
y₄ = Expert₄(x)
```

### 步骤 5：按 Gate 权重合并

常见实现会将选中的权重重新归一化：

```text
g₂ = 0.63 / (0.63 + 0.23) ≈ 0.73
g₄ = 0.23 / (0.63 + 0.23) ≈ 0.27

y = 0.73 y₂ + 0.27 y₄
```

总图：

```text
                         ┌→ Expert 2(x) ─×0.73─┐
x → Router → Top-2 ──────┤                     ├→ 相加 → y
                         └→ Expert 4(x) ─×0.27─┘
```

不同 MoE 实现会在归一化、噪声、路由分组和缩放上有所差异，但核心过程一致。

---

## 6. 一个 Token 是只选一次专家吗？

通常不是。**每个 MoE 层都会重新路由。**

```text
Token “Transformer”
  │
MoE Layer 1 → 选 E2, E5
  │ 隐藏状态改变
MoE Layer 2 → 选 E1, E7
  │ 隐藏状态再次改变
MoE Layer 3 → 选 E4, E6
```

同一个 Token 在不同层可以选择不同专家；同一个单词在不同上下文也可能走不同专家：

```text
“苹果发布了手机”中的“苹果” → 路由组合 A
“我吃了一个苹果”中的“苹果” → 路由组合 B
```

Router 读取的是上下文化隐藏状态，而不只是 Token ID。

---

## 7. Top-1 和 Top-2 有什么区别？

### Top-1

```text
一个 Token → 一个 Expert
```

优点：计算和通信较少，Switch Transformer 采用简化的 Top-1 路由思路。缺点是一个专家承担全部当前 FFN 变换，路由错误或负载不均的影响更直接。

### Top-2

```text
一个 Token → 两个 Expert → 加权融合
```

优点：组合能力与平滑性更强；代价是 Expert 计算和通信约增加。

### Top-k 的权衡

```text
k 增大
├── 每个 Token 可组合更多 Expert
├── 激活计算量增加
├── 通信量增加
└── 总参数量不变
```

“专家数 N”与“每 Token 选择数 k”必须同时报告。

---

## 8. 总参数和激活参数到底怎么计算？

把模型参数分成：

```text
P_shared：每个 Token 都使用的共享参数
P_expert：一个 Expert 的参数
N：每层/模型的 Expert 数
k：每 Token 激活的 Expert 数
```

简化关系：

```text
总参数 P_total ≈ P_shared + N × P_expert

每 Token 激活参数 P_active ≈ P_shared + k × P_expert
```

注意：真实模型有很多 MoE 层，且共享 Expert、路由器、MTP 模块等会让精确计算更复杂。

### 一个玩具例子

假设：

```text
共享 Attention 等参数：4B
8 个 Expert，每个 2B
Top-2
```

则：

```text
总参数 = 4B + 8×2B = 20B
激活参数 = 4B + 2×2B = 8B
```

每个 Token 大致使用 8B 参数做计算，但 checkpoint 必须包含 20B 参数。

---

## 9. 为什么 Mixtral 8×7B 不是简单的 56B？

名字容易造成误解：并不是 8 个完整 7B 模型简单拼在一起。

Mixtral 8×7B 的 Attention 等参数共享，主要是每层 FFN 有 8 个 Expert，每 Token 选 2 个。官方报告给出的量级是：

```text
总参数约 47B
每 Token 激活参数约 13B
8 个 FFN Expert，Top-2
```

原因：

```text
总参数 ≠ 8 × 完整 Mistral 7B
因为 Attention、Embedding、Norm 等没有复制 8 份
```

这个例子说明不能只根据模型名称乘法估参数。

---

## 10. DeepSeek-V3 的 671B / 37B 怎么理解？

DeepSeek-V3 技术报告披露：

```text
总参数：671B
每 Token 激活参数：约 37B
架构：DeepSeekMoE + MLA
```

含义：

```text
整个模型仓库/集群里有约 671B 参数
                   │
一个 Token 经过每层时只路由到少数专家
                   │
该 Token 的实际前向路径累计使用约 37B 参数
```

### 不能错误理解为

```text
错误 1：模型文件只有 37B
错误 2：部署只需保存 37B
错误 3：速度一定等同普通 Dense 37B
错误 4：所有 Token 永远使用同一组 37B
```

正确理解：不同 Token 可能选择不同专家，因此整个服务必须能访问全部专家权重。

---

## 11. 为什么 MoE 可以增加容量而不同比例增加 FLOPs？

Dense FFN：

```text
每个 Token 使用唯一大 FFN
总参数增加 → 每 Token 计算通常也增加
```

Sparse MoE：

```text
增加 Expert 数 N：总参数容量增加
保持 Top-k 不变：每 Token 只算 k 个 Expert
```

例如从 8 个专家扩展到 64 个，但始终 Top-2：

```text
总 Expert 参数：约扩大 8 倍
单 Token Expert 计算：仍然是 2 个 Expert
```

这就是 MoE 的主要吸引力：提高“参数容量 / 每 Token 计算量”的比值。

但是 Router、通信、内存访问和负载不均会增加额外开销，因此实际速度不会只由理论 FLOPs 决定。

---

## 12. 容量增加为什么可能提升模型能力？

一个 Dense FFN 必须用同一组参数吸收所有输入模式：

```text
同一 FFN
├── 中文
├── 英文
├── 代码
├── 数学
└── 对话
```

MoE 提供更多参数子空间：

```text
Router 根据隐藏状态组合 Expert
├── 某些 Expert 更常处理代码模式
├── 某些 Expert 更常处理特定语言模式
├── 某些 Expert 更常处理符号/推理模式
└── 许多 Expert 学到难以人工命名的特征
```

这是一种直觉，不应过度拟人化。Expert 的真实分工通常是分布式、重叠和层相关的，并不一定能贴上明确领域标签。

---

## 13. Router 是怎样训练的？

Router 与整个语言模型一起端到端训练。

```text
Token 隐藏状态
  ↓ Router 选择 Expert
Expert 输出
  ↓ 后续层
Next-token Loss
  ↓ 反向传播
更新 Expert 参数 + Router 参数 + 共享参数
```

如果某条路由组合有助于降低语言模型 Loss，梯度会推动 Router 更倾向相应选择。

但 Top-k 是离散选择，只有被选专家通常获得该 Token 的主要梯度。实现会借助选中 Gate 权重、辅助 Loss、噪声或其他策略维持可训练性。

---

## 14. 最大难题：所有 Token 都挤向少数专家

假设 4 个 Expert：

```text
理想负载：
E1 █████ 25%
E2 █████ 25%
E3 █████ 25%
E4 █████ 25%

路由坍缩：
E1 ███████████████ 75%
E2 ███             15%
E3 ██              10%
E4                  0%
```

后果：

- 热门专家计算溢出或排队；
- 冷门专家几乎得不到训练；
- GPU 负载不均，其他设备等待；
- 总容量名义很大，实际只用少数专家；
- Token 可能被丢弃或转移。

这叫负载均衡问题。

---

## 15. Capacity Factor 与 Token Dropping

训练时通常为每个 Expert 规定 Batch 内最多处理多少 Token。

若 Batch 中有 `T` 个 Token、`N` 个 Expert、Top-1，理想每个专家约：

```text
T / N 个 Token
```

容量可设为：

```text
Expert Capacity = capacity_factor × T / N
```

例如：

```text
T = 1024
N = 8
capacity_factor = 1.25

capacity = 1.25 × 1024/8 = 160 Token/Expert
```

若 E1 收到 220 个 Token：

```text
160 个正常处理
60 个如何处理？
├── Drop：跳过 Expert 路径
├── Reroute：改送其他专家
├── 增大 Capacity：更多显存/计算
└── 使用无丢弃的动态实现
```

不同系统策略不同。Token Dropping 可能损伤质量；过大 Capacity 又浪费计算。

---

## 16. 辅助负载均衡 Loss

经典方法在语言模型 Loss 外增加负载均衡辅助项：

```text
L_total = L_language + λ L_balance
```

它希望：

- 各 Expert 被选频率较均匀；
- Router 概率质量也较均匀；
- 避免少数 Expert 独占 Token。

直觉：

```text
语言模型 Loss：选最有助于预测的专家
Balance Loss：不要所有 Token 都选同一个专家
```

两者会冲突。`λ` 太小压不住坍缩；太大会迫使不合适的均匀路由，损害模型主任务。

---

## 17. DeepSeek-V3 的辅助 Loss-Free 负载均衡

DeepSeek-V3 报告提出辅助损失自由的负载均衡策略：不直接用较强辅助 Loss 干扰主模型目标，而是为 Expert 的路由分数引入动态偏置。

直觉示意：

```text
Expert 太拥挤 → 路由 bias 略降低
Expert 太空闲 → 路由 bias 略提高

原始 affinity score + 动态 balance bias → 选择 Top-k
```

这样可以调节选择频率，同时减少负载均衡辅助 Loss 对主任务梯度的干扰。

“auxiliary-loss-free”不表示完全没有任何负载控制或监控；它表示核心均衡机制不依赖传统的强辅助损失形式。具体实现仍应以技术报告和代码为准。

---

## 18. Shared Expert 为什么出现？

传统 Routed Experts 可能重复学习所有 Token 都需要的通用知识：

```text
E1 学一份通用语法
E2 又学一份通用语法
E3 再学一份通用语法
→ 参数冗余
```

DeepSeekMoE 引入共享 Expert：

```text
Token
├→ Shared Experts：每个 Token 都经过，学习公共知识
└→ Routed Experts：Router 选择少数，学习差异化知识
              │
              ▼
           合并输出
```

目标：

- Shared Experts 吸收公共模式；
- Routed Experts 更专注于差异化特征；
- 减少多个路由 Expert 重复学习通用知识。

激活参数计算也要把 Shared Experts 算进去。

---

## 19. Fine-Grained Expert Segmentation 是什么？

假设传统设计有 8 个大 Expert，每 Token 选 2 个：

```text
[大E1][大E2][大E3]...[大E8]
Top-2 组合数量有限
```

DeepSeekMoE 将大 Expert 切成更多小 Expert，同时选择更多个小 Expert，使总激活计算相近：

```text
[小E1][小E2][小E3]...[小E32]
例如 Top-8 小 Expert
```

直觉收益：

```text
少量大模块组合 → 更细粒度的小模块组合
组合方式更多 → 更灵活地拼接能力
```

如果一个大 Expert 的 FFN 宽度是小 Expert 的 4 倍：

```text
传统 Top-2 大 Expert ≈ 激活 2×大宽度
细粒度 Top-8 小 Expert ≈ 激活 8×(大宽度/4)
                     ≈ 相近计算量
```

代价是路由、调度和通信更复杂。

---

## 20. Shared Expert 与 Routed Expert 的完整图

```text
                         ┌→ Routed E1
                         ├→ Routed E2
x → Router → Top-k ──────┼→ Routed E3 ──┐
│                        └→ ...          │
│                                        ├→ 加权/相加 → MoE Output
├────────→ Shared Expert 1 ──────────────┤
└────────→ Shared Expert 2 ──────────────┘
```

每个 Token：

- 必定经过 Shared Experts；
- 只经过 Top-k Routed Experts；
- 下一层会重新路由。

---

## 21. MoE 训练时为什么需要 All-to-All 通信？

多个 GPU 各自存不同 Expert：

```text
GPU0：E0, E1
GPU1：E2, E3
GPU2：E4, E5
GPU3：E6, E7
```

GPU0 当前 Batch 的 Token 可能被路由到 E3、E6：

```text
各 GPU 上的 Token
  │ Router 决定目标 Expert
  ▼
All-to-All：把 Token 隐藏状态发送到持有该 Expert 的 GPU
  │
  ▼
各 Expert 本地计算
  │
  ▼
All-to-All：把 Expert 输出发回原 Token 所在 GPU
```

通信图：

```text
GPU0 Tokens ─┬→ GPU1 Experts
             └→ GPU3 Experts
GPU1 Tokens ─┬→ GPU0 Experts
             └→ GPU2 Experts
...
```

Dense Tensor Parallel 常大量使用 All-Reduce；Expert Parallel 的典型关键通信是 All-to-All。

如果网络慢、Token 分布不均或消息太碎，通信时间可抵消稀疏计算收益。

---

## 22. Expert Parallel（EP）是什么？

EP 把不同专家分布在不同设备：

```text
EP size = 4
Rank 0：部分 Experts
Rank 1：部分 Experts
Rank 2：部分 Experts
Rank 3：部分 Experts
```

这降低单 GPU 的 Expert 权重存储，但一次 MoE 层需要跨设备 Dispatch 和 Combine。

常与其他并行组合：

```text
DP：不同数据副本
TP：共享 Attention/单 Expert 内矩阵切分
EP：不同 Expert 分布
PP：不同层分布
```

例如 64 GPU 集群可能用：

```text
TP=4 × EP=8 × DP=2 = 64
```

实际组合受模型架构、拓扑和框架约束。

---

## 23. 为什么部署时仍要存全部专家？

考虑一批请求：

```text
Token A → E1, E4
Token B → E3, E8
Token C → E2, E7
Token D → E5, E6
```

虽然每个 Token 只用 2 个专家，一整个服务中的不同 Token 可能用遍所有专家。

```text
单 Token 激活少数 Expert
≠
系统永远只需要那几个 Expert
```

如果专家不在 GPU：

- 从 CPU/SSD 动态加载会有巨大延迟；
- 缓存热门 Expert 仍可能遇到冷门 Expert；
- Router 的选择依上下文动态变化，难提前完全预测。

因此大 MoE 通常把所有专家分布在多 GPU/节点上，或采用量化/Offload 等折中。

---

## 24. 为什么激活 37B 也不一定等同 Dense 37B 速度？

Dense 37B：

```text
规则的大矩阵乘
→ Kernel 容易高效
→ 通信模式相对规律
```

MoE 激活 37B：

```text
Router
→ Token 分组/排序
→ All-to-All Dispatch
→ 多个大小不一的 Expert GEMM
→ All-to-All Combine
→ 负载不均和 Padding
```

理论 FLOPs 相近，端到端时间仍可能不同。MoE 的效率依赖：

- Batch/并发是否足够大；
- 每个 Expert 能否形成足够大的 GEMM；
- GPU 间网络；
- Expert 是否均衡；
- Fused MoE Kernel；
- Expert Parallel 布局；
- 量化支持。

小 Batch、单用户本地推理时，MoE 不一定充分发挥吞吐优势。

---

## 25. 一个具体 Batch 的路由例子

假设有 6 个 Token、4 个 Expert、Top-2：

| Token | 第一专家 | 第二专家 |
|---|---|---|
| 我 | E1 | E3 |
| 喜欢 | E1 | E2 |
| 写 | E2 | E4 |
| Python | E4 | E2 |
| 代码 | E4 | E1 |
| 。 | E3 | E1 |

按 Expert 重排：

```text
E1 收到：我、喜欢、代码、。
E2 收到：喜欢、写、Python
E3 收到：我、。
E4 收到：写、Python、代码
```

计算过程：

```text
原 Token 顺序
  ↓ Dispatch / Permute
按 Expert 分组做 Batched GEMM
  ↓ Combine / Unpermute
恢复原 Token 顺序，并按 Gate 权重相加
```

这解释了为什么 MoE Kernel 需要高效的 Token Permutation、Grouped GEMM 和通信融合。

---

## 26. Training 时专家怎样形成分工？

一个简化的正反馈过程：

```text
某 Expert 偶然更擅长某类隐藏模式
  ↓
Router 更常把类似 Token 送给它
  ↓
它获得更多这类 Token 的梯度
  ↓
对这类模式更擅长
```

这可能形成 specialization，也可能导致路由坍缩。负载均衡、Router 噪声、数据多样性和共享 Expert 用来控制这两面。

Expert 分工并非一定对应“数学/英语/代码”这种人类类别。研究中常见的是：

- 某些专家偏好标点、数字或特定语言；
- 某些专家在某层处理局部结构，在另一层处理更抽象模式；
- 多个 Expert 功能重叠；
- 同一 Expert 的功能难以用单标签概括。

---

## 27. Router Z-loss、噪声与稳定性

路由训练还可能使用：

### Router Z-loss

抑制 Router logits 绝对值无限增大，避免 Softmax 过度饱和，提高数值稳定性。

```text
logits 太极端 → 概率近乎 one-hot → 梯度与低精度稳定性变差
```

### Noisy Gating

训练时给路由分数加入噪声，鼓励探索不同 Expert，减少过早固定。

### Router 精度

即使模型主体用 BF16/FP8，Router 的某些计算可能保留更高精度，以防小数值差异导致错误 Top-k。

这些并非所有模型都使用同样方案。

---

## 28. MoE 的训练 Loss 是什么？

核心语言模型 Loss 并没有因为 MoE 改变：

```text
L_LM = 下一 Token 的交叉熵
```

可能再加：

```text
L_total = L_LM
        + λ_balance L_balance
        + λ_z L_router_z
        + 其他正则项
```

所以 MoE 是**模型架构和条件计算方式**，不是一种像 DPO/GRPO 那样的新训练目标。

Pre-training、SFT、DPO、RL 都可以应用于 MoE，但路由与专家并行会让训练系统更复杂。

---

## 29. SFT 和 RL 阶段 MoE 有什么额外问题？

### SFT

SFT 数据通常比预训练少，Expert 可能：

- 路由分布发生漂移；
- 少数专家过拟合；
- 小 Batch 导致每 Expert Token 太少；
- 全参数更新的通信成本高。

LoRA 是否加到 Router、Attention、所有 Expert 或部分 Expert，会影响参数量和效果。

### RL

在线 RL 的生成 Batch、序列长度和奖励筛选不断变化：

- Expert 负载更不稳定；
- Token-level ratio 与专家路由变化相互作用；
- 训练 Policy 与采样 Engine 的路由/精度一致性重要；
- 大 MoE 权重同步开销高。

GSPO 论文特别报告序列级优化对 MoE RL 稳定性有帮助，但应根据具体实现验证。

---

## 30. Dense 与 MoE 的全面比较

| 维度 | Dense | Sparse MoE |
|---|---|---|
| 每 Token 参数 | 使用全部层参数 | 使用共享参数 + 少数专家 |
| 总参数容量 | 与每 Token 计算紧密相关 | 可大幅高于激活计算 |
| 计算模式 | 规则大 GEMM | Router + Dispatch + Grouped GEMM |
| 通信 | TP 常用 All-Reduce | EP 常用 All-to-All |
| 负载均衡 | 不涉及专家 | 核心难题 |
| 单机部署 | 相对简单 | 仍需存全部 Expert |
| 小 Batch 效率 | 通常较稳定 | Expert GEMM 可能太小 |
| 大规模训练 | 架构简单但计算贵 | 计算经济但系统复杂 |
| 量化生态 | 通常更成熟 | 需支持 Router/Expert/Fused MoE |

---

## 31. MoE 与多头注意力不是一回事

二者都出现“多个”，但作用不同：

```text
Multi-Head Attention
├── 每个 Token 通常计算所有 Attention Heads
├── 多个头看不同表示子空间
└── 重点：Token 间交换信息

Mixture of Experts
├── 每个 Token 只选择少数 Experts
├── 多套 FFN 参数提供条件计算
└── 重点：每个 Token 做不同的非线性变换
```

GQA 减少的是 K/V Head；MoE 稀疏的是 Expert FFN。两者可以同时存在。

---

## 32. MoE 与 Ensemble 也不是一回事

传统 Ensemble：

```text
完整模型 A ─┐
完整模型 B ─┼→ 合并最终预测
完整模型 C ─┘
```

MoE：

```text
一个模型内部的某些层
Token → Router → 少数 FFN Experts → 继续后续层
```

MoE 通常端到端联合训练，Expert 是模型内部组件，不是多个独立完整模型。

---

## 33. MoE 与 RAG 不是一回事

```text
MoE：从模型内部参数中选择计算路径
RAG：从外部文档库检索信息放进上下文
```

MoE 的 Expert 通常不是可独立更新的知识数据库，也不会自动给出来源；RAG 可以更新文档并提供证据。二者可以组合。

---

## 34. MoE 与 LoRA Adapter 路由的区别

```text
MoE Expert：通常是预训练架构中的大 FFN 模块，每层动态按 Token 路由
LoRA Adapter：小型低秩参数，通常按任务/租户静态选择，也可设计动态路由
```

二者都可称“专家化”，但参数规模、训练阶段和路由粒度不同。

---

## 35. MoE 推理的显存粗算

假设总参数 100B：

```text
BF16 权重理论下限：100B × 2 bytes ≈ 200 GB
INT8：约 100 GB + 元数据
INT4：约 50 GB + 元数据
```

即使激活参数只有 20B，BF16 也不能按：

```text
20B × 2 = 40GB   ← 用它估计全部权重显存是错的
```

40GB 只是激活权重的字节量级直觉，实际系统仍需保存 200GB 总 BF16 权重，并额外容纳 KV Cache、激活、通信 Buffer 与框架开销。

---

## 36. MoE 推理的延迟与吞吐

### 低并发

```text
每步 Token 少
→ 每个 Expert 分到的 Token 更少
→ GEMM 形状小
→ GPU 利用率低
→ Router/通信开销占比高
```

### 高并发

```text
许多请求连续批处理
→ 各 Expert 收到更多 Token
→ Grouped GEMM 更大
→ 更能摊薄路由/通信开销
```

因此 MoE 可能在服务吞吐上表现很好，但单请求 latency 未必按激活参数等比例降低。

---

## 37. MoE 量化为什么更复杂？

不同 Expert 的权重和激活分布可能不同：

```text
E1：高频、激活范围 A
E2：低频、激活范围 B
E3：含离群通道
...
```

量化需考虑：

- 每 Expert 是否独立 Calibration；
- Router 保持什么精度；
- Grouped GEMM 是否支持量化格式；
- Expert Parallel 通信前后数据类型；
- 热门/冷门 Expert 的校准样本是否足够；
- 量化是否改变 Top-k 选择。

省下总权重显存对大 MoE 很重要，但必须配套高效 Fused MoE Kernel，才可能转化为速度收益。

---

## 38. MoE 模型适合哪些情况？

适合：

- 有大规模训练和部署集群；
- 希望增加参数容量，但控制每 Token FLOPs；
- 数据覆盖多语言、多领域、多种模式；
- 推理服务并发较高，可形成大 Expert Batch；
- 有高速互联和成熟 MoE Kernel/并行框架。

不一定适合：

- 单 GPU/低内存本地部署；
- 极低延迟、小 Batch；
- 集群网络较慢；
- 团队缺少分布式调试能力；
- 总权重存储比计算预算更受限。

---

## 39. 从模型配置文件读 MoE

常见字段名称因模型而异：

```text
num_experts / n_routed_experts：路由专家总数
num_experts_per_tok / top_k：每 Token 选择几个
n_shared_experts：共享专家数
moe_intermediate_size：单 Expert FFN 中间维度
first_k_dense_replace：前若干层仍用 Dense FFN
moe_layer_freq：每隔多少层使用 MoE
router_aux_loss_coef：负载均衡 Loss 系数
```

不要只读 `num_experts`。还需确认：

```text
哪些层是 MoE？
每层有几个 Expert？
每 Token 选几个？
是否有 Shared Expert？
单 Expert 多宽？
Attention/Embedding 多大？
总参数和激活参数的统计口径是什么？
```

---

## 40. 一个最小 MoE 伪代码

下面为了说明逻辑，未实现高效的 Token Dispatch：

```python
class SimpleMoE(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(hidden_size) for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x):              # x: [tokens, hidden]
        router_logits = self.router(x) # [tokens, experts]
        probs = router_logits.softmax(dim=-1)

        top_weights, top_ids = probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        for token_idx in range(x.shape[0]):
            for slot in range(self.top_k):
                expert_id = top_ids[token_idx, slot]
                weight = top_weights[token_idx, slot]
                output[token_idx] += weight * self.experts[expert_id](x[token_idx])

        return output
```

真实实现会：

```text
按 Expert 对 Token 排序/Dispatch
→ 使用 Grouped GEMM 并行计算
→ Combine 恢复 Token 顺序
→ 多机时加入 All-to-All
```

逐 Token Python 循环会非常慢，只适合教学。

---

## 41. 手算练习

给定：

```text
Shared 参数 = 6B
Routed Experts = 16 个
每个 Expert = 1.5B
Top-k = 2
```

### 总参数

```text
P_total = 6B + 16×1.5B = 30B
```

### 激活参数

```text
P_active = 6B + 2×1.5B = 9B
```

### BF16 权重理论下限

```text
30B × 2 bytes = 60 GB
```

不是 `9B×2=18GB`，因为部署仍需保存全部 16 个 Expert。

若改为 Top-4：

```text
总参数仍是 30B
激活参数变为 6B + 4×1.5B = 12B
```

---

## 42. DeepSeek-V3 路由的一张整体图

这是帮助理解的简化图，不代表全部实现细节：

```text
Token Hidden State
       │
       ├────────────────────────→ Shared Experts ────────┐
       │                                                 │
       ▼                                                 │
Router Affinity Scores                                   │
       │ + Dynamic Balance Bias                          │
       ▼                                                 │
Grouped / Top-k Routed Experts                           │
       │                                                 │
       ├→ Fine-grained Expert A ─┐                       │
       ├→ Fine-grained Expert F ─┤                       │
       ├→ Fine-grained Expert M ─┼→ Weighted Combine ────┤
       └→ ...                   ──┘                       │
                                                         ▼
                                                   MoE Layer Output
```

理解时分开记：

- MLA：主要优化 Attention/KV 表示；
- DeepSeekMoE：主要优化 FFN Expert 组织和路由；
- MTP：辅助训练目标；
- 671B/37B：总参数与每 Token 激活参数量级。

它们不是同一个技术。

---

## 43. 最容易犯的十个错误

1. **“8 个专家就是 8 个完整模型”**：通常错误，常见 MoE 主要复制 FFN。
2. **“每 Token 只选 2 个专家，所以只需存 2 个”**：错误，不同 Token 会选不同专家。
3. **“激活 37B 就等于 Dense 37B 的延迟和显存”**：错误。
4. **“Expert 都有清晰的人类职业分工”**：不一定。
5. **“MoE 是多头注意力”**：错误，MoE 常用于 FFN。
6. **“MoE 改变了交叉熵训练目标”**：错误，它主要改变架构和计算路径。
7. **“总 Expert 越多越好”**：路由、存储、通信和数据会成为瓶颈。
8. **“Top-k 越大越好”**：计算和通信也随之增加。
9. **“理论 FLOPs 低就一定端到端快”**：忽视 All-to-All 和小 GEMM。
10. **“负载完全平均就是最佳路由”**：过度均匀可能妨碍真正的专家化。

---

## 44. 一张最终知识地图

```text
Sparse MoE
├── 结构
│   ├── Shared Dense Components
│   ├── Router / Gate
│   ├── Routed Experts
│   └── Shared Experts（部分架构）
├── 路由
│   ├── logits → softmax
│   ├── Top-1 / Top-2 / Top-k
│   └── weighted combine
├── 参数
│   ├── Total Parameters：全部专家都算
│   └── Active Parameters：共享 + 每 Token Top-k
├── 训练难题
│   ├── Router Collapse
│   ├── Load Balancing
│   ├── Capacity / Token Dropping
│   └── 数值稳定
├── 系统难题
│   ├── Token Dispatch / Combine
│   ├── Grouped GEMM
│   ├── Expert Parallel
│   └── All-to-All Communication
└── 收益与代价
    ├── 更高参数容量 / 每 Token FLOPs
    └── 更高存储、通信、调度与部署复杂度
```

---

## 45. 复习题

1. MoE 通常替换 Transformer 的 Attention 还是 FFN？
2. Router 的输入和输出分别是什么？
3. Top-2 如何将两个 Expert 的输出合并？
4. 同一个 Token 在不同 MoE 层是否一定选择同一 Expert？
5. 总参数和激活参数的简化公式是什么？
6. 为什么 Mixtral 8×7B 的总参数不是 56B？
7. DeepSeek-V3 的 671B/37B 分别表示什么？
8. 为什么激活 37B 仍需存储全部 671B 参数？
9. 负载不均会造成哪些训练和系统问题？
10. Capacity Factor 控制什么？
11. Shared Experts 为什么能减少 Routed Experts 的冗余？
12. Fine-grained Expert Segmentation 的目的是什么？
13. Expert Parallel 为什么需要 All-to-All？
14. 为什么 MoE 在低并发时未必比 Dense 高效？
15. MoE、MHA、Ensemble、RAG 有什么区别？

---

## 46. 参考资料

1. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*：<https://arxiv.org/abs/1701.06538>
2. Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*：<https://arxiv.org/abs/2006.16668>
3. Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*：<https://arxiv.org/abs/2101.03961>
4. Jiang et al., *Mixtral of Experts*：<https://arxiv.org/abs/2401.04088>
5. Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*：<https://arxiv.org/abs/2401.06066>
6. DeepSeek-AI, *DeepSeek-V3 Technical Report*：<https://arxiv.org/abs/2412.19437>
7. Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models*：<https://arxiv.org/abs/2202.08906>
