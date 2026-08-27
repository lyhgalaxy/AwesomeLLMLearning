# LLM 学习笔记：第 11 章——推理加速、模型量化与服务部署

> 学习目标：理解一次请求从 API 到 GPU 的完整路径；知道 vLLM、SGLang、DeepSpeed、FlashAttention 和模型量化分别解决哪一层问题；能为具体负载设计和验证部署方案。
>
> 软件能力更新很快。本章原理具有延续性，命令参数应以实际安装版本的官方文档为准。资料核对日期：2026-08-27。

---

## 1. 一次 LLM 请求经历了什么？

```text
客户端
  │ HTTP / gRPC / OpenAI-compatible API
  ▼
网关：鉴权、限流、配额、超时
  │
  ▼
Tokenizer / Chat Template
  │ 文本 → Token IDs
  ▼
请求调度器
  ├── Continuous Batching
  ├── Prefill/Decode 调度
  ├── Prefix Cache
  └── 抢占、优先级、公平性
  │
  ▼
模型执行器
  ├── FlashAttention/FlashInfer 等内核
  ├── GEMM/MoE 内核
  ├── Tensor/Expert/Pipeline Parallel
  └── 量化计算
  │
  ▼
KV Cache 管理器
  │
  ▼
逐 Token 流式返回 → Detokenizer → 客户端
```

“部署一个模型”不仅是调用 `model.generate()`。真正的在线系统需要同时优化计算、显存、调度、网络和可靠性。

---

## 2. 先定义性能指标

### 2.1 TTFT：首 Token 延迟

Time To First Token，从请求到第一个输出 Token 的时间：

```text
TTFT ≈ 排队 + Tokenize + Prefill + 第一次 Decode + 网络
```

长 Prompt、排队和 Prefill 计算通常显著影响 TTFT。

### 2.2 TPOT：每输出 Token 时间

Time Per Output Token，首 Token 后相邻 Token 的平均生成时间。用户感受到的生成速度约为：

```text
输出 tokens/s ≈ 1 / TPOT
```

### 2.3 ITL、E2E 与吞吐

- ITL：Inter-Token Latency，相邻 Token 的实际延迟分布；
- E2E Latency：整个回答完成耗时；
- Throughput：系统每秒完成的请求数或处理的 Token 数；
- Goodput：满足 SLO 的有效吞吐，而不是不顾延迟塞入的最大吞吐；
- P50/P95/P99：不能只看平均延迟，尾延迟决定高峰体验。

### 2.4 延迟与吞吐通常冲突

```text
Batch 增大
├── GPU 利用率 ↑
├── 总吞吐 ↑
└── 单请求排队/尾延迟可能 ↑
```

因此不存在脱离业务目标的“最快引擎”。聊天强调低 TTFT/TPOT，离线批处理强调总 Token/s，Agent 可能更依赖 Prefix Cache 和结构化输出。

---

## 3. Prefill 与 Decode 是两种不同工作负载

### 3.1 Prefill：一次读完整个 Prompt

```text
Prompt：x₁ x₂ x₃ ... xₙ
           │ 一次前向，可在 Token 维并行
           ▼
生成每层所有 Prompt Token 的 K/V Cache
```

Prefill 通常矩阵较大、并行度高，更偏 **compute-bound**。长 Prompt 会增加 Attention 计算和 TTFT。

### 3.2 Decode：每次只生成一个新 Token

```text
第 n+1 步：读取历史 KV → 生成 y₁
第 n+2 步：读取更长 KV → 生成 y₂
第 n+3 步：读取更长 KV → 生成 y₃
```

每步矩阵的 Token 维很小，却要读取大模型权重和 KV Cache，常偏 **memory-bandwidth-bound**。

```text
Prefill：大块并行计算 → 影响 TTFT
Decode：许多小步串行   → 影响 TPOT
```

这解释了为什么一种优化可能提高 Prefill，却不明显改善 Decode；部署基准必须分别报告。

---

## 4. 显存都被谁占用了？

推理显存粗略组成：

```text
总显存
├── 模型权重
├── KV Cache
├── 激活与临时 Workspace
├── CUDA Graph / 编译缓存
├── 通信 Buffer
└── 框架与显存碎片
```

### 4.1 权重占用

理论下限：

```text
FP32：参数数 × 4 bytes
BF16/FP16：参数数 × 2 bytes
INT8：参数数 × 1 byte
INT4：参数数 × 0.5 byte
```

例如 7B BF16 权重约：

```text
7×10⁹ × 2 bytes ≈ 14 GB
```

实际还有 scale、zero point、对齐、未量化层和运行时开销。

### 4.2 KV Cache 占用

对常规 Attention，元素量近似：

```text
2 × layers × concurrent_tokens × kv_heads × head_dim
↑ K、V
```

再乘数据类型字节数。它与并发请求的“当前总 Token 数”近似线性增长。

```text
上下文翻倍 → 单请求 KV 约翻倍
并发翻倍   → 总 KV 约翻倍
MHA→GQA    → kv_heads 减少，KV 显著下降
```

量化权重只解决权重显存，不自动解决长上下文 KV Cache。

---

## 5. FlashAttention：相同数学结果，更少显存搬运

### 5.1 标准实现的问题

普通 Attention 概念上会产生 `S×S` 分数矩阵：

```text
QKᵀ → [S,S] Scores → Softmax → [S,S] Weights → ×V
```

GPU 不只有“计算速度”，还有存储层次：

```text
HBM/显存：容量大，访问相对慢
   ⇅ 数据搬运常是瓶颈
SRAM/片上存储：容量小，访问快
```

传统实现反复把大型中间矩阵写入和读出 HBM。

### 5.2 FlashAttention 的方法

FlashAttention 对 Q/K/V 分块，在片上 SRAM 中计算局部块，并用数值稳定的在线 Softmax 合并结果：

```text
Q block ─┐
K block ─┼→ SRAM 中计算局部 Attention → 在线累积 Output
V block ─┘
                 │
                 └→ 不完整写出整个 S×S 矩阵
```

它的关键是 **IO-aware**：减少 HBM 读写，而不是把标准 Attention 换成近似公式。

### 5.3 它改变和不改变什么？

| 项目 | FlashAttention 的影响 |
|---|---|
| Attention 数学语义 | 保持精确，允许浮点舍入级差异 |
| 标准 Attention FLOPs 数量级 | 仍近似 `O(S²d)` |
| 中间显存 | 显著下降，不物化完整注意力矩阵 |
| HBM IO | 显著减少 |
| 训练长序列 | 常明显受益 |
| 整体端到端速度 | 取决于 Attention 在总耗时中的比例 |

所以“FlashAttention 把 Attention 复杂度从平方变线性”是错误说法。它主要优化内存 IO 与中间存储。

### 5.4 FlashAttention 与 vLLM 的关系

```text
FlashAttention：单个/一类 Attention 算子如何高效计算
vLLM/SGLang：许多请求如何排队、批处理、管理 KV 并调用算子
```

推理引擎可以选择 FlashAttention、FlashInfer、Triton 等后端，两者不是互相替代的同类产品。

---

## 6. KV Cache：为什么生成时不重新计算整段历史？

若没有 Cache，生成每个 Token 都会重新计算历史所有 Token 在每层的 K/V：

```text
第1步：算 Prompt
第2步：重算 Prompt + y₁
第3步：重算 Prompt + y₁ + y₂
```

使用 KV Cache：

```text
Prefill：保存 Prompt 的 K/V
Decode 1：只计算 y₁ 的新 K/V，读取历史 Cache
Decode 2：只计算 y₂ 的新 K/V，读取历史 Cache
```

代价是显存随上下文和并发增长。LLM 服务系统的核心问题之一就是：怎样紧凑地分配、复用和回收 KV Cache。

---

## 7. vLLM 与 PagedAttention

### 7.1 连续显存分配的问题

若提前为每个请求按最大长度预留连续 KV 区域：

```text
请求 A：[已用已用已用 | 预留预留预留预留]
请求 B：[已用 | 预留预留预留预留预留预留]
请求 C：[已用已用 | 预留预留预留]
```

实际输出长度未知，会造成内部浪费；请求结束和新请求加入又会产生碎片。

### 7.2 PagedAttention 的思想

借鉴操作系统虚拟内存分页，把一个请求的 KV 切成固定大小 Block：

```text
请求 A 的逻辑 Block： A1 → A2 → A3
                         │    │    │
Block Table              ▼    ▼    ▼
物理 KV 池： [B2][A1][C1][A3][A2][空闲]...
```

逻辑上连续的上下文不必存放在物理连续显存中。需要新空间时按 Block 分配，结束后归还。

收益：

- 减少预留浪费和显存碎片；
- 提高可容纳的并发 Token 数；
- 支持共享 Prefix Block；
- 配合调度器实现高吞吐服务。

### 7.3 vLLM 不只有 PagedAttention

官方文档当前列出的能力包括：

```text
vLLM
├── Paged KV Cache
├── Continuous Batching
├── Prefix Caching
├── Chunked Prefill
├── Speculative Decoding
├── CUDA/HIP Graph 与编译优化
├── 多种量化与 Attention/GEMM 内核
├── Tensor/Pipeline/Data/Expert/Context Parallel
├── Multi-LoRA
└── OpenAI-compatible API
```

具体模型、硬件、量化组合是否支持，要查当前版本兼容矩阵。

---

## 8. Continuous Batching：批次不再等最慢请求结束

### 8.1 静态批处理

```text
Batch 1：A A A A A 完成
         B B B 完成 [空闲][空闲]
         C C C C C 完成
等整个 Batch 完成后才能加入 D
```

不同回答长度导致短请求完成后留下空位。

### 8.2 连续批处理

每个 Decode step 后重新调度：

```text
Step 1：[A B C]
Step 2：[A B C]  B完成
Step 3：[A D C]  立即加入 D
Step 4：[A D E]  C完成，加入 E
```

它提高 GPU 利用率与吞吐，但调度策略必须兼顾：

- 新请求 TTFT；
- 老请求 TPOT；
- 长 Prompt 与短 Decode 的竞争；
- 优先级、公平性和抢占；
- KV Cache 是否足够。

---

## 9. Chunked Prefill：防止长 Prompt 阻塞 Decode

若一个超长 Prompt 一次完成 Prefill，已有用户的 Decode 可能等待很久：

```text
时间 →
传统：[超长 Prefill================] [Decode]

分块：[Prefill块] [Decode] [Prefill块] [Decode] ...
```

Chunked Prefill 将长 Prompt 分块并和 Decode Token 混合调度，在吞吐、TTFT 和 TPOT 之间折中。块太小会增加调度/内核开销，太大仍可能阻塞。

---

## 10. Prefix Cache：相同前缀不要重复 Prefill

许多请求共享系统提示和 Few-shot 示例：

```text
请求 A：[系统提示][公共示例][用户问题 A]
请求 B：[系统提示][公共示例][用户问题 B]
请求 C：[系统提示][公共示例][用户问题 C]
```

缓存公共前缀的 KV 后：

```text
公共 Prefix KV ─┬→ 只计算问题 A
                ├→ 只计算问题 B
                └→ 只计算问题 C
```

Prefix Cache 主要降低重复 Prefill 计算和 TTFT；不会减少不同输出的 Decode 工作。

缓存命中要求 Token 序列完全兼容。多一个空格、不同 Chat Template、随机化系统提示或不同 LoRA Adapter 都可能导致无法共享或需要隔离命名空间。

---

## 11. SGLang 与 RadixAttention

### 11.1 为什么使用 Radix Tree？

Agent、RAG、Tree-of-Thought、并行采样会产生树状共享前缀：

```text
系统提示 + 示例
├── 问题 A
│   ├── 候选回答 A1
│   └── 候选回答 A2
└── 问题 B
    ├── 候选回答 B1
    └── 候选回答 B2
```

Radix Tree 把共同 Token 前缀合并存储：

```text
[系统+示例]
      ├── [问题A] ─┬── [A1]
      │            └── [A2]
      └── [问题B] ─┬── [B1]
                   └── [B2]
```

SGLang 的 RadixAttention 用树结构管理和复用 KV Cache，并结合淘汰策略处理显存压力。

### 11.2 SGLang 的定位

SGLang 最初包含：

```text
前端语言：表达 generation、select、fork/join、控制流
后端 Runtime：调度、RadixAttention、结构化输出等
```

当前工程生态持续演进，但其核心优势仍可理解为：对多轮、多调用、共享前缀和结构化生成工作负载做系统级优化。

### 11.3 vLLM 与 SGLang 怎么选？

| 维度 | vLLM | SGLang |
|---|---|---|
| 基本定位 | 通用高吞吐 LLM 推理与服务 | 高性能推理 + 复杂 LM Program/结构化工作负载 |
| KV 核心思想 | Paged KV 管理、Prefix Cache | Radix Tree 前缀复用、Paged/统一缓存演进 |
| 常见优势场景 | 通用聊天、批量生成、成熟 API 服务 | 多轮 Agent、共享前缀、分支采样、结构化输出 |
| 是否支持批处理/并行/量化 | 支持，依版本和硬件 | 支持，依版本和硬件 |

这不是固定性能排名。两者功能快速互相吸收；必须在**同一模型、精度、硬件、请求分布、SLO 和版本**下实测。

---

## 12. DeepSpeed：训练加速和推理加速要分开看

### 12.1 DeepSpeed Training

DeepSpeed 在训练中常用于：

```text
ZeRO-1：切分 Optimizer States
ZeRO-2：再切分 Gradients
ZeRO-3：再切分 Parameters

加上：Data/Tensor/Pipeline Parallel、Offload、Mixed Precision
```

示意：4 张 GPU 存储同一个训练状态。

```text
普通数据并行：
GPU0 [完整参数][完整梯度][完整优化器]
GPU1 [完整参数][完整梯度][完整优化器]
...

ZeRO 分片：
GPU0 [状态分片0]
GPU1 [状态分片1]
GPU2 [状态分片2]
GPU3 [状态分片3]
```

代价是需要通信和按需聚合；并不是 ZeRO Stage 越高就一定越快。

### 12.2 DeepSpeed Inference

DeepSpeed Inference Engine 通过 `deepspeed.init_inference()` 提供：

- Tensor Parallel；
- Kernel Injection/优化内核；
- CUDA Graph；
- Triton 内核；
- 量化配置；
- MoE Expert Parallel 等。

概念配置示例：

```python
config = {
    "dtype": "fp16",
    "tensor_parallel": {"tp_size": 4},
    "kernel_inject": True,
    "enable_cuda_graph": False,
}

engine = deepspeed.init_inference(model, config=config)
```

参数名称及支持模型依版本变化。

### 12.3 DeepSpeed 与 vLLM/SGLang 不是完全同类

```text
DeepSpeed：大规模训练、模型并行、推理内核/执行
vLLM：在线生成服务的 KV 内存与请求调度
SGLang：在线 Runtime + 前缀复用 + 结构化/多调用程序
```

生产架构应按需求选，而不是把所有框架机械叠加。训练用 DeepSpeed、部署用 vLLM/SGLang 是常见组合；也可用 DeepSpeed Inference 满足特定并行或已有系统需求。

---

## 13. 并行策略：模型放不下一张卡怎么办？

### 13.1 Tensor Parallel（TP）

把单层大矩阵沿维度切到多卡：

```text
输入 X
├→ GPU0：W 分片0 ─┐
├→ GPU1：W 分片1 ─┼→ All-Reduce/All-Gather → 输出
└→ GPU2：W 分片2 ─┘
```

优点：每层都利用多卡、降低单卡权重；代价：几乎每层通信，依赖 NVLink/NVSwitch 等高速互联。

### 13.2 Pipeline Parallel（PP）

按层切模型：

```text
GPU0：Layers 0-9 → GPU1：10-19 → GPU2：20-29 → GPU3：30-39
```

通信数据相对集中，但可能出现流水线气泡；单请求低 Batch 时利用率不佳。

### 13.3 Data Parallel（DP）

每组 GPU 放完整模型副本，处理不同请求：

```text
流量 → Load Balancer
       ├→ Replica A
       ├→ Replica B
       └→ Replica C
```

提高吞吐与可用性，但每个副本都要存完整模型或完整 TP 组。

### 13.4 Expert Parallel（EP）

MoE 专家分布在不同 GPU：

```text
Token → Router → All-to-All → 对应专家 GPU → All-to-All 返回
```

减少每卡专家权重，但 All-to-All 通信和负载不均是关键瓶颈。

### 13.5 Context/Sequence Parallel

把长序列沿 Token 维切分，降低超长上下文单卡压力。需要专门 Attention 通信算法，适合长上下文或多卡 Prefill。

### 13.6 组合示例

8 张 GPU 不一定设置 `TP=8`：

```text
方案 A：TP=8，单个大模型实例
方案 B：TP=4 × DP=2，两组副本提高吞吐
方案 C：TP=2 × PP=2 × DP=2
方案 D：MoE 使用 TP + EP
```

选择取决于权重能否放下、互联拓扑、请求并发和延迟目标。

---

## 14. 模型量化到底是什么？

模型通常以 BF16/FP16 存储。量化用更少 bit 表示数值：

```text
原浮点权重 w
   │ 选择 scale / zero-point / codebook
   ▼
整数或低精度值 q
   │ 推理内核按需反量化或直接低精度计算
   ▼
近似得到原矩阵乘结果
```

最简单的均匀量化：

```text
q = clamp(round(w / scale) + zero_point)
w_hat = scale × (q - zero_point)
```

量化误差为 `w - w_hat`。

## 15. 位宽符号怎么读？

```text
W4A16：Weights 4-bit，Activations 16-bit
W8A8：Weights 8-bit，Activations 8-bit
FP8：权重/激活或特定张量使用 8-bit 浮点，需看具体方案
KV8/FP8 KV：KV Cache 量化
```

只写“INT4 模型”信息不足，还要知道：

- 对称还是非对称；
- per-tensor、per-channel 还是 per-group；
- group size；
- 哪些层保持高精度；
- 量化格式和推理内核是否匹配；
- Calibration 数据是什么。

---

## 16. PTQ 与 QAT

### 16.1 PTQ：训练后量化

Post-Training Quantization 在已有模型上转换精度：

```text
FP16/BF16 模型
  │ 少量 Calibration 数据统计/重建
  ▼
INT8/INT4/FP8 模型
```

优点：快、便宜、无需完整训练；低 bit 时可能损失质量。

### 16.2 QAT：量化感知训练

Quantization-Aware Training 在训练或微调中模拟量化误差：

```text
前向：模拟低精度量化
反向：用近似梯度更新浮点参数
最终：导出低精度权重
```

质量潜力更高，成本和工程复杂度也更大。

### 16.3 QLoRA 不是部署量化方法本身

QLoRA 使用量化 Base 权重降低**微调显存**，训练 LoRA Adapter。它解决的是低成本训练；最终部署格式和推理内核仍需单独决定。

---

## 17. 常见量化方案

### 17.1 GPTQ

GPTQ 是一次性权重量化方法，利用近似二阶信息逐层/逐块补偿量化误差，常用于 3/4-bit Weight-only PTQ。

```text
选择一批 Calibration 激活
  ↓
估计层内误差敏感性
  ↓
逐步量化权重，并补偿尚未量化权重
```

### 17.2 AWQ

AWQ（Activation-aware Weight Quantization）用激活统计识别重要权重通道，并通过缩放保护显著通道，常见 W4A16。

```text
Calibration Activations
  ↓ 找到高影响通道
通道缩放 → 权重 4-bit 量化 → 专用 Kernel
```

### 17.3 SmoothQuant

激活比权重更难量化，因为存在离群值。SmoothQuant 用数学等价缩放把部分量化困难从 Activation 迁移到 Weight：

```text
原来：Weight 易量化 | Activation 有大离群值
缩放：Weight 稍难    | Activation 更平滑
结果：更适合 W8A8 矩阵乘
```

### 17.4 GGUF/llama.cpp 量化

GGUF 是 llama.cpp 生态常见模型文件格式，可包含多种 K-quant 等量化方案，适合 CPU、Apple Silicon 和消费级 GPU 混合卸载。格式名不等同算法质量；不同 `Q4_*` 方案的位宽、分组和混合精度不同。

### 17.5 FP8

FP8 具有指数位，动态范围与 INT8 不同。现代 GPU 对 FP8 Tensor Core 支持可能带来训练和推理收益，但需要匹配硬件、scale 管理、Kernel 和模型 checkpoint。不能只因“8 比 16 小”就假设精确 2× 加速。

---

## 18. 为什么量化省显存却不一定加速？

要真正加速必须有适配 Kernel：

```text
量化权重更小
  ├── 内存读取减少 → Decode 可能更快
  └── 但若先用慢 Kernel 反量化为 FP16
        ├── 额外开销
        └── 可能反而更慢
```

影响端到端性能的因素：

- GPU 是否原生支持该低精度；
- GEMM 的形状和 Batch；
- 量化 Kernel 是否成熟；
- 反量化、scale 读取与格式转换；
- Attention/KV 是否仍是瓶颈；
- 多卡通信是否占主导。

所以要分别报告：权重大小、峰值显存、Prefill tokens/s、Decode tokens/s、TTFT、TPOT 和质量。

---

## 19. KV Cache 量化

长上下文高并发时，KV Cache 可超过权重之外的剩余显存。将 KV 从 BF16 降到 FP8/INT8 可近似减半 Cache 字节数：

```text
BF16 KV：2 bytes/element
FP8 KV：约 1 byte/element + scale 元数据
```

收益：更大并发或上下文。风险：Attention 读取时引入量化误差与转换开销；不同层/头的 K/V 分布敏感性不同。应做长上下文、困惑度和业务任务评测。

---

## 20. Speculative Decoding：让小模型起草，大模型并行验收

标准 Decode 每次用大模型生成一个 Token：

```text
大模型 → y₁ → 大模型 → y₂ → 大模型 → y₃
```

推测解码：

```text
小 Draft 模型快速提出：y₁ y₂ y₃ y₄
             │
             ▼
大 Target 模型一次并行验证多个候选
             │
             ├── 接受前 k 个
             └── 从首个不接受处修正
```

在正确采样算法下可保持 Target 分布。实际收益取决于：

- Draft 比 Target 快多少；
- 接受率；
- 验证 Kernel；
- Batch 和并发；
- 任务语言与 Prompt；
- Draft/Target Tokenizer 是否兼容。

如果 Draft 预测常错，额外计算可能不划算。EAGLE、Medusa、n-gram speculation 等使用不同草稿来源。

---

## 21. CUDA Graph、Kernel Fusion 与编译

### CUDA Graph

把一组 GPU 操作捕获后重复回放，减少 CPU 启动 Kernel 的开销：

```text
普通：CPU launch op1 → op2 → op3 → ... 每步重复
Graph：捕获 [op1 op2 op3] → 一次回放
```

动态图形、不同 Shape 和动态控制流会增加捕获难度，也可能需要为多种 Batch Shape 预留显存。

### Kernel Fusion

```text
RMSNorm → Linear → Bias → Activation
```

若分成多个 Kernel，中间结果反复写 HBM；Fusion 在一个/少数 Kernel 内完成，减少启动和 IO。

### torch.compile / Triton

图编译可做算子融合、Shape 特化和生成优化 Kernel。首次编译/warmup 较慢；线上应区分冷启动和稳定状态性能。

---

## 22. 结构化输出为什么需要专门优化？

若输出必须符合 JSON Schema，每一步只有部分 Token 合法：

```text
当前：{"name": "Alice", "age":
合法下一步：数字、空格等
非法下一步：任意破坏 JSON 的 Token
```

约束解码用 FSM/Grammar 构造合法 Token Mask。朴素方法在 CPU 上逐步计算可能拖慢生成；SGLang 等系统使用压缩 FSM、缓存或专门 Runtime 减少开销。

结构化输出保证语法结构，不保证字段事实正确或业务约束正确。

---

## 23. Prefill–Decode 分离

Prefill 与 Decode 的硬件需求不同，可以分离到不同 Worker：

```text
请求 → Prefill Worker（计算密集）
              │ 传输 KV Cache
              ▼
       Decode Worker（带宽敏感）→ 输出
```

潜在收益：分别调优和扩容；挑战：KV 传输量大、调度复杂、跨机网络可能抵消收益。适合规模足够大且负载稳定的服务，不是单机部署的默认方案。

---

## 24. 从单机到生产集群

### 24.1 单机开发

```text
客户端 → 单个 vLLM/SGLang Server → 1～多 GPU
```

先验证模型、模板、量化、显存和输出正确性。

### 24.2 多副本生产

```text
                     ┌→ Replica 1（TP group）
Load Balancer/Router ├→ Replica 2（TP group）
                     └→ Replica 3（TP group）
```

Router 可以根据：

- 活跃 Token 数；
- KV Cache 使用率；
- Prefix 亲和性；
- 模型/LoRA Adapter；
- 优先级和租户；
- GPU 健康状态。

### 24.3 控制面与数据面

```text
控制面：部署、扩缩容、版本、健康检查、配置
数据面：实际 Tokenize、调度、GPU 推理、流式返回
```

模型滚动升级要考虑缓存隔离、Tokenizer/Template 版本和正在生成的长请求。

---

## 25. 容量规划的基本步骤

### 步骤 1：记录真实请求分布

```text
输入长度 P50/P95/P99
输出长度 P50/P95/P99
并发与到达率
共享前缀比例
模型和 Adapter 数量
目标 TTFT/TPOT/E2E
```

### 步骤 2：判断权重能否放下

```text
可用显存 > 权重 + 运行时保留 + 最低 KV Cache
```

若放不下：选择量化、TP/PP、CPU Offload 或更小模型。

### 步骤 3：估算 KV 容量

按模型层数、KV heads、head dim、dtype 和最大活跃 Token 估算，再用引擎实际日志校准。

### 步骤 4：按真实负载压测

```text
并发：1, 2, 4, 8, 16, ...
输入/输出：使用生产分布，不只固定 128/128
持续时间：覆盖 warmup 和稳定期
记录：TTFT、TPOT、吞吐、错误、显存、功耗
```

### 步骤 5：用 SLO 而非峰值吞吐定容量

最大吞吐点可能 P99 已不可接受。生产容量应保留故障、流量突增和长请求余量。

---

## 26. 一个公平的 vLLM/SGLang/其他引擎基准

必须固定：

```text
同一模型 checkpoint
同一量化和 dtype
同一 GPU、驱动、CUDA 与拓扑
同一 Chat Template/Tokenizer
同一输入和输出长度分布
同一采样参数
同一并发/到达过程
同一 Prefix Cache 开关与命中条件
同一质量容差
```

报告：

| 类型 | 指标 |
|---|---|
| 质量 | 任务准确率、PPL、量化回归 |
| 延迟 | TTFT、TPOT/ITL、E2E 的 P50/P95/P99 |
| 吞吐 | input/output/total tokens/s、requests/s |
| 资源 | 峰值显存、KV 使用率、GPU 利用率、功耗 |
| 稳定 | OOM、超时、取消、错误率、长时间泄漏 |

论文或博客中的“快 3×”只能说明其测试条件，不能直接替代自己的基准。

---

## 27. 部署选择决策树

```text
只做本地单用户、CPU/Apple Silicon？
  └→ llama.cpp / GGUF 常合适

GPU 通用在线聊天、批量生成、OpenAI API？
  └→ 先基准 vLLM

Agent、多轮、共享前缀、分支采样、结构化输出很多？
  └→ 同时重点基准 SGLang

训练系统已深度使用 DeepSpeed，需要特定 TP/MoE 推理？
  └→ 评估 DeepSpeed Inference

NVIDIA 生产环境追求特定模型极致性能？
  └→ 同时评估 TensorRT-LLM 等硬件定向方案
```

最终选择仍以版本兼容性、团队维护能力和真实负载实测为准。

---

## 28. 常见错误认识

1. **FlashAttention = 推理服务框架**：错误，它主要是 Attention 计算内核/算法。
2. **PagedAttention = Prefix Cache**：不完全等同；分页主要解决 KV 分配，Prefix Cache 解决重复前缀复用。
3. **量化到 4-bit 一定快 4 倍**：错误；显存字节减少不等于端到端线性加速。
4. **MoE 只需存激活参数**：错误；全部专家权重仍需存储/分布。
5. **最大上下文等于高质量有效上下文**：错误；还要做长文检索、位置和任务评测。
6. **Batch 越大越好**：错误；吞吐提升可能损害 TTFT/尾延迟。
7. **TP 卡数越多越快**：错误；通信可超过计算收益。
8. **Prefix Cache 会加速所有请求**：错误；只有共享完全兼容前缀时命中。
9. **DeepSpeed、vLLM、SGLang 三选一**：错误；它们作用层次有重叠也有差异。
10. **结构化解码保证答案正确**：错误；它只约束语法/Schema。

---

## 29. 生产监控面板

```text
流量：QPS、并发、输入/输出 Token 分布
延迟：TTFT、TPOT、E2E P50/P95/P99
调度：Queue Time、Active/Waiting Requests、Preemption
缓存：KV 使用率、Prefix Hit Rate、Eviction、碎片
GPU：利用率、HBM、带宽、功耗、温度、通信
质量：空回答、重复、截断、JSON 失败、量化回归
可靠：OOM、超时、取消、5xx、Worker 重启
成本：每百万输入/输出 Token 的 GPU 时间与费用
```

性能优化前先判断瓶颈属于：CPU Tokenize、排队、Prefill Compute、Decode Bandwidth、KV 容量、GPU 通信还是网络返回。

---

## 30. 安全与可靠性

- API 鉴权、租户隔离、请求和 Token 限额；
- Prompt/输出日志脱敏，避免记录密钥和个人信息；
- 模型权重与自定义代码供应链审查；
- 禁止无条件加载不可信 remote code；
- 请求取消后及时回收 KV；
- 超长 Prompt、超大 `max_tokens` 和并发限流；
- 健康检查、熔断、重试幂等与流式断连处理；
- 模型/Tokenizer/Chat Template/量化版本共同纳入版本管理；
- 对量化模型重新执行安全与拒答评测，不能只检查 PPL。

---

## 31. 动手实验

### 实验 A：Prefill/Decode 分离测量

固定模型，测试：

```text
输入 128 / 输出 1024   → Decode-heavy
输入 8192 / 输出 128   → Prefill-heavy
输入 4096 / 输出 1024  → Mixed
```

分别记录 TTFT、TPOT 和吞吐。

### 实验 B：Prefix Cache

准备 100 个请求，共享 2K Token 系统提示；再准备 100 个无共享前缀请求。对比缓存开关和命中率，观察 TTFT。

### 实验 C：量化

同一模型比较 BF16、AWQ/GPTQ W4A16、FP8（若硬件支持）：

```text
显存 | Prefill tok/s | Decode tok/s | PPL | 业务准确率 | 输出差异
```

### 实验 D：并行策略

同一 8-GPU 节点比较 `TP=8` 与 `TP=4, DP=2`。单请求延迟可能前者更好，总吞吐可能后者更好，结果依模型和互联而定。

### 实验 E：vLLM 与 SGLang

至少测试两类负载：随机独立聊天、共享长前缀的多轮/Agent。若只测试其中一种，会掩盖另一种系统设计的优势。

---

## 32. 复习题

1. Prefill 与 Decode 为什么分别偏计算密集和带宽密集？
2. KV Cache 节省了什么计算，又增加了什么成本？
3. FlashAttention 是否改变标准 Attention 的平方 FLOPs 数量级？
4. PagedAttention 怎样减少 KV 显存浪费？
5. Continuous Batching 与静态 Batch 的区别是什么？
6. Prefix Cache 对什么请求最有效？
7. Radix Tree 为什么适合分支式 Agent 工作流？
8. ZeRO-1/2/3 分别切分什么训练状态？
9. TP、PP、DP、EP 的主要通信和适用场景是什么？
10. W4A16 与 W8A8 分别代表什么？
11. PTQ、QAT、QLoRA 为什么不是同一概念？
12. 为什么 4-bit 量化可能省显存但不加速？
13. Speculative Decoding 的速度取决于哪两个核心因素？
14. 怎样公平比较 vLLM 与 SGLang？

---

## 33. 参考资料

以下优先使用原始论文和官方文档：

1. Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*：<https://arxiv.org/abs/2205.14135>
2. Dao, *FlashAttention-2*：<https://arxiv.org/abs/2307.08691>
3. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*：<https://arxiv.org/abs/2309.06180>
4. vLLM 官方文档：<https://docs.vllm.ai/en/stable/>
5. Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*：<https://arxiv.org/abs/2312.07104>
6. SGLang 官方仓库：<https://github.com/sgl-project/sglang>
7. DeepSpeed Inference 官方文档：<https://deepspeed.readthedocs.io/en/stable/inference-init.html>
8. Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*：<https://arxiv.org/abs/1910.02054>
9. Frantar et al., *GPTQ*：<https://arxiv.org/abs/2210.17323>
10. Lin et al., *AWQ*：<https://arxiv.org/abs/2306.00978>
11. Xiao et al., *SmoothQuant*：<https://arxiv.org/abs/2211.10438>
12. Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*：<https://arxiv.org/abs/2211.17192>
13. Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*：<https://arxiv.org/abs/2302.01318>

---

## 34. 本章知识地图

```text
LLM 部署性能
├── 算子层
│   ├── FlashAttention
│   ├── Kernel Fusion / Triton
│   └── CUDA Graph / Compile
├── 数值层
│   ├── GPTQ / AWQ / SmoothQuant / FP8
│   ├── PTQ / QAT
│   └── KV Cache Quantization
├── 内存层
│   ├── KV Cache
│   ├── PagedAttention
│   └── Prefix/Radix Cache
├── 调度层
│   ├── Continuous Batching
│   ├── Chunked Prefill
│   └── Prefill–Decode Disaggregation
├── 并行层
│   ├── TP / PP / DP / EP / CP
│   └── DeepSpeed / 分布式执行
├── 解码层
│   ├── Speculative Decoding
│   └── Structured/Constrained Decoding
└── 服务层
    ├── vLLM / SGLang / DeepSpeed Inference
    ├── 网关、路由、扩缩容
    └── SLO、监控、安全与成本
```
