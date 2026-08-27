# LLM 学习笔记：第 9 章——主流模型、架构与训练框架

> 信息快照：2026-08-27。模型迭代很快，本章以可核对的公开报告举例，不把未披露信息猜成事实。

## 1. 先区分“开源”与“开放权重”

```text
完整开放
├── 架构
├── 权重
├── 训练代码
├── 数据及其来源
├── 训练日志/超参数
└── 允许研究、修改和再分发的许可证

常见“开放模型”
├── 权重 ✓
├── 推理代码 ✓
├── 技术报告 部分
├── 完整训练数据 ✗
└── 许可证 可能有使用限制
```

因此 Llama、Qwen、Gemma 等更准确的常用描述是“开放权重模型”，是否符合严格 Open Source 定义要逐项检查。

## 2. Dense 与 MoE

### Dense

每个 Token 经过每层全部 FFN 参数：

```text
Token → Attention → FFN 全部参数 → 下一层
```

### MoE

Router 为每个 Token 选择少数专家：

```text
                         ┌→ Expert 1
Token → Router → Top-k ──┼→ Expert 2  → 加权合并
                         ├→ Expert 3
                         └→ ... Expert N
```

总参数可以很大，但单 Token 只激活一部分。它降低每 Token 计算相对总容量的比例，却不会让全部权重消失：模型存储、跨设备通信、负载均衡和部署仍然困难。

## 3. 参数量要看哪一个？

```text
Dense 32B：总参数 ≈ 每 Token 使用参数（粗略）
MoE 235B-A22B：总计约 235B，每 Token 激活约 22B
```

比较时至少报告：总参数、激活参数、上下文、精度/量化、是否多模态。仅说“235B 比 70B 大”无法推断实际速度或质量。

## 4. 代表性开放权重家族

| 家族/公开版本示例 | 架构与规模 | 已公开训练信息 | 说明 |
|---|---|---|---|
| Llama 3/3.1 | Dense Decoder；8B、70B、405B；GQA | Llama 3 8B/70B 模型卡披露 15T+ Token；3.1 报告含 405B | 128K 词表；自定义许可证 |
| Qwen3 | Dense 0.6B～32B；MoE 30B-A3B、235B-A22B | 技术报告披露多阶段预训练与后训练，但并非完整原始数据 | 同时覆盖 dense/MoE，支持 thinking/non-thinking 模式的版本设计 |
| DeepSeek-V3 | MoE 671B，总激活约 37B；MLA、DeepSeekMoE | 披露 14.8T 预训练 Token，之后 SFT 与 RL | 大容量 MoE；本地部署仍需存放全部权重 |
| Gemma 3 | 1B、4B、12B、27B；较大版本支持图像输入 | 官方模型卡披露 128K 上下文、140+语言（1B 有差异需看卡） | 开放权重，受 Gemma 条款约束 |
| Mistral/Mixtral | Dense 与稀疏 MoE Decoder | 不同版本披露程度不同 | 滑动窗口、GQA/MoE 等高效设计常见 |

表中信息不是排行榜。不同报告的评测 Prompt、采样、量化和数据污染控制不一致，不能只比较一列平均分。

## 5. Llama 路线

```text
Llama 1：强调用更多公开数据训练较小模型
  ↓
Llama 2：Base + Chat，报告 SFT/RLHF
  ↓
Llama 3：128K 词表、GQA、15T+ Token（8B/70B 模型卡）
  ↓
Llama 3.1：扩展到 405B 与长上下文、多语言/工具能力
```

其主干仍是相对“标准”的 Decoder-only：RMSNorm、RoPE、SwiGLU、GQA。价值之一是形成了广泛微调和部署生态。

## 6. Qwen 路线

Qwen 家族强调多语言、代码、数学和工具能力。Qwen3 同时提供 Dense 与 MoE：

```text
小型本地：0.6B / 1.7B / 4B / 8B
中大型 Dense：14B / 32B
MoE：30B-A3B / 235B-A22B
```

`A3B`/`A22B` 表示每 Token 激活参数量，不是模型文件只含这些参数。不同具体 checkpoint 的上下文、许可和量化需查模型卡，不能从家族名推断。

## 7. DeepSeek 路线

DeepSeek-V3：

```text
总参数 671B
每 Token 激活约 37B
14.8T 预训练 Token
MLA：压缩 KV 表示以降低推理缓存
DeepSeekMoE：细粒度专家与共享专家
MTP：训练中预测多个未来 Token 的辅助目标
```

DeepSeek-R1 在 V3 Base 基础路线中强调推理 RL；R1-Zero 展示不以 SFT 冷启动也可经 RL 出现推理行为，但存在可读性和语言混杂，正式 R1 使用冷启动与多阶段训练改善。

## 8. 闭源模型怎样记录？

GPT、Claude、Gemini 等闭源产品常不披露精确参数量、完整架构和训练数据量。正确表格应写：

| 项目 | 公开状态 |
|---|---|
| 参数量 | 未披露，不用传闻填表 |
| Dense/MoE | 若官方未确认则记未知 |
| 训练 Token | 未披露 |
| 上下文/模态 | 可按官方产品文档记录具体版本 |
| 训练方法 | 只记录官方报告明确内容 |

API 模型名称可能指动态更新的服务，而非永远不变的权重 checkpoint；比较时记录测试日期和版本。

## 9. 从技术报告读取模型的模板

```text
身份：Base / Instruct / Chat / Reasoning / Multimodal
架构：Decoder-only？Dense/MoE？Attention 类型？位置编码？
规模：总参数、激活参数、层数、隐藏维度、专家数
数据：训练 Token、语言/领域、截止时间、是否公开
训练：预训练目标、CPT、SFT、偏好/RL
上下文：训练长度、声明最大长度、长文实测
许可：商用、再分发、衍生模型限制
评测：版本、Prompt、污染控制、推理预算
部署：权重精度、显存、吞吐、框架支持
```

## 10. 常见开源训练框架

### 底层与大规模预训练

| 框架 | 主要作用 | 典型特点 |
|---|---|---|
| PyTorch FSDP | 参数/梯度/优化器分片 | PyTorch 原生生态 |
| DeepSpeed | ZeRO、并行和训练优化 | 大模型训练与推理工具丰富 |
| Megatron-LM/Core | 张量、流水线、上下文并行 | 大规模 Transformer 训练 |
| Colossal-AI | 多维并行与显存优化 | 一体化大模型系统 |

### 微调与对齐

| 框架 | 主要用途 |
|---|---|
| Transformers + PEFT | 模型加载、LoRA/QLoRA、训练器 |
| TRL | SFT、DPO、PPO/GRPO 类训练 |
| LLaMA-Factory | 配置化 SFT/PEFT/偏好训练 |
| Axolotl | YAML 驱动的微调流程 |
| verl | 大规模 RL、混合训练与推理后端 |
| OpenRLHF | Ray/vLLM 等组合的 RLHF |

### 推理

```text
vLLM：PagedAttention、连续批处理
SGLang：结构化生成与高性能服务
TensorRT-LLM：NVIDIA 推理优化
llama.cpp：CPU/消费级硬件与 GGUF 量化生态
```

项目是否“开源”不代表所有模型许可证兼容；框架许可与权重许可是两件事。

## 11. 选择模型的决策图

```text
数据必须本地？──否→ 可比较闭源 API 与开放权重服务
  │是
  ▼
可用显存/内存？
├─ 很小 → 1B～8B 量化
├─ 中等 → 8B～32B，张量并行/量化
└─ 集群 → 70B+ 或 MoE
  ↓
任务：中文/代码/数学/多模态/长文
  ↓
实测自己的保留集：质量、延迟、吞吐、成本、安全
```

## 12. 参考资料

1. Meta, *The Llama 3 Herd of Models*：<https://arxiv.org/abs/2407.21783>
2. Llama 3 Model Card：<https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md>
3. Qwen Team, *Qwen3 Technical Report*：<https://arxiv.org/abs/2505.09388>
4. DeepSeek-AI, *DeepSeek-V3 Technical Report*：<https://arxiv.org/abs/2412.19437>
5. DeepSeek-AI, *DeepSeek-R1*：<https://arxiv.org/abs/2501.12948>
6. Google, Gemma 3 Model Card：<https://ai.google.dev/gemma/docs/core/model_card_3>
7. Rajbhandari et al., *ZeRO*：<https://arxiv.org/abs/1910.02054>
8. Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*：<https://arxiv.org/abs/2104.04473>
