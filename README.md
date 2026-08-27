# Awesome LLM Learning

一套面向大语言模型初学者与工程实践者的中文学习笔记，从 NLP 基础、Tokenizer 和 Transformer 原理，逐步延伸到训练、强化学习、MoE、推理加速、量化与生产部署。

笔记尽量使用通俗语言，并通过流程图、结构图、注意力矩阵、对比表、公式拆解和小型代码实验帮助理解。涉及模型参数、训练数据及新算法时，优先引用原始论文、官方技术报告和官方文档。

## 开始阅读

完整目录与学习路线：[LLM 学习笔记总目录](./llm/LLM学习笔记-总目录.md)

建议按照章节顺序学习：

```text
语言模型基础
  ↓
NLP 模型发展史
  ↓
Transformer 与三类架构
  ↓
Scaling Law 与数据工程
  ↓
Pre-train、CPT、SFT、LoRA、RLHF
  ↓
PPO、DPO、GRPO、DAPO、GSPO
  ↓
主流模型、工程复现与部署
  ↓
MoE 混合专家模型
```

## 章节目录

1. [NLP 与语言模型基础](./llm/LLM学习笔记-第01章-NLP与语言模型基础.md)
2. [从 N-gram 到 Transformer：NLP 模型发展史](./llm/LLM学习笔记-第02章-NLP模型发展史.md)
3. [深入理解 Transformer](./llm/LLM学习笔记-第03章-深入理解Transformer.md)
4. [BERT、GPT 与 Encoder/Decoder/Encoder–Decoder](./llm/LLM学习笔记-第04章-BERT与GPT三类架构.md)
5. [Scaling Law 与能力增长](./llm/LLM学习笔记-第05章-Scaling-Law与能力增长.md)
6. [LLM 数据工程](./llm/LLM学习笔记-第06章-LLM数据工程.md)
7. [LLM 完整训练链路](./llm/LLM学习笔记-第07章-LLM完整训练链路.md)
8. [LLM 强化学习方法](./llm/LLM学习笔记-第08章-LLM强化学习方法.md)
9. [主流模型与开源训练框架](./llm/LLM学习笔记-第09章-主流模型与开源训练框架.md)
10. [工程实践与复现路线](./llm/LLM学习笔记-第10章-工程实践与复现路线.md)
11. [LLM 推理加速、量化与部署](./llm/LLM学习笔记-第11章-LLM推理加速量化与部署.md)
12. [深入理解 MoE 混合专家模型](./llm/LLM学习笔记-第12章-深入理解MoE混合专家模型.md)

## 主要内容

- Tokenizer 是否需要模型、BPE/WordPiece/Unigram 如何训练，以及数据量与词表大小如何选择；
- Causal LM、MLM、Span Corruption、Prefix LM 的训练目标和信息可见范围；
- N-gram、Word2Vec、TextCNN、RNN、LSTM、Attention 到 Transformer 的演进；
- Self-Attention、Q/K/V、MHA、MQA、GQA、RoPE、FFN、RMSNorm 与 KV Cache；
- Encoder-only、Decoder-only、Encoder–Decoder 的结构与应用差异；
- Scaling Law、Chinchilla、能力涌现、训练数据配比、去重和污染检测；
- Pre-training、CPT、SFT、LoRA、QLoRA、RLHF 和 DPO；
- PPO、GRPO、DAPO、GSPO 的原理、差异与适用场景；
- vLLM、SGLang、DeepSpeed、FlashAttention、PagedAttention 和 RadixAttention；
- GPTQ、AWQ、SmoothQuant、FP8、KV Cache 量化和推测解码；
- MoE Router、Top-k、Shared Expert、负载均衡、Expert Parallel 与 All-to-All。

## 笔记结构

所有学习资料都位于 [`llm/`](./llm/) 目录中。每章通常包含：

- 本章知识地图；
- 核心概念与结构图；
- 输入、输出和损失函数；
- 技术收益、代价与常见误区；
- 对比表、手算示例或代码实验；
- 复习题；
- 原始论文和官方文档链接。

## 使用说明

模型、框架和算法仍在快速演进。笔记中的模型规格和软件能力均应结合标注日期理解；进行实际训练或部署时，请再次核对所用版本的官方模型卡、许可证和文档。

欢迎通过 Issue 或 Pull Request 指出错误、补充资料或改进讲解。

## License

本仓库暂未声明独立许可证。在添加明确许可证前，请勿默认其中内容可按任意开源协议再分发；引用的论文、数据集、模型和软件分别遵循其原始许可。
