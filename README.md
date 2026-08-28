# Awesome LLM & VLM Learning

一套面向大语言模型与视觉语言模型初学者、工程实践者的中文学习笔记。内容从 NLP、Tokenizer 和 Transformer 原理，延伸到 LLM 训练与部署；同时覆盖视觉基础、ViT、CLIP/SigLIP、DINO、VLM 架构、训练、强化学习、评估和生产部署。

笔记尽量使用通俗语言，并通过流程图、结构图、注意力矩阵、对比表、公式拆解和小型代码实验帮助理解。涉及模型参数、训练数据及新算法时，优先引用原始论文、官方技术报告和官方文档。

## 开始阅读

项目分为两条学习路线：

- [LLM 学习笔记总目录](./llm/LLM学习笔记-总目录.md)
- [VLM 学习笔记总目录](./vlm/VLM学习笔记-总目录.md)

### LLM 路线

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
  ↓
LLM 评估与业务验收
```

### VLM 路线

```text
视觉任务与图像表示
  ↓
ViT → CLIP / SigLIP → DINO
  ↓
视觉编码器与 LLM 的连接和融合
  ↓
VLM 发展史、重点模型与原生多模态
  ↓
数据工程、Pre-training、CPT、SFT、LoRA
  ↓
多模态 DPO / PPO / GRPO
  ↓
评估、推理加速、量化与部署
  ↓
端到端工程实践
```

## LLM 章节目录

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
13. [LLM 评估体系、指标与 Benchmark](./llm/LLM学习笔记-第13章-LLM评估体系与Benchmark.md)

## VLM 章节目录

1. [VLM 基础与任务边界](./vlm/VLM学习笔记-第01章-VLM基础与任务边界.md)
2. [计算机视觉与图像表示基础](./vlm/VLM学习笔记-第02章-计算机视觉与图像表示基础.md)
3. [ViT 与视觉 Transformer](./vlm/VLM学习笔记-第03章-ViT与视觉Transformer.md)
4. [CLIP、ALIGN、SigLIP 与 SigLIP2](./vlm/VLM学习笔记-第04章-CLIP与SigLIP图文对齐.md)
5. [DINO、DINOv2 与 DINOv3](./vlm/VLM学习笔记-第05章-DINO自监督视觉学习.md)
6. [VLM 的连接与融合架构](./vlm/VLM学习笔记-第06章-VLM连接与融合架构.md)
7. [VLM 发展史](./vlm/VLM学习笔记-第07章-VLM发展史.md)
8. [重点开源 VLM 系列解析](./vlm/VLM学习笔记-第08章-重点开源VLM系列解析.md)
9. [原生多模态模型](./vlm/VLM学习笔记-第09章-原生多模态模型.md)
10. [VLM 数据工程](./vlm/VLM学习笔记-第10章-VLM数据工程.md)
11. [VLM 完整训练链路](./vlm/VLM学习笔记-第11章-VLM完整训练链路.md)
12. [VLM 微调方法与开源训练框架](./vlm/VLM学习笔记-第12章-VLM微调与训练框架.md)
13. [多模态偏好优化与强化学习](./vlm/VLM学习笔记-第13章-多模态偏好优化与强化学习.md)
14. [VLM 评估体系与 Benchmark](./vlm/VLM学习笔记-第14章-VLM评估体系与Benchmark.md)
15. [VLM 推理、加速、量化与部署](./vlm/VLM学习笔记-第15章-VLM推理加速量化与部署.md)
16. [端到端工程实践](./vlm/VLM学习笔记-第16章-端到端工程实践.md)

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
- Base/Chat/Reasoning/RAG/Agent 模型的能力、事实性、安全、鲁棒性、系统性能、成本和业务效果评估。
- 图片张量、Patch、ViT、CLIP/SigLIP、DINO 自监督视觉学习；
- Projector、Q-Former、Resampler、动态分辨率和视觉 Token 压缩；
- LLaVA、Qwen-VL、InternVL、DeepSeek-VL 与原生多模态模型；
- VLM 数据、Pre-training、CPT、SFT、LoRA、DPO、PPO 和 GRPO；
- OCR、Grounding、视频、GUI Agent、幻觉评估与多模态部署。

## 笔记结构

学习资料分别位于 [`llm/`](./llm/) 和 [`vlm/`](./vlm/) 目录中。每章通常包含：

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
