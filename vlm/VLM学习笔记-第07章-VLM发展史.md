# 第 7 章：VLM 发展史

> 本章不是模型名单，而是一条“问题—方法—新问题”的演进线。参数和数据只记录论文或官方公开值；闭源模型未披露部分不作猜测。

## 1. 总时间线

```text
专用视觉语言任务
  ↓
单流/双流跨模态预训练：VisualBERT、ViLBERT、UNITER
  ↓
大规模图文双塔：CLIP、ALIGN
  ↓
统一生成与数据治理：OFA、BLIP
  ↓
冻结强视觉/语言模型并搭桥：Flamingo、BLIP-2
  ↓
视觉指令微调：LLaVA、MiniGPT-4、Qwen-VL
  ↓
高分辨率、OCR、Grounding、视频：LLaVA-NeXT、Qwen2-VL、InternVL、DeepSeek-VL
  ↓
视觉推理、GUI Agent、长视频：Qwen2.5-VL、Qwen3-VL 等
  ↓
基础预训练即混合多模态：Qwen3.5 等原生多模态路线
```

## 2. 早期：为单个任务设计模型

早期视觉语言研究主要围绕 Image Caption、VQA、图文检索。常见流水线：

```text
CNN 提取区域特征 + RNN/BERT 编码文字
              ↓
注意力融合
              ↓
为某个任务设计的输出头
```

特点：每个任务使用专门结构和标注，迁移能力有限；但区域特征、注意力对齐和 VQA 数据为后续打下基础。

## 3. 2019～2020：跨模态 BERT 预训练

VisualBERT、ViLBERT、LXMERT、UNITER 等把检测器区域特征与文本 Token 送入 Transformer。

- Single-stream：图文 Token 早期放入同一 Transformer；
- Dual-stream：视觉与文字分别编码，再通过跨模态层交互。

常见目标包括 Masked Language Modeling、Image-Text Matching、Masked Region Modeling。局限是依赖目标检测器和区域标注，流水线复杂，开放生成能力弱。

## 4. 2021：CLIP 与 ALIGN 的规模化表征路线

CLIP 用海量自然语言图文对训练双塔，使图片与文字进入共享向量空间。核心变化：

- 不再依赖固定类别标签；
- 用自然语言描述类别；
- 零样本分类和跨数据集迁移显著增强；
- 图文检索可离线编码。

ALIGN 进一步探索大规模弱清洗网页图文数据。这一阶段解决“开放词汇视觉语义”，但双塔相似度模型还不能像 LLM 一样长文本对话。

## 5. 2022：统一生成、数据治理与少样本学习

### 5.1 BLIP

BLIP 同时面向理解和生成，并提出 Captioning and Filtering（CapFilt）：用模型生成更丰富描述、过滤噪声图文对。贡献不仅是结构，还强调数据质量。

### 5.2 Flamingo

Flamingo 连接预训练视觉模型和冻结语言模型：Perceiver Resampler 把视觉特征压成固定数量，Gated Cross-Attention 把视觉信息插入语言层。它能处理交错图片/视频与文字，并展示多模态 In-context Few-shot Learning。

它说明：不必从头联合训练所有大模块，也可以让强视觉骨干和强 LLM 协作。

## 6. 2023：BLIP-2 与“轻量桥接”

BLIP-2 冻结图像编码器和 LLM，用 Q-Former 缩小模态鸿沟：

```text
阶段1：冻结图像编码器，学习视觉—语言表征
阶段2：冻结 LLM，学习视觉到语言的生成
```

该路线降低训练参数量，并系统说明“已有视觉基础模型 + 已有 LLM + 可训练桥梁”如何组合。

## 7. 2023：LLaVA 与视觉指令微调

LLaVA 的核心结构很简洁：CLIP Vision Encoder + Linear Projector + LLM。关键贡献是用语言模型辅助生成视觉指令数据，再做端到端视觉指令微调。

```text
图片描述/框等符号信息
  ↓ 交给强语言模型生成问答
多模态指令数据
  ↓
对齐预训练 + 指令微调
```

它把 VLM 从“特定 Benchmark 模型”推进到“通用视觉助手”，也推动了开源复现生态。

## 8. LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5 表明 MLP Projector、更高分辨率视觉编码器、质量较好的公开 VQA 数据和格式规范能建立很强基线。13B 版本论文报告使用约 1.2M 公开数据并可在单个 8×A100 节点约一天完成训练。

LLaVA-NeXT 继续改善：

- AnyRes 和图像切块；
- 更高分辨率与更多长宽比；
- 更强 OCR、文档和视觉推理；
- 视频和不同 LLM Backbone 扩展。

演进启示：视觉 Token 组织、数据混合和训练细节，可能和连接器是否复杂同样重要。

## 9. Qwen-VL：OCR 与 Grounding

Qwen-VL 使用视觉 Transformer、跨模态模块和 Qwen LLM，训练分预训练、多任务预训练和指令微调等阶段。系列早期就强调：

- 中英文多语言；
- 文档与文字读取；
- 边界框形式的视觉定位；
- 多轮图文对话。

这使“视觉助手”不只是自然图片聊天，也进入文档、Grounding 等可落地任务。

## 10. InternVL：扩展视觉骨干与动态高分辨率

InternVL 系列重点包括大规模视觉语言基础模型、强视觉编码器、像素重排/投影接口，以及动态高分辨率切块。系列后续版本持续加强 OCR、文档、多图、视频和推理。

学习时要分清：InternViT 视觉骨干参数、LLM 参数和整个 VLM 参数，不能把模型名称中的数字自动当成视觉塔大小。

## 11. DeepSeek-VL 与 DeepSeek-VL2

DeepSeek-VL 强调真实场景视觉理解和保持语言能力的数据配比。DeepSeek-VL2 的两项关键升级：

1. 动态切块视觉编码，适配高分辨率和不同长宽比；
2. 使用 DeepSeekMoE 语言模型与 MLA，提高参数容量和推理效率。

官方系列的激活参数分别约 1.0B、2.8B、4.5B。激活参数不是总权重，也不表示只需加载这些权重。

## 12. Qwen2-VL：Naive Dynamic Resolution 与 M-RoPE

Qwen2-VL 让不同分辨率图片映射成不同数量视觉 Token，并用 Multimodal RoPE（M-RoPE）分别表达：

- 文本的一维顺序；
- 图像的高度、宽度位置；
- 视频的时间、高度、宽度位置。

官方公开视觉编码器约 600M 参数。动态 Token 改善原始尺寸适配，但让服务显存和延迟依输入图片波动。

## 13. Qwen2.5-VL：视觉编码与时间尺度

Qwen2.5-VL 强化视觉识别、精确框/点定位、文档结构抽取、图表和长视频。技术重点包括：

- 原生动态分辨率 ViT；
- 多数层使用 Window Attention，仅部分层全局注意力；
- ViT 采用 RMSNorm、SwiGLU 等与 LLM 更一致的组件；
- 动态 FPS 和绝对时间对齐；
- 坐标可使用图片实际尺寸尺度表达。

这表明视觉编码器本身也在针对 VLM 重新训练，而非永远冻结通用 CLIP。

## 14. Qwen3-VL：更深融合与视觉 Agent

Qwen3-VL 官方公开的架构升级包括：

- Interleaved-MRoPE：增强时间、宽、高位置建模；
- DeepStack：融合多级 ViT 特征；
- Text–Timestamp Alignment：更精确视频时间定位；
- Dense 与 MoE 版本；
- Instruct 与 Thinking 版本；
- GUI Agent、空间理解、OCR、视觉编码等能力增强。

“Thinking”表示后训练与推理行为版本，不能直接等同于视觉编码器不同。

## 15. Qwen3.5：原生多模态转折

Qwen3.5 官方称其为原生视觉语言模型，强调在基础预训练中早期融合文本与视觉，而非纯文本基础模型完成后才外挂视觉能力。

公开的 Qwen3.5-397B-A17B：

- 总参数约 397B；
- 每次前向激活约 17B；
- 高稀疏 MoE；
- Gated DeltaNet 与 Gated Attention 混合；
- Multi-Token Prediction；
- 混合文本、图片、视频预训练；
- 面向多模态推理和数字 Agent。

原生多模态不等于“没有视觉编码过程”，而是视觉能力进入基础模型的核心预训练和统一优化。详细见第 9 章。

## 16. 闭源模型怎样学习

Gemini、GPT-4V/4o 等推动原生多模态、实时交互和工具使用，但内部参数、训练数据和完整结构没有全部公开。

记录规范：

- 只写 System Card、Technical Report、官方博客/API 文档披露内容；
- “支持图片/视频”不推断具体视觉塔；
- “端到端”不推断所有模块完全共享；
- 不用传闻参数填表；
- Benchmark 必须核对版本和评测设置。

## 17. 历史演进的五条主线

| 主线 | 早期问题 | 演进方向 |
|---|---|---|
| 视觉表征 | 固定类别/区域特征 | 开放词汇、密集特征、高分辨率 |
| 融合 | 专用跨模态层 | Projector、Q-Former、深层融合、原生融合 |
| 数据 | 小型人工标注 | 网页图文、交错文档、合成指令、视频/Agent |
| 训练 | 单任务微调 | 预训练、对齐、SFT、偏好与 RL |
| 输出 | 类别/短答案 | 长文本、坐标、时间、工具与动作 |

## 18. 不应得出的错误结论

1. 新模型只是参数更大：分辨率、Token、数据和训练阶段同样关键。
2. 所有系列沿同一路线演进：不同团队在连接、视觉塔和原生融合上选择不同。
3. 高 Benchmark 分数证明真实视觉可靠：可能存在污染、语言捷径和 Judge 偏差。
4. 原生多模态必然完全优于模块化：模块化仍有可替换、易调试和低成本优势。
5. 闭源模型的结构可以从 API 行为反推：行为不足以唯一确定内部结构。

## 19. 总结与复习题

```text
跨模态 BERT：区域特征与文本联合编码
CLIP：开放词汇图文表征
Flamingo/BLIP-2：冻结大模型、学习桥梁
LLaVA：视觉指令微调与开源助手
Qwen/InternVL/DeepSeek：高分辨率、OCR、视频、Grounding
Qwen3-VL：多级视觉融合与 Agent
Qwen3.5：基础预训练阶段的原生多模态融合
```

复习：双流和单流是什么？CLIP 没解决什么？Q-Former为何出现？LLaVA 的关键贡献为何不只是线性层？动态分辨率带来什么？M-RoPE 表达哪些轴？MoE 激活参数是什么意思？原生多模态怎样定义？

## 20. 参考资料

1. VisualBERT：<https://arxiv.org/abs/1908.03557>
2. ViLBERT：<https://arxiv.org/abs/1908.02265>
3. UNITER：<https://arxiv.org/abs/1909.11740>
4. CLIP：<https://arxiv.org/abs/2103.00020>
5. BLIP：<https://arxiv.org/abs/2201.12086>
6. Flamingo：<https://arxiv.org/abs/2204.14198>
7. BLIP-2：<https://arxiv.org/abs/2301.12597>
8. LLaVA：<https://arxiv.org/abs/2304.08485>
9. LLaVA-1.5：<https://arxiv.org/abs/2310.03744>
10. Qwen-VL：<https://arxiv.org/abs/2308.12966>
11. Qwen2-VL：<https://arxiv.org/abs/2409.12191>
12. Qwen2.5-VL：<https://arxiv.org/abs/2502.13923>
13. Qwen3-VL：<https://github.com/QwenLM/Qwen3-VL>
14. DeepSeek-VL2：<https://arxiv.org/abs/2412.10302>
15. Qwen3.5 官方介绍：<https://qwen.ai/blog?id=qwen3.5>

> 信息核对日期：2026-08-28。
