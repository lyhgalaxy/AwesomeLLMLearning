# 第 8 章：重点开源 VLM 系列解析

> 本章用同一套模板比较模型系列。数字只代表指定版本，不能把上下文、视觉塔、LLM、总参数和激活参数混为一谈。

## 1. 统一拆解框架

```text
Model = Image Processor
      + Vision Encoder
      + Connector / Fusion
      + LLM Backbone
      + Position & Token Strategy
      + Pretrain / SFT / Preference-RL Recipe
```

比较时固定十二项：输入模态、视觉编码器、连接器、LLM、分辨率、视觉 Token、位置编码、训练阶段、数据、损失、参数/许可证、评估设置。

## 2. LLaVA 系列

### 2.1 LLaVA

```text
图片 → CLIP ViT-L/14 → Linear Projector → Vicuna LLM
```

训练分两步：先用图文对齐数据训练 Projector，再用视觉指令数据训练多模态助手。视觉指令数据由强语言模型结合 Caption/框等符号信息生成。核心价值是简单结构、可复现训练和视觉指令微调范式。

### 2.2 LLaVA-1.5

主要改进：CLIP ViT-L 336px、两层 MLP Projector、增加公开 VQA/学术任务数据、统一短答案格式。论文的 13B checkpoint 使用约 1.2M 公开数据，展示简单连接器也可建立强基线。

### 2.3 LLaVA-NeXT

重点转向 AnyRes、高分辨率切块、不同 LLM Backbone、视频与更丰富任务。全局缩略图保整体，局部块保细节。代价是视觉 Token 与延迟随块数增长。

### 2.4 系列启示

- 数据质量和任务格式可比复杂连接器更重要；
- 固定低分辨率是 OCR/文档瓶颈；
- 模块化易替换，但视觉与语言往往分别预训练。

## 3. Qwen-VL 系列

### 3.1 Qwen-VL

由 ViT、跨模态适配模块和 Qwen LLM 组成，强调中英文、OCR、Grounding 和多轮对话。原论文描述三阶段训练：大规模预训练、多任务预训练、监督微调。

### 3.2 Qwen2-VL

```text
任意尺寸图片/视频
 → 约600M参数 ViT
 → Naive Dynamic Resolution
 → 2×2 Merge 等视觉 Token 处理
 → Qwen2 LLM + M-RoPE
```

M-RoPE 分配时间、高度、宽度三个轴，同时兼容文本一维位置。官方发布 2B、7B、72B 等语言规模版本；完整参数需按模型卡核对。

### 3.3 Qwen2.5-VL

- 动态分辨率 ViT 从头训练；
- 大多数视觉层使用 Window Attention，少数全局层；
- 视觉侧使用 RMSNorm、SwiGLU；
- 动态 FPS 和绝对时间对齐；
- 更强 OCR、文档/图表、框/点定位、长视频和 GUI 操作；
- 公开 3B、7B、32B、72B 等版本。

### 3.4 Qwen3-VL

- Dense 与 MoE；
- Instruct 与 Thinking；
- Interleaved-MRoPE；
- DeepStack 多级 ViT 特征注入；
- Text–Timestamp Alignment；
- GUI Agent、3D Grounding、长上下文视频能力。

公开规格包括 2B/4B/8B/32B Dense 及 30B-A3B、235B-A22B MoE 等版本；A3B/A22B 表示近似激活规模，不是总参数。

### 3.5 系列启示

Qwen-VL 从“连接视觉塔的 LLM”逐步发展到专门训练动态 ViT、多轴位置、多层融合、视频时间和 Agent，再在 Qwen3.5 转入原生多模态基础预训练。

## 4. InternVL 系列

InternVL 路线强调扩大视觉基础模型、强图文对齐和动态高分辨率。

```text
原图 → 按长宽比选择 Tile 网格
     → 多个局部 Tile + 可选 Thumbnail
     → InternViT
     → Pixel Shuffle/MLP 映射与压缩
     → InternLM/Qwen 等 LLM
```

动态切块有利于文档和小字，但 `max_num` 等块数配置直接决定显存。系列模型组合较多，参数表必须注明 InternVL 版本、视觉塔、LLM 和最大 Tile，不能只写“InternVL-XXB”。

InternVL2/2.5 等扩展多图、视频、OCR、数学与多语言，并使用逐步训练、视觉指令微调及偏好优化。训练细节以对应技术报告为准。

## 5. DeepSeek-VL 系列

### 5.1 DeepSeek-VL

使用混合视觉编码思路兼顾语义与细节，并重视 vision-language 数据与纯文本数据配比，以减少多模态训练导致语言能力退化。

### 5.2 DeepSeek-VL2

```text
高分辨率图片 → Dynamic Tiling
              → 视觉编码与投影
              → DeepSeekMoE + MLA
```

两项关键升级：动态切块视觉编码；语言侧使用 DeepSeekMoE 和 Multi-head Latent Attention。官方 Tiny/Small/完整版本激活参数约 1.0B/2.8B/4.5B，支持 VQA、OCR、文档/表格/图表和 Grounding。

MoE 减少单次激活计算，但部署仍需存放/分片更多专家权重，并处理专家通信。

## 6. Gemma 3

Gemma 3 的多模态版本将 SigLIP 视觉编码器接入 Gemma Decoder，并面向不同模型尺寸、长上下文和多语言。学习时应分别确认：某一参数规模是否支持图片、视觉编码器输入尺寸、图像 Token 数、量化和许可证。

它适合展示“小到大统一模型家族”与边缘部署，但不可把 hosted API 能力自动等同于所有开放 checkpoint。

## 7. Molmo

Molmo 强调高质量人工/半人工图像描述数据、开放权重和视觉 Grounding。其路线说明高密度、细粒度描述可以减少对纯粹超大数据量的依赖。

Pointing 能力需要专门坐标数据与评估；会生成点格式不等于定位准确。

## 8. SmolVLM

SmolVLM 系列聚焦小型、可部署视觉语言模型，使用视觉 Token 压缩、较小 LLM 与高效训练。适合学习：

- 小模型怎样控制视觉序列；
- 单 GPU/边缘设备如何微调；
- 模型缩小后 OCR、推理和幻觉怎样变化。

小模型的优势是成本和延迟，不应只用大型综合 Benchmark 判断价值。

## 9. 横向架构表

| 系列 | 典型视觉策略 | 连接/融合 | 代表优势 | 主要代价 |
|---|---|---|---|---|
| LLaVA | CLIP/SigLIP、AnyRes | Linear/MLP 拼接 | 简洁、生态成熟 | 原始模块较拼装式 |
| Qwen-VL | 专门 ViT、动态分辨率 | Merge/投影、深层融合演进 | OCR、视频、Agent、定位 | 配置和视觉 Token 较复杂 |
| InternVL | 大视觉塔、动态 Tiles | Pixel Shuffle/MLP | 高分辨率和多图 | Tile 数导致成本波动 |
| DeepSeek-VL2 | Dynamic Tiling | 投影 + MoE LLM | 少激活参数、文档/图表 | 专家权重与部署复杂 |
| Gemma 3 | SigLIP | 视觉适配到 Decoder | 多尺寸、部署生态 | 版本能力需逐个核对 |
| Molmo | 强视觉骨干 | 生成式接口 | 数据质量、Pointing | 生态与任务覆盖不同 |
| SmolVLM | 小视觉/Token 压缩 | 轻量连接 | 低资源部署 | 上限和细节能力受限 |

## 10. 训练阶段比较

不同论文名称不统一，可映射成：

```text
A. 视觉骨干/图文编码器预训练
B. Connector 或模态对齐
C. 大规模多模态联合预训练/CPT
D. 多模态 SFT
E. 偏好优化/RL
```

看到“Pretrain”要问：从头训练谁？是否冻结 LLM？数据是 Caption 还是交错文档？Loss 是对比还是 Next-Token？

## 11. 参数量怎样正确报告

```text
总参数 = Vision + Connector + LLM/Experts + 其他模块
激活参数 = 一次前向实际参与计算的近似参数
可训练参数 = 当前阶段 requires_grad 的参数
```

三者完全不同。LoRA 可训练参数很少，但推理仍需基础模型；MoE 激活少，但所有专家权重通常仍需加载或分布式存储。

## 12. 选型决策树

```text
OCR/文档？ → 高分辨率、动态 Token、OCR Benchmark
长视频？   → 帧采样、时间编码、上下文和服务限制
GUI Agent？ → Grounding、工具格式、闭环成功率
边缘部署？ → 小模型、视觉 Token、量化、首 Token 延迟
训练研究？ → 官方训练代码、Base checkpoint、数据格式
商业使用？ → 许可证、隐私、内容安全、模型卡
```

## 13. 常见误区

1. 名称数字就是完整精确参数；
2. 激活参数等于显存只需这么多；
3. 支持任意分辨率等于没有像素上限；
4. Benchmark 分数可跨不同 Prompt 直接比较；
5. Instruct 与 Thinking 只是解码开关；
6. 动态切块必然保留所有细节；
7. 官方 API 与开源 checkpoint 能力完全相同；
8. 连接器简单说明训练简单；
9. 开放权重等于训练数据和训练代码全开放。

## 14. 总结与复习

模型系列的真正区别不在名称，而在视觉骨干、Token 预算、融合位置、训练阶段和数据。请任选两个系列，用统一十二项模板比较，并解释它们在 OCR、视频和边缘部署上的取舍。

## 15. 参考资料

1. LLaVA：<https://arxiv.org/abs/2304.08485>
2. LLaVA-1.5：<https://arxiv.org/abs/2310.03744>
3. LLaVA-NeXT：<https://llava-vl.github.io/blog/2024-01-30-llava-next/>
4. Qwen-VL：<https://arxiv.org/abs/2308.12966>
5. Qwen2-VL：<https://arxiv.org/abs/2409.12191>
6. Qwen2.5-VL：<https://arxiv.org/abs/2502.13923>
7. Qwen3-VL：<https://github.com/QwenLM/Qwen3-VL>
8. InternVL：<https://github.com/OpenGVLab/InternVL>
9. DeepSeek-VL：<https://github.com/deepseek-ai/DeepSeek-VL>
10. DeepSeek-VL2：<https://arxiv.org/abs/2412.10302>
11. Gemma 3：<https://ai.google.dev/gemma/docs/core/model_card_3>
12. Molmo：<https://arxiv.org/abs/2409.17146>
13. SmolVLM：<https://huggingface.co/blog/smolvlm>

> 信息核对日期：2026-08-28。
