# 第 12 章：VLM 微调方法与开源训练框架

> 本章从工程角度回答：有限 GPU 下训练哪些模块、数据怎样喂给模型、LoRA 插在哪里，以及怎样避免“程序跑通但图片没有参与训练”。

## 1. 训练范围地图

```text
Vision Encoder ─ Connector ─ LLM
      │              │         │
冻结/部分/LoRA     全训/LoRA  冻结/LoRA/全训
```

| 方案 | 可训练量 | 适用情况 | 主要风险 |
|---|---:|---|---|
| 只训 Connector | 最少 | 初始对齐、验证数据 | 上限有限 |
| Connector+LLM LoRA | 少 | 通用领域 SFT | 视觉域适配弱 |
| 再加 Vision LoRA | 中 | 工业/医疗等视觉域 | 配置复杂 |
| 部分解冻 ViT | 中高 | 需改变高层视觉语义 | 遗忘、显存 |
| 全参数 | 最大 | 大数据大算力 | 成本和灾难性遗忘 |

## 2. LoRA 原理

冻结原权重 `W`，只学习低秩增量：

```text
W' = W + (α/r)BA
A∈R^(r×d_in), B∈R^(d_out×r), r≪d
```

训练参数减少，但前向仍需基础权重。`r` 越大容量越强、参数和显存越高；`alpha` 控制缩放，Dropout 可正则化。

## 3. LoRA 插在哪里

### LLM

常见 `q_proj/k_proj/v_proj/o_proj` 与 MLP 投影。只训 Q/V 是轻量方案，全线性层容量更强。

### Vision Encoder

可对 ViT Attention 和 MLP 投影加 LoRA。领域视觉差距大时有帮助，但数据小会破坏通用视觉。

### Connector

Connector 通常较小，可直接全量训练；对大 Q-Former/Resampler 也可 LoRA。

模块名称因架构不同，必须用 `named_modules()` 核对，不能照抄另一模型的 Target Modules。

## 4. QLoRA

QLoRA 把冻结基础权重量化（典型 4-bit），梯度更新 LoRA。它降低权重存储，不代表：

- 激活也只有 4-bit；
- Vision Encoder 自动量化；
- 长视觉序列不再 OOM；
- 训练后无需基础模型。

多模态 QLoRA 的瓶颈常转向视觉激活、长上下文和图片解码。

## 5. 数据格式

TRL 推荐视觉对话样本带 `image/images`，消息 `content` 为类型列表：

```python
{
 "images": [image],
 "messages": [
  {"role":"user", "content":[
    {"type":"image"},
    {"type":"text", "text":"图中有几个杯子？"}]},
  {"role":"assistant", "content":[
    {"type":"text", "text":"三个。"}]}
 ]
}
```

不同模型 Chat Template 会生成不同特殊 Token。应使用模型官方 Processor，不手工猜 `<image>` 展开方式。

## 6. Label Mask

```text
System/User/Image Tokens → label=-100
Assistant Answer Tokens  → 真实 label
Padding                   → label=-100
```

是否训练 System/User 取决于配方，但必须明确。调试时打印解码后的 `input_ids` 和非 `-100` Labels，检查错位。

## 7. Truncation 陷阱

若普通 `max_length` 从左/右截断，可能删掉图片 Token 或答案。TRL 官方 DPO VLM 示例建议在未验证前使用 `max_length=None`。生产训练应按 Processor 展开后的真实长度分布设计截断，而不是只看原始文字长度。

## 8. 分辨率与 Batch

按视觉 Token 数分桶，比按图片数分桶更稳定：

```text
Batch Budget = Σ text_tokens + λ Σ visual_tokens
```

设置 `min_pixels/max_pixels`、最大图片数、最大帧数，并记录被降采样比例。Gradient Accumulation 形成全局 Batch。

## 9. 训练加速

- BF16：较稳定低精度；
- FlashAttention：降低注意力 IO/显存；
- Gradient Checkpointing：用重算换激活显存；
- DeepSpeed ZeRO/FSDP：分片参数、梯度、优化器；
- CPU/NVMe Offload：省 GPU，可能严重降速；
- Packing：减少 Padding，但多模态边界更难；
- 冻结 Vision：可预计算视觉特征，但随机增强与解冻时不适用。

## 10. 框架选择

| 框架 | 强项 | 注意点 |
|---|---|---|
| Transformers | 模型/Processor 基础接口 | 训练逻辑需自行组合 |
| PEFT | LoRA/Adapter/量化适配 | 核对多模态模块名 |
| TRL | SFT/DPO/GRPO 与统一数据 | 版本变化快、模型支持需核对 |
| LLaMA-Factory | 配置化多模型训练 | 模板与支持矩阵依版本 |
| MS-SWIFT | 多模态、PEFT、RL 工程生态 | 使用官方版本文档 |
| DeepSpeed | ZeRO、大规模分布式 | 配置与通信调优复杂 |
| Accelerate | 启动与分布式抽象 | 不替代具体 Trainer |
| 官方仓库 | 最贴近模型原配方 | 依赖固定、通用性不同 |

## 11. TRL 的任务数据类型

- SFT：Language Modeling 或 Prompt-Completion；
- DPO：Prompt + Chosen + Rejected；
- Reward Model：Preference；
- GRPO：Prompt-only + Reward Functions。

视觉样本额外包含 image/images。混合纯文本与视觉数据需要核对当前 Transformers/TRL 版本支持。

## 12. DeepSpeed ZeRO

```text
ZeRO-1：分片优化器状态
ZeRO-2：再分片梯度
ZeRO-3：再分片参数
```

ZeRO-3 最省单卡常驻显存，但参数收集、LoRA 保存、生成 Rollout 和外部推理引擎集成更复杂。不能只因 OOM 就无脑选最高阶段。

## 13. 最小训练配置清单

```yaml
model: exact checkpoint
processor: exact checkpoint
trainable: connector + llm_lora
vision_lr: 0
projector_lr: ...
llm_lora_lr: ...
max_pixels: ...
max_images: ...
max_text_length: ...
precision: bf16
gradient_checkpointing: true
deepspeed/fsdp: ...
seed: ...
```

还要保存软件版本、Git SHA、数据 manifest、模板、LoRA Target、有效 Batch 和评测配置。

## 14. 训练前单元测试

1. 图片与占位符数量一致；
2. Processor 输出视觉张量和 Grid 信息正确；
3. Labels 只覆盖目标区；
4. 随机替换图片会改变 Logits；
5. Connector/LoRA 有非零梯度；
6. 冻结模块没有梯度；
7. 一小批数据能过拟合；
8. 保存后加载输出一致；
9. 坐标变换可往返；
10. 多图顺序正确。

## 15. 微调实验矩阵

固定数据与评测，比较：

```text
A 只训 Connector
B Connector + LLM LoRA
C Connector + LLM LoRA + Vision LoRA
D 部分解冻 ViT
```

报告可训练参数、峰值显存、Tokens/s、训练时长、VQA/OCR/Grounding、文本回归和图片依赖性。不能只比较训练 Loss。

## 16. 常见失败

- 训练后完全不看图：占位/Processor/梯度断开；
- 答案区不收敛：Label Mask 错；
- 保存后能力消失：LoRA/Processor 未保存；
- OOM 不稳定：动态像素或视频帧未分桶；
- Loss 很低但泛化差：模板重复/数据泄漏；
- OCR 退化：Processor 使用错误分辨率；
- DPO 报长度错：图片 Token 被 Truncate；
- ZeRO 合并失败：分片 checkpoint 处理错误。

## 17. 常见误区

1. LoRA 只适用于 LLM；
2. QLoRA 让所有内存都缩到 4-bit；
3. Target Module 可跨模型复制；
4. 框架支持 VLM 就支持该模型全部训练方式；
5. Processor 与 Tokenizer 可互换；
6. 图片数相同则 Batch 成本相同；
7. 训练跑通等于图片参与梯度；
8. Adapter 文件单独就是完整模型。

## 18. 总结与复习

```text
先用最小训练范围验证数据与梯度，再逐步解冻。
VLM 微调的关键不是一条命令，而是 Processor、模板、
图片映射、Label Mask、视觉 Token 预算和保存加载闭环。
```

复习：LoRA 增量公式是什么？Vision LoRA 何时需要？QLoRA 没减少什么？图片 Token 被截断有何后果？ZeRO-1/2/3 分别分片什么？

## 19. 参考资料

1. LoRA：<https://arxiv.org/abs/2106.09685>
2. QLoRA：<https://arxiv.org/abs/2305.14314>
3. PEFT：<https://github.com/huggingface/peft>
4. TRL：<https://github.com/huggingface/trl>
5. TRL 数据格式：<https://github.com/huggingface/trl/blob/main/docs/source/dataset_formats.md>
6. TRL VLM DPO：<https://github.com/huggingface/trl/blob/main/docs/source/dpo_trainer.md>
7. LLaMA-Factory：<https://github.com/hiyouga/LLaMA-Factory>
8. MS-SWIFT：<https://github.com/modelscope/ms-swift>
9. DeepSpeed：<https://www.deepspeed.ai/>
10. Accelerate：<https://huggingface.co/docs/accelerate/>

> 信息核对日期：2026-08-28；实际命令以锁定版本官方文档为准。
