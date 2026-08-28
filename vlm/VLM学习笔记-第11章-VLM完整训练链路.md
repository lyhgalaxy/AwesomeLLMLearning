# 第 11 章：VLM 完整训练链路

> 同一个“Pretrain”在不同论文中可能指完全不同阶段。本章用统一坐标系说明每阶段训练谁、吃什么数据、优化什么损失、获得什么能力。

## 1. 总流程

```text
A 视觉骨干预训练
  ↓
B 图文表征/连接器对齐
  ↓
C 大规模多模态联合预训练
  ↓
D 领域持续预训练 CPT
  ↓
E 多模态监督微调 SFT
  ↓
F 偏好优化 / 强化学习
```

模型可以跳过或合并阶段。判断时不要看名字，要看初始化、可训练模块、数据和 Loss。

## 2. 阶段 A：视觉编码器预训练

目标是把像素变成通用视觉特征。路线包括：

| 数据 | 目标 | 代表 |
|---|---|---|
| 图片+类别 | Cross-Entropy | 监督 ViT |
| 图片+文本 | 对比/Sigmoid | CLIP/SigLIP |
| 无标签图片 | 自蒸馏/遮蔽预测 | DINO/MAE |

输出通常是预训练 ViT。此阶段可能与最终 LLM 无关，也可能在原生多模态模型中联合发生。

## 3. 阶段 B：连接器对齐

典型 LLaVA 式配置：

```text
Vision Encoder：冻结
Projector：训练
LLM：冻结
数据：图片—Caption
损失：答案 Caption 的 Next-Token Cross-Entropy
```

梯度只更新 Projector，让视觉特征映射到 LLM 可利用空间。若数据只含短 Caption，得到的是基础看图描述能力，不等于复杂指令跟随。

BLIP-2 则用 Q-Former 两阶段分别做表征学习和视觉到语言生成，说明“对齐阶段”内部也可能有多个目标。

## 4. 阶段 C：多模态联合预训练

数据可混合图文对、交错文档、OCR、视频和纯文本。模块可能全部或部分解冻。

```text
多模态 Batch Scheduler
  ├─ 纯文本 → 保持语言能力
  ├─ 图文对 → 视觉语义
  ├─ 交错文档 → 多图长上下文
  ├─ OCR/文档 → 细粒度文字
  └─ 视频 → 时间理解
```

核心通常仍是自回归 Next-Token Loss，也可叠加对比、匹配、定位和辅助视觉损失。

## 5. 阶段 D：CPT

CPT（Continual Pre-Training）是在已有模型上用领域或新分布数据继续预训练。

VLM CPT 示例：工业缺陷图片+报告、医疗影像+描述、企业文档、特定 GUI 截图轨迹。它主要学习领域概念和输入分布，数据不必全部是指令问答。

风险：领域过窄导致通用视觉/语言遗忘；解决方式包括混入通用数据、较小学习率、冻结部分模块和持续通用回归评估。

## 6. 阶段 E：SFT

样本是图片/视频、用户指令和理想答案。训练目标：

```text
L_SFT = -Σ_{t∈assistant} log p(y_t | image, prompt, y_<t)
```

通常只对 Assistant Token 计算 Loss；若错误地训练用户问题和图片占位，会浪费监督或破坏模板。

SFT 教会格式、任务和交互风格，但不会自动修复视觉塔无法读取的小字。

## 7. 阶段 F：偏好与 RL

- DPO：图片+Prompt+Chosen/Rejected；
- PPO：策略生成、奖励模型打分、价值估计和约束更新；
- GRPO：同一视觉 Prompt 采样一组答案，用组内相对奖励；
- 可验证奖励：答案、坐标、OCR、格式、工具执行结果。

详见第 13 章。

## 8. 损失函数地图

| 损失 | 输入监督 | 学到什么 |
|---|---|---|
| 分类 CE | 图片类别 | 固定视觉类别 |
| InfoNCE/CLIP | 匹配图文对 | 共享表征空间 |
| Sigmoid Pair Loss | 图文正负标签 | 独立配对概率 |
| ITM | 图文是否匹配 | 跨模态匹配 |
| MLM/MIM | 遮蔽 Token/Patch | 上下文与视觉结构 |
| Next-Token CE | 图片+前文+答案 | 视觉条件生成 |
| Coordinate CE | 区域描述+坐标 Token | Grounding |
| DPO Loss | Chosen/Rejected | 偏好相对概率 |
| Policy Gradient | Rollout+Reward | 直接优化期望奖励 |

## 9. Next-Token Loss 怎样训练视觉能力

```text
图片中真实文字“128.50”
问题：“总额是多少？”
答案 Token：“128.50 元”
```

答案预测错误产生梯度，经 LLM、Projector 回传到可训练视觉层。只要视觉证据对减小 Loss 有用，网络就会学习读取它；但若语言先验可轻松猜中，模型可能不学视觉，因此需反事实和困难数据。

## 10. 冻结/解冻课程

一种稳健路线：

```text
Step1 只训新 Projector，稳定尺度
Step2 解冻 LLM LoRA，学习任务
Step3 以更小 LR 解冻 ViT 高层，适配领域
Step4 必要时全参数联合，持续回归通用能力
```

不是硬规则。原生预训练可能从早期联合更新；小数据直接全解冻更容易过拟合。

## 11. 分组学习率

常见相对关系：

```text
LR_projector > LR_LLM > LR_vision
```

新连接器可较快学习，预训练骨干用更小 LR 防遗忘。还需 Warmup、Gradient Clipping、Weight Decay 与不同参数组规则。

## 12. Batch 的真实单位

多模态 Batch 不能只报告样本数：图片分辨率和视频帧数会让 Token 差异巨大。应报告：每设备样本、梯度累积、全局 Batch、文本 Token、视觉 Token、最大像素、帧数和有效 Token 吞吐。

## 13. Packing 与 Padding

- Padding：补齐最长序列，简单但浪费；
- Packing：多个短样本放入同一序列，提高利用率；
- 多模态 Packing 必须维护图片与占位符对应、Attention Mask、Position ID 和 Label Mask。

错误 Packing 会让样本互相看见或图片错配。

## 14. 训练内存

```text
权重 + 梯度 + 优化器状态 + 激活 + 临时通信 + 视觉输入
```

缓解方法：BF16/FP8、Gradient Checkpointing、FlashAttention、ZeRO/FSDP、LoRA/QLoRA、视觉 Token 上限、Sequence Parallel。

QLoRA 降低基础权重存储，但激活和视觉长序列仍可能是瓶颈。

## 15. 数据课程与混合

可由简单到复杂：清晰 Caption → VQA/OCR → 多图/视频 → 推理/Agent。也可始终混合，防能力遗忘。

调度器应按 Token/算力采样，并监控每类 Loss、梯度和验证指标，避免大图片任务独占 GPU 时间。

## 16. 训练监控

至少记录：总/分任务 Loss、学习率、梯度范数、视觉 Token 数、文本 Token 数、吞吐、显存、坏样本率、NaN、各模块参数/梯度范数，以及固定小评测集。

Loss 下降不代表看图能力变强，必须运行图片替换/移除对照、OCR、Grounding 和通用文本回归。

## 17. 常见失败

| 失败 | 可能原因 | 排查 |
|---|---|---|
| 不看图 | 语言捷径、视觉 Mask 错 | 图片替换测试、检查梯度 |
| 输出乱码 | Template/Label Shift 错 | 查看 Token 与 Label |
| OCR 差 | 分辨率/压缩不足 | 可视化实际输入 |
| 坐标错 | Resize 后标签未变 | 坐标往返单测 |
| 语言退化 | 视觉数据过多/全解冻 | 混纯文本、降 LR |
| Loss NaN | FP16、长序列、异常图 | BF16、Clip、过滤 |
| OOM 波动 | 动态分辨率/视频帧 | 按视觉 Token 分桶 |
| 多图混淆 | 占位顺序错误 | 保存 Processor 展开结果 |

## 18. 训练验收门禁

每阶段保存独立 checkpoint，并比较：通用文本、Caption/VQA、OCR、Grounding、多图/视频、幻觉、安全、显存和速度。只有目标能力提升且关键回归不过阈值才进入下一阶段。

## 19. 常见误区

1. 所有论文的 Pretrain 是同一阶段；
2. 冻结模块就不占显存；
3. SFT 可以补回被 Resize 丢掉的信息；
4. 只要 Loss 降就说明图文对齐成功；
5. 全解冻必然更好；
6. LoRA 只需考虑 LLM；
7. 样本 Batch 相同就计算相同；
8. CPT 必须是问答格式；
9. RL 可以跳过基本 SFT 和奖励验证。

## 20. 总结与复习

```text
视觉预训练学像素表示；连接器对齐学接口；
联合预训练学广泛多模态分布；CPT 学领域；
SFT 学任务和指令；偏好/RL 优化行为与结果。
```

复习：如何判定一个 Pretrain 阶段？Next-Token Loss 怎样更新视觉塔？为何分组 LR？动态分辨率如何影响 Batch？怎样证明模型真的看图？

## 21. 参考资料

1. CLIP：<https://arxiv.org/abs/2103.00020>
2. BLIP-2：<https://arxiv.org/abs/2301.12597>
3. LLaVA：<https://arxiv.org/abs/2304.08485>
4. Qwen-VL：<https://arxiv.org/abs/2308.12966>
5. VILA：<https://arxiv.org/abs/2312.07533>
6. DeepSpeed ZeRO：<https://arxiv.org/abs/1910.02054>
7. LoRA：<https://arxiv.org/abs/2106.09685>
8. QLoRA：<https://arxiv.org/abs/2305.14314>

> 信息核对日期：2026-08-28。
