# 第 5 章：DINO、DINOv2 与 DINOv3

> DINO 的核心不是图文对齐，而是只用图片学习通用视觉表示。本章重点解释教师—学生、自蒸馏、多视图训练，以及 DINO 与 CLIP/SigLIP 的本质区别。

## 1. 知识地图

```text
同一张无标签图片
 ├─ 全局裁剪/增强 → Teacher → 目标分布 ─┐
 └─ 全局+局部裁剪 → Student → 预测分布 ─┤→ 一致性损失
                                      │
Student 梯度更新 ─→ EMA ─→ Teacher 参数 ─┘

DINO：自蒸馏证明路线
DINOv2：数据治理、iBOT式目标与规模化
DINOv3：7B 模型、1.7B 图片、Gram Anchoring、密集特征
```

## 2. 为什么做视觉自监督学习

人工分类标签昂贵且语义有限；网页 Caption 又可能缺失、噪声大或语言偏置强。自监督学习从图片本身构造训练信号：不要求人工告诉模型“这是猫”，而要求同一图片不同视图的表示保持一致，或重建被遮住区域。

目标不是直接输出类别，而是得到可迁移的视觉骨干。

## 3. DINO 是什么

DINO 全称 Self-Distillation with No Labels。它有结构相同的 Student 和 Teacher，Teacher 不通过普通反向传播直接更新，而由 Student 参数的指数移动平均（EMA）得到。

```text
θ_teacher ← m θ_teacher + (1-m) θ_student
```

`m` 接近 1，Teacher 变化更平滑，给 Student 提供稳定目标。

## 4. Multi-Crop 多视图

同一图片产生不同增强视图：

- Global Crop：覆盖较大区域；
- Local Crop：覆盖局部区域；
- 颜色、模糊、翻转等变化。

Teacher 通常看全局视图，Student 看全局和局部视图。Student 必须让局部表示与 Teacher 的全局语义保持一致，从而学习对象级语义，而非死记像素。

## 5. DINO 损失

Teacher 和 Student 输出类别式概率分布：

```text
p_s(x)=softmax(g_s(x)/τ_s)
p_t(x)=softmax((g_t(x)-c)/τ_t)
L = - Σ p_t log p_s
```

- `τ_t` Sharpening：让 Teacher 目标更有区分度；
- `c` Centering：抑制某些维度长期占优；
- Cross-Entropy：Student 匹配 Teacher 在另一个视图上的分布。

这里的输出维度不是人工类别标签，训练产生的是内部原型/分布。

## 6. 为什么不会全部输出同一个向量

自监督一致性存在 Collapse 风险：若所有图片都输出同一个向量，表面上也一致。DINO 使用 Teacher 动量、Centering、Sharpening、多视图和训练配方避免坍塌。

不能只复制一个损失公式而忽略温度、动量、增强和优化配置；它们共同决定训练是否稳定。

## 7. DINO 学到了什么

DINO 的 ViT 注意力和 Patch 特征会自然呈现对象边界和语义组织。它既能提供全局特征，也能保留密集 Patch 表示，适合：

- k-NN/线性分类；
- 图像检索；
- 检测与分割；
- 深度等密集任务；
- 作为下游视觉骨干。

## 8. DINOv2 的重点

DINOv2 将自监督视觉学习扩展为通用视觉基础模型，重点不只是算法名字变化，还包括：

- 从大规模原始图片池中检索和治理训练数据；
- 融合图像级与 Patch 级训练信号；
- 使用 iBOT 风格的 Masked Image Modeling；
- KoLeo 等正则化以改善表示分布；
- 大模型训练和蒸馏；
- 发布多种尺寸模型和预计算特征能力。

官方论文的最大 ViT-g/14 约 1B 参数，数据集 LVD-142M 约 1.42 亿张治理后的图片。参数和数据应按具体版本核对。

## 9. 图像级与 Patch 级目标

```text
图像级：同一图片不同视图的整体表示应一致
Patch级：被遮住或对应位置的局部表示应可预测
```

图像级目标擅长全局语义；Patch 级目标更直接促进密集特征。两者结合有助于分类与检测/分割共同迁移。

## 10. DINOv3 为什么出现

继续扩大自监督模型和训练时间时，最终全局语义可能增强，但 Patch 间局部结构在长训练中退化，影响密集任务。DINOv3 引入 Gram Anchoring 来稳定密集特征，并进一步扩展数据、模型与后处理。

官方披露的关键规模：

- 最大模型约 7B 参数；
- 训练图片约 17 亿张；
- 相比 DINOv2，模型约扩大 6 倍、数据约扩大 12 倍；
- 重点增强跨自然图像、遥感等领域的通用密集特征。

## 11. Gram Anchoring 直觉

Gram Matrix 描述一组特征之间的相似关系。简化地，若 Patch 特征为 `F∈R^(N×D)`：

```text
G = FFᵀ
```

`G_ij` 表示 Patch i 与 Patch j 的关系。Gram Anchoring 用参考特征关系约束当前模型，防止长训练过程中局部结构逐渐失真。

重要：它不是让每个 Patch 特征数值完全不变，而是锚定特征之间的结构关系。

## 12. DINOv3 后处理

DINOv3 还提供面向不同分辨率、模型尺寸和文本对齐的后处理/蒸馏策略。需要区分：

- 核心自监督视觉预训练；
- 从大模型蒸馏到小模型；
- 分辨率适配；
- 训练后添加文本对齐能力。

添加文本对齐不意味着 DINO 的基础训练突然等同于 CLIP。

## 13. DINO 与 CLIP/SigLIP

| 维度 | DINO 系列 | CLIP/SigLIP 系列 |
|---|---|---|
| 主要数据 | 无标签图片 | 图片—文本对 |
| 核心监督 | 同图多视图/遮蔽预测 | 图文匹配与不匹配 |
| Teacher | EMA Teacher 常是核心 | 经典 CLIP 不依赖 EMA Teacher |
| 强项 | 通用视觉、局部/密集特征 | 图文语义、检索、零样本分类 |
| 语言能力 | 基础预训练不需要文本 | 自带文本编码器 |
| 开放词汇 | 需额外分类器/文本对齐 | 天然可与类别文本比较 |
| VLM 用途 | 强视觉特征 | 强图文语义接口 |

两者可以组合：用 DINO 提供细节、CLIP/SigLIP 提供语言对齐，或在一个视觉编码器中加入多种训练目标。

## 14. DINO 是否“需要模型”

需要。DINO 不是数据增强算法，而是一套训练视觉神经网络的方法。训练时至少有 Student 与 Teacher 两份网络状态；推理时通常只保留训练好的 Backbone。

Teacher 不是另一个人工预训练模型，也不需要人工标签，它来自 Student 的 EMA。

## 15. 数据需要多少

研究级通用骨干可用百万到十亿级图片，领域模型不应机械复制规模：

- 从头训练大 ViT：需要大量、多样、去重数据和强算力；
- 从公开 DINO 权重继续训练：可使用较小领域无标签集；
- 线性 Probe/Adapter：少量标注数据即可测试迁移；
- 医疗、工业、遥感：领域相似性与覆盖比单纯图片数更重要。

必须设置独立验证、近重复去除并监测 Collapse、特征方差、k-NN 和下游 Probe。

## 16. 怎样用于 VLM

```text
DINO Patch Features ─┐
                     ├→ 拼接/融合 → Projector → LLM
CLIP/SigLIP Features ─┘
```

挑战：DINO 特征没有天然语言坐标，需要连接器和图文任务把它们映射到 LLM；多视觉塔会增加参数、Token、显存和数据对齐难度。

## 17. 怎样评估视觉骨干

- Frozen k-NN：不训练分类器；
- Linear Probe：冻结骨干，只训练线性头；
- Fine-tuning：完整下游适配；
- 检测/分割/深度等密集任务；
- 跨领域迁移；
- 不同分辨率和鲁棒性；
- VLM 下游 OCR、Grounding、VQA。

只报告 ImageNet 分类不足以证明通用视觉能力。

## 18. 常见误区

1. DINO 是目标检测器：它是视觉表征学习方法。
2. DINO 训练完全不需要模型：恰恰需要 Student/Teacher 网络。
3. Teacher 由人工标签训练：Teacher 来自 Student EMA。
4. 无监督意味着不需要训练信号：训练信号由多视图和遮蔽任务自动构造。
5. DINO 与 CLIP 只是数据量不同：监督来源与目标本质不同。
6. DINOv3 的 7B 是 LLM 参数：它指大型视觉骨干规模。
7. 全局分类强就密集特征强：两类能力需分别评估。
8. Gram Anchoring 是固定所有特征：它约束关系结构而非逐值冻结。

## 19. 总结与复习题

```text
DINO：同一图片不同视图之间做教师—学生自蒸馏；
DINOv2：数据治理 + 图像级/patch级目标 + 规模化；
DINOv3：继续扩展到 7B/1.7B 图片，并用 Gram Anchoring
        解决长训练中的密集特征退化。
```

复习：Teacher 怎样更新？Centering/Sharpening 防什么？DINO 与 CLIP 数据和损失有何区别？图像级和 Patch 级目标分别学什么？Gram Matrix 描述什么？为什么 DINO 接 LLM 仍需图文对齐？

## 20. 参考资料

1. DINO：<https://arxiv.org/abs/2104.14294>
2. iBOT：<https://arxiv.org/abs/2111.07832>
3. DINOv2：<https://arxiv.org/abs/2304.07193>
4. DINOv2 官方仓库：<https://github.com/facebookresearch/dinov2>
5. DINOv3：<https://arxiv.org/abs/2508.10104>
6. DINOv3 官方研究页：<https://ai.meta.com/research/dinov3/>
7. MAE：<https://arxiv.org/abs/2111.06377>

> 信息核对日期：2026-08-28。
