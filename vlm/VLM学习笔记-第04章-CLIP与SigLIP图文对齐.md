# 第 4 章：CLIP、ALIGN、SigLIP 与 SigLIP2

> 本章讲“图像与文本怎样在表征空间对齐”。这些模型通常输出 Embedding 或相似度，不等同于能够自由对话的生成式 VLM。

## 1. 知识地图

```text
图片 ─→ 图像编码器 ─→ 图像向量 ─┐
                                  ├→ 相似度 → 对齐损失
文字 ─→ 文本编码器 ─→ 文本向量 ─┘

CLIP：Batch 内 Softmax 对比学习
ALIGN：用更大规模、较少清洗的网页图文数据扩展
SigLIP：每个图文对做独立 Sigmoid 二分类
SigLIP2：加入 Caption、自蒸馏、遮蔽预测、在线数据治理等
```

## 2. 为什么需要图文对齐

传统分类器只能输出训练时固定类别。图文模型把类别写成自然语言：

```text
图片向量 vs “a photo of a cat”
图片向量 vs “a photo of a dog”
```

相似度最高的文本可作为预测，因此能够零样本迁移到新类别，也能做图文双向检索。

## 3. 双塔结构

```text
image_i → Vision Encoder → v_i → normalize
text_j  → Text Encoder   → t_j → normalize

s_ij = v_i · t_j / τ
```

`τ` 是温度，控制分布尖锐程度。向量 L2 归一化后点积等于余弦相似度。

双塔的优势是图片和文本可离线编码，检索时只需向量搜索；代价是深层跨模态交互有限。

## 4. 训练数据是什么

基本样本是一对 `(image, text)`：

```json
{"image": "dog.jpg", "text": "a dog catching a frisbee"}
```

数据来源可以是网页图片与 Alt Text、人工 Caption、OCR 文本或合成描述。数量大不等于质量高，常见问题包括弱相关、广告模板、重复、语言失衡、人物隐私和不安全内容。

## 5. CLIP 的 Batch 内对比学习

假设 Batch 有 B 个匹配图文对，得到 `B×B` 相似度矩阵：

```text
             文1   文2   文3
图片1        高    低    低
图片2        低    高    低
图片3        低    低    高
```

对角线是正样本，其余通常视为负样本。CLIP 同时计算：

- Image-to-Text Cross-Entropy：每张图找正确文字；
- Text-to-Image Cross-Entropy：每段文字找正确图片。

```text
L_CLIP = (L_i2t + L_t2i) / 2
```

### 5.1 为什么 Batch 大有帮助

Batch 越大，每个正样本可看到更多批内负样本。但也会出现 False Negative：两张不同图片可能都正确对应“a dog”，却被当作负样本。

分布式训练需要跨设备聚合向量，通信和显存随之增加。

## 6. 手算一个极小例子

假设一张猫图对两段文字的 Logit 为 `[4,1]`：

```text
P(猫文本|猫图) = exp(4)/(exp(4)+exp(1)) ≈ 0.953
```

若正确文本是第一个，损失：

```text
-log(0.953) ≈ 0.048
```

模型通过提高正确图文相似度、降低其他配对相似度来减小损失。

## 7. CLIP 怎样做零样本分类

```text
候选类别：cat, dog, car
  ↓ Prompt 模板
“a photo of a cat”
“a photo of a dog”
“a photo of a car”
  ↓ 文本编码器
类别向量
  ↓ 与图片向量比较
最高相似度类别
```

Prompt 模板会影响结果。可对多个模板的文本向量或预测做集成。零样本不是“完全没见过相关概念”，而是没有在该下游数据集上专门训练分类头。

## 8. ALIGN 的思路

ALIGN 同样是双塔图文对齐模型，关键路线是用大规模噪声网页图文对扩展训练，减少昂贵清洗。它表明模型和数据规模能够吸收部分噪声，但不意味着数据治理不重要。

需要区分：论文中的规模化结论、数据可复现性和实际开放程度。学习笔记只记录论文公开信息。

## 9. SigLIP 为什么改损失

SigLIP 对每个图文组合做独立二分类：匹配为正，不匹配为负。

设 `y_ij ∈ {+1,-1}`：

```text
L_ij = -log sigmoid(y_ij × (v_i·t_j×scale + bias))
```

再对图文组合聚合。它不要求像 CLIP 那样分别做全局 Softmax 归一化。

直观区别：

```text
CLIP：在这一批文字中，哪一个最匹配这张图？
SigLIP：这一张图与这一段文字是否匹配？
```

Sigmoid 目标有利于更灵活的分布式实现，并在论文实验中表现出较好的 Batch Size 效率；但它仍然需要负配对，不能理解为只训练正样本。

## 10. SigLIP2 改进什么

SigLIP2 不只是“把 SigLIP 放大”，而是组合多种目标：

- 基础图文 Sigmoid 对齐；
- Captioning-based Pretraining；
- 自蒸馏；
- Masked Prediction；
- 在线数据治理；
- 更多语言和去偏数据混合；
- 原生长宽比和多分辨率变体。

结果目标从全局检索进一步扩展到定位、密集预测、多语言和作为 VLM 视觉编码器的迁移。

官方论文发布 B、L、So400m 和 g 等不同规模，最大视觉编码器约 1B 参数。具体 checkpoint 的分辨率和配置应查模型卡。

## 11. CLIP 与 SigLIP 对比

| 维度 | CLIP | SigLIP |
|---|---|---|
| 核心目标 | Batch 内双向 Softmax | 图文对独立 Sigmoid |
| 正样本 | 对角匹配图文对 | 匹配对标签为正 |
| 负样本 | Batch 内其他组合 | 非匹配组合标签为负 |
| 全局归一化 | 需要 Softmax 竞争 | 不需要全局 Softmax |
| Batch 依赖 | 大 Batch 提供更多负样本 | 对 Batch 的使用方式不同 |
| 输出 | 图文 Embedding/相似度 | 图文 Embedding/相似度 |

两者都不是靠 Next-Token Prediction 生成长答案。

## 12. 图文表征对齐不等于接入 LLM

```text
CLIP 视觉空间 D_v
        ↓ Projector / Resampler
LLM 隐藏空间 D_l
        ↓ 多模态生成训练
可被 LLM 用于预测答案 Token
```

即使图像与文本在 CLIP 空间相似，生成式 LLM 仍需知道这些视觉 Token 怎样插入上下文、怎样影响下一个 Token。LLaVA 等模型还会训练连接器并做视觉指令微调。

## 13. 用作 VLM 视觉编码器

常见方式：

1. 加载预训练 CLIP/SigLIP Vision Encoder；
2. 丢弃图文检索投影头，取 Patch 特征；
3. 选择最后层、倒数层或多层特征；
4. 通过 Projector 映射至 LLM 隐藏维；
5. 用图文对和指令数据训练；
6. 视数据与资源决定冻结或解冻视觉塔。

用于检索时优秀，不保证 OCR、计数和定位一定优秀，因为全局对齐目标可能忽略细粒度信息。SigLIP2 加入密集和定位相关目标正是为改善这类能力。

## 14. 检索怎样评估

### 14.1 Recall@K

Text-to-Image Recall@K：正确图片是否出现在前 K 个检索结果。

```text
100 个查询中，正确图片有 82 次进入 Top-5
Recall@5 = 82%
```

Image-to-Text 与之反向。还可使用 Median Rank、Mean Rank 等。

### 14.2 零样本分类

报告 Top-1/Top-5 Accuracy，同时固定：Prompt 模板、类别名称、图像处理器、是否模板集成。

### 14.3 作为 VLM 骨干

不能只看 ImageNet 或检索分数，还要看 OCR、Grounding、VQA、图表和幻觉等下游表现。

## 15. 数据与训练成本

对比学习训练成本来自：

- 图像编码器前向/反向；
- 文本编码器前向/反向；
- 大 Batch 或跨设备向量通信；
- 数据解码和增强；
- 相似度矩阵。

训练数据量没有通用固定值。若从头训练通用编码器，通常需要大规模且多样的图文对；领域迁移可从公开权重开始，用更少的高质量领域对进行继续训练或微调。

## 16. 数据质量案例

```text
图片：一只黑狗在雪地奔跑
文本A：dog.jpg                 相关但信息少
文本B：一只黑狗在雪地奔跑      高质量匹配
文本C：冬季户外用品大促         网页邻近但视觉弱相关
文本D：一只白猫                 错误配对
```

海量 A/C 类数据能提供规模，但 B 类高信息密度描述对细粒度能力更有价值；D 类会直接制造错误监督。

## 17. 常见误区

1. CLIP 是图片生成模型：它主要是图文编码和相似度模型。
2. 零样本表示没见过这个概念：通常只是未在目标数据集微调。
3. 对角线以外都是真负样本：可能存在语义等价的 False Negative。
4. Batch 越大无限更好：通信、噪声和收益递减都需考虑。
5. SigLIP 不需要负样本：非匹配组合仍是负标签。
6. Embedding 对齐后可直接聊天：还缺生成式 LLM 接口和训练。
7. 检索强就必然 OCR 强：全局语义与局部文字能力不同。
8. 数据规模能替代清洗：噪声、偏见、隐私和污染仍需治理。
9. 所有模型的 Embedding 可直接混用：维度、空间和归一化不同。

## 18. 本章总结

```text
CLIP / ALIGN / SigLIP 的核心任务：
让匹配图片和文字的向量靠近，让不匹配组合分开。

CLIP 使用 Batch 内双向 Softmax；
SigLIP 使用逐图文对 Sigmoid 损失；
SigLIP2 加入生成、自监督、数据治理和多语言能力。

它们可用于检索、零样本分类和 VLM 视觉骨干，
但表征空间对齐不等于 LLM 生成空间对齐。
```

## 19. 复习题

1. 双塔结构的输入输出是什么？
2. 为什么要归一化图文向量？
3. 温度系数有什么作用？
4. CLIP 的 `B×B` 矩阵中正负样本在哪里？
5. False Negative 是什么？
6. 零样本分类怎样构造类别文本？
7. ALIGN 的规模化路线是什么？
8. SigLIP 与 CLIP 的损失有何区别？
9. SigLIP 为什么仍需要负配对？
10. SigLIP2 增加了哪些训练信号？
11. 为什么检索能力不能代表 OCR 能力？
12. 图文表征对齐与 LLM 生成对齐有什么区别？

## 20. 参考资料

1. CLIP：<https://arxiv.org/abs/2103.00020>
2. ALIGN：<https://arxiv.org/abs/2102.05918>
3. SigLIP：<https://arxiv.org/abs/2303.15343>
4. SigLIP2：<https://arxiv.org/abs/2502.14786>
5. LiT：<https://arxiv.org/abs/2111.07991>
6. OpenCLIP：<https://github.com/mlfoundations/open_clip>
7. LLaVA：<https://arxiv.org/abs/2304.08485>

> 信息核对日期：2026-08-28。
