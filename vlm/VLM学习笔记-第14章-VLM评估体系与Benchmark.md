# 第 14 章：VLM 评估体系与 Benchmark

> VLM 评估必须同时回答：看见了吗、理解了吗、推理对吗、是否忠于图片、是否安全，以及系统是否跑得动。一个总分无法回答这些问题。

## 1. 五层框架

```text
感知：对象、属性、文字、细节
  ↓
关系：位置、数量、空间、时间
  ↓
语义与推理：文档、图表、数学、知识
  ↓
行为与安全：幻觉、拒答、工具、Agent
  ↓
系统与业务：延迟、吞吐、成本、真实成功率
```

## 2. 先固定评测对象

记录模型 checkpoint、Base/Instruct/Thinking、量化、Processor、最大像素、图片数、视频帧/FPS、Prompt、Chat Template、解码参数、工具、Judge 和代码版本。

同名模型不同设置不是同一实验。Thinking 与 Instruct 也不应混在一个表中。

## 3. 任务—指标矩阵

| 任务 | 主要指标 | 必补检查 |
|---|---|---|
| 分类/选择 VQA | Accuracy | 选项顺序、语言先验 |
| 开放 VQA | EM/F1/Judge | 答案规范化、人审 |
| OCR | CER/WER/ANLS | 版面、小字、语言 |
| Caption | CIDEr/BLEU/ROUGE/Judge | 幻觉与覆盖 |
| Grounding | IoU/Acc@阈值 | 坐标格式、空目标 |
| 图表/数学 | Accuracy | 数值单位、过程验证 |
| 视频 | Accuracy/Temporal IoU | 帧采样、字幕 |
| Agent | Success Rate | 步数、安全、恢复 |
| 性能 | TTFT/TPOT/吞吐/显存 | 输入像素与并发 |

## 4. Accuracy、EM、F1

多选/分类用 Accuracy。开放短答案可规范化大小写、空格、标点、单位后计算 Exact Match；多词集合用 Precision/Recall/F1。

规范化规则必须公开，不能为了提高分数临时删除关键单位或数字格式。

## 5. OCR 与文档指标

- CER：字符编辑距离/参考字符数；
- WER：词级编辑距离；
- ANLS：基于归一化 Levenshtein 相似度，适合存在轻微 OCR 差异的问答；
- 字段级 Precision/Recall/F1；
- 表格结构相似度或执行正确率。

仅文本正确不代表版面结构正确，应另测阅读顺序、单元格和字段关系。

## 6. Grounding

边界框 IoU：

```text
IoU = area(intersection) / area(union)
```

报告 Mean IoU、IoU≥0.5 准确率、点命中率，并加入“目标不存在”样本。坐标先统一到原图，排除 Resize/Pad 逆变换错误。

## 7. Caption 指标的局限

BLEU/ROUGE 看 n-gram 重叠，CIDEr 强调与多参考 Caption 的一致性。图片可有多种正确描述，自动重叠指标可能惩罚合理改写，也可能漏掉视觉幻觉。因此需要对象级事实检查或多模态 Judge/人评。

## 8. Benchmark 能力地图

- 综合感知：MME、MMBench、SEED-Bench；
- 学科与专家推理：MMMU；
- 数学/图表：MathVista、ChartQA；
- OCR/文档：TextVQA、DocVQA、OCRBench；
- 幻觉：POPE、HallusionBench；
- 多图：BLINK、Mantis 等；
- 视频：Video-MME、EgoSchema、MVBench、Video-MMMU；
- GUI/Agent：OSWorld、AndroidWorld 等。

Benchmark 名单会变化，应通过 LMMS-Eval 当前任务列表核对；选择覆盖业务能力，而不是追求数量。

## 9. 多模态幻觉

分类记录：对象不存在、属性错、数量错、关系错、文字错、跨图片/帧混淆、外部知识冒充视觉证据。

报告：幻觉率、对象 Precision/Recall、诱导问题错误率、校准与拒答。POPE 等主要测对象存在性，不代表覆盖所有幻觉。

## 10. 图片依赖性反事实测试

```text
A 原图
B 无图
C 替换为不相关图
D 遮挡证据区域
E 只保留证据区域
```

正确模型应对证据变化敏感。定义 Image Reliance Gap，例如正确图准确率减去无图准确率；同时防止模型看到任何扰动都机械拒答。

## 11. 语言先验与 Prompt 敏感性

交换选项顺序、改写问题、改变“请简短回答”等格式，运行多模板方差。若图片不变但答案剧烈变化，说明评测不稳定或模型依赖模板。

## 12. 多图与视频

多图要交换图片顺序、加入干扰图、测试跨图引用。视频固定解码库、采样 FPS、最大帧、字幕和时间信息。

分别评估单帧可答题与真正时间题；否则视频模型可能靠关键帧或字幕得分。Video-MME 等覆盖时长和多种领域，但业务仍需自建集。

## 13. LLM-as-a-Judge

适合开放回答，但存在位置、长度、风格、自家模型和视觉盲区偏差。多模态 Judge 应接收原图；若只给 Caption，它评价的是 Caption 条件答案。

控制方法：明确 Rubric、随机交换候选顺序、匿名模型、少量多 Judge、一致性统计、人类校准集，并保存 Judge 原始理由和版本。

## 14. 人工评估

设计盲评 Pairwise，维度包括视觉正确、完整、相关、推理、引用证据、安全。至少双人标注一部分，报告一致率/Cohen's Kappa 和分歧裁决。

人评不是“随便看几条”，需要抽样、Rubric、培训和质量控制。

## 15. 安全评估

- 图片中的 Prompt Injection；
- 敏感个人信息和人脸推断；
- 医疗/工业等高风险误导；
- 色情、暴力、自伤；
- OCR 恶意指令；
- 工具越权和 GUI 误操作；
- 正常请求过度拒答。

安全率和过度拒答率需同时报告。

## 16. 鲁棒性与公平性

测试模糊、压缩、旋转、低光、遮挡、不同分辨率、字体、语言、肤色、地区和文化图像。区分性能下降是视觉预处理、数据覆盖还是语言输出问题。

## 17. 系统指标

```text
Preprocess Latency
Vision Encode Latency
TTFT：请求到首个输出 Token
TPOT：后续每 Token 时间
Throughput：tokens/s 或 requests/s
Peak GPU Memory
Cost per request / per successful task
```

按图片像素、图片数、帧数、文本长、输出长和并发分桶。只测一张小图没有生产意义。

## 18. 统计可信度

报告样本数、均值、Bootstrap 置信区间；比较模型用 Paired Bootstrap/配对检验。多任务 Macro 平均防大数据集支配；业务集按真实流量权重另算。

错误样本不能悄悄跳过。报告解码失败、API 错误和无效输出率。

## 19. 污染

用图片感知 Hash/Embedding、问题文本相似度、答案模板和来源 URL 检查训练—测试重合。动态/私有测试集、时间切分和扰动版本可降低记忆风险。

Benchmark 高分但轻微改图后崩溃，是污染或捷径的重要信号。

## 20. 业务评测集

```text
真实流量采样 → 隐私脱敏 → 能力/风险分层
 → 标准答案与证据 → 人工复核 → 冻结版本
 → 离线回归 → Shadow → A/B → 在线监控
```

覆盖正常、长尾、对抗和拒答样本。主指标用任务成功率，护栏包括严重幻觉、安全、P95 延迟和成本。

## 21. 回归门禁示例

```yaml
ocr_anls: ">= baseline - 0.005"
grounding_iou: ">= baseline"
critical_hallucination: "<= 0.5%"
text_regression: "<= 1%"
p95_ttft_2mp: "<= 2.0s"
cost_per_success: "<= budget"
```

阈值需结合置信区间和业务风险，不使用一个综合分掩盖严重回退。

## 22. LMMS-Eval 流水线

LMMS-Eval 提供图像、视频、音频等统一任务和多模型后端。使用时锁定 Git SHA、模型适配器、任务 YAML、后处理和依赖；保存逐样本输出而非只有汇总分。

## 23. 常见误区

1. 一个综合榜单分代表所有能力；
2. 选择题分数代表开放对话；
3. Judge 一定比规则客观；
4. 视频题都需要时间理解；
5. POPE 覆盖全部幻觉；
6. 推理参数不同也可直接比；
7. 失败样本可以跳过；
8. 无图仍答对说明视觉推理强；
9. 离线高分保证线上成功。

## 24. 总结与复习

```text
VLM 评估 = 能力矩阵 + 视觉依赖 + 幻觉安全 + 系统成本 + 业务成功。
固定模型、Processor、像素、Prompt、解码和评测代码，
保存逐样本结果、置信区间和失败类型。
```

复习：ANLS 适合什么？IoU 怎样计算？为何需要无图/替换图？Judge 如何校准？视频评测必须固定什么？回归门禁为何不能只用总分？

## 25. 参考资料

1. MME：<https://arxiv.org/abs/2306.13394>
2. MMBench：<https://arxiv.org/abs/2307.06281>
3. MMMU：<https://arxiv.org/abs/2311.16502>
4. MathVista：<https://arxiv.org/abs/2310.02255>
5. POPE：<https://arxiv.org/abs/2305.10355>
6. Video-MME：<https://arxiv.org/abs/2405.21075>
7. OSWorld：<https://arxiv.org/abs/2404.07972>
8. LMMS-Eval：<https://github.com/EvolvingLMMs-Lab/lmms-eval>

> 信息核对日期：2026-08-28。
