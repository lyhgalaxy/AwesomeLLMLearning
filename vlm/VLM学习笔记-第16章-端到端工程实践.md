# 第 16 章：端到端工程实践

> 本章把前 15 章落地为四个逐级实验。默认先做小规模可复现版本，再扩模型与数据；命令需要结合实际 GPU 和当时软件版本调整。

## 1. 实验总路线

```text
实验1 图文检索：理解 Embedding 与对齐
  ↓
实验2 VLM LoRA SFT：理解数据、Processor 与梯度
  ↓
实验3 评估流水线：证明能力和视觉依赖
  ↓
实验4 服务化压测：控制像素、延迟、显存和成本
```

## 2. 统一实验规范

每个实验保存：目标、假设、环境、Git SHA、依赖锁、硬件、数据 Manifest、随机种子、配置、命令、日志、Checkpoint、逐样本结果、失败案例和结论。

目录建议：

```text
vlm-labs/
├── configs/
├── data/{raw,processed,splits}/
├── src/
├── tests/
├── outputs/{checkpoints,predictions,reports}/
└── README.md
```

## 3. 实验一：CLIP/SigLIP 图文检索

### 目标

理解 Image/Text Processor、Embedding 归一化、相似度矩阵和 Recall@K。

### 数据

先准备 500～5000 张可合法使用图片，每张 1～3 条描述；按原始对象去重切分。加入相似负例，如不同颜色、数量和动作。

### 流程

```text
图片批量编码 → L2 Normalize → image_embeddings.npy
文本批量编码 → L2 Normalize → text_embeddings.npy
矩阵乘法/向量库 → Top-K
```

### 评价

Text→Image 和 Image→Text Recall@1/5/10、Median Rank、按场景/属性分组错误。比较不同 Prompt、图片尺寸和 CLIP/SigLIP。

### 验收

能展示检索结果、复现指标，并解释至少 20 个错误属于视觉、文字歧义、数据或模型哪一类。

## 4. 实验二：小型 VLM LoRA SFT

### 模型选择

选择官方支持且硬件可运行的小模型，例如 2B～4B 级。先验证许可证、Transformers/TRL 兼容、显存和官方 Processor。

### 数据

构建 2k～20k 高质量图片指令样本，保留 10% 验证；覆盖描述、VQA、OCR、拒答。避免训练/测试图片近重复。

### 三组对照

```text
A 只训 Connector
B Connector + LLM LoRA
C Connector + LLM LoRA + Vision LoRA
```

固定数据、步数和评测。记录可训练参数、峰值显存、Tokens/s、Loss、OCR/VQA、无图对照和文本回归。

### 调试顺序

1. 单样本打印 Processor 结果；
2. 检查图片占位数量；
3. 解码 Labels；
4. 检查非零梯度；
5. 32 条样本过拟合；
6. 保存/加载一致；
7. 再启动完整训练。

### 验收

不仅 Loss 下降，还要目标任务提升、图片替换能改变答案、通用能力不严重回退，并说明最佳训练范围。

## 5. 实验三：VLM 评估流水线

### 测试集

至少建立：通用 VQA、OCR/文档、Grounding、幻觉诱导、安全拒答和业务真实样本，每类含正常、困难、对抗样本。

### 配置冻结

```yaml
model_revision: ...
processor_revision: ...
max_pixels: ...
max_frames: ...
prompt_template: ...
temperature: 0
max_new_tokens: ...
judge_model: ...
eval_git_sha: ...
```

### 流水线

```text
Manifest → 推理 → 原始响应 JSONL
 → 规则/IoU/OCR 指标
 → Judge（开放题）
 → 人工抽检
 → Bootstrap CI + 分组报告
 → 回归门禁
```

### 反事实集

为部分题生成无图、替换图、遮挡关键区域版本。报告原图准确率、无图准确率和视觉依赖差值。

### 验收

同配置重复结果稳定；失败不被静默跳过；报告能下钻到样本；门禁能阻止一个已知退化模型。

## 6. 实验四：服务化与性能

### 服务

使用模型官方支持的 vLLM/SGLang/Transformers 后端，提供 OpenAI-compatible 或自定义多模态 API。不要为追求统一接口牺牲 Processor 正确性。

### 压测矩阵

| 维度 | 建议档位 |
|---|---|
| 图片 | 1/4/8 张 |
| 分辨率 | 0.25MP/1MP/4MP |
| 视频 | 8/32/128 帧 |
| 文本 | 短/中/长 |
| 并发 | 1/4/16/目标峰值 |
| 精度 | BF16/FP8/INT8/INT4（按支持） |

记录预处理、视觉编码、TTFT、TPOT、吞吐、P95/P99、显存、错误率、准确率和每请求成本。

### 缓存实验

同一图片多轮问答，对比无缓存、预处理缓存、视觉特征缓存。验证模型/Processor 更新后缓存失效。

### 验收

形成容量模型：每个输入档位在目标 SLA 下的单实例并发、QPS、显存和成本，并给出限流与降级策略。

## 7. 可选实验五：多模态 GRPO

仅在 SFT 和评估稳定后进行。选可验证任务（数学、OCR 或 Grounding），为每 Prompt 采样多回答，组合正确性和低权重格式奖励。

先检查 Reward：正确答案高于错误；无图猜测不会高分；大框不会投机；组内奖励有方差。小步训练并监控 KL、熵、长度和幻觉。

## 8. 失败案例模板

```markdown
样本ID：
原始图片/实际处理尺寸：
Prompt/答案/模型输出：
预期证据区域：
错误类型：感知/定位/OCR/推理/格式/幻觉/安全
是否无图也答同样：
根因假设：
验证实验：
修复与回归结果：
```

## 9. 实验报告核心表

| Run | Trainable | Data | Max Pixels | Peak GB | tok/s | VQA | OCR | Hallucination | TTFT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | ... | ... | ... | ... | ... | ... | ... |
| A | connector | ... | ... | ... | ... | ... | ... | ... | ... |
| B | +LLM LoRA | ... | ... | ... | ... | ... | ... | ... | ... |
| C | +Vision LoRA | ... | ... | ... | ... | ... | ... | ... | ... |

不选择“所有分数平均最高”，而根据业务主指标、护栏和成本做决策。

## 10. 安全与许可清单

- 数据和模型许可证；
- 人脸、账号、文档隐私；
- 图片 URL SSRF 与解压炸弹；
- Prompt Injection；
- 高风险领域免责声明与人工复核；
- 日志图片保存策略；
- 模型输出过滤和操作权限；
- 第三方 Judge/API 数据传输。

## 11. 完成定义

```text
代码可从空环境复现
数据来源与切分可追踪
训练前后有同配置评测
视觉依赖被验证
错误样本可下钻
服务性能按输入分桶
安全和许可有记录
结论能说明收益、代价和适用范围
```

## 12. 常见误区

1. Demo 能回答就算实验完成；
2. 训练 Loss 是唯一指标；
3. 无需保存 Processor；
4. 只用随机图片划分即可防泄漏；
5. Judge 分数无需人审；
6. 平均延迟可以做容量规划；
7. INT4 精度只测文本题；
8. 环境命令不需锁版本；
9. 上传公开仓库前不检查数据和权重许可。

## 13. 总结与最终复习

四个实验分别验证表征、训练、评估和部署。最终应能从一条失败样本追溯到原始资产、Processor、视觉 Token、模型版本、输出和评分器，并用对照实验验证修复。

请完成最终问题：模型是否真的看图？提高分辨率的收益和成本是什么？哪一层适合 LoRA？量化损害了哪类能力？业务选择为何不是只看 Benchmark？

## 14. 参考资料

1. OpenCLIP：<https://github.com/mlfoundations/open_clip>
2. Transformers：<https://github.com/huggingface/transformers>
3. TRL：<https://github.com/huggingface/trl>
4. PEFT：<https://github.com/huggingface/peft>
5. LMMS-Eval：<https://github.com/EvolvingLMMs-Lab/lmms-eval>
6. vLLM：<https://docs.vllm.ai/>
7. SGLang：<https://docs.sglang.ai/>
8. MLflow：<https://mlflow.org/docs/latest/>

> 信息核对日期：2026-08-28。执行时先根据硬件和锁定版本生成具体配置。
