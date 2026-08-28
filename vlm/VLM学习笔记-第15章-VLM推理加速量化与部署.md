# 第 15 章：VLM 推理、加速、量化与部署

> VLM 服务比纯 LLM 多出图片解码、预处理和视觉编码，还存在动态视觉 Token。优化必须分阶段测量，不能只看输出 tokens/s。

## 1. 请求数据流

```text
上传/URL → 下载与安全检查 → 解码 → Resize/Normalize
 → Vision Encoder → Connector → 视觉 Tokens
 → LLM Prefill → Decode → 后处理/工具调用
```

端到端延迟：

```text
T_total=T_io+T_preprocess+T_vision+T_prefill+T_decode+T_post
```

## 2. 指标

- TTFT：请求到首 Token，包含前处理、视觉编码和 Prefill；
- TPOT：Decode 阶段每个 Token 时间；
- ITL：相邻 Token 延迟；
- Throughput：请求/秒、输出 Tokens/秒；
- Peak Memory、GPU 利用率、队列时间；
- 每成功任务成本。

VLM 必须按像素、图片数和帧数分桶报告。

## 3. 视觉 Token 是核心预算

```text
视觉 Tokens ≈ 图片块数 × 每块 Patch Tokens ÷ Merge Ratio
```

它影响 Vision Compute、LLM Prefill、Attention 和 KV Cache。限制 `max_pixels`、最大图片数/帧数通常比只限制文字长度更有效。

## 4. 图片预处理优化

- 限制文件大小、像素和解压炸弹；
- 使用高效解码库与线程池；
- 避免重复 Base64 编解码；
- Resize/Normalize 批处理；
- 缓存内容 Hash 对应的处理结果；
- 监控坏图、EXIF 方向和颜色空间。

缓存前处理结果需把 Processor 版本、像素配置纳入 Key。

## 5. 视觉特征缓存

适合重复询问同一图片：

```text
cache_key=hash(image_bytes, processor, model, pixel_config)
```

缓存 Vision/Connector 输出可跳过视觉前向。模型或连接器版本变更必须失效；敏感图片需加密、权限和 TTL。

## 6. Continuous Batching

请求不必等同批全部完成，Decode 时动态加入/移除序列。VLM 难点是 Prefill 成本差异大：一张缩略图与长视频不能只按请求数排队。

调度应感知预估视觉 Token、文本长度、优先级和 SLA，并防大请求阻塞小请求。

## 7. Chunked Prefill

把长 Prefill 分块，与 Decode 请求交错，提高延迟公平性。多模态实现还要正确处理视觉 Embedding 边界和位置。

## 8. KV Cache

LLM 层为上下文保存 Key/Value。视觉 Token 作为前缀也增加 KV Cache。PagedAttention 用分页方式管理非连续 KV，减少碎片并支持动态 Batch。

量化 KV Cache 可省显存，但可能影响长上下文、Grounding 和细粒度答案，需业务回归。

## 9. vLLM 与 SGLang

vLLM 提供 PagedAttention、Continuous Batching、Prefix/Prompt Cache、量化与多模态模型支持；SGLang 强调高性能结构化生成、Radix Cache、调度与多模态服务。

支持矩阵随版本快速变化。部署前核对：精确模型类、图片/视频输入、最大图片数、量化、TP、CUDA/驱动和 Processor 行为。

## 10. 量化对象

```text
Vision Encoder | Connector | LLM Weights | Activations | KV Cache
```

不要只写“模型 INT4”。需说明谁被量化、格式、Group Size、校准集和未量化层。

### Weight-only

GPTQ、AWQ 等主要压缩权重；AWQ 使用激活统计保护重要通道。适合 LLM 权重，视觉塔兼容和收益需单独验证。

### W8A8/FP8

权重与激活都低精度，吞吐潜力高，但依赖硬件 Kernel 和校准。敏感 LayerNorm、Router 或输出层可能保留 BF16。

### KV 量化

降低并发显存，长序列收益明显；但不是权重量化。

## 11. 校准数据

VLM 量化校准集需覆盖真实图片分辨率、OCR、文档、自然图、多图/视频和文本长度。只用纯文本校准可能无法覆盖视觉引起的激活分布。

比较量化前后：VQA、OCR、Grounding、幻觉、文本能力、TTFT、TPOT、峰值显存和吞吐。

## 12. 并行

- DP：多副本处理不同请求；
- TP：单模型矩阵跨 GPU；
- PP：层跨设备，在线延迟有 Pipeline Bubble；
- EP：MoE 专家分布，多 All-to-All；
- Vision/Language 分离：可把视觉编码和 LLM 部署到不同资源池。

小模型优先多副本扩吞吐；大模型才需要 TP/EP。跨节点网络可能决定性能。

## 13. 视频服务

视频下载、解码和抽帧可能比模型更慢。设置时长、文件、分辨率、帧数和总像素上限；异步预处理、镜头采样和取消机制。记录实际采样帧与时间戳，便于解释答案。

## 14. 容量规划

按流量分桶压测：单图小图、高分辨率文档、多图、短视频、长视频。对每桶测 P50/P95/P99、显存、队列、失败率与单位成本，再按业务比例加权。

容量不是 `平均QPS × 平均延迟` 的简单值，还要考虑长尾、突发和动态 Batch 干扰。

## 15. API 设计

接口明确：图片 URL/Base64/对象存储、数量与大小上限、超时、幂等 ID、模型版本、像素配置、输出 Schema、流式响应和错误码。

服务端不应任意访问内网 URL；防 SSRF、恶意文件、元数据泄漏和图片 Prompt Injection。

## 16. 监控

监控阶段延迟、视觉/文本 Token、输入像素、帧数、Batch、Cache Hit、GPU、OOM、解码错误、拒答、幻觉抽检、安全命中和成本。日志避免保存原始敏感图片，必要时用脱敏 ID。

## 17. 灰度发布

```text
离线回归 → 压测 → Shadow → 小流量 Canary → A/B → 全量
```

设置自动回滚：严重安全事件、错误率、P95、OOM、关键业务成功率越界。模型、Processor、推理引擎和 Prompt 均独立版本化。

## 18. 常见误区

1. VLM 延迟等于 LLM Decode 延迟；
2. 图片大小可用 JPEG 文件 KB 代表；
3. 视觉特征可跨模型版本缓存；
4. INT4 会让所有显存降四倍；
5. 引擎支持模型名就支持其全部多模态功能；
6. 动态 Batch 按请求数即可；
7. 平均延迟可代表生产体验；
8. 图片 URL 输入没有安全风险。

## 19. 总结与复习

```text
先拆阶段、再测瓶颈；视觉 Token 是能力与成本的共同变量。
量化要写清对象，调度要感知像素，缓存要绑定版本，
上线要同时守住正确性、安全、P95 与成本。
```

复习：TTFT 包含哪些阶段？视觉缓存 Key 有什么？PagedAttention 解决什么？为什么 VLM 校准不能只用文本？EP 的主要通信是什么？

## 20. 参考资料

1. vLLM：<https://arxiv.org/abs/2309.06180>
2. vLLM 官方文档：<https://docs.vllm.ai/>
3. SGLang：<https://arxiv.org/abs/2312.07104>
4. SGLang 文档：<https://docs.sglang.ai/>
5. FlashAttention：<https://arxiv.org/abs/2205.14135>
6. AWQ：<https://arxiv.org/abs/2306.00978>
7. GPTQ：<https://arxiv.org/abs/2210.17323>
8. SmoothQuant：<https://arxiv.org/abs/2211.10438>

> 信息核对日期：2026-08-28；部署支持以锁定版本文档为准。
