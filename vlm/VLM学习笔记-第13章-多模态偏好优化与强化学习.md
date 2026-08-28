# 第 13 章：多模态偏好优化与强化学习

> PPO、DPO、GRPO 的数学骨架来自语言模型，但 VLM 多了视觉依赖、坐标、视频时间、图片编码成本和多模态奖励可靠性。

## 1. 全流程

```text
图片/视频 + Prompt
 → Policy 生成一个或多个回答
 → 规则/答案/Judge/环境打分
 → 优势或偏好信号
 → 更新 Policy，同时限制偏离 Reference
```

目标不只是回答流畅，还包括看对图、少幻觉、格式正确、定位准确和任务成功。

## 2. 为什么 SFT 后还要优化

SFT 模仿训练答案，但数据通常只有一个参考，且不会直接优化最终成功率。偏好/RL 可比较多个候选、用可执行结果评分，并探索 SFT 数据之外的策略。

前提是奖励可靠。错误奖励会让模型更擅长钻漏洞。

## 3. Reward Model

输入图片、Prompt 和回答，输出标量分数。可用 Chosen/Rejected 训练：

```text
L_RM = -log σ(r(chosen)-r(rejected))
```

VLM Reward Model 必须真正接收图片，否则只能评价文风。用图片替换/移除测试验证其视觉依赖性。

## 4. PPO

PPO 使用 Policy、Reference、Reward Model/规则和 Value Model。简化目标：

```text
r_t(θ)=π_θ(a_t|s_t)/π_old(a_t|s_t)
L_clip=min(r_t A_t, clip(r_t,1-ε,1+ε)A_t)
```

再加入 KL、价值损失等。优点是通用、可在线优化；代价是需要 Rollout、Value、多个模型状态，VLM 图片编码进一步增加显存。

## 5. DPO

DPO 直接用偏好对，使 Policy 相对 Reference 更偏向 Chosen：

```text
Δ = [logπ(y_w|x)-logπ_ref(y_w|x)]
  - [logπ(y_l|x)-logπ_ref(y_l|x)]
L_DPO=-log σ(βΔ)
```

其中 `x` 包含图片和 Prompt。优点是不训练显式 Reward/Value；局限是依赖离线偏好质量，不进行在线探索。

VLM DPO 要保证 Chosen/Rejected 使用同一视觉输入，且截断不能删除视觉 Token。

## 6. GRPO

同一 Prompt 采样 G 个回答：

```text
o_1...o_G → reward r_1...r_G
A_i=(r_i-mean(r))/std(r)
```

再用 clipped policy objective 和 KL 更新。它不依赖独立 Value Model，适合有可验证答案的推理任务。

若一组奖励全相同，标准化后没有有效优势；因此题目难度、采样多样性和奖励分辨率很重要。

## 7. VLM 奖励设计

### 7.1 答案奖励

数学/选择题可规范化后 Exact Match；开放回答可用规则、语义匹配或 Judge，但要防 Judge 偏差。

### 7.2 格式奖励

检查 `<think>`、JSON、坐标、工具调用格式。格式奖励不能压过正确性，否则模型会输出漂亮但错误的答案。

### 7.3 OCR 奖励

字符/词 Edit Distance、ANLS、字段精确率。需处理大小写、空格、金额和单位规范化。

### 7.4 Grounding 奖励

```text
r = IoU(pred_box, gt_box)
```

或阈值命中、点是否落在区域。先校验坐标格式和原图变换。

### 7.5 视频奖励

答案正确 + 时间区间 IoU + 事件顺序。防止模型从字幕或问题文本泄漏答案。

### 7.6 Agent 奖励

环境任务成功、步骤成本、无效动作、越权操作和安全惩罚。最终成功通常比模仿单步轨迹更重要。

## 8. 视觉依赖性奖励

对同一问题比较原图、替换图和无图回答：

```text
正确图高分；错误/无图仍自信回答 → 惩罚
```

也可要求答案引用区域证据。但避免奖励模型只学习固定拒答模板。

## 9. Outcome 与 Process Reward

- Outcome：只看最终答案；客观、便宜，但中间可能瞎猜；
- Process：逐步评价推理；信号密集，但多模态过程难标注，Judge 可能无法验证每一步视觉证据。

可组合最终可验证结果、关键中间量和格式，而不是把冗长 CoT 当作高质量。

## 10. 防 Reward Hacking

常见漏洞：重复答案提高关键词分、输出超大框提高覆盖、总是拒答规避幻觉、利用 Judge 文风偏好、读取图片中的提示注入。

措施：多奖励约束、对抗样本、隐藏测试、规则与 Judge 交叉、人审高分异常、监控长度/框面积/拒答率。

## 11. Rollout 成本

每个 Prompt 采样 G 个回答，视觉编码若重复 G 次很浪费：

```text
图片 → Vision Encoder 一次 → 缓存视觉特征
                     ├→ completion 1
                     ├→ completion 2
                     └→ completion G
```

只有视觉编码器冻结且增强固定时才能安全缓存。Policy 更新视觉塔后，缓存需失效。

## 12. 在线推理与训练解耦

Rollout Server 用 vLLM 等高吞吐生成，Trainer 做梯度更新，再同步权重。需处理 Policy Staleness、版本一致性、图片可访问性和故障恢复。

TRL 当前支持 VLM GRPO，常见 `colocate` 与 `server` 模式；具体兼容模型和参数随版本变化。

## 13. PPO/DPO/GRPO 比较

| 方法 | 数据 | Value Model | 在线采样 | VLM 适用 |
|---|---|---:|---:|---|
| PPO | Prompt+Reward | 需要 | 是 | 复杂行为/环境，成本高 |
| DPO | Chosen/Rejected | 不需要 | 否 | 事实性、风格、安全偏好 |
| GRPO | Prompt+可评分 Rollouts | 不需要独立 Value | 是 | 数学、OCR、Grounding、推理 |

## 14. GSPO

GSPO 类方法强调序列级策略比率/优化单位，以改善长回答和训练稳定性。用于 VLM 时仍需处理视觉 Prompt 固定、序列长度、奖励归因与 KL。应按所采用论文和框架的精确定义实现，不能只替换算法名称。

## 15. GAPO 命名警告

GAPO 对应多个不同论文：Group-Aware、Group Adaptive、Gradient-Adaptive、Gap-Aware 等。学习和配置时必须写：完整论文名、公式、年份、代码来源与适用任务。

若用户没有指定论文，不把 GAPO 当作唯一标准算法；放入前沿扩展。

## 16. 多模态 RL 实验配置

```yaml
model: exact VLM checkpoint
prompts: image + question
num_generations: 8
max_completion_length: ...
reward:
  correctness: 1.0
  format: 0.1
  grounding: 0.5
  hallucination: -0.5
kl_beta: ...
vision_pixels: ...
rollout_backend: ...
```

记录每项奖励分布、组内标准差、KL、长度、熵、Clip Fraction、拒答率、图片依赖性和验证集成功率。

## 17. 训练门禁

1. SFT 基线稳定；
2. Reward 与人评相关且看图；
3. 随机策略和强基线奖励排序正确；
4. 奖励不可被简单模板攻破；
5. 小规模 RL 无崩溃；
6. 目标 Benchmark 提升；
7. 幻觉、安全、语言和视觉通用能力不过度回退；
8. 人工检查高奖励异常样本。

## 18. 常见误区

1. VLM RL 只是在文本 GRPO 中加图片字段；
2. Reward Model 会自动看图；
3. 格式正确等于答案正确；
4. GRPO 完全没有 Reference/KL；
5. DPO 不需要数据质量；
6. 组内样本越多无限更好；
7. 长 CoT 就是好推理；
8. GAPO 只有一个定义；
9. RL 能补回视觉编码器丢失的像素信息。

## 19. 总结与复习

```text
DPO 学离线偏好；PPO 用 Value 做在线策略优化；
GRPO 用同 Prompt 组内相对奖励省去独立 Value。
VLM 的关键是奖励必须验证视觉证据，并控制视觉 Rollout 成本。
```

复习：DPO Δ 包含什么？GRPO 奖励全同为何无信号？怎样奖励框？如何证明 Reward 看图？视觉缓存何时失效？GAPO 为什么必须写全称？

## 20. 参考资料

1. PPO：<https://arxiv.org/abs/1707.06347>
2. DPO：<https://arxiv.org/abs/2305.18290>
3. GRPO/DeepSeekMath：<https://arxiv.org/abs/2402.03300>
4. Vision-R1：<https://arxiv.org/abs/2503.06749>
5. TRL GRPO VLM：<https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md>
6. TRL DPO VLM：<https://github.com/huggingface/trl/blob/main/docs/source/dpo_trainer.md>
7. RLAIF-V：<https://arxiv.org/abs/2405.17220>
8. DAPO：<https://arxiv.org/abs/2503.14476>

> 信息核对日期：2026-08-28。前沿算法和框架接口应锁定版本后再复现。
