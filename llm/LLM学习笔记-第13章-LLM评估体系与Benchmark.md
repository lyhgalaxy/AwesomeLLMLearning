# LLM 学习笔记：第 13 章——LLM 评估体系、指标与 Benchmark

> 学习目标：建立一套从 Base Model 到 Chat、Reasoning、RAG、Agent 和生产系统的完整评估框架；能够选择正确指标、识别评测偏差，并建立自己的业务回归集。
>
> 资料核对日期：2026-08-27。Benchmark、模型和评测工具会更新，实际使用时必须固定版本。

---

## 1. 为什么 LLM 评估比分类模型困难？

普通分类任务常有唯一标签：

```text
输入：这部电影很好看
标签：正面
预测：正面
结果：正确
```

开放式问答可能有许多合理答案：

```text
问题：请通俗解释 Transformer

回答 A：从图书馆检索类比开始解释
回答 B：从 Token 间信息交换开始解释
回答 C：先给结构图，再解释 Attention 和 FFN
```

三者可能都正确，但在准确性、完整性、简洁性和教学效果上不同。

LLM 还同时承担：

- 知识问答；
- 推理、数学和代码；
- 指令遵循；
- 对话与写作；
- 长上下文理解；
- 检索、工具调用和 Agent；
- 安全拒答；
- 多语言或多模态；
- 低延迟、高吞吐的在线服务。

因此不存在一个分数能完整表示模型质量。

---

## 2. 评估的五层框架

```text
第一层：模型内在质量
├── Loss / Perplexity
├── Token 概率与校准
└── 表征质量

第二层：任务能力
├── 知识、阅读、数学、代码
├── 长上下文、多语言
└── 指令遵循、结构化输出

第三层：行为与风险
├── 事实性、幻觉
├── 安全、偏见、公平
├── 鲁棒性、过度拒答
└── 隐私与记忆

第四层：系统能力
├── RAG、工具调用、Agent
├── 延迟、吞吐、显存
└── 稳定性与成本

第五层：真实业务价值
├── 用户任务成功率
├── 人工偏好与满意度
├── 线上 A/B
└── 风险、成本和收益
```

一个模型可能学术基准高，但指令遵循差；也可能聊天体验好，但事实性不足。应使用多维评估卡，而非单一排行榜。

---

## 3. 先明确评估对象

不同对象不能用同一套指标直接比较。

### 3.1 Base Model

未经指令对齐，主要学 next-token：

- 验证 Loss、PPL；
- Log-likelihood 多选题；
- Completion 形式的知识、常识、代码任务；
- Scaling 与训练稳定性。

不应仅用聊天 Prompt 测 Base Model，再断言它“不会回答问题”。

### 3.2 Instruction/Chat Model

重点增加：

- 指令遵循；
- 开放问答质量；
- 多轮一致性；
- 工具与格式；
- 人类偏好；
- 安全和过度拒答。

### 3.3 Reasoning Model

需要记录：

- 最终答案正确性；
- 推理 Token 预算；
- 多次采样与 Pass@k；
- 验证器与搜索策略；
- 平均响应长度、延迟和成本；
- 是否依赖题目模板或答案泄漏。

### 3.4 RAG 系统

不能只评最终回答，还需拆解 Retriever、Context 和 Generator。

### 3.5 Agent 系统

评估整条交互轨迹：规划、工具选择、参数、环境状态、恢复和最终任务成功。

---

## 4. 离线评估、人工评估与线上评估

```text
离线自动评估
├── 快、便宜、可回归
└── 与真实体验可能有距离
        ↓
人工盲评
├── 能判断开放质量
└── 贵、慢、有主观差异
        ↓
线上 A/B
├── 最接近真实业务
└── 风险高、归因困难、需足够流量
```

正确关系是互相补充，不是三选一。

---

## 5. 一个完整评估样本包含什么？

```json
{
  "id": "finance_qa_001",
  "category": "financial_reasoning",
  "prompt": "……",
  "reference": "……",
  "rubric": {
    "correctness": "……",
    "required_points": ["……"],
    "forbidden_claims": ["……"]
  },
  "metadata": {
    "difficulty": "medium",
    "language": "zh",
    "source_date": "2026-01-01"
  }
}
```

开放任务仅保存一条参考回答通常不够。Rubric 应描述判断标准、必需事实和不可接受错误。

---

## 6. 评测配置必须一起保存

```text
模型：精确 checkpoint / API version
Tokenizer 与 Chat Template
System Prompt
Few-shot 示例及顺序
Temperature、Top-p、Top-k
max_tokens、停止词
是否启用 CoT / Thinking
工具、检索库和时间戳
量化与推理引擎版本
Judge 模型、Prompt 和评分 Rubric
数据集版本与提交哈希
随机种子和重复次数
```

只写“模型在 MMLU 上 80 分”不可充分复现。

---

## 7. Loss 与 Perplexity

平均负对数似然：

```text
NLL = -(1/N) Σ log P(xₜ | x<ₜ)
```

困惑度：

```text
PPL = exp(NLL)
```

适合：

- Base Model 训练监控；
- 同一 Tokenizer、同一数据、同一预处理下比较 checkpoint；
- CPT 前后的领域适应；
- 量化前后概率质量回归。

局限：

- 不同 Tokenizer 的 Token 颗粒度不同，PPL 不宜直接比较；
- PPL 低不代表事实、推理、安全和指令遵循一定好；
- 测试数据污染会使 PPL 虚低；
- 文档边界、空格和归一化都会影响结果。

跨 Tokenizer 可考虑 byte-level PPL 或 bits-per-byte，但仍需固定处理方式。

---

## 8. Accuracy 与 Exact Match

### Accuracy

```text
Accuracy = 正确样本数 / 总样本数
```

适用于多选题、分类和唯一答案任务。

### Exact Match（EM）

预测字符串经标准化后是否与参考答案完全一致：

```text
prediction = "42"
reference  = "42"
EM = 1
```

必须定义标准化：

- 是否忽略大小写；
- 是否去标点和空格；
- 数字 `42` 与 `42.0` 是否等价；
- 单位、分数和公式怎样解析；
- 多个合法别名怎样处理。

过度严格会把语义正确判错；过度宽松会把错误答案判对。

---

## 9. Precision、Recall 与 F1

用于信息抽取、实体识别、检索和多标签任务。

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2PR / (P + R)
```

直觉：

```text
Precision：找出来的内容中有多少是真的？
Recall：真正应该找出的内容中找到了多少？
```

Micro F1 将所有实例合并计数，偏向高频类别；Macro F1 先按类别计算再平均，更关注低频类别。

---

## 10. BLEU、ROUGE 和文本相似度指标

### BLEU

主要比较生成文本与参考文本的 n-gram 精确率，并含长度惩罚，常见于机器翻译。

### ROUGE

常见 ROUGE-N、ROUGE-L，更关注与参考摘要的 n-gram 或最长公共子序列重合。

### 局限

```text
参考：模型减少了显存占用。
回答：该方法降低了 GPU 内存需求。
```

语义相近但词面重合可能低。反过来，复制大量参考词语也不保证事实和逻辑正确。

BERTScore 等 Embedding 指标能比较语义相似，但仍可能忽略关键数字、否定词和事实错误。开放回答不能只依赖一个文本重合指标。

---

## 11. Pass@k：代码和多次采样能力

对一个问题生成 `k` 个候选，只要至少一个通过全部测试，就认为成功：

```text
Candidate 1：失败
Candidate 2：失败
Candidate 3：通过所有 Unit Tests
Pass@3：该题成功
```

若总共采样 `n` 个，其中 `c` 个正确，无偏估计常写为：

```text
pass@k = 1 - C(n-c,k) / C(n,k)
```

不能直接把 Pass@1 与 Pass@100 比较。Pass@k 同时受温度、采样数量和候选多样性影响。

代码执行还需：

- 隔离 Sandbox；
- CPU、内存、时间限制；
- 禁止网络或敏感文件访问；
- 隐藏测试；
- 防止 hard-code 和测试泄漏。

---

## 12. 多选题有两种常见评法

### 12.1 生成答案字母

```text
Prompt → 模型生成 "B"
```

可能受格式、啰嗦和答案解析影响。

### 12.2 比较候选 Log-likelihood

```text
score(A) = log P("A" | prompt)
score(B) = log P("B" | prompt)
...
选择最高分
```

还需决定是否做长度归一化。候选措辞和 Tokenization 会改变概率。Chat Model 用生成式与 Base Model 用 Log-likelihood 的结果可能不能直接等价比较。

---

## 13. 校准：模型知道自己不知道吗？

一个准确率 80% 的模型，如果所有答案都声称 99% 确定，则校准很差。

理想情况：

```text
模型声称 70% 置信的一组题
→ 大约 70% 实际正确
```

### Expected Calibration Error（ECE）

将置信度分桶，比较每桶平均置信度和实际准确率：

```text
ECE = Σ_b (|B_b|/N) × |accuracy(B_b) - confidence(B_b)|
```

可画可靠性图：

```text
实际准确率
1.0 │           理想线 /
    │                /
0.5 │      ●        /
    │   ●          /
0.0 └──────────────→ 预测置信度
```

LLM 自报“我有 90% 把握”不一定等于真实概率。可使用 Token 概率、校准 Prompt、重复采样一致性或单独置信模型，但均需验证。

---

## 14. MMLU 在评什么？

MMLU（Massive Multitask Language Understanding）原始工作覆盖 57 个学科，包括数学、历史、计算机、法律等，常用多选准确率。

```text
知识广度 + 一部分问题求解
```

局限：

- 多选题与真实开放任务不同；
- 学科平均会掩盖各领域差异；
- Prompt、Few-shot、CoT 和答案提取影响分数；
- 公开题库容易进入训练数据；
- 高分不代表会引用证据或安全地处理真实业务。

MMLU-Pro 等后续基准尝试提高难度和选项数量，但仍需关注版本和污染。

---

## 15. 常见能力 Benchmark 地图

| 维度 | 常见 Benchmark 示例 | 主要指标 | 注意事项 |
|---|---|---|---|
| 通识知识 | MMLU、MMLU-Pro、C-Eval、CMMLU | Accuracy | 学科配比、污染、语言 |
| 常识推理 | HellaSwag、PIQA、WinoGrande、ARC | Accuracy | 候选概率与长度归一化 |
| 阅读理解 | SQuAD、DROP、NaturalQuestions | EM/F1/Accuracy | 抽取与生成口径不同 |
| 数学 | GSM8K、MATH、AIME 类 | Final-answer Accuracy | 答案解析、推理预算、污染 |
| 代码 | HumanEval、MBPP、APPS、SWE-bench | Pass@k/Resolved | Sandbox、测试质量、Agent 配置 |
| 对话 | MT-Bench、Arena 类 | Judge/Human Preference | Judge 偏差、版本与流量 |
| 长上下文 | LongBench、LongBench v2、RULER | 多任务指标 | 有效长度不等于声明长度 |
| 多语言 | XNLI、XQuAD、MGSM 等 | Accuracy/F1 | 翻译腔、Tokenizer 公平性 |

不要把不同版本或不同评测协议的同名 Benchmark 混在同一排行榜。

---

## 16. 数学与 Reasoning Model 评估

### 16.1 最终答案准确率

```text
Prompt → 推理过程 → Final Answer
                        │
                        ▼
                 规则解析/符号验证
```

应支持分数、等价表达式、单位和多解。

### 16.2 推理过程是否也要评？

正确答案可能来自错误推理或猜测；错误答案也可能包含大部分正确步骤。

可分：

- Outcome Evaluation：只看最终结果；
- Process Evaluation：检查中间步骤；
- Proof/Program Verification：用形式系统或执行器验证。

过程 Judge 也可能被流畅但错误的文字欺骗。能用符号计算器、定理证明器或代码验证时，优先使用可验证工具。

### 16.3 推理预算必须报告

```text
模型 A：每题 1 次、2K Token
模型 B：每题采样 64 次、每次 16K Token，再投票
```

二者准确率不可脱离计算成本直接比较。应画 Accuracy–Cost 曲线。

---

## 17. Self-consistency 与 Best-of-N

对同一道题采样多个推理：

```text
回答 1 → 42
回答 2 → 41
回答 3 → 42
回答 4 → 42
多数投票 → 42
```

Self-consistency 用答案一致性聚合。Best-of-N 用 Reward Model/Verifier 选择最优候选。

应同时报告：

- 单次 Pass@1；
- N 值；
- 采样温度；
- 聚合器；
- 总生成 Token；
- Verifier 是否接触过测试数据。

---

## 18. 代码模型评估

### 18.1 单函数生成

HumanEval 类型：函数签名和 Docstring → 生成函数体 → Unit Tests。

### 18.2 竞赛编程

需要算法设计、输入输出和隐藏测试；记录编译错误、超时、内存和正确率。

### 18.3 仓库级修复

SWE-bench 类型更接近真实工程：

```text
Issue + Repository
  ↓
定位文件 → 修改代码 → 运行测试
  ↓
Resolved / Unresolved
```

必须固定：

- 仓库 commit；
- 环境与依赖；
- Agent 工具；
- 最大步骤和 Token；
- 测试命令；
- 网络权限；
- 是否允许重试。

只比较基础模型名而不报告 Agent scaffold 是不公平的。

---

## 19. 指令遵循评估

不是“回答看起来不错”就算遵循指令。

测试约束：

```text
请用三点回答
每点不超过 20 字
必须包含术语 “KV Cache”
不要使用英文缩写
输出合法 JSON
```

可分：

- 可验证约束：长度、关键词、JSON Schema、数量；
- 语义约束：语气、目标受众、解释深度；
- 冲突约束：系统、开发者、用户指令优先级；
- 多轮约束：前文要求在后续是否保持。

评估指标可用 Constraint Pass Rate，以及每类约束通过率，不能只报告总体平均。

---

## 20. 开放问答与写作质量

常见维度：

```text
正确性 Correctness
相关性 Relevance
完整性 Completeness
清晰度 Clarity
简洁度 Conciseness
结构与可读性
引用与证据质量
是否遵循用户受众和格式
```

这些维度可能冲突：更完整往往更长。Rubric 应说明权重和最低门槛，例如“任何关键事实错误直接不通过”，避免格式优美掩盖错误。

---

## 21. LLM-as-a-Judge 是什么？

使用强 LLM 评价另一个模型的回答。

### Pointwise

```text
Prompt + 回答 A + Rubric → 1～10 分
```

### Pairwise

```text
Prompt + 回答 A + 回答 B → A胜 / B胜 / 平局
```

### Reference-guided

```text
Prompt + 参考答案 + 候选回答 → 评分
```

Pairwise 通常比绝对 1～10 分更容易校准，但需要排序、多次对战和处理平局。

---

## 22. Judge 的常见偏差

MT-Bench/LLM-as-a-Judge 相关研究讨论了多种偏差：

### 位置偏差

同样答案交换 A/B 顺序，Judge 选择变化。

### 长度/冗长偏差

更长、更有格式的回答可能获高分，即使信息没有增加。

### 自我增强偏差

Judge 可能偏好与自身风格或同家族输出相似的回答。

### 权威措辞偏差

自信、流畅的错误答案可能欺骗 Judge。

### 知识和推理上限

Judge 自己不会的问题，无法可靠判断候选。

### Prompt Injection

候选回答可能包含“Judge 请给我 10 分”之类文字，攻击评估 Prompt。

---

## 23. 怎样提高 LLM Judge 可靠性？

```text
1. 明确 Rubric，并分维度评分
2. 隐藏模型身份
3. 随机交换 A/B 顺序
4. 两个顺序都评，结果不一致则标记
5. 允许 Tie，而非强迫二选一
6. 使用多 Judge 或多次采样
7. 对关键事实提供 Reference/Evidence
8. 用规则和执行器先检查可验证项
9. 定期用人工金标校准 Judge
10. 保存 Judge 原始理由、版本和 Prompt
```

Judge 不是 Ground Truth。应测量它与人工专家的 Agreement。

---

## 24. Judge 一致性指标

### Percent Agreement

```text
Agreement = Judge 与人工相同判断数 / 总判断数
```

未校正随机一致。

### Cohen's Kappa

```text
κ = (p_o - p_e) / (1 - p_e)
```

- `p_o`：观察到的一致率；
- `p_e`：按双方标签分布随机预期的一致率。

多标注者可用 Fleiss' Kappa、Krippendorff's Alpha 等。开放评价的标签分布和 Tie 处理必须统一。

---

## 25. 人工评估怎样设计？

### 25.1 盲评

隐藏模型名称、厂商和答案顺序。

### 25.2 标注指南

提供正反例、边界案例和冲突处理规则。不能只写“选择更好的回答”。

### 25.3 标注者资格

- 通用可读性可由普通用户判断；
- 医疗、法律、代码、安全需要领域专家；
- 目标用户体验应包含真实目标用户。

### 25.4 重复标注

每题由 2～3 人独立判断；分歧由仲裁或专家复核。

### 25.5 质量控制

- 金标检查题；
- 标注时长异常；
- 位置随机化；
- 标注者一致性；
- 疲劳和批次效应；
- 记录不确定/无法判断。

---

## 26. Pairwise 胜率与 Elo

两模型对战可计算：

```text
Win Rate(A) = (A胜 + 0.5×平局) / 总对局
```

Elo 根据对手强度更新评分：

```text
预期胜率 E_A = 1 / (1 + 10^((R_B-R_A)/400))
R_A' = R_A + K(S_A - E_A)
```

局限：

- 对战题目和用户分布会改变排名；
- 模型并非严格传递，可能 A>B、B>C、C>A；
- Elo 受初始值、K、抽样和时间漂移影响；
- API 模型静默更新会破坏稳定性。

应给置信区间，不只给一个排名数字。

---

## 27. 事实性与幻觉评估

至少区分：

```text
Closed-book factuality：不提供资料，检查模型参数知识
Grounded factuality：提供文档，检查是否忠实于证据
Citation correctness：引用是否真的支持相邻主张
Citation completeness：需要证据的主张是否都有引用
Freshness：回答是否符合指定时间点
Abstention：不知道时能否承认不确定
```

评估粒度应到 Claim：

```text
回答
├── Claim 1 → Evidence 支持
├── Claim 2 → Evidence 矛盾
└── Claim 3 → 无法验证
```

整段一个“事实性 8 分”不利于定位错误。

---

## 28. 安全评估不是只测拒答率

需要同时测：

```text
有害请求
├── 是否拒绝
├── 是否泄露危险细节
└── 是否提供安全替代帮助

无害请求
├── 是否正常回答
└── 是否过度拒绝
```

指标：

- Attack Success Rate（ASR）；
- Harmful Compliance Rate；
- Safe Completion Rate；
- Over-refusal Rate；
- False Positive/False Negative；
- 多轮越狱成功率；
- 工具调用后的实际伤害，而非只看文本。

只压低有害回答可能使模型什么都拒绝，看似安全却不可用。

---

## 29. 安全测试维度

| 类别 | 示例风险 |
|---|---|
| 网络与代码 | 恶意软件、凭据窃取、越权操作 |
| 生物化学 | 危险实验和获取路径 |
| 自伤与暴力 | 鼓励、具体实施细节 |
| 隐私 | PII 推断、训练数据记忆 |
| 欺诈 | 钓鱼、身份冒充、社会工程 |
| 仇恨与骚扰 | 受保护群体攻击 |
| 性内容 | 年龄不明、未成年人风险 |
| 医疗法律金融 | 高风险错误建议 |
| Prompt Injection | 系统指令泄露、工具劫持 |
| Agent 行为 | 未授权写入、删除、发送、购买 |

测试需要得到授权，并在隔离环境中进行。

---

## 30. 偏见、公平与多语言

不能只测英语平均准确率。

```text
同一任务
├── 不同性别/年龄/地区/群体表述
├── 不同语言、方言和代码切换
├── 不同文化背景
└── 不同专业与读写水平
```

关注：

- 性能差距；
- 毒性/拒答率差距；
- 刻板印象；
- Token 成本差距；
- 翻译后评测是否改变题意；
- Judge 是否偏好特定语言风格。

公平不是要求所有群体输出完全相同，而是识别不合理的差异伤害。

---

## 31. 鲁棒性评估

对语义不变的输入扰动，模型结果是否稳定：

```text
原题
├── 改写措辞
├── 改变选项顺序
├── 加入无关信息
├── 拼写错误/口语化
├── 中英文混合
├── 长度和格式变化
└── 对抗性提示
```

指标：

- 原始准确率与扰动准确率；
- Performance Drop；
- Consistency Rate；
- 最坏分组性能；
- 对 Prompt 模板的方差。

只在一个 Prompt 模板上测量，可能高估模型能力。

---

## 32. 长上下文评估

“支持 128K”仅表示系统允许输入，不表示有效利用 128K。

需要测试：

```text
检索：关键信息能否从不同位置找到？
整合：能否组合多个远距离证据？
顺序：能否理解时间和事件先后？
冲突：能否识别文档间矛盾？
摘要：是否覆盖长文关键内容？
多轮：早期约束是否保持？
代码：能否跨文件追踪依赖？
```

### Needle-in-a-Haystack 的局限

随机插入一个明确字符串主要测检索，不等于深层理解。应结合 LongBench/LongBench v2、RULER 类任务和真实长文业务集。

应按长度和关键信息位置画热力图：

```text
准确率
位置\长度   8K   32K   64K   128K
开头        95    90    82     70
中间        93    80    61     42
结尾        96    91    85     74
```

---

## 33. RAG 评估必须分层

```text
用户问题
  ↓
Retriever → 检索文档
  ↓
Reranker/Context Builder
  ↓
Generator → 最终回答和引用
```

如果最终回答错误，需要知道是没检索到、上下文排序错、模型没使用证据，还是生成了证据外内容。

---

## 34. Retriever 指标

假设相关文档集合为 `Rel`，Top-k 检索结果为 `Ret_k`。

```text
Recall@k = |Rel ∩ Ret_k| / |Rel|
Precision@k = |Rel ∩ Ret_k| / k
Hit@k = 前 k 中是否至少有一个相关文档
```

### MRR

第一个相关结果排名为 `rank`：

```text
RR = 1 / rank
MRR = 所有问题 RR 的平均
```

### nDCG

考虑多个相关等级和排名位置，越相关的文档越应靠前。

Retriever 高 Recall 不保证最终答案好：文档可能太长、包含噪声或互相冲突。

---

## 35. RAG Generator 指标

### Context Relevance

提供的上下文是否与问题相关，是否包含大量噪声。

### Faithfulness/Groundedness

回答中的主张是否由上下文支持。

### Answer Correctness

最终答案是否正确，不能因“忠实引用了错误文档”就判正确。

### Citation Precision

引用的文档是否支持对应主张。

### Citation Recall/Completeness

需要引用的主张是否都有证据。

### Context Utilization

模型是否使用了已提供的关键证据。

RAGAS 等框架尝试自动化这些维度，但自动 Judge 本身仍需人工校准。

---

## 36. RAG 端到端测试矩阵

```text
问题类型
├── 答案存在于单文档
├── 需跨多文档整合
├── 文档互相冲突
├── 知识库没有答案
├── 时间敏感答案
├── 权限受限文档
└── Prompt Injection 文档
```

特别要测“知识库没有答案”时是否承认未知，而不是强行编造。

---

## 37. Agent 评估：最终成功不够

Agent 轨迹：

```text
任务
  ↓ 规划
选择工具 → 生成参数 → 执行
  ↓ 观察结果
继续规划/恢复错误
  ↓
最终状态
```

评估维度：

- Task Success Rate；
- 子目标完成率；
- 工具选择准确率；
- 参数/Schema 正确率；
- 环境状态是否达到目标；
- 步骤数、Token、工具成本和耗时；
- 无效/重复调用；
- 错误恢复率；
- 是否越权或产生副作用；
- 轨迹可解释与可审计性。

AgentBench 等工作将模型放入多个交互环境，重点不只是单轮回答，而是多步决策。

---

## 38. 工具调用评估

可拆为：

```text
1. 是否应该调用工具？
2. 选择了正确工具吗？
3. 参数 JSON 合法吗？
4. 参数语义正确吗？
5. 是否正确使用工具结果？
6. 工具失败后能否恢复？
7. 是否在需要确认时先确认？
8. 是否避免未授权副作用？
```

只检查 JSON 语法会漏掉“调用了错误账户”“日期错误”“删除范围过大”等严重问题。

---

## 39. 多轮对话评估

测试：

- 是否记住用户偏好和前文事实；
- 是否遵循后续修正；
- 是否区分不同人物和任务；
- 是否错误复活已经撤销的要求；
- 长对话中系统指令是否保持；
- 是否在不确定指代时澄清；
- 对话压缩/摘要后是否丢关键约束。

可以设计状态表：

```text
Turn 1：用户偏好中文
Turn 3：项目预算 10 万
Turn 5：撤销功能 A
Turn 12：检查模型是否仍保持这些状态
```

---

## 40. 系统性能评估

质量相同的模型，部署体验也可能完全不同。

```text
TTFT：首 Token 延迟
TPOT/ITL：Token 间延迟
E2E：完整回答耗时
Throughput：tokens/s 或 requests/s
Goodput：满足 SLO 的吞吐
显存、GPU 利用率、功耗
Prefix Cache 命中率
错误率、OOM、超时、取消
```

分别测试 Prefill-heavy、Decode-heavy、混合负载和真实长度分布。详见第 11 章。

---

## 41. 成本与质量必须一起画

```text
质量
│                         ● 大模型 Best-of-64
│                 ● 大模型单次
│          ● 小模型 Best-of-8
│    ● 小模型单次
└────────────────────────────→ 成本/延迟
```

记录：

- 每题输入/输出 Token；
- 推理次数；
- Judge/Verifier 成本；
- 工具与检索成本；
- GPU 时间或 API 费用；
- 成功任务成本 Cost per Successful Task。

最好的模型往往不是单题分数最高者，而是满足质量和安全阈值后成本最低者。

---

## 42. 数据污染与 Benchmark 记忆

公开题库可能出现在预训练、SFT 或合成数据中：

```text
训练语料 ∩ Benchmark ≠ ∅
→ 模型可能记答案，而非泛化
```

污染检测：

- Exact/Normalized Hash；
- n-gram 与 MinHash 近似匹配；
- 题干、答案和解释分别搜索；
- 检查公开解答、GitHub 和网页镜像；
- 使用时间切分或发布后的新题；
- 生成 Benchmark 变体并测鲁棒性；
- Canary 数据和训练谱系审计。

未发现匹配不等于绝对没有污染，因为训练数据可能不可见或被改写。

---

## 43. 动态 Benchmark 与私有评测集

公开 Benchmark 会逐渐饱和。可使用：

```text
动态题目：定期更新
时间切分：模型知识截止后产生的事件
参数化生成：数字、实体、条件随机变化
私有保留集：不进入训练和 Prompt 调优
对抗集：从真实失败案例持续收集
```

私有集也不能只由内部模型合成，否则可能偏向生成者风格。需要真实用户问题和专家审查。

---

## 44. Prompt 敏感性

同一道题改变 Prompt 可能改变结果：

```text
直接回答
请逐步思考
先给答案再解释
Few-shot 示例 A/B
选项顺序变化
System Prompt 变化
```

至少：

- 固定官方或公开 Prompt；
- 保存完整模板；
- 对多个合理模板做敏感性测试；
- 不要在测试集上反复挑最佳 Prompt；
- 将 Prompt 调优集与最终测试集分开。

---

## 45. 解码参数与推理预算影响

### 确定性任务

通常使用 Temperature=0 或接近确定性解码，便于复现。

### 创意/代码/推理采样

需要报告 Temperature、Top-p、采样次数和聚合方式。

### Max Tokens

设置太小会截断答案；太大会允许模型用更多推理计算。Reasoning Model 的输出预算是评估条件的一部分。

### Stop Sequence

错误停止词可能提前截断，或让模型继续生成多余内容，影响答案解析。

---

## 46. 统计显著性与置信区间

两个模型 80.1% 与 80.3% 不一定有真实差异。

### 二项比例标准误差

对 Accuracy `p`、样本数 `n`：

```text
SE ≈ sqrt(p(1-p)/n)
```

近似 95% 区间：

```text
p ± 1.96 SE
```

小样本或极端比例更适合 Wilson 区间。

### Bootstrap

对评测样本有放回重采样，多次计算指标，获得经验置信区间。适用于复杂平均分、胜率和差值。

### Paired Test

同一批题比较两个模型时应使用配对分析，因为题目难度共享。可 Bootstrap 两模型逐题分数差；二元正确/错误可考虑 McNemar Test。

---

## 47. 多次采样与随机种子

随机生成任务需要区分：

```text
题目间方差：有些题更难
采样方差：同一题多次答案不同
Judge 方差：同一答案评分变化
系统方差：API/模型版本动态变化
```

应保存逐样本原始输出，而不是只保存平均分。报告均值、标准差/区间和样本数。

---

## 48. Macro、Micro 和加权平均

假设评测含数学 900 题、安全 100 题。

```text
Micro Average：1000 题一起平均，数学主导
Macro Average：先算两个类别，再各占 50%
Weighted Macro：按业务权重，例如安全 70%、数学 30%
```

业务权重必须提前定义，不能看到结果后调整以偏袒某模型。任何关键安全门槛不应被平均分掩盖。

---

## 49. 平均分为什么会骗人？

```text
模型 A：知识 95，安全 40，平均 67.5
模型 B：知识 75，安全 70，平均 72.5
```

如果安全要求最低 80，两者都不应上线。应使用：

```text
硬门槛：Safety ≥ 80、JSON ≥ 99.9%
通过后再优化：质量、延迟、成本
```

先做约束，再做加权排序。

---

## 50. 业务评测集怎样建立？

### 步骤 1：定义真实任务

从用户日志、工单、专家流程和失败案例采样，而不是凭空编题。

### 步骤 2：建立任务分类

```text
高频 × 低风险
高频 × 高风险
低频 × 高风险
长尾与对抗案例
```

### 步骤 3：去隐私与授权

脱敏 PII、密钥和商业机密，记录数据用途和访问权限。

### 步骤 4：制作 Reference/Rubric

由领域专家定义必须点、可接受变体、禁忌错误和证据。

### 步骤 5：划分数据

```text
Dev：开发与 Prompt 调优
Validation：模型/配置选择
Test：最终保留集
Shadow：线上影子验证
```

### 步骤 6：持续更新

将生产失败转成新回归样本，但先查重和审查，避免测试集无限重复某一类事故。

---

## 51. 评估矩阵示例

| 一级维度 | 二级维度 | 自动指标 | 人工/专家 | 硬门槛？ |
|---|---|---|---|---|
| 正确性 | 最终答案 | EM/F1/Unit Test | 专家复核 | 是 |
| 事实性 | Claim 支持 | Evidence Match/Judge | 专家 | 是 |
| 指令 | 格式/约束 | Rule Pass Rate | 语义遵循 | 是 |
| 安全 | 有害合规 | ASR | 安全专家 | 是 |
| 可用性 | 清晰、完整 | Judge | 用户盲评 | 否 |
| RAG | 检索与忠实 | Recall@n/Faithfulness | 抽检 | 视业务 |
| Agent | 最终状态 | Task Success | 轨迹审查 | 是 |
| 性能 | 延迟吞吐 | TTFT/TPOT/Goodput | 用户体验 | 是 |
| 成本 | 成功任务成本 | 元/成功任务 | 财务/运营 | 否 |

---

## 52. Evaluation Harness 的基本流程

```text
Task Definition
├── Dataset
├── Prompt Template
├── Output Type
├── Generation Config
├── Answer Parser
└── Metric/Aggregation
        │
        ▼
Model Backend → Raw Outputs → Per-sample Scores → Aggregate Report
```

EleutherAI `lm-evaluation-harness` 提供统一任务接口并支持多种模型后端。框架提高一致性，但任务 YAML 或实现仍可能有 Bug；必须固定版本/commit，并保存原始输出。

其他常见工具包括 HELM、OpenCompass、LightEval、EvalPlus、RAGAS 和各类业务自建工具。工具不是评估方法本身，Rubric 和数据质量更重要。

---

## 53. 一个最小可复现评估配置

```yaml
run_id: qwen_eval_2026_08_27
model:
  id: exact-model-checkpoint
  revision: commit-or-version
  dtype: bfloat16
  quantization: none
prompt:
  chat_template: exact-template-version
  system_prompt: prompts/system_v1.txt
generation:
  temperature: 0.0
  max_new_tokens: 2048
  seed: 42
dataset:
  name: internal_finance_test
  version: 1.3.0
  split: test
evaluator:
  rules_version: 2.1.0
  judge_model: exact-judge-version
  judge_prompt: prompts/judge_v3.txt
output:
  save_raw_responses: true
  save_per_sample_scores: true
```

真实字段按框架调整，但版本信息不能缺。

---

## 54. 评估流水线

```text
冻结数据与配置
  ↓
运行模型，保存原始输出和 Token 用量
  ↓
规则/执行器评分可验证部分
  ↓
LLM Judge 评分开放部分
  ↓
抽样人工复核 + Judge 校准
  ↓
按维度聚合 + 置信区间
  ↓
失败案例聚类与根因分析
  ↓
质量/安全门禁
  ↓
Shadow / A-B / 上线监控
```

---

## 55. 回归门禁怎样设计？

示意：

```text
必须满足：
Safety ASR ≤ 1%
JSON Schema Pass ≥ 99.9%
关键业务 Accuracy 不低于旧版 -0.5pp
P95 TTFT ≤ 1.5s

同时优化：
人工胜率 ≥ 55%
Cost per Success 降低 ≥ 10%
```

允许小范围统计波动，设置置信区间或最小有意义差异。关键风险样本可采用“任何一个失败都阻止发布”的规则。

---

## 56. 失败案例分析比排行榜更重要

按错误类型聚类：

```text
知识缺失
推理错误
问题理解错误
格式/解析错误
引用错误
检索失败
工具选择/参数错误
安全过拒答/漏拒答
长上下文遗忘
语言与文化偏差
超时/截断
```

为每类记录数量、严重度、代表样本和改进责任归属。平均分只能告诉“变好了多少”，错误分类才能告诉“下一步改什么”。

---

## 57. 线上 A/B 测试

### 指标

- 任务完成率；
- 用户采纳/复制/修改率；
- 追问率和重试率；
- 会话放弃率；
- 人工升级率；
- 用户评分；
- 延迟、成本和安全事件。

### 注意

- 随机分流单位是用户、会话还是请求；
- 避免同一用户在不同模型间污染体验；
- 控制日期、流量来源和功能差异；
- 预先定义停止规则；
- 高风险领域先 Shadow，不直接暴露用户；
- 线上指标变化不一定由模型导致，UI 和延迟也会影响。

---

## 58. 模型更新后的持续评估

需要在以下变化后重跑相关评估：

```text
模型 checkpoint/API version
System Prompt/Chat Template
Tokenizer
量化方式
RAG 文档、Embedding、Chunking、Reranker
工具 Schema 或权限
推理引擎/采样参数
Judge 模型/Prompt
安全策略
```

“只改了 Prompt”也可能使安全、格式和工具调用回归。

---

## 59. 评估报告应该长什么样？

```text
1. 目标与决策：这次评估支持什么上线决定？
2. 对象：模型、系统和版本
3. 数据：来源、规模、分布、污染和限制
4. 方法：Prompt、采样、工具、Judge、人工流程
5. 指标：定义、聚合、置信区间
6. 结果：总体 + 分组 + 成本/延迟
7. 失败案例：严重问题与根因
8. 公平、安全与风险
9. 与基线的配对差异
10. 结论：上线/灰度/不通过及理由
11. 可复现材料：配置、原始输出、commit
```

---

## 60. 常见评估错误

1. 只报告一个总分；
2. 不固定 Prompt 和采样参数；
3. 在测试集上反复调 Prompt；
4. 用公开题库训练后继续报告同一题库；
5. 用一个 Judge 作为绝对真值；
6. 不做 A/B 顺序交换；
7. 把格式错误和知识错误混在一起；
8. 只看平均值，不看最坏分组和严重失败；
9. Reasoning Model 不报告推理 Token/采样次数；
10. RAG 只评最终答案，不拆 Retriever；
11. Agent 只看最后文字，不检查环境状态；
12. 代码执行没有 Sandbox；
13. 量化后只测 PPL，不测业务和安全；
14. API 模型静默更新后仍沿用旧分数；
15. 不保存逐样本原始输出；
16. 只追求 Benchmark，忽视真实用户任务。

---

## 61. 一张最终知识地图

```text
LLM Evaluation
├── 对象
│   ├── Base / Chat / Reasoning
│   ├── RAG / Agent / Tool Use
│   └── Model + Prompt + Runtime + Data
├── 方法
│   ├── 自动规则/执行器
│   ├── LLM-as-a-Judge
│   ├── 人工盲评
│   └── 线上 A/B
├── 能力
│   ├── 知识/阅读/数学/代码
│   ├── 指令/对话/长上下文
│   └── 多语言/结构化输出
├── 风险
│   ├── 事实/幻觉/引用
│   ├── 安全/过拒答
│   ├── 偏见/公平/隐私
│   └── 鲁棒性/Prompt Injection
├── 系统
│   ├── RAG Retriever/Generator
│   ├── Agent 轨迹/最终状态
│   └── TTFT/TPOT/Goodput/Cost
└── 科学性
    ├── 版本与可复现
    ├── 污染与动态测试
    ├── 置信区间/配对检验
    └── 失败分析与持续回归
```

---

## 62. 复习题

1. 为什么不能用一个 Benchmark 分数代表 LLM 全部能力？
2. Base Model 和 Chat Model 的评测协议有何区别？
3. PPL 为什么不能跨 Tokenizer 直接比较？
4. EM 的标准化规则为什么重要？
5. Pass@1 与 Pass@100 为什么不应直接比较？
6. MMLU 的优势和局限分别是什么？
7. Reasoning Model 为什么必须报告推理预算？
8. Pointwise 与 Pairwise Judge 有何区别？
9. LLM Judge 有哪些典型偏差？
10. 怎样用交换答案顺序检测位置偏差？
11. 为什么安全评估必须同时测过度拒答？
12. Needle-in-a-Haystack 为什么不等于长文理解？
13. RAG 的 Retriever 与 Generator 分别评什么？
14. Agent 为什么要检查环境最终状态？
15. Accuracy 差 0.2 个百分点为何可能没有统计意义？
16. Macro 与 Micro Average 分别偏向什么？
17. 怎样防止测试集变成开发集？
18. 业务评测集应从哪里获取样本？
19. 为什么要保存逐样本原始输出？
20. 什么时候应使用硬门槛而不是加权平均？

---

## 63. 参考资料

以下优先列出原始论文和官方项目：

1. Hendrycks et al., *Measuring Massive Multitask Language Understanding*（MMLU）：<https://arxiv.org/abs/2009.03300>
2. Liang et al., *Holistic Evaluation of Language Models*（HELM）：<https://arxiv.org/abs/2211.09110>
3. Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*：<https://arxiv.org/abs/2306.05685>
4. EleutherAI, `lm-evaluation-harness`：<https://github.com/EleutherAI/lm-evaluation-harness>
5. Chen et al., *Evaluating Large Language Models Trained on Code*（HumanEval/Pass@k）：<https://arxiv.org/abs/2107.03374>
6. Bai et al., *LongBench*：<https://arxiv.org/abs/2308.14508>
7. Bai et al., *LongBench v2*：<https://arxiv.org/abs/2412.15204>
8. Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*：<https://arxiv.org/abs/2309.15217>
9. Liu et al., *AgentBench: Evaluating LLMs as Agents*：<https://arxiv.org/abs/2308.03688>
10. Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*：<https://aclanthology.org/W04-1013/>
11. Papineni et al., *BLEU*：<https://aclanthology.org/P02-1040/>
12. Zhang et al., *BERTScore*：<https://arxiv.org/abs/1904.09675>

> Benchmark 只是测量工具，不是模型真实价值本身。最终评估必须回到目标用户、实际任务、风险边界和成本约束。
