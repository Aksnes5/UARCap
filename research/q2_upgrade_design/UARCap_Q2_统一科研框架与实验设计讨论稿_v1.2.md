# UARCap Q2 升级研究：基于现成高质量视频事实评价数据集的统一科研框架与实验设计讨论稿（v1.2）

> **当前状态**：`DESIGN_FOR_DISCUSSION`（完整自洽的 `v1.2` 研究设计讨论稿，供用户、ChatGPT 与导师审阅讨论；**本轮未编写正式实验代码，未下载大型视频数据集或新模型权重，未构造正式数据，未执行 GPU 推理，未启动人工正式标注，未冻结新研究协议**）  
> **版本保护声明**：本稿在继承 `v0.8`（`6421b6b9acd395420ce29dd630b8f1c5b15f84ec`）、`v1.0`（`fa292301a2f3edc337318e8c606e1aa2c6ae7e9e`）与 `v1.1`（`845a91b8b921c91a590ac2bf2aa5e044f96e720b`）全部有效科学资产与理论收敛的基础上，完成**数据基础、评价体系与资源路径的重大重构**；所有历史版本及冻结实验文件均完整保留、不做覆盖。  
> **项目隔离声明**：本项目仅限 UARCap 外部检索增强视频描述研究，与法律 RAG 项目完全隔离。

---

## 摘要与核心设计转型

经过前两轮理论审查，各方已就 UARCap Q2 升级的科学定位达成稳定共识——坚持**方案 III：外部检索证据对视频描述事实正确性、核心信息覆盖与行为错误风险的联合诊断研究**（不再开发新的动态检索器或重排序算法，原 E4 正式移出主研究范围）。然而，`v1.1` 路线暴露出一个严峻的现实工程瓶颈：在仅有简短参考字幕的 MSR-VTT 上从零建立细粒度视频事实真值并完成跨模型与新数据集标注，完整方案预计需要 **130–196 人时** 的高强度人工视频核验，极易拖垮个人与小团队的科研节奏。

本轮（`v1.2`）围绕“**能否直接复用国际高水平公开视频事实评价数据集，在不牺牲科学严谨性的前提下显著压缩重复人工标注成本**”这一核心命题，对 **ARGUS（ICCV 2025）、DREAM-1K / AutoDQ（Tarsier / Tarsier2）、CapQuiz（ACL 2026）、VDC（ICLR 2025）、VCapsBench（AAAI/arXiv 2025）、VidHalluc（CVPR 2025）、FactVC（EMNLP 2023）与 PE-Video** 开展了**面向真实数据文件、官方代码仓库与实际样本记录的一手审查**（详表见附录 `FACTUAL_VIDEO_DATASET_AUDIT.md` 与 `DATASET_SCHEMA_AND_EVALUATOR_MATRIX.md`），取得了五项关键一手发现：
1. **CapQuiz（ACL 2026）目前尚未公开发布数据集与下载链接**，官方页面与 ACL Anthology 仍处于“will release”状态，**严禁设为必选依赖**；
2. **VDC（ICLR 2025）的 `VDCScore_qa/short.jsonl` 存在明显质量缺陷**（答案栏包含 `"environment"`、`"lighting"`、`"object"`、`"action"` 等未清洗类别占位词），且描述偏 LLM 生成华丽文风，**不适合作为事实真值基准**；
3. **ARGUS（ICCV 2025）与 DREAM-1K（AutoDQ）的官方最终得分均存在“将未提及误作幻觉”的源码级陷阱**：`JARVVVIS/argus/code/compute_metrics.py` 第 118 行将 `underdetermined` 与 `contradiction` 一律按代价 `1.0` 计入 `ArgusCost-H`；`bytedance/tarsier/evaluation/metrics/evaluate_dream_gpt.py` 第 254 行在计算 Precision 时传入了 `is_recall=True`，导致 `neutral` 同样被判作精确率错误——**绝不能盲信官方标量总分，必须读取并独立汇报三值 NLI 原始类别（`entailment / contradiction / neutral-underdetermined`）**；
4. **DREAM-1K（`omni-research/DREAM-1K`，Apache 2.0，1,000 视频仅 `3.85 GB`，专业人工细粒度描述）** 是当前最均衡、最轻便、最适合承载 **E2（自然检索配对事实变化与跨模型/跨数据集主验证）** 的公开数据集（以开放下载的 **ARGUS** 为备选）；
5. **VidHalluc（`chaoyuli/VidHalluc`，CVPR 2025，完全开放下载）** 直接提供了视觉高度相似的视频片段及**经核验的真实主动作与高混淆冲突动作选项**（如 `"throwing grass in the garbage"` vs. `"raking leaves"`），天然适配 **E3（视觉固定下的预定义目标事实最小配对干预）**，可大幅压低 E3 靶事实从零构造的人力成本！

基于上述一手审查，本稿确立 **“路线 C：现成高质量公开数据集主体 + 三值解耦本地开源评估 + 少量精准双盲人工校准与桥接”** 架构：保留 `v0.8` 全部 MSR-VTT/MSVD 冻结实验不重做，取消额外引入 VATEX 及大规模 MSR-VTT 从零标注，将完整 Q2 方案人工核验压缩至约 **28–45 人时**（GPU 约 **18–32 小时**，本地开源跨家族裁判 `$0` 商业 API 费用），并先设仅需 **~0.5–1.5 GPU 小时 + ~8–12 人时** 的**方案 A（数据与评价兼容性试点）**作为前置止损闸门。

---

## 1. 现有研究成果与不可逾越的科学边界

新方案不是另起炉灶，而是直接建立在 UARCap 已经冻结并经独立复算核验的实证基座之上：

1. **百万级嵌套外部检索记忆库（VideoUFO）**：包含 `1,091,712` 个视频片段、`345,437` 个来源视频与六个严格嵌套规模（`10K / 50K / 100K / 250K / 500K / FULL`），BN 服务器（NVIDIA RTX A6000 48GB）已完整保存 VideoUFO 视频与特征、X-CLIP 检索索引、外部片段特征缓存及冻结 Video-LLaMA / Vicuna-7B / cross-gating 检查点。
2. **规模—可得性解耦与跨基准预算梯度（`v0.8` 冻结资产）**：在修复后 MSR-VTT 验证集（`val497`，`361` 源簇）上，从 `10K` 扩至 `FULL` 使所选 top-1 X-CLIP 相似度单调上升（CF `+0.0259`，D `+0.0319`）并更换近全量 top-1 候选（`494/497`、`495/497`），但 72 项验证网格中 0 项通过 BH 校正（查询级相似度与 CIDEr 变化秩相关 $\rho = .011 / .007$）。在测试基准上，MSR-VTT `test2690`（`2,004` 源簇，原训练预算 $k=7$）呈单调下降（`D1 71.21 → D3 70.08 → D5 69.48 → D7 68.27`，`D1 − D7 = +2.93`, BH $q = .0007$）；MSVD `test670`（`522` 源簇，原训练预算 $k=3$）同步下降（`156.39 → 151.45 → 144.39 → 141.59`，训练区间内 `D1 − D3 = +4.94`, $q = .0069$；超预算外推 `D3 − D7 = +9.86`）。
3. **预注册实验 A（匹配零外部证据对照，MSR-VTT `n = 2,690`, `2,004` 源簇）**：保留同一检查点、查询视频、提示骨架与 cross-gating 融合路径，空外部输入 `A0 = 69.81`，`A1 = 71.21`，`A7 = 68.27`。**`A1 − A0 = +1.40`（`95%` CI `[+0.45, +2.36]`，A 族 Holm $p = .0066$，五检验联合 Holm $p = .0165$）** 为当前最稳健的新增实证结论；**`A7 − A0 = −1.54`（`[−2.94, −0.18]`）** 通过 A 族 Holm（$p = .0271$），未通过五检验联合 Holm（$p = .0813$）。`A1/A7` 与历史 `D1/D7` 全部 2,990 条输出一致，属于端点复现核验而非独立重复样本。
4. **预注册实验 B（逐查询锁定视觉张量的文本扩展，MSR-VTT `n = 2,618`, `1,969` 源簇）**：`B3 = 69.88`、`B7HQ = 68.62`、`B7LQ = 68.75`。逐查询 trace 证实 3 条检索视觉特征、相似度权重、聚合张量、注意力 K/V 与融合投影完全一致。但三项主比较均未通过 B 族 Holm（`B7HQ−B3 = −1.27`, $p = .0573$；`B7LQ−B3 = −1.13`, $p = .0612$；`B7HQ−B7LQ = −0.14`, $p = .7698$），且 HQ/LQ 仅为平均间隔 `0.00835` 的 X-CLIP 相似度代理，不代表真实语义或事实质量，因此未确认一般性文本过载机制。
5. **严格排除的历史无效结论**：受早期 raw-video 解码缺陷污染的未修复验证集强文本过载与域反转主张永久排除；所有新设计均严守既有统计边界（五项 A/B 对比均未越过 `±1.0` CIDEr 实践清晰界；A/B 各自拟合参考 IDF 不可跨实验直比均值）。

---

## 2. 统一科学问题

本版完全继承前两轮收敛后的统一研究主线，不再扩散新方向，将整篇未来论文严格锚定在一个核心科学问题及三个子问题上：

> **核心科学问题**：  
> **相对于架构匹配的零外部证据条件，百万级外部检索记忆如何改变视频描述中的事实正确性、错误负担与核心信息覆盖；外部证据中的特定事实冲突是否在视觉输入固定时因果地诱发对应生成错误；以及这些效用—事实风险规律在不同模型接口与数据集之间呈现何种适用边界？**

- **RQ1（配对事实变化，对应 E2）**：超越单一 CIDEr，启用外部证据（$k=1, 7$，必要时 $k=3$）相对于零外部证据（$k=0$）如何分别改变视频描述的错误事实负担、视频支持事实与核心事件覆盖（以及纠正、新增错误、遗漏与不可核验比例）？
- **RQ2（受控行为因果效应，对应 E3）**：在查询视频、模型权重与实际视觉输入保持固定的条件下，仅改变外部文本中预定义目标事实的表达（`Sup / Con / Neu` 相对共享 `Base`），目标事实冲突是否因果地提高对应错误事实的生成概率并抑制目标正确事实？
- **RQ3（跨模型与跨数据集适用边界，对应 E5）**：上述配对事实变化与受控干预效应，在共同查询子集上的冻结 Video-LLaMA 与现代开源视频 VLM 之间、以及在原 MSR-VTT 与新细粒度视频事实评价数据集之间，是否保持一致抑或呈现出明确的系统与接口依赖边界？

**方法范围硬裁决**：本轮**正式不开发**新的动态检索算法、复杂重排序器或风险门控器（原 E4 永久移出本轮范围）；新论文的定位是**外部检索增强视频描述的事实效用、错误风险及受控证据干预诊断研究**。

---

## 3. `v1.1` 纯自建人工标注路线的现实瓶颈

`v1.1` 在理论上已经相当严密，但进入成本核算后暴露出致命的可实施性短板：
1. **底层数据天然欠缺细粒度事实真值**：MSR-VTT 与 MSVD 每视频仅提供若干条 8–10 词的简短人类概括句，未记录视频中大部分次要实体、动作细节或明确反事实，因此 `v1.1` 不得不要求标注员从零观看数百段视频、手写核心事实、切分并审核数千条原子命题、做双人三值核验及跨字幕语义对齐；
2. **总人时突破小团队承载极限**：按 `v1.1` 逐工序实算，仅 MSR-VTT 标定 + 试点 + 正式 E2/E3 + VATEX 确认集合计需要 **130–196 人时**（即便经过 L1–L4 分层压缩仍需 **107–145 人时**），对于同时推进法律 RAG 论文的研究者而言周期风险极高；
3. **重复造轮子与审稿外部效度顾虑**：国际视频理解社区在 2023–2026 年已投入大量专业人力建成了 DREAM-1K、ARGUS、VidHalluc、VCapsBench 等公开细粒度视频事实评价基准。完全绕开这些社区公认基准而仅靠内部小规模自标 MSR-VTT，既耗时费力，在 Q2 审稿时也容易面临“自建标注基准外部公信力有限”的追问。

---

## 4. 现成细粒度视频数据集的一手实际审查（非摘要转述）

本轮通过直接调用 Hugging Face Hub API、Datasets Server 样本预览接口及 GitHub 原始评价源码，对 **8 个候选公开数据集** 完成了逐项一手审查（完整链接、文件树与样例记录见附录 `FACTUAL_VIDEO_DATASET_AUDIT.md` 与 `DATASET_SCHEMA_AND_EVALUATOR_MATRIX.md`）。

### 4.1 逐数据集一手能力对照矩阵

| 数据集与原始来源 | 开放下载与规模 | 许可 | 人工标注可信度 | 事实级真值与 schema | 明确错误 / 反事实 | 信息遗漏与核心事件 | 官方评价器源码实查与本地可运行性 | 原短字幕兼容性 | E2 适用性 | E3 适用性 |
|---|---|---|---|---|---|---|---|---|---|---|
| **DREAM-1K** (`omni-research/DREAM-1K`, ByteDance Tarsier) | **可下载**（`gated: auto` 一键同意）；**1,000 视频仅 `3.85 GB`**（`video.zip`） | **Apache 2.0** | **极高**（专业人工细粒度描述，涵盖电影/YouTube/短片/动画） | `json/metadata.json` 含详细人工事件描述 | 无内置错误选项 | `AutoDQ` 评估动态事件 Recall/Precision | `evaluate_dream_gpt.py` 硬编码 Azure GPT-4，且 **L254 传错 `is_recall=True` 把 `neutral` 计作 Precision 错误**；仅抽动态动作。**可轻松改用本地开源 LLM 输出三值 NLI 解耦评估！** | **较好**（多为短视频片段，篇幅适中，配短字幕核心事件子集可行） | **强烈推荐（E2 主数据）** | 中（需从人工描述改写反事实） |
| **ARGUS** (`tomg-group-umd/argus`, ICCV 2025) | **直接开放**（`gated: False`）；**500 视频 `16.75 GB`**（含 8 条同源分镜需源聚类） | `cc-by-nc-sa-4.0` | **中高**（500 视频平均 ~477 词，15–90 秒；混合来源既有数据集+人工增补） | Parquet **仅 2 列**：`clip_name` + `human_caption`（无预拆评价点） | 无内置错误选项 | `ArgusCost-O` 测遗漏（但 477 词对 9 词短字幕遗漏率逼近 90%+） | `nli_eval.py` 原生支持开源裁判（`deepseek_v3` / `qwen_2_5_72b` / `llama_3_3_70b`）并输出三值（`entailment / contradiction / underdetermined`）；但 **`compute_metrics.py` L118 把 `underdetermined` 与 `contradiction` 合并按 cost=1 扣分！** | **偏弱**（15–90 秒长片 + 477 词文学性稠密描述，短字幕地板效应重） | **备选（须拆解原始三值 NLI，禁用单一 `ArgusCost-H`）** | 弱 |
| **VidHalluc** (`chaoyuli/VidHalluc`, CVPR 2025) | **直接开放**（`gated: False`）；含 `ACH/TSH/STH` 视频包与 JSON | `cc-by-nc-nd-4.0` | **高**（高视觉相似视频对 + 明确动作/时序真值与强混淆干扰项） | `ach_mcq.json`（4 选 1 主动作）、`ach_binaryqa.json`（同源成对 Yes/No）、`tsh.json`（双动作时序） | **极强！**直接提供同场景高混淆错误动作（如 `throwing grass` vs. `raking leaves`） | 聚焦核心显著动作与时序，不测泛背景覆盖 | 原任务为问答/MCQ，**但极其适合直接提供 E3 的 $(f^+, f^-)$ 靶动作对**；且文件名带 YouTube ID（如 `vcCwvRYqU2I_clip_4`），便于 VideoUFO 零重叠审计与源聚类！ | **极高**（短片段 + 单一显著主动作，完美契合 8–12 词短字幕！） | 弱（非全句覆盖） | **强烈推荐（E3 主数据！）** |
| **VCapsBench** (`somos99/VCapsBench`, AAAI/arXiv 2025) | **直接开放**（`gated: False`）；**5,677 视频 ~23.4 GB**（12 个 zip）+ `8.2 MB` 100K QA | `odbl` | **中**（多阶段 VLM/LLM 合成 + 自动一致性过滤，非 100% 人工撰题） | `VCapsBench_100KQA.jsonl` 含 `duration`（多为 6–15s）、`source` 与 21 维 `dim*Q*yes/no` | 含约 20%–35% `no` 问答（偏缺失/镜头属性，少量动作冲突） | 覆盖全面，但含大量 `blur / color grading / shot type` 摄影维，短字幕几乎全答 `Unanswerable` | `LLM4eval-m.py` 内置三值 `Yes / No / Unanswerable`，默认调 Gemini / GPT-4o API，可改本地开源 LLM；须事前过滤至核心语义维（`action / entity / count / color / relation / background`） | **中上**（6–15 秒短片段天然匹配，但须筛掉摄影维） | 备选（QA 路线候选） | 中（`no` 题多为无某元素，仍需人工审动作冲突） |
| **CapQuiz** (Apple, `2026.acl-long.777`, ACL 2026) | **当前不可下载！**（Apple 页面与 ACL 2026 均无 GitHub/HF 下载链接，仍称 `"We will release..."`） | 未发布 | 论文描述高（1,204 视频、23,632 人工核验 MCQ） | 论文定义 10 类 MCQ 与 `CapP/CapR/CapF1` | 论文含干扰选项，但数据未公开 | 理论上极佳 | 代码与数据均未公开 | — | **当前排除（严禁作必选依赖）** | **当前排除** |
| **VDC** (`wchai/Video-Detailed-Caption`, ICLR 2025) | 直接开放；1,027 视频 | `apache-2.0` | **低（事实真值不合格）**——实查发现 `VDCScore_qa/short.jsonl` 答案栏充斥 `"environment"`、`"lighting"`、`"object"`、`"action"` 等占位词！ | 5 类结构化长描述（~500 词，高度 GPT-4o 合成化文风） | 无明确验证反事实 | `VDCScore` 依赖 GPT-4o QA 匹配 | 存在明显 QA 占位词污染与合成文风偏置 | 弱 | **淘汰** | **淘汰** |
| **FactVC** (`NiteRainHL/FactVC`, EMNLP 2023) | Google Drive（200 ActivityNet + YouCook2） | 科研使用 | 高（人工句级错误标注） | **致命局限**：仅针对历史旧模型（COOT/MART/PDVC）输出打 1–5 分与错词括号，**无面向新模型输出的可复用视频事实表** | 仅旧输出中的历史错词 | 无遗漏评测 | `factvc_video.pth` 为弱监督 CLIP 标量分，无法输出原子事实状态转移 | — | **不适用** | **不适用** |
| **PE-Video** (`facebook/PE-Video`, Meta) | 直接开放（2,000 个 WebDataset tar 分片，体量极大） | `cc-by-nc-4.0` | 高（其中 12 万条人工复核），但定位是百万级训练/预训练语料 | 大规模运动中心视频描述 | 无内置反事实 | 无专用事实/遗漏评测协议 | 面向表征预训练而非评测套件，下载存储成本过高 | — | 不推荐 | 不推荐 |

### 4.2 本轮一手审查揭示的三项核心警示
1. **绝不以论文摘要决定数据集**：如果只看摘要，我们可能会押宝尚未开源下载的 **CapQuiz** 或存在占位词污染的 **VDC**，导致项目一启动即卡死；
2. **绝不盲信已发表的官方标量指标**：**ARGUS 的 `ArgusCost-H`** 与 **DREAM-1K 的 `AutoDQ Precision`** 在源码层面均将“参考描述未提及（`underdetermined / neutral`）”直接当作幻觉扣分！任何参考描述无论多详细（哪怕 477 词）都无法穷尽视频世界的全部事实，因此**必须严格区分三类信息**：① 已被标注确认的正确事实；② 已被确认与视频冲突的错误/反事实；③ 参考未覆盖、真伪暂未知的事实（须由小规模人工看片抽查确定其中多少属真实细节、多少属幻觉）；
3. **数据集互补组合远胜强行单打一**：没有一个现有公开数据集同时完美解决“自由文本事实覆盖变化（E2）”与“严格高混淆动作冲突干预（E3）”，但 **`DREAM-1K`（E2 主战场） + `VidHalluc`（E3 靶冲突主战场）** 恰好在数据体量、开放性、clip 时长与科学角色上形成极高契合度的最小双集组合！

---

## 5. 评价路线比较与人工标注的可替代性边界

围绕“现成数据集究竟能在多大程度上替代人工标注”，系统比较三条评价路线：

- **路线 A（纯详细人工参考 + 官方自动评价器，零人工）**：例如直接跑 `DREAM-1K + AutoDQ` 或 `ARGUS + ArgusCost`。**不可行！** 因为官方标量合并了 `neutral/underdetermined` 与 `contradiction`，且将闭源 GPT-4 替换为本地开源模型后若无人工校准，无法证明评价器没有偏爱长句或特定模型家族。
- **路线 B（纯预定义 QA 自动评测，零人工）**：例如直接用 `VCapsBench` 的 100K QA。**存在明显盲区！** ① 仅评测题库覆盖到的维度，漏检题库外生成的新增幻觉；② TextQA 评价器极易凭语言常识盲猜答案（即便字幕未提及也能猜对“天空是蓝色的”）；③ `VCapsBench` 的 QA 由 VLM/LLM 合成且含大量摄影属性，未经筛选与人工核验不可直接当黄金事实。
- **路线 C（推荐主体系：现成高质量数据集主体 + 三值解耦开源自动评价 + 少量精准双盲人工校准与桥接）**：
  - **由现成数据集与解耦自动评价承担约 75%–85% 的重复劳动**：利用 `DREAM-1K` 的 1,000 条专业人工细粒度参考，将自动 NLI 严格解耦为三类输出（`Entailed / Contradicted / Unmentioned-Neutral`），并利用 `VidHalluc` 的既有 `(f+, f-)` 主动作对加速 E3 靶构造；
  - **人工核验精准集中在 4 处机器无法替代的高价值关口（总计仅 ~28–45 人时）**：
    1. **自动评价器校准与独立验证（约 80 个 DREAM-1K 视频，30 校准 + 50 独立验证）**：双人静音看片，直接测量本地开源三值裁判的精确率/召回率/$\kappa$，并重点核查被判为 `Unmentioned-Neutral` 的生成命题中有多大比例实际上是视频支持的真实细节、多大比例是真实幻觉；
    2. **历史 MSR-VTT 方向桥接子集（50–60 个独立来源视频）**：复用已冻结 `A0 / A1 / A7`，按统一核心事实与三值核验口径建立新旧数据集之间的方向桥接；
    3. **E3 靶事实严格冲突审核与自然度盲审（150–180 个 VidHalluc 片段）**：确认选用的干扰动作 $f^-$ 确实与原视频严格冲突（而非次要并存动作），并审核 `Neu` 的中性与视频相容性；
    4. **自然 VideoUFO 冲突检索子集抽核与分歧裁决（50–60 组）**：确保生态效度验证的真实性。

---

## 6. 继承 `v1.1` 的统一理论定义与指标口径严格划界

本版完全继承 `v1.1` 确立的结构化多模态证据输入模型：
$$y = G\big(v,\; T,\; V;\; I,\; \theta,\; d\big)$$
其中 $v$ 为查询视频、$T=(c_1,\dots,c_m)$ 为有序外部文本序列、$V=((\tilde{v}_j, w_j); \mu)$ 为实际加载的外部视觉张量及图文绑定关系、$I$ 为证据接口（Video-LLaMA 的 $I_{\text{x-gate}}$ vs. 现代 VLM 的 $I_{\text{prompt}}$）、$\theta$ 为冻结参数、$d$ 为固定解码配置。继续坚持：允许零外部证据（`external-evidence withholding` / $T=\emptyset, V=\emptyset$）、区分候选搜索与证据供给、不假设顺序无关或效用可加性、不将不同接口宣称为完全等价。

### 6.1 四状态转移与新数据集指标的诚实映射（严禁偷换概念！）
保留四状态空间 `{S（视频支持）, C（视频明确反驳）, U（参考未覆盖/无法核验）, Ø（未表达）}` 与**废除单一主观加权总分 `B_fact`** 的裁决。但针对“复用现成参考描述评价器”的特点，本版设立严格的**指标命名诚实规则**——**凡自动评价器基于参考描述（而非全片人工看片）算出的指标，绝不直接冒充人工看片真值 `N_C`、`N_S` 或 `R_core`！** 明确建立如下一一对应：

| 科学维度 | 全队列自动代理指标（基于 DREAM-1K 参考三值 NLI） | 人工看片黄金指标（80 DREAM-1K 验证集 + 60 MSR-VTT 桥接集） | E3 预定义靶指标（VidHalluc L4 五态） |
|---|---|---|---|
| **错误事实风险** | **参考显式冲突命题数 `N_Contra_ref`**（严格只计 NLI=`contradiction`，**绝不混入 `neutral`！**）与冲突发生率 | 真实视频反驳命题数 **`N_C`**（并审计 `N_Unment_ref` 中真实错误占比） | 目标错误事实采纳概率差 **`ΔP(f⁻)`**（`Con vs. Neu`） |
| **受支持事实覆盖** | **参考蕴含命题数 `N_Entail_ref`**（NLI=`entailment`）与参考事件召回率 | 真实视频支持命题数 **`N_S`** | 目标正确事实抑制 **`ΔP_supp(f⁺)`**（`Con vs. Neu`）与促进 **`ΔP_fac(f⁺)`**（`Sup vs. Neu`） |
| **固定核心事件覆盖** | **预提取核心事件覆盖率 `R_core_ref`**（在跑任何模型前，仅由 DREAM-1K 人工参考确定每视频 1–2 个主干核心事件，避免长尾背景稀释） | 独立视频先行核心事件覆盖率 **`R_core`** | 靶正确/错误共存五态分布（`NEITHER / ONLY_POS / ONLY_NEG / BOTH / AMBIGUOUS`） |
| **未覆盖 / 不确定比例** | **参考未提及/中性比例 `P_Unment_ref`**（单独报告，严禁并入错误！） | 不可核验比例 **`P_U`** | `AMBIGUOUS` 比例 |
| **社区兼容参照指标** | 官方 `AutoDQ`（Precision / Recall / F1）或官方 `ArgusCost-H/O`（明确注明其合并 `neutral` 的局限，仅作社区参照） | 传统 PTB CIDEr（仅用于 MSR-VTT 衔接参照） | — |

### 6.2 对 L1–L4 分层体系的精简保留
引入现成数据集后，L1–L4 依然清晰有效且大幅轻量化：
- **L1（基本事实三值评判）**：全队列由经校准的本地开源三值 NLI 评估器执行，80 视频独立人工验证集做黄金校准；
- **L2（固定核心事件覆盖）**：由 DREAM-1K 人工参考事前锁定每视频 1–2 个主干核心事件（无需人工看片重写 1,000 个视频的核心清单！），彻底消除反向由模型输出定核心的泄漏风险；
- **L3（精细配对状态转移：区分 `C→S` 真正纠错、`C→Ø` 删除错误、`Ø→C` / `S→C` 新增错误与 `Align(T)` 外部对齐）**：**仅在 80 视频 DREAM-1K 人工验证子集与 50–60 视频 MSR-VTT 桥接子集上执行**（或在 VidHalluc 预定义固定槽位上直接获得精确四状态跃迁，免去自由文本复杂跨字幕对齐！）；
- **L4（E3 预定义靶五态判定）**：直接依托 VidHalluc 既有 $(f^+, f^-)$ 动作对，在允许共存的五态框架下高效完成。

---

## 7. 新旧数据集之间的科学衔接（打消前后割裂担忧）

采用现成细粒度事实数据集，**绝对不需要重做历史 MSR-VTT 与 MSVD 的全部实验**，而是通过“三类角色隔离 + 最小衔接运行 + 探索性 MSR-VTT 桥接”构建严密逻辑链：

### 7.1 严格区分三类数据角色
1. **外部检索记忆库（固定不变）**：**VideoUFO**（`1,091,712` clips，已有 X-CLIP 索引与视觉特征）。新引入的 DREAM-1K / VidHalluc 仅作为**查询评价集**，绝不混入或改动 VideoUFO 检索库；
2. **传统视频描述性能基准（完整保留历史冻结结果）**：**MSR-VTT**（规模解耦、A/B 预注册实验、联合预算梯度）与 **MSVD**（跨基准联合预算梯度）；
3. **细粒度视频事实评价与靶干预基准（本轮新增）**：**DREAM-1K**（承担 E2 配对事实转移、核心覆盖与跨模型/跨数据集主验证）与 **VidHalluc**（承担 E3 预定义靶事实最小配对干预）。

### 7.2 明确不重复的旧实验与新数据集最小衔接实验
- **默认不重复**：不在 DREAM-1K / VidHalluc 上重跑 6 尺度语料扫描、全部排序算法对比、历史 Oracle、模态因子消融、prompt 顺序或 LOO 实验；
- **新数据集最小衔接运行（DREAM-1K）**：保持原冻结 Video-LLaMA 检查点、VideoUFO FULL 外部检索库、X-CLIP 检索器、D 全句 ROUGE-L 共识重排、cross-gating 融合与相同短字幕解码配置不变，对 DREAM-1K 新查询**重新提取全视频 X-CLIP 查询特征并重新检索 top-20 候选**（严禁复用 MSR-VTT 查询候选池！），运行 **`k = 0, 1, 7`（必要时增 `k = 3`）**。由此直接检验：*原系统在 MSR-VTT 上呈现的预算响应，在 DREAM-1K 的事实维度上如何展开？*

### 7.3 防止跨数据集过度解释的 MSR-VTT 桥接设计
即使在 DREAM-1K 上观测到 `k = 7` 推高 `N_Contra_ref` 并拉低核心事件覆盖，也不能直接声称“这证明历史 MSR-VTT 上 `A7 − A0 = −1.54` CIDEr 是由完全相同的事实错误造成”（因为两数据集在领域、复杂度、参考粒度与候选分布上均不同）。因此设计**最低成本 MSR-VTT 探索性桥接子集（E2c）**：
- 从历史冻结 `test2690`（`A0 / A1 / A7` 现成输出，0 新增 GPU）中按 X-CLIP 相似度分层先验随机抽取 **50–60 个来源独立视频**，用与 DREAM-1K 人工验证子集**完全相同的双盲看片三值核验与核心事件覆盖规程**进行人工评价；
- **诚实限定推断强度**：明确 50–60 个视频的桥接子集用于检验**跨数据集方向一致性与探索性机制呼应**，绝不宣称 50 个视频足以独立确认统计显著机制；若 DREAM-1K 未复现原预算效应，则按第 14 章结果分支诚实讨论数据域、任务粒度与词面/事实指标分离的系统边界。

---

## 8. 短字幕模型与详细描述数据集的任务兼容性（核心科学风险攻坚）

原 Video-LLaMA 检查点在 MSR-VTT 上微调生成 **8–12 词的单句短描述**，而 DREAM-1K（数十至上百词多事件人工描述）与 ARGUS（平均 477 词）面向细粒度详细描述。这一任务长度错配是新方案最大的潜在科学陷阱，本版通过四项刚性规范将其正面化解：

1. **配对差异有效，但须防覆盖率地板效应**：在同一模型、同一长度约束下做配对比较（`k = 0 vs 1 vs 7`），即便绝对全事件召回率偏低（例如在 8 个事件的视频里短句仅覆盖 1–2 个事件，Recall 仅 15%–20%），配对差值在错误率维度（`N_Contra_ref`）上依然具有极高分辨率——因为哪怕 9 词短句只要写错核心动作/对象就会立刻触发明确的 `contradiction`！但在覆盖率维度上，直接拿全量 10–19 条长尾细节做分母确实存在地板压缩风险；
2. **事前无泄漏的“核心事件子集”协议（化解覆盖地板效应）**：在启动任何模型推理之前，仅依据 DREAM-1K 人工参考描述（按固定客观规则：提取描述中占主导动作与主要交互对象的 **1–2 条最显著主干事件**，不看任何模型输出、不按结果筛选），建立全队列固定的核心事件子集 $\mathcal{K}_{\text{core}}(v)$，计算短字幕的 **`R_core_ref`**。这样即使短句只有 10 词，覆盖核心事件也能在 `0.0–1.0` 区间产生充足的测量分辨率；
3. **跨模型长度公平控制（强制短字幕对齐主设置）**：若原 Video-LLaMA 输出 9 词单句、现代 VLM 输出 150 词长段落，二者的错误命题数与支持命题数完全不可比！因此**E5 跨模型主设置强制统一任务口径**：给现代开源视频 VLM（如 `Qwen2.5-VL-7B-Instruct`）设定明确的**短字幕对齐指令**（`"Describe the main action and key subjects in this video in one concise sentence (around 8–18 words)."`），实测并汇报两模型的平均词数分布；同时仅在现代 VLM 内部可选追加一组**详细描述模式长度敏感性对照**（明确与短字幕主比较隔离，绝不跨长度直比绝对计数！）；
4. **绝不强迫原 Video-LLaMA 改做长段落详细描述**：原检查点从未在长段落描述上训练，强行用提示词逼其写长段只会引发重复退化与分布外崩塌，偏离原论文的真实研究任务。

---

## 9. 推荐数据集组合、备选方案与数据取舍说明

综合开放获取状态、视频存储体积、人工标注质量、clip 时长适配度与评价器源码审查结果，本版提出**最小双集互补组合（无需再额外引入 VATEX！）**：

### 9.1 推荐主组合（仅两个轻量公开数据集，总下载量适中）
1. **E2 主事实评价与跨数据集确认队列：`DREAM-1K`（`omni-research/DREAM-1K`，Apache 2.0，1,000 视频仅 `3.85 GB`）**
   - **入选理由**：① `gated: auto` 一键授权即可完整下载（无 CapQuiz 的未发布阻断）；② 体积仅 `3.85 GB`（不及 ARGUS 的四分之一、VCapsBench 的六分之一）；③ 专业人工细粒度事件描述，非 LLM 模板合成文风；④ 天然构成从未参与 `v0.1–v0.8` 任何决策的**未触碰外部确认队列**，**可直接取代 `v1.1` 拟额外引入的 VATEX**！
   - **关键改进用法**：不盲信官方 `AutoDQ` 标量（因其源码第 254 行将 `neutral` 计作 Precision 错误且硬编码 Azure GPT-4），而是复用其 1,000 条高质量人工描述，建立**三值解耦本地开源 NLI 评估（`Entailed / Contradicted / Unmentioned-Neutral`）+ 事前预提取核心事件子集 `R_core_ref`**，并同步汇报官方 `AutoDQ` 指标作社区参照。
2. **E3 受控目标事实干预专用辅助集：`VidHalluc`（`chaoyuli/VidHalluc`，CVPR 2025，完全开放 `gated: False`）**
   - **入选理由**：① `ach_mcq.json` 与 `ach_binaryqa.json` 直接提供视觉高度相似的短片段及其**经核验的真实主动作（$f^+$）与高混淆场景冲突动作（$f^-$）**（如 `throwing grass in the garbage` vs. `raking leaves`），将 E3 靶事实从零构思与筛片成本大幅压减；② 片段名自带 YouTube ID（如 `vcCwvRYqU2I_clip_4`），可直接做精确 VideoUFO 来源零重叠审计与来源簇去重；③ 聚焦单句短字幕最核心的动作/事件事实，极度贴合原 Video-LLaMA 能力域。

### 9.2 备选方案与淘汰名单
- **E2 备选数据集 1：`ARGUS`（`tomg-group-umd/argus`，ICCV 2025，`gated: False`，500 视频 `16.75 GB`）**：若 DREAM-1K 的 `auto` 授权或特定视频域存在意外障碍，可立即切换至 ARGUS——直接调用其原生支持的开源裁判接口（`deepseek_v3` / `qwen_2_5_72b_instruct`），**强制读取 `nli_eval.py` 原始三值 JSON** 以分离 `contradiction` 与 `underdetermined`，并在事前抽取每视频核心事件子集缓解 477 词描述的地板效应。
- **E2/E3 备选数据集 2：`VCapsBench`（`somos99/VCapsBench`，`odbl`，5,677 视频 ~23.4 GB）**：若团队偏好纯三值 TextQA 路线（`Yes / No / Unanswerable`），可抽取 500–600 条 6–15 秒片段并将 21 维 QA **事前过滤至 6 个核心语义维**（`action / entity / count / color / relation / background`），但须额外执行小规模人工审计核查其合成 QA 噪声。
- **明确淘汰名单**：**CapQuiz**（当前未公开下载，禁止设为必选依赖）、**VDC**（实查 `short.jsonl` 存在严重类别占位词错误且高度 GPT 合成化）、**FactVC**（仅标注旧模型历史输出，无新输出可复用的视频事实真值表）、**PE-Video**（百万级训练语料、2,000 分片过大，定位不符）、**VATEX**（功能已被 DREAM-1K 完整覆盖，正式取消以防数据战线过长）。

---

## 10. E1 / E2 / E3 / E5 完整实验设计（统一口径，详表见 `PROPOSED_EXPERIMENT_MATRIX_V12.md`）

### 10.1 Experiment E1：既有资产的探索性效用分析（0 GPU，非准入门槛）
完全复用冻结的 MSR-VTT `A0 / A1 / A7 / D3 / D5` 与 MSVD `D1 / D3 / D5 / D7` 输出，开展逐查询净效用分布、跨参考集验证的事后选择收益估计（分离 IDF 拟合 + 20 次重复劈半 + 解耦评价侧随机选臂参照）及推理前预测特征探索。明确标注 E1 属回顾性低成本探索（可按篇幅放入补充材料），**不构成 E2/E3 的硬性准入门槛**。

### 10.2 Experiment E2：基于现成高质量事实数据的配对事实变化（RQ1）
分为三步递进，全部复用 VideoUFO FULL 外部检索库，严禁提前窥探正式测试结果调参：
- **E2a：数据与评价兼容性试点（方案 A，前置必过闸门）**：
  - 抽取 **30 个 DREAM-1K 独立视频**（+ 20 个 VidHalluc 片段用于 E3 预检），跑通“VideoUFO X-CLIP 新查询检索 → Video-LLaMA 与候选现代 VLM 短字幕推理（`k = 0, 1, 7`）→ 本地开源三值 NLI 评估器”；
  - 由 2 名标注员静音看片独立核验这 30 个视频，定量检验：① 短字幕在 DREAM-1K 上是否存在覆盖地板效应；② `N_Contra_ref` 与真实人工 `N_C` 的一致性；③ `P_Unment_ref` 中有多少属于真实细节而非幻觉；④ 单视频实际 GPU 与人工吞吐。**通过兼容性检查后再进入 E2b！**
- **E2b：DREAM-1K 正式预算配对研究（核心主实验）**：
  - 在去除 E2a 试点（30 视频）及任何与 VideoUFO 存在源重叠的片段后，于剩余独立来源 DREAM-1K 视频（暂定规划 **500–800 个独立来源视频**，视 E2a 实测吞吐与来源去重留存确定）上运行 **`k = 0 (A0)、k = 1 (A1)、k = 7 (A7)`**（必要时增 `k = 3` 供斜率核查）；
  - **两项共同主要终点（来源聚类配对检验，族内 Holm $\alpha = 0.05$）**：① 参考显式冲突命题数 **`N_Contra_ref`**（严格仅计 `contradiction`）；② 预提取核心事件覆盖率 **`R_core_ref`**（或参考蕴含命题数 `N_Entail_ref`；二者与 `P_Unment_ref`、官方 `AutoDQ` 同步汇报，不压成单一加权总分）；
  - **独立黄金人工验证子集**：从 E2b 中先验随机抽取 **50 个独立来源视频**（与 E2a 30 校准视频完全隔离，绝不参与提示词调优），执行双盲静音看片核验，汇报人工黄金 `N_C`、`N_S`、`R_core` 及 L3 精细八类转移（真正纠错 `C→S` vs. 删除错误 `C→Ø`、同槽转错 `S→C`、事后 `Align(T)`）。
- **E2c：原 MSR-VTT 方向桥接子集（0 新增 GPU）**：
  - 从历史冻结 `test2690`（`A0 / A1 / A7`）中按 X-CLIP 相似度分层随机抽取 **50–60 个独立来源视频**，执行与 E2b 黄金子集完全一致的双盲人工三值核验与 `R_core` 评分，仅用于检验新旧数据集的**探索性方向一致性**（如实限定推断强度）。

### 10.3 Experiment E3：基于 VidHalluc 既有事实依据的最小配对靶干预（RQ2）
- **四臂结构与共享基础**：保留 **`Base / Sup / Con / Neu`** 四臂，主设置以 **`Base = A0`**（纯文本无外部视觉张量）实现 Video-LLaMA 与现代 VLM 完全对齐；保留一个较小的 **`Base = B3`（60–80 视频，仅 Video-LLaMA）** 锁定视觉敏感性子集衔接历史实验 B。
- **利用 VidHalluc 半自动构造 + 人工必审关口**：
  - 从经 YouTube ID 去重（确保与 VideoUFO 零源重叠、且每来源视频仅取 1 个 clip！）的 VidHalluc `ach_mcq.json` / `ach_binaryqa.json` 中抽取 **150–180 个独立来源片段**；
  - 将其真实主动作作为候选 $f^+$，从高混淆错误选项中挑选候选 $f^-$，半自动生成 `Sup / Con / Neu` 三条单句草稿，**随后强制经过轻量人工审核（必审 4 条红线）**：① 观看静音视频确认 $f^+$ 确实清晰存在；② **确认 $f^-$ 确实与视频严格冲突**（绝不能仅因为是选择题干扰项就默认反事实——排除“次要并存动作”或“仅非最佳答案”的伪冲突！）；③ 确认 `Neu` 对靶事实严格中性、与视频其余画面相容且无其他错误事实；④ 核验句法自然度并用各模型 tokenizer 实测 subword 长度匹配（配对差 $\le 2$ tokens）。
- **防“构造—评价循环自证”铁律**：
  - 虽然靶事实 $(f^+, f^-)$ 取自 VidHalluc，但评价模型输出时**绝不把原 MCQ 选项或编辑模板喂给同一套启发式字符串脚本充当唯一裁判**！主结果采用 **L4 五态独立判定（允许 $f^+$ 与 $f^-$ 同时共存：`NEITHER / ONLY_POS / ONLY_NEG / BOTH / AMBIGUOUS`）**，通过经人工抽核验证的跨家族语义判定 + 人工复核全部边界/共存案例完成；
  - **两项主终点（`Con vs. Neu`，2 项 Holm $\alpha = 0.05$）**：目标错误采纳概率差 $\Delta P(f^-)$ 与目标正确表达概率差 $\Delta P_{\text{supp}}(f^+)$；次要拆解：`Sup vs. Neu`、`Neu vs. Base` 及按 `Base` 基线分层报告错误新增与错误持续；
  - **自然 VideoUFO 冲突生态效度子集**：在 DREAM-1K / VidHalluc 真实 VideoUFO 检索池中筛选 50–60 组自然冲突 vs. 自然支持对照，明确限定为观察性关联验证，不与人工最小配对因果效应混为一谈。

### 10.4 Experiment E5：三单元跨模型与跨数据集边界验证（RQ3）
- **两套主力系统**：① 冻结 `Video-LLaMA / Vicuna-7B / cross-gating`（$I_{\text{x-gate}}$）；② **1 款现代开源视频 VLM**（候选：`Qwen2.5-VL-7B-Instruct` Apache 2.0、`LLaVA-Video-7B-Qwen2` Apache 2.0、`InternVL2.5-8B` MIT；在 E2a 试点中经 20 条 timed smoke 实测端到端峰值显存与 16 帧接口后事前锁定唯一主力，防模型选择偏差）。
- **干净共同查询对照（杜绝混杂）**：在**完全相同的 DREAM-1K 队列（E2b）与完全相同的 VidHalluc 靶干预队列（E3）**上，让两个模型均在**短字幕对齐指令**与**相同外部文本证据**下运行，直接实现固定数据集上的跨模型/跨接口比较；同时结合已有 MSR-VTT 冻结结果（及现代 VLM 在 MSR-VTT 桥接 50–60 视频或 200 视频子集上的平行短字幕运行），构成清晰的跨数据集对照（如实声明三单元 L 型结构，不夸称为完整 2×2 因子设计）。
- **MSVD 匹配 `k=0` 的取舍建议**：由于 DREAM-1K 已提供更强、更干净的跨数据集事实外部验证，**将 MSVD trace-matched 3 臂（`A0/A1/A3`，~1.5–2.2 GPU 小时）降级为可选补充实验**——若算力充裕则补跑放入正文/补充材料闭合历史 MSVD 零点缺口；若算力紧张则不强求，以免分散核心聚焦！

---

## 11. 统计分析、来源聚类与数据泄漏防御

1. **严格的外部记忆零重叠与去重审计**：
   - **YouTube 来源 ID 匹配**：VidHalluc 文件名直接含 11 位 YouTube ID（如 `vcCwvRYqU2I_clip_4`），ARGUS 与 DREAM-1K 中凡源自 YouTube 的片段均先提取精确来源 ID 与 VideoUFO 的 `345,437` 个来源 ID 做精确求交，命中者一律剔除出评价队列；
   - **来源级内部去重与聚类**：本轮实查已发现 **VidHalluc 同一 YouTube 视频含多个 clip（如 `_clip_2` 与 `_clip_4`）**、**ARGUS 含同源分镜（`-h5KF2SffqI-Scene-*`）**；因此在抽样时强制“每来源最多 1 clip”（若保留多 clip 则一律以来源视频为完整聚类单元）；
   - **无可靠来源 ID 时的内容近重复防御**：对于电影/股票素材片段（如 DREAM-1K 部分非 YouTube 片段），先执行精确文件哈希与关键帧感知哈希（pHash）+ X-CLIP 视觉向量近邻筛查（剔除视觉相似度异常接近 `> 0.98` 的疑似近重复项）；
   - **诚实声明不可排除边界**：明确说明精确 ID 与近重复审计不能排除全部跨上传重编码片段，也**无法排除现代开源 VLM 在上游预训练阶段对公开数据集或同类视频的潜在暴露**，在 Limitations 中如实记录；
   - **绝对隔离检索库与评价真值**：新数据集的任何真值描述或 QA 严禁写入 VideoUFO 索引；E3 利用 VidHalluc 事实构造干预文本明确界定为受控实验干预，绝不包装为可部署检索算法。
2. **统计推断与多重校正规范**：
   - 主推断统一采用固定种子的 **10,000 次来源聚类配对自助法（95% 逐项百分位 CI）** 与 **10,000 次双侧 Monte Carlo 符号翻转置换 $p$ 值**；
   - 主检验家族严格收敛并独立做族内 Holm 校正（$\alpha = 0.05$）：E2b 主族为 2 主终点 × 2 核心预算跃迁（`A1 vs. A0`、`A7 vs. A1`）= **4 项 Holm**；E3 主族为 `Con vs. Neu` 的 2 项靶终点 = **2 项 Holm**；现代 VLM 各自内部执行同等族内 Holm；
   - 严守“不显著不等于等效或非劣效”；严禁在看到 E2a 试点方向后将 E2a 并入 E2b 正式确认集。

---

## 12. 自动评价器的十项刚性审查、校准与最小人工核验

本轮绝不因某评价器发表于顶会就盲目信任，而是按十项标准逐一审查并设立硬性防偏差护栏：

| 审查项 | DREAM-1K 官方 AutoDQ | ARGUS 官方 ArgusCost | VCapsBench 官方 TextQA | **本方案 `v1.2` 修正护栏** |
|---|---|---|---|---|
| 1. 是否读原视频 | 否（只读参考与预测） | 否（只读参考与预测） | 否（只读预测与 QA） | 以 80 视频双盲人工静音看片作为原视频真值锚点 |
| 2. 是否读生成字幕 | 是 | 是 | 是 | 是 |
| 3. 是否需正确答案 | 需人工参考描述 | 需人工参考描述 | 需预定义 Yes/No 答案 | 仅用于评测，严禁向生成器泄漏 |
| 4. 是否允许信息不足 | Prompt 有 `neutral`，但 **L254 源码将其计作 Precision 错误！** | Prompt 有 `underdetermined`，但 **L118 源码将其计作 cost=1！** | Prompt 显式支持 `Unanswerable` | **直接读取三值 NLI 原始标签，把 `contradiction` 与 `neutral/underdetermined` 彻底分列汇报！** |
| 5. 是否误判未覆盖真实事实为幻觉 | **官方标量会误判** | **官方标量会误判** | 不会（标 `Unanswerable`），但漏检题外幻觉 | 通过人工看片抽查定量估计 `Unmentioned` 中真实细节 vs. 真实幻觉的精确比例 |
| 6. 长度偏好风险 | 长句易获高 Recall、低 Precision | 长句易获低 Omission、高 Hallucination | 长句覆盖更多题 | 跨模型强制统一短字幕指令，且在验证集上按句长分层审计评价器误差 |
| 7. 模型家族自我偏好 | 取决于裁判模型 | 取决于裁判模型 | 取决于裁判模型 | **强制跨家族裁判**：若生成器为 `Qwen2.5-VL`，则开源文本 NLI 裁判选用 `Llama-3.3` / `DeepSeek` / `InternLM` 等不同家族模型 |
| 8. QA 常识盲猜风险 | N/A | N/A | **高风险**（无字幕也可猜对常识题） | 若涉及 QA 评测，强制运行**空字幕基线**与**无关字幕基线**，剔除盲猜准确率过高的常识题，并随机化选项位置、要求输出显式字幕依据 |
| 9. 闭源 API 成本 | 官方硬编码 Azure GPT-4 | 官方支持开源 + GPT-4o | 官方默认 Gemini / GPT-4o | 优先采用本地 A6000 开源强 LLM 裁判（`$0` API 费）；严禁直接宣称与 GPT-4 等价，必须用独立验证集校准！ |
| 10. 校准与验证隔离 | — | — | — | **30 视频校准集（E2a）调优提示，50 视频独立验证集（E2b 抽样）无偏汇报人机 $\kappa$、按臂误差与长度偏置！** |

---

## 13. 真实资源成本估算与阶段性停止规则（详表见 `RESOURCE_BUDGET_V12.md`）

锚定 BN 服务器 RTX A6000（48 GB）历史实测基准（Video-LLaMA `~2.25 秒/查询`），并**完整计入详细事实评估所需的大量本地 LLM 裁判调用成本**（绝不只算生成次数！），对比三条执行方案：

| 成本与决策维度 | **方案 A：最低成本兼容性试点（推荐唯一首步）** | **方案 B：最小充分 Q2 完整方案（DREAM-1K + VidHalluc）** | **方案 C：止损回归 `v0.8`（投 SCI Q3）** |
|---|---|---|---|
| **数据下载量** | DREAM-1K `3.85 GB` + VidHalluc `ACH` 子集（约 1–2 GB） | 同左（总计仅 **~5–6 GB**；省去 ARGUS 16.75 GB、VCapsBench 23.4 GB 与 VATEX 大规模下载！） | **0 GB** |
| **新查询 X-CLIP 与检索** | 50 条视频特征提取与 VideoUFO 检索（~0.2 h） | DREAM-1K 1,000 + VidHalluc 200 视频抽特征与检索（**~1.5–2.5 GPU 小时**） | **0 h** |
| **模型生成推理（A6000）** | 30 DREAM-1K × 3 臂 × 2 模型 + 20 VidHalluc × 4 臂（~340 次生成：**~0.3–0.6 GPU 小时**） | DREAM-1K 600–800 独立源 × 3–4 臂 × 2 模型（~4,500–6,000 次）+ VidHalluc E3 160–180 × 4 臂 × 2 模型 + `Base=B3` 60–80 子集（~1,800 次）+ 可选 MSVD 3 臂（~2.0 h）：**~10–18 GPU 小时** | **0 h** |
| **本地开源三值裁判运行** | 30 视频 × 6 臂（~180 次 NLI 判断：**~0.2–0.5 GPU 小时**） | 全量 ~6,000–7,500 条生成字幕的三值事件/命题 NLI 评估（本地 A6000 运行 14B–32B AWQ 跨家族裁判，~3–5 s/条）：**~6–11 GPU 小时** | **0 h** |
| **GPU 总耗时** | **~0.7–1.5 GPU 小时** | **~18–32 GPU 小时**（含全部推理、检索与本地裁判） | **0 GPU 小时** |
| **商业 API 费用** | **$0**（本地开源裁判；可选花 ~$5–$10 抽核官方 GPT-4o/DeepSeek API 对齐度） | **$0**（本地开源跨家族裁判；可选花 ~$30–$60 在 200 视频上跑一次官方 AutoDQ API 敏感性复核） | **$0** |
| **真实人工核验工作量** | 30 DREAM-1K 静音看片核验裁判精度与 `neutral` 误报 + 20 VidHalluc 审核：**~8–12 人时** | 80 DREAM-1K（30 校准 + 50 独立验证含 L3：~12–16 h）+ 50–60 MSR-VTT 方向桥接（~8–12 h）+ 160–180 VidHalluc E3 严格冲突与自然度审核（~8–12 h）+ 50–60 自然冲突复核与裁决（~4–6 h）= **~32–46 人时**（较 `v1.1` 130–196 人时显著下降，且仍以方案 A 实测为准，不提前承诺精确压缩比！） | **0 人时** |

### 阶段性停止规则（杜绝沉没成本滚雪球）
1. **方案 A 后立刻止损（转方案 C）的条件**：① 本地开源三值裁判在 30 视频校准后与人工看片一致性仍极差（$\kappa < 0.60$）或存在严重跨臂/长度系统偏置且无法修正；**或** ② 短字幕在 DREAM-1K 核心事件子集上依然出现严重地板压缩导致完全无测量分辨率；**或** ③ VidHalluc 干扰项经人工审核超过半数并非严格视频冲突（无法低成本产出干净靶对）；
2. **E2b 完成后止损扩展的条件**：若 DREAM-1K 正式运行结果高度清晰且资源紧张，可果断放弃可选的 MSVD 3 臂与可选第 4 单元，直接聚焦核心三单元收尾成文。

---

## 14. 未来论文的统一论证结构与结果分支预案

全篇论文紧紧围绕同一个核心科学问题展开，**绝不把新数据集写成割裂的第二项研究**：

| 论文部分 | 对应 RQ | 使用数据集 | 使用模型 | 资产状态（已有 / 新增） | 核心论证与主张边界 | 移入补充材料的旧实验 |
|---|---|---|---|---|---|---|
| **第一部分：大规模外部检索的可得性与生成效用** | 背景铺垫 | VideoUFO（6 尺度）+ MSR-VTT `val497` | 冻结 Video-LLaMA | **100% 已有（`v0.8`）** | 确立“检索相似度提升 $\ne$ 固定预算生成收益”的现象起点 | 48 格验证集全表、历史阴性机制与 LOO 全表 |
| **第二部分：零外部证据参照下的净效用与跨基准预算梯度** | 背景铺垫 | MSR-VTT `test2690`（A/B 实验）+ MSVD `test670` | 冻结 Video-LLaMA | **已有为主**（可选新增 MSVD 3 臂） | 确立 `A1 > A0 > A7` 条件净效用及实验 B 的视觉锁定与窄相似度边界，自然引出为何必须转向视频事实层 | 历史 Oracle 池衰减、重拟合 IDF 与全部 B 匹配细表 |
| **第三部分：外部检索如何改变视频事实（配对事实转移）** | **RQ1** | **DREAM-1K（主）** + **MSR-VTT 50–60 视频桥接（探索）** | Video-LLaMA | **新增 E2a/E2b/E2c** | 展示三值解耦 `N_Contra_ref`、`R_core_ref`、`P_Unment_ref`、80 视频黄金人工验证与 L3 纠错/删错/污染拆解；诚实限定桥接强度 | 30 视频校准细节、官方 `AutoDQ` 标量敏感性对照 |
| **第四部分：视觉受控的靶事实冲突与行为因果效应** | **RQ2** | **VidHalluc（主）** + 自然 VideoUFO 冲突观察子集 | Video-LLaMA | **新增 E3** | 共享 `Base` 四臂（`Base/Sup/Con/Neu`）L4 五态行为因果识别（`Con vs. Neu`）+ `Base=B3` 敏感性与自然冲突生态佐证 | 全部自然度盲审与 tokenizer 长度分布细表 |
| **第五部分：跨模型接口与跨数据集适用边界** | **RQ3** | **共同 DREAM-1K & VidHalluc**（+ MSR-VTT 共享子集） | Video-LLaMA vs. **1 款现代开源视频 VLM** | **新增 E5** | 统一短字幕指令下的跨接口对比与跨数据集验证；诚实声明三单元 L 型边界与长度敏感性 | 现代模型详细描述模式长度敏感性对照 |
| **第六部分：讨论、限制与实践启示** | 综合收敛 | 全部 | 全部 | 综合分析 | 揭示传统词面指标（CIDEr）、显式事实冲突与核心事件覆盖的分离规律，给出无夸大的外部检索评测与部署建议 | — |

### 新旧衔接失败或阴性结果分支预案
- **若 DREAM-1K 未复现 MSR-VTT 的大预算退化**：绝不强行宣称普遍规律！结合 MSR-VTT 50–60 桥接集结果，系统辨析究竟源于**领域/视频复杂度差异、短字幕长度机制、还是 CIDEr 词面惩罚与真实事实错误之间的分离**，作为数据域与指标边界的重要诊断贡献如实呈现；
- **若现代 VLM 在共同 DREAM-1K / VidHalluc 查询上对外部冲突显著更具抗性**：将其升格为“跨多模态接口脆弱性分化”核心发现（对比旧式 32-token cross-gating 与原生多帧指令 VLM 的外部依赖差异），绝不强拗一致，也不宣称现代模型绝对免疫。

---

## 15. 尚待用户、ChatGPT 与导师裁决的问题

1. **是否认可“DREAM-1K（E2 主）+ VidHalluc（E3 主）”的最小公开双集架构**（并正式取消 VATEX 与大规模 MSR-VTT 从零标注），还是更倾向改用 **ARGUS** 或 **VCapsBench（过滤语义维）**？
2. **是否授权先启动极低成本的方案 A（兼容性试点：30 DREAM-1K + 20 VidHalluc，~0.7–1.5 GPU 小时 + ~8–12 人时）**，先用实测数据检验 DREAM-1K 短字幕分辨率、本地开源三值裁判精度与 VidHalluc 靶冲突纯度，再做完整方案 B 的 Go/No-Go？
3. **现代主力模型与跨家族本地裁判配对**：是否同意在方案 A 阶段先锁定一款主力模型（如 `Qwen2.5-VL-7B-Instruct` 或 `LLaVA-Video-7B-Qwen2`）并配另一不同家族开源文本模型（如 `Llama-3.3` / `DeepSeek` / `InternLM`）作为本地三值 NLI 裁判，做到 `$0` 商业 API 依赖？
4. **MSVD trace-matched 3 臂（`A0/A1/A3`，~1.5–2.2 GPU 小时）** 是作为方案 B 的标准组件保留，还是直接移出正文以进一步聚焦？
5. **时间与机会成本最终权衡**：对比方案 B（~18–32 GPU 小时 + ~32–46 人时）与方案 C（直接投稿已完备的 `v0.8` 至 SCI Q3，0 新增成本），结合法律 RAG 论文进度确认是否值得启动方案 A。

---

## 16. 真实参考文献与原始公开资源链接（全部经本轮核实）

1. **Fang, C., Li, J., Xie, Y., Jiang, W., Zeng, G., Li, Z., & Fang, Y. (2026).** *Retrieval Augmented Video Captioning with Quality-Aware Re-Ranking and Cross-Gating Fusion.* **Displays**, 93: 103420. DOI: `10.1016/j.displa.2026.103420`.
2. **Wang, J., Yuan, L., Zhang, Y., & Sun, H. (2024/2025).** *Tarsier: Recipes for Training and Evaluating Large Video Description Models* (arXiv:2407.00634) & *Tarsier2* (arXiv:2501.07888) — **DREAM-1K & AutoDQ**. HF: https://huggingface.co/datasets/omni-research/DREAM-1K ; Code: https://github.com/bytedance/tarsier (`evaluation/metrics/evaluate_dream_gpt.py`).
3. **Rawal, R., Shirkavand, R., Huang, H., Somepalli, G., & Goldstein, T. (2025).** *ARGUS: Hallucination and Omission Evaluation in Video-LLMs.* **ICCV 2025** (arXiv:2506.07371). Project: https://ruchitrawal.github.io/argus/ ; HF: https://huggingface.co/datasets/tomg-group-umd/argus ; Code: https://github.com/JARVVVIS/argus (`code/nli_eval.py`, `code/compute_metrics.py`).
4. **Li, C., Im, E. W., & Fazli, P. (2025).** *VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding.* **CVPR 2025**. HF: https://huggingface.co/datasets/chaoyuli/VidHalluc (`ach_mcq.json`, `ach_binaryqa.json`, `tsh.json`).
5. **Zhang, S.-X., Wang, H., Huang, D., Li, X., Zhu, X., & Yin, X.-C. (2025).** *VCapsBench: A Large-scale Fine-grained Benchmark for Video Caption Quality Evaluation.* arXiv:2505.23484. Code: https://github.com/GXYM/VCapsBench (`LLM4eval-m.py`); HF: https://huggingface.co/datasets/somos99/VCapsBench (`VCapsBench_100KQA.jsonl`).
6. **Wang, Z., Feng, B., Lai, Z., Li, S., Lu, Y., Cao, M., Huang, P., & Wang, X. S. (2026).** *Putting Captions to the Test: Evaluating Video Caption Quality through Multiple-Choice Question Answering (CapQuiz).* **ACL 2026** (`2026.acl-long.777`). Page: https://machinelearning.apple.com/research/video-caption-quality ; ACL: https://aclanthology.org/2026.acl-long.777/ (数据尚未公开下载).
7. **Chai, W., et al. (2025).** *AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark (VDC).* **ICLR 2025** (arXiv:2410.03051). HF: https://huggingface.co/datasets/wchai/Video-Detailed-Caption ; Code: https://github.com/wenhaochai/aurora .
8. **Liu, H., & Wan, X. (2023).** *Models See Hallucinations: Evaluating the Factuality in Video Captioning (FactVC).* **EMNLP 2023**. Code: https://github.com/NiteRainHL/FactVC .
9. **AI at Meta (2025).** *Perception Encoder / PE-Video Dataset.* HF: https://huggingface.co/datasets/facebook/PE-Video (CC BY-NC 4.0).
10. **Li, W., Li, J., Ramos, R., Tang, R., & Elliott, D. (2024).** *Understanding Retrieval Robustness for Retrieval-Augmented Image Captioning.* **ACL 2024**.
11. **Luo, W., Hu, Z., Wang, X., Yu, Z., Zhang, H., & Huang, Z. (2026).** *Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation.* **ACL 2026** (arXiv:2605.13277).
12. **Zhao, B., et al. (2026).** *MAD-RAG.* **ICML 2026**; **Sun, J., et al. (2024).** *SURf.* **EMNLP 2024**; **Taguchi, C., et al. (2025).** *Adaptive-k.* **EMNLP 2025**; **Zhang, H., et al. (2026).** *SpecUBench.* **CIKM 2026**.
13. **Wang, W., & Yang, Y. (2025).** *VideoUFO.* **NeurIPS 2025 Datasets & Benchmarks**; **Xu, J., et al. (2016).** *MSR-VTT.* **CVPR 2016**; **Chen, D., & Dolan, W. (2011).** *MSVD.* **ACL 2011**.
14. **Qwen Team (2025).** *Qwen2.5-VL-7B-Instruct (Apache 2.0)*; **Zhang, Y., et al. (2024).** *LLaVA-Video-7B-Qwen2 (Apache 2.0)*; **Chen, Z., et al. (2024/2025).** *InternVL2.5-8B (MIT)*.
