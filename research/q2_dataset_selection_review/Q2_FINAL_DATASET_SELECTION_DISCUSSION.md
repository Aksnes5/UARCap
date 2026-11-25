# UARCap Q2 全球视频事实评价数据集调研与最终选型总讨论报告（独立审阅主稿）

> **文档定位**：本报告是 UARCap Q2 数据集系统性调研的**独立中文总报告**。用户无需阅读其余六份专项文档即可直接依据本报告做出最终科研选型裁决。  
> **调研日期**：2026-11-25｜**本轮状态**：`DATASET_SELECTION_RESEARCH_COMPLETE_AWAITING_SCIENTIFIC_DECISION`  
> **严守红线**：本轮未启动 GPU、未下载完整视频或大模型、未修改任何正式实验代码（`research/uarcap_q2_experiments/`）、未预冻结新方案。  
> **证据分级标记**：`[论文原文]` = 直接核验自正式论文/arXiv HTML；`[官方源码实查]` = 直接读取 GitHub 官方仓库源码；`[公开元数据实测]` = 本轮直接下载并统计公开 JSON/Parquet/HF 文件树；`[科研推断]` = 基于上述一级证据的独立科学判断；凡公开资料未披露项均如实标注 `UNKNOWN`。

---

## 一、执行摘要与最终独立科学裁决

### 1.1 一句话结论

在系统检索全球 **18 个研究方向**、实证盘点 **22 个公开视频事实/描述/幻觉基准**、并直接审计正式论文、官方评测源码与完整公开元数据之后，本轮得出两项核心结论：

1. **不存在“单向碾压一切的完美单一数据集”，但存在清晰的科学最优帕累托前沿**：
   - **推荐首选方案（方案二）**：保留全部既有 **MSR-VTT + MSVD + 百万级 VideoUFO** 历史研究资产，将 E2 自然检索主数据集切换为 **CaReBench（严格按 FineAction 父视频去重至 ≤ 913 条独立视频）**，E3 受控冲突干预数据集**维持 VidHalluc (ACH，精确 11 位 YouTube ID 零重叠过滤)**，自动事实评价**维持 UARCap 自研三值评价器（严禁直接套用官方 CapST/AutoDQ 源码）**。
   - **并列可行备选（方案一）**：维持既有 **DREAM-1K + VidHalluc** 方案不变（需用户先在 Windows 完成 Hugging Face gated 登录授权并下载 3.59 GiB `video.zip`）。
   - **明确否决**：不采用 `VidHal` 替代 `VidHalluc` 作为 E3 主集；不采用 `ARGUS`、`VCapsBench`、`VidCapBench`、`VDC`、`CapQuiz` 作为 E2 主集；不强行用单个数据集同时承担 E2 与 E3；**不扩大数据集数量**。

2. **为什么是“略倾向 CaReBench + VidHalluc”，而非盲从任一方？**
   - 在本轮多维度加权评分（[DATASET_SCIENTIFIC_SELECTION_MATRIX.json](file:///D:/UARCap/research/q2_dataset_selection_review/DATASET_SCIENTIFIC_SELECTION_MATRIX.json)）中：
     - **默认科研权重**下：**CaReBench (`7.93`) vs DREAM-1K (`7.07`)**（领先 `+0.86`）；
     - **高事实严谨与 `Unmentioned` 严控权重**下：**CaReBench (`8.03`) vs DREAM-1K (`6.53`)**（领先 `+1.50`）；
     - **高开放域泛化与工程复用权重**下：**CaReBench (`7.82`) vs DREAM-1K (`7.62`)**（仅领先 `+0.20`，**基本打平**）；
     - **极度看重零代码改动与极短时长（8.9s）权重**下：**DREAM-1K (`7.94`) vs CaReBench (`7.70`)**（DREAM-1K 反超 `+0.24`）。
   - 同时，在 E3 受控冲突干预任务上，**VidHalluc (`8.60`) 在全部权重设定下均显著压过 VidHal (`6.69`)**。
   - 因此，最终决策可浓缩为一个清晰的科研取舍：**用户更希望在 E2 中优先守住“静态对象+动态动作参考完整、压低 `Unmentioned` 灰区与校准返工风险、零下载阻力（`MIT` / `gated=False`）”的 CaReBench，还是优先守住“5 大开放域视觉多样性、8.9s/59 词天然粒度贴合与零代码改动”的 DREAM-1K？** 若没有必须坚守 5 类影视/动画混合域的硬约束，**方案二（CaReBench + VidHalluc）是当前综合风险最低、人工成本最省、二区论文事实归因最有说服力的选择**。

---

## 二、全球调研覆盖范围与最强五强候选

### 2.1 18 个检索方向与五大功能类型

围绕 UARCap Q2 的三大核心研究问题（**RQ1**：自然检索预算 `A0/A1/A7` 下正确事实/错误事实/核心覆盖变化；**RQ2**：固定视觉输入下 `Base/Sup/Con/Neu` 外部文本冲突是否定向诱发幻觉；**RQ3**：冻结 16 帧旧模型 `Video-LLaMA/Vicuna-7B` 与 BN 既有现代模型 `Qwen3.8-27B-FP8` 跨代际复现），本轮覆盖 CVPR、ICCV、ECCV、NeurIPS、ICML、ICLR、ACL、EMNLP、AAAI、ACM MM、TMLR、arXiv、GitHub 与 Hugging Face 共 **18 个方向**：密集字幕评测、细粒度长描述基准、视频事实一致性、字幕幻觉基准、幻觉检测、事实核验、细粒度视频理解、对象中心标注、事件中心标注、时空分离理解、人工元评价、视频文本蕴含、事实证据定位、检索困难负例、视频反事实推理、原子命题基准、检索增强视频生成/冲突证据评测、细粒度视频问答。

所有 22 个实查候选按角色划分为五大功能类型：
- **类型 A（纯人工细粒度视频参考描述 → E2 主选）**：CaReBench、DREAM-1K、ARGUS、Shot2Story；
- **类型 B（结构化对象/事件/场景图 → E2 拆解辅助，须严查来源）**：CaReBench（缓存 GPT-4 字段）、Action Genome；
- **类型 C（真实视频 + 对比/冲突陈述对 → E3 主选）**：VidHalluc、VidHal、TempCompass、VideoCon、VITATECS、VIOLIN；
- **类型 D（细粒度视频问答 → 辅助敏感性验证，不能直接替代开放字幕事实率）**：VCapsBench、VidCapBench、CapQuiz、VideoHallucer、EventHallusion、COVER、MVBench；
- **类型 E（旧模型字幕的人类错误评分 → 仅可做评价器元验证，不能评我们新模型输出）**：FactVC（ActivityNet-Fact / YouCook2-Fact）。

### 2.2 硬准入过滤结果与全球最强 5 个候选

经过 7 项硬准入条件（可合法公开获取、有效独立样本量足够、真实视觉自然视频、非大规模 LLM 合成参考、支持明确事实判定、无强音频/剧情依赖、非旧模型专属静态评分），**22 个基准中仅 6 个具备实际竞争力，其中真正适合进入最终决策层的 Top-5 为**：

| 排名 | 候选数据集 | 拟定角色 | 出处与许可 | 一手实测核心特征 | 核心短板 | 默认总分 |
|---:|---|---|---|---|---|---:|
| **1** | **VidHalluc (ACH)** | **E3 首选** | CVPR 2025；`gated=False`；`cc-by-nc-nd-4.0`；ACH `15.60 GiB` | `2,159` 对、`3,897` 独立片段、**`1,586` 显式 11 位 YouTube ID**；真/干扰词长 **`4.23 vs 4.25`** 词 `[公开元数据实测]` | 聚焦动作维（不含静态颜色/数量替换）；需对 150–180 正式样本做轻量互斥性看片审核 | **8.60** |
| **2** | **CaReBench** | **E2 推荐首选** | arXiv:2501.00513v2；`gated=False`；`MIT`；`3.65–3.66 GiB` | 1,000 片段（**913 独立 FineAction 父视频**），均长 **`14.35s`**，双人+专家人工时空分离长描述 **`227.95` 词**，**94% 含 `14.5` 词开篇总览** `[公开元数据实测]` | 100% 源自 FineAction 人类活动；`events`/`objects` 为 GPT-4 提取；匿名父 ID 无法精确查重 YouTube ID | **7.93** |
| **3** | **DREAM-1K** | **E2 并列强备选** | arXiv:2407.00634；`gated: auto`；`apache-2.0` 元数据；`3.59 GiB` | 1,000 片段覆盖 **5 大开放领域**（影视/动画/YouTube/TikTok/Stock），均长 **`8.9s`**，人工描述 **`59.3` 词**，既有离线工程 100% 就绪 `[论文原文/HF实测]` | 偏事件轻静态对象（~5.79 对象/视频），增加 `Unmentioned` 漏检风险；ID `1..1000` 全匿名；需 HF gated 授权下载 | **7.07** |
| **4** | **VidHal** | **E3 多维备选** | TMLR 2025 / arXiv:2411.16771；`gated=False`；`cc-by-4.0`；`5.08 GiB` | 1,000 片段覆盖 **5 维度**（属性/对象/动作/顺序/方向），3 级最小编辑幻觉字幕组内词长差中位数 `0.0` `[公开元数据实测]` | **锚点与幻觉字幕均由 GPT-4o 生成**（仅 10% 人工复核且一致性 87%）；**`261/1,000` (26.1%) 为 CLEVRER 合成 3D 球体**；无 YouTube ID | **6.69** |
| **5** | **ARGUS** | **E2 次优备选** | ICCV 2025 / arXiv:2506.07371；`gated=False`；`cc-by-nc-sa-4.0`；`15.60 GiB` | 500 片段精细人工长描述（实测前 25 条均值 ~485 词），提出显式 `underdetermined` 幻觉分类 `[论文原文/公开元数据实测]` | 仅 500 总样本；时长 **15–90s** 严重不适配 16 帧旧模型；485 词文学描写与 10 词短字幕错配；官方 `ArgusCost-H` 仍将 `underdetermined` 计为错分 | **6.67** |

**重点淘汰项说明**：
- **CapQuiz（ACL 2026）**：设计理念优秀但**官方从未释放公开数据或代码**，硬准入淘汰。
- **VDC / AuroraCap（ICLR 2025）**：500 词参考高度由 GPT-4o 合成，本轮实测 `VDCScore_qa/short.jsonl` 存在大量未清洗类别占位词，且体量达 `74.61 GiB`，硬准入淘汰。
- **VCapsBench / VidCapBench**：均面向文生视频（T2V）细粒度描写（含大量调色/景深/模糊/运镜噪声），且为封闭式 VLM/GPT-4o QA，**无法捕捉 VideoUFO 自然检索随机注入的列表外幻觉**（例如检索注入的“厨房里的棕色小狗”若不在预设 QA 列表中，封闭 QA 完全盲视）。
- **VideoCon / VITATECS**：Hugging Face 仓库均返回 `401 Unauthorized` 且侧重检索对比而非开放字幕事实评测。

---

## 三、本轮打破预设偏见的五项一手实证发现

本轮未停留在论文摘要或二手博客，而是直接读取正式 arXiv HTML 全文、官方 GitHub 源码及完整公开 JSON，得出五项决定性一手证据：

### 发现 1：CaReBench 的人工段落极强，但它的 `events` / `objects` 结构化字段不是人工 Gold！

- **真人工 Gold 部分** `[论文原文 3.2.1–3.2.2 + 公开元数据实测]`：
  - CaReBench 每条视频的 `caption`、`spatial_caption`、`temporal_caption` 由 **2 名经过培训的标注员独立标注空间与时间维度**，再由 **专家审查员逐条核片合并**，并严格执行 4 条清洗规范（**显式删除音频/语音内容**、删除情绪/主观臆测、消除冗余、补充遗漏细节）。
  - 实测全部 1,000 条记录：`caption` 均值 **`227.95` 词**（中位数 `218`，最小 `150`，P10=`184`，P90=`283`，最大 `474`）；人工拆分的 `spatial_caption` 均值 **`113.22` 词**、`temporal_caption` 均值 **`102.26` 词**。
  - **意外惊喜**：全部 1,000 条中 **`940/1,000` (94.0%)** 以一句独立的 **General Overview（总览句）** 开头（如 *"The video shows a woman performing yoga on a mat indoors."*），均值仅 **`14.5` 词**（中位数 `14.0` 词）——天然构成与 8–18 词短字幕粒度完美对齐的**人工核心事件锚点**！
- **非人工 Gold 揭秘（重大止损发现）** `[官方源码实查 + 公开元数据实测]`：
  - 直接审计官方仓库 `MCG-NJU/CaReBench/utils/dream_gpt.py` 发现：文件第 1 行赫然保留 `# Copyright (2024) Bytedance Ltd.`，直接复刻了 DREAM-1K 的 GPT-4 抽取脚本！
  - 实测 `CaReBench/json/metadata.json` 全部 1,000 条记录：
    - `objects` 共 `10,032` 条（均值 **`10.03`/视频**，范围 4–26），其中 **`10,023 / 10,032` (`99.9%`)** 机械以 `"There is "` 或 `"There are "` 开头——逐字匹配 `dream_gpt.py` L133 的 GPT-4 prompt 模板；
    - `events` 共 `6,942` 条（均值 **`6.94`/视频**），最大值恰好卡死在 **`10`**——逐字匹配 `dream_gpt.py` L105 `"Extract at most 10 key events"`！
  - 此外，核验 `arXiv:2501.00513v2` 全文确认：论文**未报告**人工标注者间一致性（Kappa/ICC = `UNKNOWN`），也**未报告** `CapST` 与人工判断的相关系数（`UNKNOWN`）。
- **对 UARCap 的科学意义**：若采用 CaReBench，论文必须诚实写明：**我们以双人+专家人工撰写的 `caption` / `spatial_caption` / `temporal_caption` 及其人工开篇总览句作为人类参考真值（Human Reference Gold），绝不将公开 JSON 中的 `events`/`objects` 字段冒称为“人工认证原子事实”**。

### 发现 2：三大官方自动指标（AutoDQ / CapST / ArgusCost-H）全部存在 `Unmentioned` 误判缺陷

直接审计三套官方评测代码确认：
- **DREAM-1K 官方 `AutoDQ`**（`bytedance/tarsier/evaluation/metrics/evaluate_dream_gpt.py` L254）`[官方源码实查]`：计算 Precision 时传入 `is_recall=True`，导致 `neutral`（参考未提及但视频可能为真）被**直接计为精确率错误**；且事件提取 prompt 明令 `"NOT STATIC INFORMATION"`，结构性盲视静态对象幻觉。
- **CaReBench 官方 `CapST`**（`MCG-NJU/CaReBench/utils/dream_gpt.py` L446 & L448）`[官方源码实查]`：完整继承了同一行代码缺陷——计算 `events_score_p` 与 `objects_score_p` 时均传入 `is_recall=True`，依然把 `neutral` 当作幻觉扣分！
- **ARGUS 官方 `ArgusCost-H`**（`JARVVVIS/argus/code/compute_metrics.py` L306–L313）`[官方源码实查]`：虽然论文将 `underdetermined` 单独分类，但汇总公式中明确令 `underdetermined` 惩罚权重与 `contradicted` 相同（均为 `1.0`），仍混淆了“参考遗漏”与“明确编造”。
- **核心结论**：无论 E2 选 DREAM-1K 还是 CaReBench，**绝不能直接照搬原数据集的官方 GPT-4 评分脚本**；必须保留 UARCap 既有设计的**三值参考事实评价器（`ENTAILED / CONTRADICTED / UNMENTIONED`）+ 独立开源跨家族 LLM + Pilot A2/A3 盲法人工看片校准与两阶段误差校正**。

### 发现 3：CaReBench 存在明确的父视频聚类，且比 DREAM-1K 多出一项“可显式清洗”的统计优势

- `[公开元数据实测]` 统计 `CaReBench/json/metadata.json` 的 1,000 个视频文件名（如 `v_00002068_11.mp4`）发现：它们来自 **`913` 个独立 FineAction 父视频**，其中 **`68` 个父视频贡献了 2–6 个片段**（共计 155 条多片段记录，最多为 `v_00005394` 的 6 条片段）。
- 相比之下，DREAM-1K 公开视频全部匿名编号为 `1..1000`，**完全无法核查**是否存在同部电影/同条 YouTube 视频切出多个片段。
- 因此，在 CaReBench 上可以一键执行 **1 片段/父视频严格去重（1,000 → 913 独立父视频）**，彻底阻断 Pilot→Formal 同源泄漏与配对统计中的簇相关偏差；去重后剩余 913 条依然充裕支持 Pilot 30 + Formal 600–750 的统计功效要求。

### 发现 4：VidHalluc (ACH) 在 E3 任务上显著优于 VidHal

- `VidHalluc (ACH)` `[公开元数据实测]`：
  - `ach_mcq.json` 含 `2,159` 组视觉高度相似的自然动作冲突对，覆盖 `3,897` 个独立片段、**`1,586` 个显式 11 位 YouTube 来源 ID** → 可与 BN 本地 VideoUFO 的 `345,437` 个 YouTube ID 做 **100% 精确差集零重叠过滤**；
  - 真实动作均值 **`4.23` 词**（中位数 4） vs 混淆动作均值 **`4.25` 词**（中位数 4），几乎零长度偏置，且既有代码 `uarcap_q2/datasets/vidhalluc.py` 已 100% 实现并完成 CPU 回归验证。
- `VidHal` `[论文原文 arXiv:2411.16771 + 公开元数据实测]`：
  - 虽然覆盖 5 维度且词长平衡极佳（组内差中位数 `0.0`），但**所有正负对比字幕均为 GPT-4o 自动撰写**（仅抽检 100/1,000 条由人工复核，共识率 87%）；
  - 全部 1,000 条中有 **`261` 条 (26.1%) 为 CLEVRER 合成 3D 渲染球体/立方体**，部分 `order` 题甚至考查屏幕字母出现顺序（如 `e, m, n`），不适合 16 帧自然视频字幕模型；
  - 全部采用匿名编号（`attribute_1`），**完全丧失精确 YouTube ID 去重能力**。
- **裁决**：**坚决保留 `VidHalluc (ACH)` 作为 E3 主数据集**；仅在论文局限性或未来扩展中提及多维度（对象/属性）受控干预。

### 发现 5：全球文献创新性查重证实 UARCap Q2 依然独立清晰

专项核查 2024–2026 年文献（包括 CaReBench 原文、EgoInstructor CVPR 2024、CM²/HiCM²、MAD-RAG 2026、Video-DPL、EAEV 等）确认：
- **CaReBench 原文**只做标准视频检索（`ReBias`）和标准无检索长描述（`CapST`），**完全未涉及**检索增强视频字幕、外部证据预算（`A0 → A1 → A7`）的事实非单调性、或冲突外部文本的受控定向干预（`Base/Sup/Con/Neu`），因此复用 CaReBench **与原论文零贡献重叠**。
- 全球范围内尚未出现同类“固定视觉输入下自然检索预算诊断 + 最小冲突文本行为因果 + 跨代际视频模型复现”的系统实证研究。

---

## 四、DREAM-1K vs CaReBench 深度对比与短字幕公平评价设计

### 4.1 核心维度正面交锋

| 对比维度 | **DREAM-1K** | **CaReBench** | 科学胜出方与原因 |
|---|---|---|---|
| **视频领域覆盖** | **5 大开放域**（影视、动画、YouTube、TikTok、Stock） `[论文原文]` | 100% 源自 FineAction **人类日常活动**（4 大类、82 子类） `[公开元数据实测]` | **DREAM-1K 胜**（域泛化更宽，但需注意 MSR-VTT/MSVD 历史实验已补齐开放域基础） |
| **视频时长与 16 帧适配** | 均值 **`8.9s`**（1.8 fps） `[论文原文]` | 均值 **`14.35s`**（>80% 在 5–20s；1.1 fps，**短于 MSR-VTT 的 `15.01s`**） `[公开元数据实测]` | **DREAM-1K 略胜**，但 CaReBench 完全在 16 帧舒适区内 |
| **人工参考深度与静态对象密度** | 均值 **`59.3` 词**，约 `6.16` 事件 + `5.79` 对象（事件中心，静态对象偏少） `[论文原文]` | 均值 **`227.95` 词**（`113.22` 空间 + `102.26` 时间），双人+专家审校，约 `6.94` 事件 + **`10.03` 对象** `[公开元数据实测]` | **CaReBench 显著胜**（对象覆盖高出 73%，极大压低静态幻觉落入 `Unmentioned` 的风险！） |
| **天然核心事件锚点** | 需从 59 词段落抽取核心事件 | **94% 自带 14.5 词人工开篇总览句** `[公开元数据实测]` | **CaReBench 胜**（直接对齐短字幕核心覆盖 `R_core_ref`） |
| **静态 vs 动态幻觉归因** | 难以无损拆分静态与动态参考 | 原生人工拆分 **`spatial_caption` vs `temporal_caption`** `[公开元数据实测]` | **CaReBench 胜**（可额外揭示 A7 注入的错误主要是静态背景/对象幻觉还是动态动作幻觉！） |
| **父视频独立性清洗** | ID `1..1000` 不可核查 | 可精确去重为 **913 独立父视频** `[公开元数据实测]` | **CaReBench 胜** |
| **与 VideoUFO 精确 ID 查重** | 不可精确查重（匿名 ID） | 公开元数据不可精确查重（FineAction 匿名 ID） | **平手**（均须采用高相似度 + 感知哈希近重复筛查并诚实披露残余风险） |
| **短字幕粒度贴合度** | **天然贴合**（59 词参考评 10 词短字幕） | 需显式防粒度错配设计（见 4.2 节） | **DREAM-1K 胜** |
| **获取门槛与许可** | `gated: auto`（国内镜像 403，须 HF 登录下载中转），`3.59 GiB` | **`gated=False`, `MIT`, `3.65–3.66 GiB`**（直连/镜像均可立即下载） | **CaReBench 显著胜** |
| **既有离线代码改动** | **0 行改动** | **约 200 行局部改动**（复用 ~95% 既有代码） | **DREAM-1K 胜** |

### 4.2 关键方法论解答：如何用 228 词的 CaReBench 公平评价 8–18 词短字幕？

如果选择 CaReBench，审稿人必然追问：“拿 228 词的详细长描述去评 10 词单句短字幕，会不会导致短字幕召回率极低、评价失真？”本轮提前设计了四项严密的防错配机制 `[科研推断]`：

1. **绝不把“全段落 Recall”当主指标**：对 10 词短字幕而言，遗漏背景中某个挂画或衣袖褶皱并非模型缺陷。因此：
   - **主精度指标（Precision / Error Rate）**：只从模型生成的 8–18 词短字幕中拆出原子主张（通常每句 2–5 条），逐条对照完整 228 词人工参考判定 `ENTAILED / CONTRADICTED / UNMENTIONED`。**参考越长越完整，对短字幕精度判定反而越有利**——因为模型说对的静态对象更容易被确认 `ENTAILED`，而模型受 A7 检索污染编造的错误对象也更容易与 `spatial_caption` 明确冲突从而抓出 `CONTRADICTED`，而不是掉进模糊的 `UNMENTIONED`！
   - **主核心覆盖指标（`R_core_ref`）**：不拿 228 词全段落当分母，而是直接利用 CaReBench 94% 样本自带的 **14.5 词人工开篇总览句**（外加 6% 轻量抽检补齐）作为核心事件分母，与短字幕处于完全一致的信息压缩层级！
2. **新增静态 vs 动态错误双轨分解（CaReBench 专属论文亮点）**：
   - 将短字幕中判为 `CONTRADICTED` 的错误主张进一步按 CaReBench 原生人工分离的 `spatial_caption` 与 `temporal_caption` 归类为 **静态/对象属性错误** vs **动态/动作事件错误**，直接回答“为什么 A7（7 条检索文本）比 A1 更伤事实性——是因为检索带入了错误的背景对象，还是覆盖了视频主动作？”
3. **严格控制长度偏置**：
   - 同步报告每条字幕的词数、原子主张数（`n_claims`）、归一化每主张错误率（`CONTRADICTED / n_claims`）、正确主张数（`n_entailed`）与 `Unmentioned` 占比（`n_unmentioned / n_claims`），防止“说得越少越安全”的伪精度幻觉。
4. **人工看片校准解决“224px / 16 帧极微细节”边界**：
   - 人工校准界面同步展示模型实际看到的 **16 帧采样条带**，区分“视频真实但 16 帧/低分辨率不可见”与“明确视觉错误”。

---

## 五、VideoUFO 来源重叠风险、16 帧适配与模型/评价器协同

### 5.1 VideoUFO 重叠风险分级声明

百万级 VideoUFO 检索库包含 **`1,091,712` 片段 / `345,437` 独立 YouTube 来源**。本轮严格区分四类来源独立性状态（详参 [DATASET_SOURCE_OVERLAP_AND_MODEL_FIT.md](file:///D:/UARCap/research/q2_dataset_selection_review/DATASET_SOURCE_OVERLAP_AND_MODEL_FIT.md)）：

- **精确可去重（100% 零重叠）**：**VidHalluc (ACH)** 的 3,897 条片段均含显式 11 位 YouTube ID（1,586 独立来源），可直接与 BN 服务器 `/home/bn/videoufo_research/manifests/clips_full.tsv` 精确求差集并强制每来源最多 1 clip。
- **父视频聚类可去重 + 近重复筛查（保留不可识别残余风险）**：**CaReBench**（FineAction 匿名 ID `v_XXXXXXXX_N.mp4`）可 100% 去重为 **913 独立父视频**，并利用 X-CLIP 检索天然产出的 top-1 极高相似度（如 `> 0.90`）+ 帧级感知哈希（pHash/dHash）筛查疑似 VideoUFO 近重复片段，在主分析与“剔除疑似近重复”敏感性分析中双轨报告；同时论文中**诚实声明不宣称未经原始 URL 映射证实的绝对零重叠**。
- **匿名不可验证（父视频与 YouTube ID 均未知）**：**DREAM-1K**（`1..1000`）与 **VidHal**（`attribute_1`）。

### 5.2 16 帧旧模型与现代多模态模型的公平对照

- **历史冻结主模型**：`Video-LLaMA / Vicuna-7B`（`Transformers 4.28.0`，16 帧均匀采样、静音无音频、输出 8–18 词单句短字幕）。CaReBench 均值时长 **`14.35s`**（甚至短于已在历史实验稳定运行 17,940 次的 MSR-VTT `15.01s`），且专家审校阶段**显式剔除了全部音频与语音内容**，与静音 16 帧输入高度契合。
- **BN 既有现代对照模型**：首选已部署验证的 **`Qwen/Qwen3.8-27B-FP8`**（备用 `Qwen3.5-27B`），通过既有 `qwen3_vllm_worker.py` 强制喂入**完全相同的 16 帧与短字幕长度约束**，关闭/控制 thinking 模式并记录完整输入 Trace。
- **跨家族独立开源 LLM 事实评价器**：为避免与 Qwen3.8 生成器发生“同家族自我偏置”，事实评价器优先复用 BN 已有非 Qwen 家族模型（如 **`Mistral Small 3.2 24B`、`Gemma 3 27B`、`Llama 3.3 70B W4A16` 或 `Phi-4 14B`**），所有自动指标必须通过 Pilot A2/A3 盲法双人看片校准与两阶段误差校正后才进入正式统计推断。

---

## 六、三种精炼实验组合比较、双人人时预算与工程迁移范围

### 6.1 三种最小必要实验组合对比

| 维度 | **方案一（维持原案）：DREAM-1K + VidHalluc** | **方案二（推荐首选）：CaReBench (≤913) + VidHalluc** | **方案三（多维备选）：CaReBench + VidHal** |
|---|---|---|---|
| **历史资产继承** | 100% 保留 MSR-VTT + MSVD + VideoUFO | **100% 保留 MSR-VTT + MSVD + VideoUFO** | 100% 保留 MSR-VTT + MSVD + VideoUFO |
| **静态+动态事实分辨力** | 中（偏事件，静态幻觉易掉入 `Unmentioned`） | **优**（对象密度 +73%，原生空间/时间双轨归因） | 优（E2 同方案二，但 E3 真值公信力弱） |
| **YouTube 来源独立性** | E2 不可查重（含父视频）；E3 精确零重叠 | E2 父视频精确去重 + 近重复筛查；E3 **精确零重叠** | **E3 丧失精确 YouTube ID 去重** |
| **获取与许可阻力** | 中（DREAM-1K `gated: auto`） | **极低**（双集均 `gated=False`，CaReBench 为 `MIT`） | 极低（`8.74 GiB` 总视频） |
| **总视频下载体量** | `3.59 + 15.60 = 19.19 GiB` | `3.66 + 15.60 = 19.26 GiB` | `3.66 + 5.08 = 8.74 GiB` |
| **预计双人总人工工时** | **38–62 人时**（A2 返工上限 ~68） | **30–48 人时**（A2 返工上限 ~55，省 15–25%） | 36–57 人时（E3 需高强度过滤 GPT-4o 瑕疵） |
| **单卡 A6000 48GB GPU 耗时** | ≤ 35–45 GPU 小时 | **≤ 35–45 GPU 小时** | ≤ 30–40 GPU 小时 |
| **既有离线代码改动量** | **0 行** | **仅约 200 行局部改动（复用 ~95%）** | 约 400–500 行（重写 E2+E3 适配器） |
| **SCI Q2 论文聚焦度** | 清晰 | **最强且最具方法论深度** | 略弱（E3 GPT-4o 真值易受审稿人攻击） |

### 6.2 诚实的人工工时构成说明

任何外部数据集都**不能替代我们核验自己新生成的 A0/A1/A7 字幕**。方案二比方案一预计节省 **15%–25% 人工工时（约 8–14 双人人时）**，来源完全透明：
1. **核心事件定义工时从 2–4 人时降至 0–1 人时**：直接复用 CaReBench 94% 自带的人工开篇总览句；
2. **Pilot A2 自动评价器校准返工风险显著下降（7–10 vs 8–12/18 人时）**：静态对象覆盖更完整，自动三值判定落入模糊 `Unmentioned` 的比例显著降低；
3. **Formal B 两阶段人工验证集更易守住 50 独立视频下限（10–16 vs 14–24 人时）**：偏差方差更小，不必被迫自适应扩增至 80 视频；
4. **模型生成字幕全量人工标注始终为 0 人时**（全部由校准后的开源 LLM 三值评价器 + 两阶段统计校正完成）。

### 6.3 若切换至 CaReBench，既有工程仅需 4 处局部改动

既有离线工程（`research/uarcap_q2_experiments/`，已验证 59/59 项测试通过）约 **95% 模块原封不动复用**（包含全部检索、Video-LLaMA 与 Qwen3.8 生成器、三值评价器、标注 UI、双人仲裁 CLI、统计检验与调度器），仅需：
1. 新增 `uarcap_q2/datasets/carebench.py`（约 150 行：解析 `json/metadata.json`、提取 `caption`/`spatial_caption`/`temporal_caption` 与开篇单句总览、内置 **913 独立 FineAction 父视频去重**与 4 大类分层抽样）；
2. 在 `deployment/stage_assets.py` 与 `data_preparer.py` 注册 `carebench` 资产路径与分派分支；
3. 在 `deployment/run_stage_cli.py` 与配置文件新增 `--e2-dataset {dream1k,carebench}` 开关（**完整保留 `dream1k` 兼容回退**）；
4. 新增对应 CPU 单元测试并验证全量测试通过。

### 6.4 原封不动保留的历史研究资产

无论选择方案一还是方案二，以下既有 UARCap v0.8 成果 **100% 原封不动保留**（绝不重跑、不覆写）：
- **MSR-VTT 全套历史实验**：17,940 次冻结 Video-LLaMA 推理（A0–A9 与多轮预算扫面结果、v0.8 论文主表与统计结论）；
- **MSVD 全套历史实验**：独立跨数据集验证结果；
- **百万级 VideoUFO 检索索引与清单**：`1,091,712` 视频片段、`345,437` 独立 YouTube 来源及其完整 X-CLIP 嵌入检索基础设施；
- **UARCap v0.8 独立 Q3 备选投稿稿件**（完整冻结留存）。

---

## 七、推荐方案仍须通过真实试点确认的 6 项科学不确定性

在正式进入全量实验前，**CaReBench + VidHalluc（方案二）** 仍存在 6 项无法仅靠离线元数据回答、必须由 BN 服务器 Pilot A0–A3 实证检验的关键不确定性：

1. **FineAction vs VideoUFO 的真实近重复比例**：由于 CaReBench 公开 ID 为匿名 FineAction 编号，必须在实际跑通 X-CLIP 检索（Pilot A1）后核查 top-1 极高相似度分布与抽样看片，确认疑似重叠比例是否足够低（建议止损线：确证近重复 < 10%，且剔除后主结论稳定）。
2. **VideoUFO 对 CaReBench（FineAction 人类活动）的检索域匹配度**：CaReBench 聚焦 4 大类日常人类活动，需在 Pilot A1 实测 X-CLIP 检索出的 top-1 / top-7 VideoUFO 候选是否具备合理的语义相关度分布（既非全部完美匹配导致无错可析，也非全部无关噪声）。
3. **228 词长参考下，开源 LLM 三值评价器对 10 词短字幕的 `Unmentioned` 率与混淆矩阵**：需在 Pilot A2（30 视频 × 3 臂 = 90 短字幕盲法双人看片）中实证检验 `Unmentioned` 占比是否确已压低至可控区间（如 < 20–25%），且与双人仲裁的宏 F1 / 错误率偏差达标。
4. **14.5 词开篇总览句作为 `R_core_ref` 分母的完整性**：需在 Pilot A2 同步抽检确认开篇首句是否普遍涵盖视频核心动作（尤其核查 6% 非模板开头的样本），防止核心事件漏标。
5. **16 帧旧 Video-LLaMA 在 CaReBench（14.35s）上的细微静态属性盲区比例**：CaReBench 空间描述细腻，需在 Pilot A2 确认“因 224px/16 帧不可见而导致的 `Video-Unverifiable`”占比是否处于低位。
6. **VidHalluc 经 1,586 YouTube ID 精确去重后的剩余有效样本池与互斥率**：需在 BN 服务器将 `ach_mcq.json` 与 `/home/bn/videoufo_research/manifests/clips_full.tsv` 实跑精确交集运算，确认去重后仍充裕支撑 150–180 条经 5 项看片审核通过的严格互斥冲突对（若意外大幅缩水，再评估是否以 VidHal 真实自然子集做局部补充）。

---

## 八、完整可追溯一级资料与本轮专项文档索引

### 8.1 本轮 7 份审查文档索引（`D:\UARCap\research\q2_dataset_selection_review\`）

1. 全球文献与 22 基准盘点：[GLOBAL_VIDEO_DATASET_LITERATURE_SURVEY.md](file:///D:/UARCap/research/q2_dataset_selection_review/GLOBAL_VIDEO_DATASET_LITERATURE_SURVEY.md)
2. 人工标注来源与源码级 Gold 审计：[DATASET_ANNOTATION_PROVENANCE_AUDIT.md](file:///D:/UARCap/research/q2_dataset_selection_review/DATASET_ANNOTATION_PROVENANCE_AUDIT.md)
3. DREAM-1K vs CaReBench 深度交锋：[DREAM1K_VS_CAREBENCH_DEEP_COMPARISON.md](file:///D:/UARCap/research/q2_dataset_selection_review/DREAM1K_VS_CAREBENCH_DEEP_COMPARISON.md)
4. VideoUFO 重叠与 16 帧适配评估：[DATASET_SOURCE_OVERLAP_AND_MODEL_FIT.md](file:///D:/UARCap/research/q2_dataset_selection_review/DATASET_SOURCE_OVERLAP_AND_MODEL_FIT.md)
5. 结构化决策矩阵与四组权重敏感性 JSON：[DATASET_SCIENTIFIC_SELECTION_MATRIX.json](file:///D:/UARCap/research/q2_dataset_selection_review/DATASET_SCIENTIFIC_SELECTION_MATRIX.json)
6. 三种精炼实验组合与双人人时预算：[Q2_MINIMAL_EXPERIMENT_DESIGN_OPTIONS.md](file:///D:/UARCap/research/q2_dataset_selection_review/Q2_MINIMAL_EXPERIMENT_DESIGN_OPTIONS.md)
7. 中文独立总讨论报告（本文）：[Q2_FINAL_DATASET_SELECTION_DISCUSSION.md](file:///D:/UARCap/research/q2_dataset_selection_review/Q2_FINAL_DATASET_SELECTION_DISCUSSION.md)

### 8.2 可追溯一级论文、官方源码与公开元数据引用

- **CaReBench**：Xu et al. (2025), *CaReBench: Fine-Grained Video Captioning and Retrieval Benchmark*, arXiv:2501.00513v2 (`https://arxiv.org/abs/2501.00513`); 官方源码 `https://github.com/MCG-NJU/CaReBench`（实查 `utils/dream_gpt.py` L1, L105, L133, L446, L448）；公开数据集 `https://huggingface.co/datasets/MCG-NJU/CaReBench`（实测 `json/metadata.json` 全量 1,000 条与 `videos/videos.zip` 体积 `3,918,994,506 bytes`）。
- **DREAM-1K (Tarsier)**：Wang et al. (2024–2025), *Tarsier: Recipes for Training and Evaluating Large Video Description Models*, arXiv:2407.00634v2 (`https://arxiv.org/abs/2407.00634`); 官方源码 `https://github.com/bytedance/tarsier`（实查 `evaluation/metrics/evaluate_dream_gpt.py` L254）；公开仓库 `https://huggingface.co/datasets/omni-research/DREAM-1K`（实测 `gated: auto` 与 `video/video.zip` 体积 `3,849,677,473 bytes`）。
- **VidHalluc**：Li et al. (CVPR 2025), *VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding*, arXiv:2412.03735 (`https://arxiv.org/abs/2412.03735`); 官方仓库 `https://huggingface.co/datasets/chaoyuli/VidHalluc`（实测 `ach_mcq.json` 全量 2,159 对、3,897 片段、1,586 独立 11 位 YouTube ID 及 `ACH_videos.zip` 体积 `16,754,119,398 bytes`）。
- **VidHal**：Choong et al. (TMLR 2025), *VidHal: Benchmarking Temporal Hallucinations in Vision LLMs*, arXiv:2411.16771 (`https://arxiv.org/abs/2411.16771`); 公开仓库 `https://huggingface.co/datasets/anon-axolotl/VidHal`（实测 `annotations.json` / `options.json` 全量 1,000 条含 261 条 CLEVRER 合成数据及 `videos.zip` 体积 `5,455,798,276 bytes`）。
- **ARGUS**：Rawal et al. (ICCV 2025), *ARGUS: Benchmarking Hallucinations in Detailed Video Captioning*, arXiv:2506.07371 (`https://arxiv.org/abs/2506.07371`); 官方源码 `https://github.com/JARVVVIS/argus`（实查 `code/compute_metrics.py` L306–L313）；公开仓库 `https://huggingface.co/datasets/tomg-group-umd/argus`。
- **其余实查基准**：`VCapsBench` (`https://huggingface.co/datasets/somos99/VCapsBench`, arXiv:2505.23484)、`VidCapBench` (`https://huggingface.co/datasets/VidCapBench/VidCapBench`, Findings of ACL 2025)、`VideoHallucer` (`https://huggingface.co/datasets/bigai-nlco/VideoHallucer`, arXiv:2406.16338)、`TempCompass` (`https://huggingface.co/datasets/lmms-lab/TempCompass`, ACL 2024)、`VDC` (`https://huggingface.co/datasets/wchai/Video-Detailed-Caption`, ICLR 2025)、`FactVC` (`https://github.com/NiteRainHL/FactVC`, EMNLP 2023)、`CapQuiz` (ACL 2026，核验无公开数据释放)。

---

**最终交付状态**：`DATASET_SELECTION_RESEARCH_COMPLETE_AWAITING_SCIENTIFIC_DECISION`  
（等待用户裁决采用 **方案二：CaReBench + VidHalluc** 还是 **方案一：维持 DREAM-1K + VidHalluc**，授权后再行推进局部工程适配、下载准备或 BN 试点启动。）
