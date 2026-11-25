# 候选视频数据集人工标注来源与 Gold 可靠性逐层谱系审计（DATASET ANNOTATION PROVENANCE AUDIT）

> **核心防线**：严禁将 **“原始参考描述由人类撰写”** 直接等同于 **“全部结构化事实均属于人工认证 Gold”**，也严禁将 **“数据集含人类参考描述”** 等同于 **“数据集已经替我们认证了新模型输出（A0/A1/A7）的正确性”**。

---

## 一、11 项标注来源核验矩阵（六大核心入围候选）

| 核验维度 | **CaReBench** (`MCG-NJU/CaReBench`) | **DREAM-1K** (`omni-research/DREAM-1K`) | **VidHalluc (ACH)** (`chaoyuli/VidHalluc`) | **VidHal** (`anon-axolotl/VidHal`) | **ARGUS** (`tomg-group-umd/argus`) | **VCapsBench / VidCapBench** |
|---|---|---|---|---|---|---|
| **1. 视频本身是否真实公开** | 是，`videos.zip` (`3.66 GiB`, `gated=False`, MIT) `[HF实测]` | 是，`video.zip` (`3.59 GiB`, `gated=auto`, 限学术) `[HF实测]` | 是，`ACH_videos.zip` (`15.60 GiB`, `gated=False`, CC-BY-NC-ND) `[HF实测]` | 是，`videos.zip` (`5.08 GiB`, `gated=False`, CC-BY-4.0) `[HF实测]` | 是，500 个 `.mp4` (`15.60 GiB`, `gated=False`) `[HF实测]` | 是（VCapsBench `22.2 GiB`; VidCapBench `3.59 GiB`, 禁分发） `[HF实测]` |
| **2. 原始参考描述是否由人类独立撰写** | **是（100% 人工）**：Stage-I 细粒度描述 + Stage-II 时空分离 `[论文原文 3.2]` | **是（100% 人工）**：专业标注员撰写细粒度事件描述 `[论文原文]` | 否（短动作短语源自 ActivityNet/YouTube 动作标注 + 筛选配对） `[公开元数据实测]` | **否**：锚点正字幕 $y^+$ 由 **GPT-4o** 基于上游 QA/标注合成 `[论文原文 3.3]` | **是（100% 人工）**：自由形式超长人工描述（均值 ~485 词） `[公开元数据实测]` | **否**：QA 由 VLM / GPT-4o 初生成（VidCapBench 含人工修订） `[README实查]` |
| **3. 是否存在多名独立标注员** | **是**：每视频由 **2 名标注员独立标注** `[论文原文 3.2]` | 论文称 "skilled human annotators"，未公开每视频独立人数 (`UNKNOWN`) `[论文原文]` | 论文报告人工复核流程，细节偏简 (`PARTIAL`) | 仅在 100/1000 验证子集使用约 15 人排序校验 `[论文原文 3.4]` | 人工撰写，单视频多人独立数 `UNKNOWN` | VCapsBench 以模型一致性过滤为主；VidCapBench 含人工修订 |
| **4. 是否存在专家复核** | **是**：每个阶段后由专家合并润色，明确剔除主观评价、音频/语音与时空混杂 `[论文原文 3.2.2]` | 有质检流程，具体专家复核细则未量化公开 `[论文原文]` | 有人工筛选过滤 | 仅对 10% 子集（100 样本）做人工排序验证 `[论文原文 3.4]` | 有质量审核 `[论文原文]` | VidCapBench 含人工修订；VCapsBench 主要自动过滤 |
| **5. 是否报告标注一致性指标** | **否（未报告 kappa / AC1）** `[论文全文核查]` | **否（参考描述未报告 kappa）** `[论文全文核查]` | 未报传统 kappa | **是（仅 10% 子集）**：排序共识一致率 **87%** `[论文原文 3.4]` | 未报参考描述 kappa | VidCapBench 通过机评一致性切分 AE/HE |
| **6. 事件列表是否由人工标注** | **否！实查系 GPT-4 提取**（`events` max=10，对应 `dream_gpt.py` L105 `"Extract at most 10 key events"`） `[官方源码实查]` | **否**：评测时由 GPT-4/GPT-3.5 临时提取（`evaluate_dream_gpt.py`） `[官方源码实查]` | N/A（动作短语级标签） | N/A（整句对比字幕） | **否**：`parquet` 仅含 `clip_name` 与 `human_caption`，无预拆事件 `[公开元数据实测]` | N/A（QA 对形式） |
| **7. 对象列表是否由人工标注** | **否！实查系 GPT-4 提取**（实测 `10,023 / 10,032 = 99.9%` 以 `"There is/are "` 开头，逐字匹配 `dream_gpt.py` L133 提示词！） `[公开元数据实测 + 官方源码实查]` | **否**（且官方 AutoDQ 明确忽略静态信息 `"NOT STATIC INFORMATION"`） `[官方源码实查]` | N/A | N/A | **否** | N/A |
| **8. 结构化事实是否由大模型从人工描述提取** | **是**：`metadata.json` 中的 `events`（6,942 条）与 `objects`（10,032 条）为官方预跑 GPT-4 提取缓存 `[官方源码实查]` | **是**：AutoDQ 运行期由 GPT-4 从参考描述提取 | N/A | N/A | **是**：`nli_eval.py` 运行期由 LLM 拆解与判别 | **是**：QA 直接由 VLM/GPT-4o 生成 |
| **9. 大模型提取结果是否经过人工逐条核验** | **否（论文与代码均无逐条人工核验 `events`/`objects` 记录）** `[论文原文 + 官方源码实查]` | **否** | N/A | **仅 10%（100/1000）** 经人工排序验证，余 90% 未逐条人工核验 `[论文原文 3.4]` | **否** | VidCapBench 有人工修订，VCapsBench 未逐条人工核验 |
| **10. 真假事实或负例是否经过独立人工验证** | N/A（无内置负例/反事实对） | N/A（无内置负例） | 部分：基于高视觉相似配对构造，个别次要动作可能并存，**仍需我们抽样后轻量人工看片确认严格不存在** | 负字幕 $y^-$ 由 **GPT-4o** 合成，10% 子集人工一致率 87%（即约 13% 可能存在排序/事实瑕疵） `[论文原文 3.3–3.4]` | N/A | 否（VLM/GPT-4o 生成） |
| **11. 原作者人类评价的真实对象** | **仅针对参考字幕制作**；论文**未报告 `CapST` 与人工评分的相关系数** `[论文全文核查]` | **针对模型输出两两胜负（Table 4）**：300 视频 × 4 组模型对，3 人各看 100 对（每对仅 1 人），仅验证系统级排序一致，**非原子错误级相关性** `[论文原文 5.1.2 & App. D]` | 针对模型多选/二值幻觉表现 | 针对 100 样本的 3 字幕幻觉等级排序一致性（87%） `[论文原文 3.4]` | 针对模型自由字幕的幻觉/遗漏人类相关性检验 | VidCapBench 针对 T2V 生成相关性 |

---

## 二、最关键的一手源码打假发现（打破常见误解）

### 发现 A：CaReBench 的 `events` 与 `objects` **不是** 人工认证 Gold 清单！
- 直接下载 `MCG-NJU/CaReBench` 的 `json/metadata.json`（3.70 MB，1,000 条）与官方仓库 `MCG-NJU/CaReBench/utils/dream_gpt.py` 后，发现两项铁证：
  1. `utils/dream_gpt.py` 文件头版权为 `# Copyright (2024) Bytedance Ltd. and/or its affiliates`，系直接分叉自 DREAM-1K 的 `evaluate_dream_gpt.py`，并新增了 `call_azure_gpt_api_for_objects`（第 119–143 行）；
  2. 该提示词第 133 行明确规定：`Every object is represented by a brief sentence starting with "There is/are", including its attributes.`。对 `metadata.json` 全量 1,000 条记录的 `10,032` 条 `objects` 实测统计：**`10,023 / 10,032`（`99.9%`）以 `"There is "` 或 `"There are "` 开头**，`events` 严格封顶于 `max = 10`（匹配第 105 行 `"Extract at most 10 key events"`）。
- **结论**：CaReBench 真正高价值的人工 Gold 资产是 **双人独立撰写+专家复核的 `caption`、`spatial_caption`、`temporal_caption`（尤其是 940/1,000 条开篇自带的单句人工总览）**；其 `events` 与 `objects` 只能视为 **GPT-4 预提取参考候选**，不可在论文中宣称为“人工认证结构化 Gold”。

### 发现 B：三大官方自动评价器（AutoDQ、CapST、ArgusCost-H）源码均存在将 `Unmentioned` 误计为错误的致命缺陷！
1. **DREAM-1K `AutoDQ`**（`bytedance/tarsier/evaluation/metrics/evaluate_dream_gpt.py` L221 & L254）：计算 Precision 时错误传入 `is_recall=True`，将 `neutral`（参考未提及）直接等价为 `contradiction` 计作 Precision 幻觉错误。
2. **CaReBench `CapST`**（`MCG-NJU/CaReBench/utils/dream_gpt.py` L446 & L448）：原样继承上述代码，`events_score_p` 与 `objects_score_p` **均传入 `is_recall=True`**，同样把参考未提及事实误计为 Precision 错误！且 CaReBench 论文**未报告任何 `CapST` 与人类评价的相关系数**。
3. **ARGUS `ArgusCost-H`**（`JARVVVIS/argus/code/compute_metrics.py` L118–119）：`if node.verdict != "entailment": cost_matrix[i, j] = 1`，同样把 `underdetermined` 与 `contradiction` 等价计入幻觉代价 `1.0`。

**直接科研推论**：无论选择 DREAM-1K、CaReBench 还是 ARGUS，**都不能直接拿官方单一 Precision/F1 指标作为 UARCap 的主事实错误终点**；必须坚持使用我们已经开发并可解释的 **三值参考事实评价器（`ENTAILED` / `CONTRADICTED` / `UNMENTIONED`）+ 少量双盲人工校准**，官方指标仅可作为社区对照报告！

---

## 三、各候选数据中“可复用 Gold / 仅作自动候选 / 必须我们少量校准”的分层谱系

| 候选 | 可直接作为可靠参考依据的人工 Gold | 仅能作为自动候选（不可标榜人工 Gold） | 我们仍必须执行的最小人工核验 |
|---|---|---|---|
| **CaReBench** | 1,000 条（913 独立父视频）双人+专家人工 `caption`、`spatial_caption`、`temporal_caption`；940 条开篇单句人工总览 | `metadata.json` 中的 `events`（6,942 条）与 `objects`（10,032 条） | Pilot A2 30 视频跨家族三值评价器双盲校准（+ Formal B 50 视频独立验证集） |
| **DREAM-1K** | 1,000 条人工事件参考描述 `caption` | AutoDQ 抽出的动态事件列表 | Pilot A2 30 视频校准 + 核心事件锚点校验（+ Formal B 50–80 视频独立验证集，因 `Unmentioned` 风险更高） |
| **VidHalluc (ACH)** | 1,586 独立 YouTube 源的高视觉相似动作对候选骨架 | 未看片核验的 MCQ 干扰选项（部分可能并存） | 抽样 150–180 条 E3 候选的轻量 5 项看片适用性核验 + Pilot A3 E3 双目标评价器人工核验 |
| **VidHal** | 100/1,000 人工排序校验子集 | 其余 900 条 GPT-4o 合成对比字幕 | 若作 E3，必须剔除 261 条 CLEVRER 合成视频并对入选条目逐条人工看片核验 |
