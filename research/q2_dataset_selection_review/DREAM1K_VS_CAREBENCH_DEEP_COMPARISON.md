# DREAM-1K 与 CaReBench 深度独立对比（DREAM-1K VS CAREBENCH DEEP COMPARISON）

> **声明**：本对比基于直接下载的 `CaReBench/json/metadata.json`（全量 1,000 条实测）、`CaReBench` 官方仓库源码、`Tarsier/DREAM-1K` 论文与 `evaluate_dream_gpt.py` 源码独立核验，**不照搬任何先前假设或 ChatGPT 初步倾向**。

---

## 一、核心统计与结构字段一手实测对比表

| 对比维度 | **DREAM-1K** (`omni-research/DREAM-1K`) | **CaReBench** (`MCG-NJU/CaReBench`) | 证据来源 |
|---|---|---|---|
| **视频片段总数** | 1,000 | 1,000 | `[论文原文 / 元数据实测]` |
| **独立父视频数量** | `UNKNOWN`（匿名 ID `1..1000`） | **913 独立 FineAction 父视频**（68 个父视频含 2–6 条片段，共 155 条多片段，最多 6 条 `v_00005394`） | `[公开元数据实测]` |
| **平均视频时长** | **8.9 s** | **14.35 s**（>80% 在 5–20s；仅 5.8% <5s 或 >30s；**短于 MSR-VTT 均长 15.01s**） | `[论文原文 Table 1 & 3.3]` |
| **16 帧采样等效帧率** | ~1.80 fps | ~1.11 fps（高于 MSR-VTT 16 帧等效 1.07 fps） | `[数学推算]` |
| **参考描述词数** | 均值 **59.3 词** | 均值 **227.95 词**（中位数 218，最小 150，P10=184，P90=283，最大 474） | `[论文原文 / 元数据实测]` |
| **空间/时间独立描述** | 无（单一事件导向段落） | 有：`spatial_caption`（均值 **113.22 词**）+ `temporal_caption`（均值 **102.26 词**），人工分离且剔除音频/主观词 | `[公开元数据实测]` |
| **开篇单句核心总览** | 无强制统一总览字段 | **940 / 1,000（94.0%）以总览句开头**（均值 **14.5 词**，中位数 14.0 词） | `[公开元数据实测]` |
| **结构化事件 / 动作数** | 统一提取约 **6.16** / 视频 `[CaReBench对比/待复核]` | 实测 `events` 均值 **6.94**（中位数 7，范围 2–10，词长均值 5.7 词，总计 6,942 条，GPT-4 提取） | `[公开元数据实测]` |
| **结构化对象 / 属性数** | 统一提取约 **5.79** / 视频 `[CaReBench对比/待复核]` | 实测 `objects` 均值 **10.03**（中位数 10，范围 4–26，词长均值 7.28 词，总计 10,032 条，99.9% 以 `"There is/are"` 开头，GPT-4 提取） | `[公开元数据实测]` |
| **显式时间锚点** | 无独立时间戳字段 | 仅 8 / 1,000 条文本含显式秒数（主要以 `first/then/next` 顺序叙述） | `[公开元数据实测]` |
| **视频领域覆盖** | **5 大开放领域**：真人电影、动画、YouTube、TikTok 短视频、Stock 素材 | **1 个上游数据集（FineAction）4 类人类活动**：Socializing & Relaxing (340)、Sports & Exercise (260)、Personal Care (210)、Household Activities (190)，82 个动作子类 | `[论文原文 / 元数据实测]` |
| **原始 YouTube ID 可追溯性** | **不可追溯**（匿名 `1..1000`） | **公开元数据不可直接追溯**（FineAction 匿名 ID `v_00000030_5.mp4`，非 11 位 YouTube ID） | `[公开元数据实测]` |
| **压缩包体量与许可** | `video.zip` = **3.59 GiB**（3,849,677,473 B）；`gated: auto`；metadata `apache-2.0` / 视频限学术 | `videos.zip` = **3.66 GiB**（3,918,994,506 B）；**`gated: False`**；**`MIT` 许可** | `[HF API 实测]` |
| **官方评价指标缺陷** | `AutoDQ` 忽略静态信息 + Precision 误传 `is_recall=True` 惩罚 `neutral` | `CapST` 直接分叉自 `AutoDQ`，**同样在 L446/L448 误传 `is_recall=True` 惩罚 `neutral`**，且未报人工相关性 | `[官方源码实查]` |

---

## 二、CaReBench 相对 DREAM-1K 的“对象优势（10.03 vs 5.79）”对 UARCap 究竟重不重要？

**独立科研判断：非常重要，且直接切中 UARCap E2 实验最危险的方法学软肋——`Unmentioned` 漏检偏置！**

分析如下：
1. **动作密度差异不大（6.94 vs 6.16，仅高 12.7%）**，两者对主动态事件的描述均足够充分；
2. **静态对象/属性密度差距显著（10.03 vs 5.79，高 73.2%；且 `spatial_caption` 专门记录人物服饰、颜色、手持物、背景陈设与相对位置）**。
3. **这对 UARCap 的 A0 → A1 → A7 诊断为何至关重要？**
   - 当我们从百万级 VideoUFO 检索 7 条外部字幕（A7）注入模型时，外部文本注入的典型幻觉不仅包括“换动作”（如把 `throwing grass` 说成 `raking leaves`），更大量包括**“添对象、改属性、错背景”**（如在原视频根本不存在的地方加入 `"near a wooden table"`, `"wearing a red hat"`, `"holding a black bag"`, `"in a kitchen"`）。
   - 在 **DREAM-1K（59.3 词，事件中心）** 中，参考描述往往只写主动作而省略配角衣着与背景物体；此时如果 A7 幻觉出 `"wearing a red jacket"` 或 `"standing next to a blue car"`，三值参考评价器只能判为 **`UNMENTIONED`**（因为参考没提颜色/车辆）——结果在 Pilot A2 中触发我们最担心的 **“A7 漏检偏置（`Unmentioned` 中隐藏真实错误导致 A7−A0 风险差被系统性低估 >40%）”**，甚至可能导致自动评价器无法通过 A2 校准红线！
   - 在 **CaReBench（227.95 词，人工独立审校对象、颜色、背景、位置）** 中，参考描述明确写出了人物衣着、颜色、器具与环境背景；因此 A7 注入的虚假对象/属性/背景能被显著更高比例地直接判为 **`CONTRADICTED`**，大幅压低 `UNMENTIONED` 灰区比例，并使我们能够额外报告 **静态对象事实变化 vs 动态动作事实变化** 的精细归因！

---

## 三、CaReBench 的真实短板与“短字幕粒度错配”严格评估

我们绝不能因为上述优势就盲目美化 CaReBench。经实测 20 条代表性样本（间隔 50 抽样，涵盖全部 4 大类），CaReBench 存在 **三项实质性短板**：

### 短板 1：领域集中于 FineAction 人类活动（外部效度低于 DREAM-1K）
- CaReBench 100% 为人类日常/运动/个人护理/家务活动（如 `climbing_tree`, `tie_shoelaces`, `brush_teeth`, `play_the_drums`, `apply_eyebrows`），**缺乏电影剪辑、动画、动物、自然风光、车辆街景**。
- **缓解评估**：UARCap 并非孤立使用 CaReBench——我们的历史主资产 **MSR-VTT（20 类开放领域）+ MSVD（开放 YouTube 短片段）** 已经覆盖了开放域；而在 FineAction 这种“类内高度视觉相似、细微动作/对象易混淆”的场景中，恰恰最能暴露百万级 VideoUFO 检索带来的相似场景混淆与事实污染！

### 短板 2：228 词超详参考 vs 8–18 词短字幕的粒度错配（如何避免系统性惩罚短输出或制造 A7 虚假覆盖优势？）
- 实查 `CaReBench` Sample 0（`v_00000030_5.mp4`，`Personal Care / apply_eyebrows`）：
  - **开篇单句总览（13 词）**：`"This video showcases a woman meticulously perfecting her eyebrows using an eyebrow pencil."`
  - **全文 200+ 词细节**：涵盖发色渐变、黑色眼影、银色亮片眼线、假睫毛、黄白大戒指、先点左眉前端再点右眉前端等极细微信息。
- 如果直接拿 Video-LLaMA 的 10 词短字幕去算 **“对全部 17 条原子事实（7 事件 + 10 对象）的 Recall”**，必然出现两个问题：
  1. 绝对 Recall 被压低到 10%–20%，天花板效应反向变成地板效应；
  2. 如果 A7 仅仅因为输出比 A0 长 3 个词而碰巧多提到一个背景词，传统 Recall 可能给予虚假覆盖奖励。
- **科学破解方案（CaReBench 原生结构恰好提供了解法！）**：
  1. **精度/错误主终点不受参考过长惩罚**：对于生成字幕的每一条原子命题（短字幕通常 1–3 条），参考描述**越完整越有利**——短但正确的字幕（如 `"A woman is applying makeup to her eyebrows."`）其全部命题都能被 228 词参考干净判为 **`ENTAILED`**（0 错误、0 误伤！）；
  2. **核心事件覆盖 `R_core_ref` 不使用全部 17 条细节，而直接锚定于 CaReBench 人工撰写的开篇单句总览（940/1,000 条原生存在，均值 14.5 词）**，使短字幕的核心覆盖评价与参考长度完全解耦；
  3. **长度控制与归一化**：同时报告 **每句绝对错误数（`n_contra_ref`）** 与 **每命题错误率（`p_contra_ref = n_contra / n_claims`）**，防止 A7 仅靠拉长句子制造虚假覆盖或扭曲风险差。

### 短板 3：父视频聚类与 FineAction 上游来源不透明
- 1,000 条 CaReBench 片段来自 **913 个独立 FineAction 父视频**（68 个父视频存在多片段，最多 6 条）。若直接全部纳入会导致同源泄漏，**必须先按 FineAction 父视频去重为至多 913 条独立片段**（每父视频严格 1 条）。
- 与 DREAM-1K 一样，CaReBench 公开元数据不含 11 位 YouTube ID，无法直接与 VideoUFO 做 YouTube ID 精确交集去重（详见第 4 份报告）。

---

## 四、20 条 CaReBench 真实样本抽检摘要（间隔 50 抽样，编号 0–950）

经对 `carebench__metadata.json` 中 20 条跨类样本的逐条核查，确认：
- **零缺失字段**：`video / category / subcategory / caption / spatial_caption / temporal_caption / events / objects` 全部 1,000/1,000 非空；
- **语言质量高且客观**：未见 GPT 套话（如 `"In this captivating video..."`），明确无音频/字幕推测；
- **开篇总览极度稳定**：如 `[0]` `"This video showcases a woman meticulously perfecting her eyebrows using an eyebrow pencil."`、`[200]` `"This video captures the entire process of a foreign Black woman clapping and welcoming guests..."`、`[400]` `"This video captures the delightful process of a little baby crawling toward a toy on the floor."`、`[600]` `"This video showcases the preparation process of a man preparing to throw a shot put."`、`[800]` `"This video depicts a woman assisting a man by washing his hands."`——全部与 16 帧短视频单句描述高度对齐！
