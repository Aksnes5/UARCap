# 数据集 Schema 与自动评价器十项审查矩阵（DATASET_SCHEMA_AND_EVALUATOR_MATRIX）

> **状态**：`DESIGN_FOR_DISCUSSION`｜配套主稿：`UARCap_Q2_统一科研框架与实验设计讨论稿_v1.2.md`

---

## 一、核心数据集 Schema、开放性与适用性矩阵

| 数据集 | 人工标注可信度 | 事实级真值字段 | 明确错误 / 反事实 | 信息遗漏评价 | 核心事件可得性 | 自动评分与源码局限 | 本地可运行性 | 开放性 & 下载量 | 原短字幕兼容性 | E2 适用性 | E3 适用性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DREAM-1K** | **高**（专业人工描述） | `metadata.json` 细粒度事件描述 | 无 | 有（事件 Recall） | 可事前从人工描述提 1–2 主干事件 | `AutoDQ` L254 将 `neutral` 计作错且硬编码 Azure GPT-4；**改三值解耦本地 LLM 后极佳** | **高**（改本地开源裁判） | `gated: auto` / `apache-2.0` / **`3.85 GB`** | **较好**（配核心事件子集） | **主推** | 中 |
| **VidHalluc** | **高**（高相似对 + 人工动作/时序真值） | `ach_mcq.json` / `ach_binaryqa.json` / `tsh.json` | **强**（同场景强混淆冲突动作） | 仅主动作/时序 | 天然锚定显著主动作 | 原为问答评测；**用于 E3 靶构造 + L4 五态极契合** | **极高** | `gated: False` / `cc-by-nc-nd-4.0` / 约 1–2 GB(`ACH`) | **极高**（短片段单主动作） | 弱 | **主推** |
| **ARGUS** | 中高（500 篇 ~477 词混合来源） | Parquet 仅 `clip_name` + `human_caption` | 无 | 有（`ArgusCost-O`） | 需从 477 词长文事前抽核心 | `compute_metrics.py` L118 把 `underdetermined` 与 `contradiction` 同扣 1 分 | **高**（原生支持开源裁判） | `gated: False` / `cc-by-nc-sa-4.0` / **`16.75 GB`** | 偏弱（15–90s + 477 词地板重） | **备选** | 弱 |
| **VCapsBench** | 中（VLM 合成 + 一致性过滤） | `VCapsBench_100KQA.jsonl`（21 维 Yes/No） | 有（约 20–35% `no` 题，偏缺失/属性） | 有（`Unanswerable`） | 须事前过滤至 6 个语义维 | `LLM4eval-m.py` 支持三值，但默认 Gemini/GPT-4o 且存常识盲猜风险 | 中（须改本地 LLM + 空字幕检验） | `gated: False` / `odbl` / **`~23.4 GB`** | 中上（6–15s 短片） | 备选 | 中 |
| **CapQuiz** | 论文描述高 | 未公开 | 论文有 MCQ 干扰项 | 论文有 `CapR` | 论文有 | 代码与数据均未公开 | **不可用** | **未公开发布** | — | **排除** | **排除** |
| **VDC** | 低（实查 `short.jsonl` 含占位词） | `VDC_1k.jsonl` + 5 个 QA jsonl | 无 | 有（`VDCScore`） | 受占位词污染 | 依赖 GPT-4o 且 `short.jsonl` 存在明显脏数据 | 低 | `gated: False` / `apache-2.0` | 弱 | **淘汰** | **淘汰** |
| **FactVC** | 高（仅评历史旧输出） | 仅旧模型输出 1–5 分 | 仅旧输出错词 | 无 | 无 | 弱监督 CLIP 标量 | — | Google Drive | — | **不适用** | **不适用** |

---

## 二、自动评价器十项刚性审查对照表（对应需求第 XIII 节）

| 审查标准 | 官方 AutoDQ (DREAM-1K) | 官方 ArgusCost (ARGUS) | 官方 TextQA (VCapsBench) | **UARCap v1.2 修正评估协议** |
|---|---|---|---|---|
| 1. 是否读原视频 | 否（只读参考与预测） | 否（只读参考与预测） | 否（只读预测与 QA） | 以 **80 视频双盲静音人工看片**作原视频真值验证锚点 |
| 2. 是否读生成字幕 | 是 | 是 | 是 | 是 |
| 3. 是否需正确答案/参考 | 需人工参考描述 | 需人工参考描述 | 需 Yes/No 答案 | 仅限评测阶段使用，严禁向生成器泄漏 |
| 4. 是否允许信息不足 | Prompt 含 `neutral`，但源码 L254 并入错误 | Prompt 含 `underdetermined`，但源码 L118 并入 cost=1 | 显式支持 `Unanswerable` | **严格保留并分列三值原始输出**：`N_Contra_ref`（仅 `contradiction`）、`N_Entail_ref`、`P_Unment_ref` |
| 5. 是否误判未覆盖真实事实为幻觉 | **会**（因 `neutral` 计错） | **会**（因 `underdetermined` 计错） | 不会（标 `Unanswerable`），但漏检题外错 | 独立记录 `P_Unment_ref`，并在 80 视频人工看片集上定量估计其中真实细节 vs. 真实幻觉占比 |
| 6. 是否偏好长描述 | 长句提 Recall、降 Precision | 长句降 Omission、升 Hallucination | 长句答对更多题 | 跨模型强制**短字幕对齐指令（8–18 词）**，并在验证集按句长分层审计评价器误差 |
| 7. 是否偏好同模型家族 | 取决于裁判 | 取决于裁判 | 取决于裁判 | **强制跨家族裁判**：如用 `Qwen2.5-VL` 作生成器，则文本 NLI 裁判选用 `Llama-3.3` / `DeepSeek` / `InternLM` 等不同家族 |
| 8. QA 常识盲猜风险 | N/A | N/A | **存在显著风险** | 任何 QA 评测均强制跑**空字幕基线**与**无关字幕基线**、随机化选项位置、要求输出显式字幕依据 |
| 9. 闭源 API 成本 | 硬编码 Azure GPT-4 | 支持 GPT-4o 及开源 API | 默认 Gemini / GPT-4o | 主评测采用 BN 本地 A6000 开源强 LLM（`$0` API 费）；可选 200 视频商业 API 敏感性对照（~$30–60） |
| 10. 替代闭源模型后的等价性边界 | — | — | — | **绝不宣称本地开源裁判与 GPT-4 等价**；强制隔离 **30 视频校准集（E2a）** 与 **50 视频独立验证集（E2b 抽样）** 汇报人机 $\kappa$ 与误差矩阵 |
