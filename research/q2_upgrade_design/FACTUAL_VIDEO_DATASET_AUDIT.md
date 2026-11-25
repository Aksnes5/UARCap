# 现成细粒度视频事实评价数据集一手审查报告（FACTUAL_VIDEO_DATASET_AUDIT）

> **状态**：`DESIGN_FOR_DISCUSSION`｜**核查日期**：2026-11-25｜**核查方式**：直接访问 Hugging Face Hub API、Datasets Server 样本行接口、官方项目页与 GitHub 原始代码（仅只读元数据与 1–8 条样例行，未下载大型视频文件）

---

## 一、8 个候选数据集逐项一手实查档案

### 1. DREAM-1K（ByteDance Tarsier / Tarsier2）—— **推荐 E2 主数据集**
- **原始链接**：HF https://huggingface.co/datasets/omni-research/DREAM-1K ｜代码 https://github.com/bytedance/tarsier ｜论文 arXiv:2407.00634 & arXiv:2501.07888
- **开放与许可**：`gated: auto`（一键同意即开放下载），**`apache-2.0`**
- **实查文件与体量**：`json/metadata.json` + `video/video.zip`（**实测仅 `3.85 GB`**，1,000 个视频片段，涵盖电影、动画、YouTube、短视频与素材库）
- **标注可信度**：专业人工撰写细粒度事件描述，明显优于 GPT 模板化华丽长文
- **官方评价源码实查（`evaluation/metrics/evaluate_dream_gpt.py`）**：
  - 提示词要求先抽最多 10 条动态动作/运动事件（`"NOT STATIC INFORMATION"`），再做三值关系判断：`entailment / neutral / contradiction`；
  - **关键源码发现（L221 & L254）**：虽然 L221 定义了 `pos_classes = ['entailment'] if is_recall else ['entailment', 'neutral']`，但 **L254 计算 Precision 时实际传入了 `is_recall=True`**！这意味着官方 AutoDQ **把参考描述未提及的 `neutral` 命题直接计作 Precision 幻觉错误**，且硬编码 `azure_gpt4_client`。
- **UARCap 复用结论**：**强烈推荐作为 E2 主数据集与跨数据集确认队列**（取代 VATEX）；复用其 1,000 条高质量人工描述与仅 3.85 GB 视频，但**改造评价输出为严格分列三值（`Entailed / Contradicted / Unmentioned-Neutral`）+ 事前核心事件子集 `R_core_ref`**，并用跨家族本地开源 LLM + 80 视频双盲人工校准/验证，同时保留官方 `AutoDQ` 作社区参照。

### 2. VidHalluc（CVPR 2025）—— **推荐 E3 靶事实干预主数据集**
- **原始链接**：HF https://huggingface.co/datasets/chaoyuli/VidHalluc ｜论文 CVPR 2025
- **开放与许可**：**`gated: False`**（直接公开），`cc-by-nc-nd-4.0`
- **实查文件清单**：`ach_mcq.json`、`ach_binaryqa.json`、`tsh.json`、`sth.json`、`data/ACH_videos.zip`、`data/TSH_videos.zip`、`data/STH_videos.zip`
- **真实样例记录（`ach_mcq.json` & `ach_binaryqa.json`）**：
  - `vcCwvRYqU2I_clip_4`：主动作真值 `"throwing grass in the garbage"`，高混淆错误选项含 `"raking leaves"`、`"collecting trash from the yard"`、`"watering plants"`；
  - `OBb4013eIc8_clip_1` vs. `OBb4013eIc8_clip_4`：同一 YouTube 视频不同片段成对反转 `Yes / No` 主动作（`"turning the steering wheel"` vs. `"wakeboarding"`）。
- **关键审计提醒**：① 片段名自带 11 位 YouTube ID，极易执行与 VideoUFO 的精确零重叠过滤；② 同一 YouTube 来源包含多个 `_clip_*`，抽样必须强制每来源至多 1 clip；③ 错误选项未必 100% 严格反事实（可能属并存次要动作），因此抽出的 150–180 个 E3 样本仍须轻量人工看片核验严格冲突。
- **UARCap 复用结论**：**强烈推荐作为 E3 专用辅助数据集**，直接提供高视觉相似、强语义混淆的 $(f^+, f^-)$ 动作对，极大压缩 E3 靶事实构造成本。

### 3. ARGUS（ICCV 2025）—— **E2 备选数据集**
- **原始链接**：项目 https://ruchitrawal.github.io/argus/ ｜HF https://huggingface.co/datasets/tomg-group-umd/argus ｜代码 https://github.com/JARVVVIS/argus ｜论文 arXiv:2506.07371
- **开放与许可**：**`gated: False`**，`cc-by-nc-sa-4.0`
- **实查文件与体量**：500 个视频直接托管于 `videos/*.mp4`（**实测总计 `16.75 GB`**，其中 `8` 条含 `-Scene-` 标记的片段共享 `2` 个 YouTube 前缀，须做来源聚类）；`data/train-00000-of-00001.parquet` **仅含 2 列**：`clip_name` 与 `human_caption`（实查前 5 条词数分别为 `557 / 461 / 733 / 419 / 565` 词，15–90 秒片段），**不含预拆评价点或错误选项**。
- **官方评价源码实查（`code/nli_eval.py`, `code/configs.py`, `code/compute_metrics.py`）**：
  - `configs.py` 原生支持开源裁判（`deepseek_v3`、`qwen_2_5_72b_instruct`、`llama_3_3_70b`）与 `gpt-4o`；
  - `nli_eval.py` 输出三值裁决：`entailment / contradiction / underdetermined`；
  - **关键源码发现（`compute_metrics.py` L118–119）**：`if node.verdict != "entailment": cost_matrix[i, j] = 1`——**官方 `ArgusCost-H` 将 `underdetermined` 与 `contradiction` 等价计入幻觉代价 `1.0`**！
- **UARCap 复用结论**：列为 **E2 备选**（体量比 DREAM-1K 大 4 倍且 477 词文学性长描述与 9 词短字幕错配更重；若选用必须直接解析 `nli_eval.py` 原始三值 JSON，禁止仅报合并后的 `ArgusCost-H`）。

### 4. VCapsBench（AAAI / arXiv 2025）—— **QA 路线备选**
- **原始链接**：代码 https://github.com/GXYM/VCapsBench ｜HF https://huggingface.co/datasets/somos99/VCapsBench ｜论文 arXiv:2505.23484
- **开放与许可**：**`gated: False`**，`odbl`
- **实查文件与体量**：`VCapsBench_100KQA.jsonl`（`8.2 MB`，109,796 条 QA）+ `videos5k-0.zip` 至 `videos5k-11.zip`（**合计 ~23.4 GB**，5,677 视频）。实查前 8 行时长多为 **5.9–14.7 秒**（与 MSR-VTT 极度接近！），每视频 13–22 条 `dim*Q*yes/no` 问答（含 2–9 条 `no` 题）。
- **关键局限**：① 问答由 VLM/LLM 合成加一致性过滤，非 100% 人工撰题；② 大量摄影/T2V 维度（`blur / color palette / color grading / shot type / lighting`）不适用于短字幕，必须事前过滤至核心语义维（`action / entity / count / color / relation / background`）；③ `LLM4eval-m.py` 虽内置三值 `Yes / No / Unanswerable`，但默认调 Gemini / GPT-4o API，且需防常识盲猜。
- **UARCap 复用结论**：可作为纯 QA 路线的备选方案，优先级次于人工描述质量更高的 DREAM-1K + VidHalluc 组合。

### 5. CapQuiz（ACL 2026）—— **当前不可用（排除）**
- **原始链接**：Apple 页面 https://machinelearning.apple.com/research/video-caption-quality ｜ACL Anthology https://aclanthology.org/2026.acl-long.777/
- **实查状态**：两处官方页面均**无任何 GitHub 或 Hugging Face 下载链接**，摘要明确仍停留在 `"We will release the benchmark to facilitate reproducible research."`。
- **UARCap 复用结论**：理论契合度极高，但**当前无法获取任何视频或题目，严禁设为研究依赖**。

### 6. VDC（ICLR 2025, AuroraCap）—— **淘汰**
- **原始链接**：HF https://huggingface.co/datasets/wchai/Video-Detailed-Caption ｜代码 https://github.com/wenhaochai/aurora ｜许可 `apache-2.0`
- **实查致命缺陷**：直接读取 `VDCScore_qa/short.jsonl` 第 1 行发现答案栏为未清洗的类别占位词（`"environment"`, `"lighting"`, `"object"`, `"action"`），且 `VDC_1k.jsonl` 约 500 词描述高度 GPT-4o 合成化，不具备事实真值公信力。

### 7. FactVC（EMNLP 2023）与 PE-Video（Meta）—— **淘汰**
- **FactVC**（https://github.com/NiteRainHL/FactVC）：实查 `data/activitynet/factuality_annotation.json` 仅对旧模型（`COOT / MART / PDVC`）历史输出做 1–5 句级评分，无面向新模型输出的可复用视频事实表。
- **PE-Video**（https://huggingface.co/datasets/facebook/PE-Video，`cc-by-nc-4.0`）：2,000 个 WebDataset `.tar` 分片、百万级训练/预训练语料，定位与存储开销不适合作为评测基准。
