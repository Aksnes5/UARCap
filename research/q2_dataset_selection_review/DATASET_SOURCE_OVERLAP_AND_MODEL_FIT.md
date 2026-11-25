# 数据来源重叠风险（VideoUFO）与 16 帧旧模型适配性审计（DATASET SOURCE OVERLAP AND MODEL FIT）

> **硬性边界声明**：仅通过高 X-CLIP 相似度或感知哈希**不能证明**来源级零重叠；同样，来源 ID 不可取得也**不代表必然存在**重叠。本报告严格区分 **已证明重叠 / 精确可去重 / 疑似近重复可筛查 / 不可识别残余风险** 四类状态。

---

## 一、与百万级 VideoUFO（1,091,712 片段 / 345,437 独立 YouTube 来源）的重叠风险审计

| 候选数据集 | 上游视频来源 | 公开元数据中的 ID 形态 | 精确 YouTube ID 去重可行性 | 父视频级聚类去重可行性 | 近重复风险筛查可行性 | 不可消除残余风险与论文诚实措辞 | 预期有效独立样本量 |
|---|---|---|---|---|---|---|---|
| **VidHalluc (ACH)** | ActivityNet / YouTube 公开视频 `[公开元数据实测]` | **显式 11 位 YouTube ID**（如 `vcCwvRYqU2I_clip_4`） | **100% 可行（精确零重叠！）**：直接与 BN `/home/bn/videoufo_research/manifests/clips_full.tsv` 的 345,437 个 YouTube ID 求差集 | **100% 可行**：3,897 片段归属 **1,586 个独立 YouTube 来源**，强制每源最多 1 clip | 精确去重后叠加高余弦相似度核查 | **极低**（仅剩极少量二次搬运不同 YouTube ID 的残余风险） | 去重+5项人工过滤后 **600–1,000+ 独立来源**，远超 E3 所需 150–180 |
| **CaReBench** | **FineAction**（上游为 YouTube-8M、Kinetics-400、FCVID 与 YouTube 关键词抓取） `[论文原文]` | FineAction 匿名父 ID + 片段号（如 `v_00002068_11.mp4`） | **公开元数据不可直接精确去重**（`v_00002068` 非 YouTube ID；需 FineAction 原始 URL 映射表，公开包通常仅含匿名 ID） | **100% 可行**：实测 1,000 片段来自 **913 独立 FineAction 父视频**（68 父视频含 2–6 片段），可严格去重至 ≤913 | **可行**：下载后通过 X-CLIP top-1 极高相似度（如 >0.92）+ 帧级感知哈希（pHash/dHash）+ 人工抽检高相似对筛查疑似近重复 | **中等（不可识别风险）**：FineAction 与 VideoUFO 同源 YouTube，不能宣称“来源级零重叠”，须披露为“父视频严格去重 + 近重复筛查，保留不可识别 YouTube 重叠残余风险” | **913 独立父视频** → 排除 Pilot 30 与疑似近重复后约 **750–850 独立视频**（轻松满足 600–750 正式样本） |
| **DREAM-1K** | 5 类混合源（Live-action 电影、动画、YouTube、TikTok 短视频、Stock） `[论文原文]` | 匿名序号 `1..1000` | **不可直接精确去重**（无原始 YouTube ID、无 URL、无拍摄/抓取日期） | **不可验证**（无法从 `1..1000` 判断是否存在同片多片段） | **可行**：下载后 X-CLIP top-1 高相似 + 帧感知哈希 + 文本近重复筛查 | **中等（不可识别风险）**：YouTube 子集（约 200 条）与 VideoUFO 存在不可识别重叠风险，电影/动画/Stock/TikTok 重叠概率相对更低 | **1,000 片段** → 排除 Pilot 30 与近重复后约 **850–930** |
| **VidHal** | MVBench (534)、Perception Test (259)、TempCompass (177)、AutoEval-Video (30) `[公开元数据实测]` | 匿名维度编号（`attribute_1`, `action_1`...） | **不可精确去重**（无 YouTube ID） | 部分来自 Synthetic/CLEVRER（261 条须直接剔除） | 仅可做帧哈希/高相似筛查 | **中高**：部分 MVBench/AutoEval 片段源自 YouTube/ActivityNet/Charades，且剔除合成/多帧方向题后真实自然样本缩水 | 1,000 → 剔除 261 合成 + 剔除不可观察方向/顺序后仅剩约 **350–500 自然视频** |
| **ARGUS** | YouTube + Mixkit + Pexels/UUID 混合 `[公开元数据实测]` | 部分含 11 位 YouTube ID（如 `BQkZzDHzvIo`），部分含 `-Scene-002` 同源前缀，部分为 SHA 哈希 | **仅部分可行**（YouTube 命名子集可精确去重，哈希命名子集不可） | 含 `-Scene-` 同源片段需按前缀聚类 | 可行 | 仅 500 总样本，去重与排除 Pilot 后样本量偏小 | **约 420–450**（不足 600–800 正式规划） |

---

## 二、无需 GPU 的后续 BN 现场来源重叠可执行核验方案（适用于 CaReBench 与 DREAM-1K）

1. **阶段 0（精确 ID 查重 — 仅对含 YouTube ID 的数据集，如 VidHalluc）**：
   - 已有代码 `uarcap_q2/datasets/vidhalluc.py::deduplicate_and_filter_vidhalluc()` 直接读取 `/home/bn/videoufo_research/manifests/clips_full.tsv` 的 345,437 个 YouTube ID，实现 O(1) 精确剔除与每来源至多 1 clip。
2. **阶段 1（父视频聚类去重 — CaReBench 专享优势）**：
   - 对 `v_XXXXXXXX_N.mp4` 按 `v_XXXXXXXX` 分组，每父视频仅保留 1 条片段（1,000 → **913 独立父视频**），彻底消除同父视频跨样本相关性与 Pilot→Formal 同源泄漏。
3. **阶段 2（疑似跨库近重复筛查 — 在 Pilot A1 / Formal B 检索后零额外推理成本执行）**：
   - 利用 X-CLIP 检索天然产出的 top-1 余弦相似度分布，自动标记 top-1 相似度高于预设告警阈（例如 `> 0.90` 或 top 2% 极端高相似尾）的查询视频；
   - 对告警样本做 **轻量人工看片/关键帧核对**（比对查询帧与 VideoUFO top-1 片段描述及缩略信息），将确认为同一视频片段的查询列入 `suspected_videoufo_overlap_audit.json` 并同时报告 **全样本主分析 + 排除疑似近重复敏感性分析**。
   - **诚实披露原则**：论文中明确声明“对 VidHalluc 完成了精确 YouTube ID 零重叠过滤；对 E2 自然检索集完成了父视频去重与高相似度疑似近重复敏感性剔除，但不宣称未经原始 URL 映射证明的绝对零重叠”。

---

## 三、与历史 16 帧 Video-LLaMA 模型的适配性评估（Section X）

历史主模型为冻结的 **Video-LLaMA / Vicuna-7B (`Transformers 4.28.0`)**：**16 帧均匀采样、静音无音频、输出 8–18 词单句简短字幕**；现代对照模型为既有 **`Qwen3.8-27B-FP8`（备用 `Qwen3.5-27B`）**，使用**完全相同的 16 帧视觉输入与短字幕指令约束**（确保跨模型严格同视觉控制）。

| 适配性核验维度 | **DREAM-1K** | **CaReBench** | **ARGUS** | **VidHalluc (ACH)** | **VidHal** |
|---|---|---|---|---|---|
| **视频时长分布** | **均值 8.9s**（极短） | **均值 14.35s**（>80% 在 5–20s；短于 MSR-VTT 15.01s） | **15–90s**（偏长，16 帧易漏关键事件） | 短动作片段（通常 3–15s） | 多数 3–15s |
| **是否需要音频/旁白** | 否（视觉事件） | **显式剔除音频/语音**（专家审校规则第 4 条 `[论文 3.2.2]`） | 部分长片含叙事语境 | 否（纯视觉主动作） | 否 |
| **是否依赖 OCR / 字幕** | 少量影视/TikTok 含字幕 | 极少（专家保留字幕变化仅作次要说明） | 较多长片镜头文字 | 否 | 部分 `order` 题考查屏幕字母顺序（如 `order_1: e, m, n`），不适合旧 16 帧模型！ |
| **极快速动作/多镜头切换** | 影视/短视频存在一定镜头切换 | 多数为单场景连续人类活动，镜头稳定、主动作连贯 | 15–90s 多镜头切换频繁 | 动作连贯清晰 | `direction`/`order` 依赖细微时序方向与帧序，16 帧旧模型辨别难；261 条 CLEVRER 为合成 3D 球体 |
| **16 帧可观察性判断** | **优**（1.8 fps，主事件几乎全落入 16 帧） | **良–优**（1.1 fps，优于 MSR-VTT；主动作与静态对象均稳定可见，仅极细微局部如“银色亮片眼线”可能超出 224×224 低分辨率观察上限） | **差**（90s 视频 16 帧仅 0.18 fps，大量事实不在输入帧内） | **优**（单主动作清晰，且有 Pilot A3 5 项人工核验确认 16 帧可见性） | **中**（仅 `action`/`object`/`attribute(count/color)` 真实视频子集适合，需剔除合成与 OCR/细微速度题） |
| **“视频真值 vs 16 帧可见性”区分机制** | 通过人工标注界面同步展示 16 采样帧条带，区分“视频真实但 16 帧不可见” | 同上，且人工标注规程将超出 224px 分辨率或帧间遗漏的极微属性归入 `Video-Unverifiable` 而非惩罚模型 | 帧间遗漏严重，易混杂视觉盲区与检索错误 | 已在 `E3ApplicabilityChecklist` 中硬编码 `target_in_sampled_window` 与 `action_clear_in_sampled_frames` | 需额外筛选与看片 |

**适配性核心结论**：
- **DREAM-1K（8.9s）与 CaReBench（14.35s）均完全适配 16 帧静音 Video-LLaMA**（CaReBench 甚至比历史已成功运行 17,940 次的 MSR-VTT 均长 `15.01s` 更短！），而 **ARGUS（15–90s）明显不适配 16 帧旧模型**；
- 在 E3 候选方面，**VidHalluc (ACH)** 的自然动作片段显著优于含 26.1% CLEVRER 合成球体和屏幕字母顺序题的 **VidHal**。
