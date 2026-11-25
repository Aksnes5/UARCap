# UARCap Q2 数据集选型最终裁决前定向补充审查报告（Q2 DATASET FINAL TARGETED REVIEW）

> **审查日期**：2026-11-25｜**审查范围**：仅针对新增 3 个候选（`VidHalLoc`、`VidOmni-Bench`、`HLVC-Dataset`）开展一级论文/官方仓库/公开元数据实证核验，并对上一轮报告中的 6 项科学表述进行正式纠偏，**不重启全领域调研、不重编前七份报告、不新增正式实验数量、不启动 GPU、不下载大型视频集、不修改实验代码**。  
> **证据分级标记**：`[论文原文]` = 直接读取 arXiv HTML/PDF 全文；`[官方源码实查]` = 直接读取 GitHub 源码/仓库树；`[公开元数据实测]` = 直接统计公开 JSONL/HF 文件树及 40 条 MP4 `mvhd` 头部 Range 探测；`[科研推断]` = 基于一级证据的独立科学裁决。

---

## 一、核心裁决速览（直接回答最终 6 问）

| 关键问题 | 独立科研裁决 | 核心一级证据 |
|---|---|---|
| **1. CaReBench 是否正式确定为 E2 首选？** | **是（条件性正式确定为 E2 首选；DREAM-1K 降为备用）** | `gated=False`、`MIT`、`3.66 GiB`；双人+专家时空分离参考有效缓解静态事实 `Unmentioned` 风险；913 独立 FineAction 父视频去重与排除后保守可支撑 **600–750** 正式视频（不再机械承诺 800）；须在 Pilot A1/A2 通过 4 项实证门控，若触发止损线再回退 DREAM-1K。 |
| **2. VidHalLoc 与 VidHalluc 哪个更适合 E3？** | **维持 `VidHalluc (ACH)` 作为 E3 首选；不以 `VidHalLoc` 替换** | `VidHalLoc` 虽然文本最小编辑极干净且覆盖静/动态 8 维，但 40 条 MP4 头部实测时长中位数 **`34.6s`**、均值 **`57.4s`**、P75 **`103.9s`**（`≤20s` 仅占 **`22.5%`**），且多为长视频局部细节、完全匿名无 YouTube ID、60% 样本未经人工审核且未标记审核子集；与 16 帧单句短字幕的中心视觉冲突控制严重错配。`VidHalluc (ACH)` 虽仅聚焦动作维，但短视频单主动作清晰、支持已知 YouTube ID 层面精确零交集过滤且既有代码零改动。 |
| **3. VidOmni-Bench 和 HLVC 是否能实际减少人工核验？** | **均不能（两者当前均无公开可用数据，且方法学角色均不匹配）** | **`VidOmni-Bench`** 官方 GitHub 仓库当前仅含 1 个 13 字节拼写错误空文件 `REAME.md`（零标注/零代码发布），且伦理声明永久不分发任何原始视频，且为 578.9 词 MLLM 长字幕 Type E 数据（无人工参考文本，无法验证参考型文本 NLI 评价器）；**`HLVC-Dataset`** 全文及全网零代码/零数据发布，直接排除。 |
| **4. 最精炼的最终数据集组合是什么？** | **历史资产（MSR-VTT + MSVD + 百万级 VideoUFO）+ E2：`CaReBench`（父视频去重）+ E3：`VidHalluc (ACH)`** | 零新增实验模块、论文叙事最集中：MSR-VTT/MSVD 开放域历史基线 → CaReBench 细粒度静态/动态自然检索诊断 → VidHalluc 已知 YouTube ID 零交集动作冲突因果干预 → 冻结 Video-LLaMA 与既有 `Qwen3.8-27B-FP8` 跨代际复现。 |
| **5. 哪些关键条件仍须通过试点验证？** | **6 项硬性 Pilot A0–A3 验证门控**（详见第五节） | CaReBench 与 VideoUFO 近重复比例、CaReBench 域内检索梯度、Pilot A2 真实 `Unmentioned` 率与人工校准偏差、开篇总览句核心覆盖完整性、16 帧微细节不可见率、VidHalluc 已知 YouTube ID 差集后的有效样本池与 5 项互斥通过率。 |
| **6. 最终选择后需要哪些最小工程改动？** | **复用 ~95% 既有代码；仅局部新增/修改 4 处（约 200 行）** | 新增 `uarcap_q2/datasets/carebench.py`；在 `stage_assets.py` / `data_preparer.py` / `run_stage_cli.py` 与配置模板增加 `--e2-dataset {dream1k,carebench}` 开关（保留 `dream1k` 兼容回退）；补 1 组 CPU 单元测试。E3 (`vidhalluc.py`) **0 改动**。 |

---

## 二、三大新增候选的一手实证专项审查

### 2.1 候选一：`VidHalLoc`（`arXiv:2609.09895` + `wesfggfd/VidHalLoc`）

本轮完整下载并解析了 29 页 arXiv PDF 全文（含 Appendix C 外部人工审核细则）、公开 HF 文件树、`release_audit.json`、全部 2,000 条样本（`Main/Eval/Test.jsonl` 与 `Validation.jsonl`），并对 `videos/video_caption/` 随机抽取 40 条 `.mp4` 执行了纯 HTTP Range 头部（`mvhd` atom）时长实测（未下载完整视频）。

#### （1）12 项必查指标一手核验表

| 必查项 | 一手实证结果 | 证据来源 |
|---|---|---|
| **公开数据文件与视频可获取性** | 完全公开可获取：`gated=False`；含 `data/{Main,Eval,Test,Validation}.jsonl`、`release_audit.json`，以及完整视频目录 `videos/video_caption/`（`549` 个 `.mp4`，`5.21 GiB`）与 `videos/videoqa/`（`541` 个 `.mp4`，`5.01 GiB`）。 | `[公开元数据实测]` |
| **下载体积与许可** | 全量视频 `10,974,941,960 bytes`（**`10.22 GiB`**，其中字幕子集仅 **`5.21 GiB`**）；HF 仓库与 `LICENSE` 声明为 **`Apache-2.0`**（原始上游视频仍受各自学术许可约束）。 | `[公开元数据实测]` |
| **字幕子集 vs 问答子集样本量** | `Main (1,200) + Eval (400) + Test (400) = 2,000` 条（`Validation` 为 `Main` 的 100 条子集）；其中 **`video_captioning = 980` 条**（`Main 588 / Eval 196 / Test 196`），**`video_qa = 1,020` 条**（`Main 612 / Eval 204 / Test 204`）。8 个细类各恰好 `250` 条（字幕子集内：`EEH 105 / ECH 125 / EQH 137 / AVH 130 / SRH 132 / APH 129 / TRH 118 / CPH 104`）。 | `[公开元数据实测]` |
| **实际独立视频数量** | 全库 **`1,090`** 个独立视频；其中 **`video_captioning = 549`** 个独立视频（静态 Ontology `458` 个、动态 Dynamic `311` 个；剔除否定 `EEH` 与运镜 `CPH` 后，洁净静态 `414` 个、洁净动态 `230` 个、同视频同时具备洁净静+动态 `129` 个）；`video_qa = 541` 个独立视频。`Main/Eval/Test` 间视频零重叠。 | `[公开元数据实测]` |
| **视频时长分布与 16 帧适配性（核心硬伤！）** | 论文 Table 4 披露各类别**平均时长高达 `48.63s`–`61.05s`**（`APH 48.63s`, `ECH 48.88s`, `SRH 48.95s`, `EQH 49.25s`, `AVH 50.03s`, `EEH 50.88s`, `TRH 53.44s`, `CPH 61.05s`）。本轮对 `videos/video_caption/` 随机 40 条 `.mp4` 做 `mvhd` 头部实测（40/40 解析成功）：**min `3.3s`、P10 `10.6s`、P25 `23.1s`、中位数 `34.6s`、均值 `57.4s`、P75 `103.9s`、P90 `128.4s`、max `140.9s`**；其中 **`≤15s` 仅 `8/40 (20.0%)`、`≤20s` 仅 `9/40 (22.5%)`、`≤30s` 仅 `14/40 (35.0%)`、`>45s` 高达 `18/40 (45.0%)`**！对 16 帧均匀采样旧模型（0.11–0.46 fps），长视频极易漏采局部证据。 | `[论文原文 Table 4 + 公开元数据实测]` |
| **正负事实对的人工核验比例与流程** | **全部 2,000 条样本由多智能体工作流 `VideoHALO` 自动生成，非全量人工撰写**。Appendix C（PDF pp. 15–17）披露：随机分层抽检 **`800/2,000 (40.0%)`**（每类 100 条，含 387 条字幕 + 413 条 QA，覆盖 649 个视频），由 2 名独立人工审核员 + 1 名作者看片核验正例受支持、反例受矛盾、仅在目标槽位不同及类别归属（样本准确率 `98.75%`，双人一致率 `96.63%`，`κ=0.962`）。**关键局限**：其余 **60% (`1,200` 条) 未经人工核验**，且公开 JSONL **没有字段标明哪些样本属于已审核的 800 条**！ | `[论文原文 Sec 3.2 & App. C + 公开元数据实测]` |
| **原始视频来源与可追溯 ID** | 上游来自 5 个公开数据集：**VidOR、COIN（YouTube 教学长视频）、Perception Test、UCF101-DS、UCF101（YouTube 动作视频）**。但公开发布已**全部匿名重命名**为 `captioning_0005.mp4` / `videoqa_0004.mp4`，**无任何原始数据集标签、无 YouTube ID、无时间区间标注**。 | `[论文原文 Sec 3.2 + 公开元数据实测]` |
| **VideoUFO 来源去重可行性** | **无法完成已知 YouTube ID 层面精确去重**（尽管 COIN 与 UCF101 均源自 YouTube）；仅能在下载后通过 X-CLIP 高相似度/感知哈希做疑似近重复筛查。 | `[公开元数据实测]` |
| **正负事实的互斥性、长度平衡与自然性** | 在 `video_captioning`（980 条）中文本质量**非常出色**：`answer` 均长 **`10.83` 词** vs `counterfactual_answer` 均长 **`10.79` 词**，`|词长差|` 均值仅 **`0.22` 词**（中位数 `0.0`，`99.7%` ≤ 2 词），`0%` Yes/No 前缀，`99.1%` 严格满足最小单槽编辑（≤ 4 词差异），音频相关仅 `1/980 (0.1%)`。仅存在两类局部瑕疵：`EntityExistence` 多为纯否定改写（`is located` → `is not located`），`CameraPredicate` 中有 19 条包含代码风格 `zoom_in` 下划线词。 | `[公开元数据实测]` |
| **能否直接构造 `Base/Sup/Con/Neu`** | **语法层面可以直接构造**（`F` 作 `Sup`，`CF` 作 `Con`，其他视频无关句作 `Neu`），但存在**核心科学混杂**：大量样本描述的是 35–140 秒长视频中的**局部非核心细节**（如 *"There are 3 tomatoes on the vine in the upper right corner"*、*"The barber's shirt is blue"*），在 16 帧单句（8–18 词）自由描述的 `Base` 条件下模型几乎不会自然提及，导致 `Sup` 相对 `Base` 的上升主要反映“话题提示效应（topic priming）”，而非主视觉语义上的真假冲突覆盖！ | `[公开元数据实测 + 科研推断]` |
| **短视频静态+动态干预候选是否充足** | **偏紧且存在断档风险**。按 `≤20s` 约占 `22.5%` 推算：`video_captioning` 全部 549 个视频中短视频仅约 **`~124` 个**；细拆到核心动态动作类 `ActionPredicate`（129 个字幕视频），`≤20s` 仅约 **`~29` 个**！即便把 `video_qa` 中非 Yes/No 样本并入并放宽至 `≤30s`（~35%），剔除 `EEH/CPH`、Pilot A3、疑似近重复及 16 帧不可见样本后，洁净动态动作短视频依然捉襟见肘，迫使实验接受 40–100 秒长视频，从而破坏 16 帧控制前提。 | `[公开元数据实测 + 科研推断]` |
| **人工核验成本是否低于 VidHalluc** | **不会低于，反而大概率更高**。原因：(1) 公开 JSONL 未标记 800 条人工审核子集，选出的 150–180 条仍须 100% 人工看片确认真假互斥与 16 帧可见性；(2) 看一条 35–140 秒视频搜寻局部属性/数量细节的单条核片耗时，显著高于看一条 5–15 秒 `VidHalluc` 单主动作片段。 | `[科研推断]` |

#### （2）独立裁决：`VidHalLoc` vs `VidHalluc (ACH)` 谁更适合承担 E3？

**结论：维持 `VidHalluc (ACH)` 作为 E3 主数据集，不以 `VidHalLoc` 替换。**

本轮并未因 `VidHalLoc` 更晚发表（2026 年 9 月）或类别更全（覆盖静+动态 8 维）就迎合替换，也没有因为 `VidHalluc` 已有代码就机械护短。真正的决定性科学差异在于 **E3 的行为因果识别条件**：
1. **16 帧单句短字幕的“主事件中心性”要求**：UARCap 的 E3 是在冻结 16 帧 `Video-LLaMA` 和 16 帧 `Qwen3.8-27B-FP8` 上生成 **8–18 词单句总述字幕**。`VidHalluc (ACH)` 的短视频聚焦单一中心视觉动作，目标事实在 `Base` 条件下天然处于描述中心，使得 `Base / Sup / Con / Neu` 能够干净识别“视觉输入固定时，冲突文本是否压制正确主事实并诱发对应错误”；而 `VidHalLoc` 面向的是“长视频多帧评测器评测（evaluating hallucination detectors）”，近半数视频超过 45 秒（最高 140.9 秒）且大量样本考查局部属性/角落数量，与 16 帧单句总结任务存在结构性粒度错配。
2. **来源可控性**：`VidHalluc (ACH)` 提供 **1,586 个显式 11 位 YouTube ID**，可与本地 VideoUFO 345,437 ID 完成已知 ID 层面的精确无交集过滤；`VidHalLoc` 虽大量源自 YouTube（COIN/UCF101），却已全量匿名化，完全丧失已知 ID 查重能力。
3. **论文逻辑一致性如何诚实闭环？** E2（CaReBench）负责自然检索下**静态对象 + 动态动作**两类事实错误的全面诊断；E3（VidHalluc ACH）负责视觉严格锁定的**核心动作事实冲突**受控行为因果检验，论文中坦诚将“静态属性/数量维度的受控干预”列为明确研究边界与未来扩展（并可在 Related Work / Discussion 中客观引用 `VidHalLoc` 说明其长视频局部细节定位价值）。这样既严守实验控制纯度，又零新增工程返工与时长风险。

---

### 2.2 候选二：`VidOmni-Bench`（EMNLP 2026，`arXiv:2609.21521v2`）

本轮完整读取 `arXiv:2609.21521v2` 全文（含 Ethics Statement 与 Appendix B/C/D）并直接检查论文唯一官方仓库 `https://github.com/changbeendata/VidOmni-Bench` 及 Hugging Face 检索结果。

#### （1）关键事实核验
1. **实际发布状态（直接阻断！）** `[官方源码实查]`：截至 2026-11-25，官方 GitHub 仓库 `changbeendata/VidOmni-Bench` **完全为空**，仓库中仅有 1 个拼写错误文件 `REAME.md`（全文仅 13 字节：`VidOmni-Bench`），**无任何标注 JSON、无任何元数据、无任何代码**；Hugging Face 上检索 `VidOmni` 返回空列表 `[]`。
2. **原始视频永久不分发** `[论文原文 Ethics Statement]`：论文明确声明 *"we do not redistribute raw video files for any of the 500 videos in VidOmni-Bench, regardless of source. Instead, we will release only video metadata, the generated dense captions, and the annotations upon publication."* 500 条视频分散于 11 个上游库（Video-ACID、SpatialVID、Mannequin Challenge、ActivityNet、Assembly101、Epic-Kitchens、Ego4D、UCF-Crime、Sports-1M、Video-MME、XD-Violence）加 CC YouTube 爬取，获取成本极高。
3. **数据结构、时长与人工审核流程** `[论文原文 Sec 3.2–3.4 & App. B.3, D.1]`：
   - 500 条视频分 5 种复杂度（各 100 条），跨 3 档时长：**Short（`≤ 3 min`，即 4 秒–180 秒，约 167 条）**、Medium（4–30 min）、Long（≥ 30 min，最长 90 min）；433/500 含音轨（Table 13 证实音频对验证有显著影响）。
   - 全部 2,500 条描述由 5 个现代 Video-LLM（`InternVL3.5 38B`、`Qwen3-VL 32B`、`video-SALMONN 2+ 72B`、`Keye-VL 1.5 8B`、`Gemini 3 Flash`）生成，均值 **`578.9` 词/条**（总计 64K 句），85.6%–94.0% 的描述含错误句。
   - 人工标注：每条由 ≥2 名（总计 5 名筛选后标注员）对句子判 `Correct / Incorrect / Unknown`（原始 `Fleiss' κ = 0.50`，`P_bar = 0.764`，后经作者逐条仲裁），再对 `Incorrect` 句标 10 类词级错误。

#### （2）六项专项论证裁决
- **能否复用其人工标签作为评价器独立验证数据？能否减少 Pilot A2 人工校准？** **均不能。**
  1. 当前根本无公开数据可下；
  2. 更根本的方法学障碍在于：**`VidOmni-Bench` 不含人类参考描述（纯 Type E）**，其标注对象是“**直接看原始视频的 Video-LLM 验证器**”（输入 `Video + 578.9 词长段落句子 → 0/1`），而 UARCap E2 的自动事实评价器是“**基于 CaReBench 人类参考文本的跨家族纯文本 LLM 三值 NLI 评价器**”（输入 `Human_Ref + 8–18 词短字幕主张 → ENTAILED/CONTRADICTED/UNMENTIONED`）。两者输入模态、评价机理、字幕分布（579 词现代多句长文 vs 10 词检索增强单句短字幕）完全不同；**即使未来数据公开，也绝不能替代 UARCap Pilot A2 针对我们自己新生成 `A0/A1/A7` 短字幕的双盲人工看片校准与两阶段误差校正**。
- **是否存在适合 E3 的已人工认证错误事件？** 不适合。错误句子嵌套在 578.9 词长描述中且无配对最小编辑正例，绝大多数对应多分钟长视频且不提供视频文件。
- **是否需要下载全部长视频 / 能否只用公开标注和少量短视频？** 即使只用文本标注，缺乏配对人类参考文本也无法驱动我们的参考型 NLI 评价器；若跑视频验证器又必须自行向 Ego4D/Epic-Kitchens/Assembly101/UCF-Crime 等多源申请下载视频，得不偿失。
- **是否存在足够完整的人工参考事实支持 E2？** **完全不存在**（零人工撰写参考描述）。
- **唯一可保留的文献引用价值**：`VidOmni-Bench` Section 4.3 与 Table 11 实证发现 **Video-LLM 评判自身生成字幕时存在严重自我偏好偏差（self-preference bias）**，跨家族验证可显著缓解——这正好为 UARCap 选用非 Qwen 家族开源 LLM（如 Mistral/Gemma/Llama）评价 `Qwen3.8-27B-FP8` 输出提供了最新 EMNLP 2026 权威支撑！

---

### 2.3 候选三：`HLVC-Dataset`（`arXiv:2510.25225`，LY Corporation）

本轮完整读取 `arXiv:2510.25225v1` 全文并检索 GitHub 与 Hugging Face。

- **实际数据发布情况** `[论文原文 + 公开仓库检索]`：论文全文（含 Section 3 Dataset、Section 6 Conclusion、Acknowledgement）**未提供任何 GitHub、Hugging Face、Zenodo 或项目主页链接**；GitHub 检索返回 `0` 个仓库，Hugging Face 检索返回 `0` 个数据集。
- **裁决**：按明确指令**直接排除**（且其本质同为 Type E——对 MSR-VTT 与 FAVD-Bench 上旧 `VideoLLaMA`/`VideoChat` 输出做词级幻觉标注，缺乏新模型参考复用性）。

---

## 三、对上一轮报告 6 项科学表述的正式修正

本轮逐项复核原始论文与源码，对上一轮七份审查文档中的 6 处表述做出正式、透明的科学修正（后续所有论文撰写与协议均以本修正为准）：

1. **修正 AutoDQ 与 CapST 的 Precision 性质（撤回“程序 Bug / 源码缺陷”指控）** `[论文原文 + 官方源码实查]`：
   - 核验 `arXiv:2407.00634v2`（Tarsier/DREAM-1K）Section 4.2 原文确认：作者明确将 Precision 定义为 *“the ratio of events in $D_{\text{model}}$ that are entailed by $D_{\text{ref}}$”*（生成事件中被参考文本蕴含的比例）。因此，官方源码 `evaluate_dream_gpt.py` L254 传入 `is_recall=True`（以及 CaReBench `dream_gpt.py` L446/L448 的同名调用）仅计入 `entailment`、不计入 `neutral`，**严格符合原指标定义，并非程序 Bug**（源码中 `else ['entailment', 'neutral']` 只是未被调用的保留分支；CaReBench 原文 Section 3.4.2 公式 (2)-(3) 仅存在符号笔误，代码逻辑与 AutoDQ 一致）。
   - **重新精确区分三个概念**：
     - **参考支持率（Reference Support Rate）**：$P(\text{ENTAILED} \mid D_{\text{ref}})$ — AutoDQ / CapST 的原生 Precision，受参考描述完备性直接制约；
     - **明确参考冲突率（Explicit Reference Contradiction Rate）**：$P(\text{CONTRADICTED} \mid D_{\text{ref}})$ — 将参考未提及（`UNMENTIONED`）与明确冲突显式剥离；
     - **真实视频事实错误率（True Video Factual Error Rate）**：$P(\text{FALSE} \mid V)$ — 针对真实视频画面的错误率，必须通过 Pilot A2 盲法人工看片校准与 Formal B 两阶段统计校正从三值参考标签无偏估计。
   - 结论不变但论证更严谨：我们不直接使用 AutoDQ/CapST 并非因为其代码有 Bug，而是因为**参考支持率本身无法区分“参考遗漏（`UNMENTIONED`）”与“明确幻觉（`CONTRADICTED`）”**，不满足 UARCap 的事实错误诊断需求。
2. **修正 CaReBench 父视频独立性表述**：
   - `1,000` 片段去重至 **`913` 个不同 FineAction 父视频 ID**（`v_XXXXXXXX`），仅消除**已知 FineAction 内部同父视频多片段聚类**，**不等于 913 个已证明绝对独立的原始 YouTube 来源**（不同父视频仍存在跨搬运/剪辑重叠的理论可能，且匿名 ID 无法与 VideoUFO 精确匹配）。残余来源重叠风险须明确披露并通过检索后高相似近重复筛查做敏感性分析。
3. **修正 CaReBench 开篇总览句定位**：
   - `94.0%`（`940/1,000`）样本的开篇单句总览（均长 14.5 词）只能作为**人工撰写的核心事件候选（core-event candidates）**，**不能自动认证其已覆盖全部核心事实**（尤其 15–20s 片段中的次要核心事件可能位于 `temporal_caption` 后段），必须在 Pilot A2 中人工抽检验证覆盖完整性并制定 6% 非模板项的统一抽取规则。
4. **修正人工成本节省表述**：
   - 上一轮报告中“CaReBench 相比 DREAM-1K 预计节省 15%–25% 双人人时”**仅属于实验前预算推测，绝非实测结果**；实际是否节省取决于 Pilot A2 实测 `Unmentioned` 率、首轮校准是否一次过线以及 Formal B 是否需要自适应扩增样本量。
5. **重新核算 CaReBench 正式有效样本量（取消“机械承诺 800 独立正式视频”）**：
   - 精确扣除链路如下：
     - 原始公开片段：`1,000`；
     - FineAction 父视频去重（1 片段/父视频）：**`913`**（`Socializing, Relaxing: 311`、`Sports, Excercise: 239`、`Personal Care: 186`、`Household Activities: 177`）；
     - 扣除不相交 Pilot A1/A2 集合（`30` 父视频）：**`883`**；
     - 扣除独立人工校准验证集 Formal B（`50` 至 `80` 父视频）：剩余 **`833` 至 `803`**；
     - 扣除视频解码异常及疑似 VideoUFO 近重复剔除（保守按 `2%–8%` 即 `15–65` 条估算）：**纯未标注正式分析池约 `740–815` 条**（若含 Formal B 一并纳入两阶段合并估计则总正式池约 `790–850` 条）。
   - 因此，正式方案应**稳健承诺 `600–750` 条有效独立正式视频**（留足排除余量），绝不机械承诺 800 条。
6. **修正 VidHalluc 精确 YouTube ID 过滤的证明边界**：
   - 利用 `ach_mcq.json` 的 1,586 个 11 位 YouTube ID 与 VideoUFO 的 345,437 个 YouTube ID 求差集，**只证明“已知 YouTube ID 层面无交集”**，**不证明不存在二次转载（不同 YouTube ID）或未知来源近重复**。后续仍需在检索阶段保留高相似度近重复审计，并在论文中严格按此边界措辞。

---

## 四、最精炼的最终数据集组合与论文结构闭环

坚持**不新增实验数量、不并列两套 E3、不堆砌评价器验证章节**的原则，本轮建议冻结以下最精炼闭环作为最终 Q2 研究配置：

- **历史资产（100% 原封不动保留）**：MSR-VTT（17,940 次历史预算实验）+ MSVD（历史跨集验证）+ 百万级 VideoUFO（`1,091,712` 片段 / `345,437` 独立 YouTube 来源检索库）。
- **E2 自然检索事实诊断（首选确定）**：**`CaReBench`**（913 FineAction 父视频去重 → Pilot 30 + Formal B 50–80 + 稳健目标 **600–750** 正式视频；复用双人+专家 `caption`/`spatial_caption`/`temporal_caption` 人工参考，绝不将 GPT-4 提取的 `events`/`objects` 冒称为人工 Gold）；`DREAM-1K` 降为 Pilot 止损回退备用。
- **E3 受控冲突干预（维持原首选）**：**`VidHalluc (ACH)`**（已知 YouTube ID 层面精确无交集过滤 + 每 ID 至多 1 clip + 5 项人工看片互斥与 16 帧可见性审核 → 150–180 条高质量动作冲突干预对）。
- **模型与自动评价体系**：冻结历史 16 帧 `Video-LLaMA/Vicuna-7B` + BN 既有 16 帧 `Qwen/Qwen3.8-27B-FP8`（备用 `Qwen3.5-27B`）；自动评价保持 UARCap 跨家族开源 LLM 三值参考事实评价器（引用 `VidOmni-Bench` EMNLP 2026 自我偏好发现支撑跨家族选择）+ Pilot A2/A3 盲法双人看片校准与两阶段误差校正。

---

## 五、仍须通过 BN 真实试点（Pilot A0–A3）确认的 6 项关键条件

1. **CaReBench 与 VideoUFO 的实测近重复比例**（Pilot A1 X-CLIP top-1 极高相似尾部抽样看片核验；若确证重叠异常偏高则触发止损评估）。
2. **VideoUFO 对 CaReBench（FineAction 人类活动）的检索相关性梯度**（确认 top-1 vs top-7 检索结果同时包含有效线索与自然干扰，避免全命中或全无关退化）。
3. **Pilot A2 三值参考事实评价器的实测 `Unmentioned` 率与人工校准偏差**（30 视频 × 3 臂 = 90 短字幕盲法双人看片，检验实际校准是否一次达标以及真实人工工时）。
4. **CaReBench 开篇总览句（94% 模板 + 6% 非模板）作为 `R_core_ref` 核心事件候选的实际覆盖完整性**（Pilot A2 同步抽检，若漏盖核心事件比例高则改用 `temporal_caption` 辅助核心句提取）。
5. **16 帧 / 224px 旧 Video-LLaMA 在 CaReBench 细微空间属性上的 `Video-Unverifiable` 比例**（确认不会因低分辨率看不清背景细节而扭曲事实错误率估计）。
6. **VidHalluc (ACH) 与 BN 本地 VideoUFO 345,437 YouTube ID 差集后的剩余独立来源池及 5 项互斥看片通过率**（Pilot A3 确认稳妥产出 150–180 条有效 E3 样本）。

---

## 六、最终确认后的最小工程改动清单（零提前修改，等待授权后执行）

确认采用 **CaReBench (E2) + VidHalluc (E3)** 后，既有离线工程（`research/uarcap_q2_experiments/`，已验证 59/59 项测试通过）约 **95% 原封不动复用**（含 `vidhalluc.py`、全部检索/生成/三值评价/标注 UI/双人仲裁/统计与调度模块），仅需 **4 处局部改动（共约 200 行）**：

1. **新增 [`uarcap_q2/datasets/carebench.py`](file:///D:/UARCap/research/uarcap_q2_experiments/uarcap_q2/datasets)**（约 150 行）：解析 `CaReBench/json/metadata.json`，加载 `caption` / `spatial_caption` / `temporal_caption` 与开篇总览候选句，内置 FineAction 父视频前缀（`v_XXXXXXXX`）去重与 4 大类分层抽样；
2. **扩展 [`uarcap_q2/datasets/data_preparer.py`](file:///D:/UARCap/research/uarcap_q2_experiments/uarcap_q2/datasets/data_preparer.py) 与 `deployment/stage_assets.py`**：注册 `carebench_metadata` / `carebench_videos` 资产校验与 `dataset == "carebench"` 分派分支；
3. **扩展 `deployment/run_stage_cli.py` 与 YAML 配置模板**：新增 `--e2-dataset {dream1k,carebench}` 命令行与配置开关（**完整保留 `dream1k` 零破坏回退路径**）；
4. **新增 1 组 `carebench` CPU 单元测试**：覆盖父视频去重、开篇总览提取、分层抽样与全链路 Mock 冒烟测试，确保全量测试 100% 通过后推送 GitHub 供 BN 服务器拉取执行。

---

## 七、本轮补充审查可追溯一级引用

- **VidHalLoc**：Chen, X., Mahmood, A., & Dras, M. (2026), *Can We Trust Video Hallucination Detectors? VidHalLoc for Evaluating the Evaluators*, `arXiv:2609.09895v1` (`https://arxiv.org/abs/2609.09895`, `https://arxiv.org/pdf/2609.09895`, Table 3–4 & Appendix C pp. 15–17); 公开仓库 `https://huggingface.co/datasets/wesfggfd/VidHalLoc`（实测 `release_audit.json`、`data/{Main,Eval,Test,Validation}.jsonl` 及 40 条 `videos/video_caption/*.mp4` 头部 `mvhd` 时长）。
- **VidOmni-Bench**：Kim, C., Chang, J., Kim, K., Shinoda, R., Saito, K., & Kim, D. (EMNLP 2026), *VidOmni-Bench: A Benchmark for Fine-Grained Video Understanding via Spatio-Temporal Event Verification across Complexity and Duration*, `arXiv:2609.21521v2` (`https://arxiv.org/abs/2609.21521`, ACL Anthology `https://aclanthology.org/2026.emnlp-main.104/`, Sec. 3–4, Ethics Statement, App. B–D); 官方仓库 `https://github.com/changbeendata/VidOmni-Bench`（实查仅含 13 字节 `REAME.md`）。
- **HLVC-Dataset**：Nakada, S., Saito, K., Ishikawa, Y., Munakata, H., Komatsu, T., & Kondo, M. (2025), *Hallucination Localization in Video Captioning*, `arXiv:2510.25225v1` (`https://arxiv.org/abs/2510.25225`, LY Corporation；核验无公开数据或代码发布）。
- **AutoDQ & CapST 原文与源码复核**：Wang et al. (2024), `arXiv:2407.00634v2` Sec. 4.2 + `bytedance/tarsier/evaluation/metrics/evaluate_dream_gpt.py` L221, L254；Xu et al. (2025), `arXiv:2501.00513v2` Sec. 3.4.2 + `MCG-NJU/CaReBench/utils/dream_gpt.py` L445–448。

---

**本轮定向补充审查结论状态**：数据集选型讨论正式收尾，等待用户最终确认采用 **`CaReBench (E2) + VidHalluc (E3)`**（保留历史 MSR-VTT/MSVD/VideoUFO 与 `DREAM-1K` 备用回退），授权后即执行 4 处最小离线代码适配并推进 BN 真实试点。
