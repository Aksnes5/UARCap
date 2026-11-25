# 全球公开视频事实评价数据集系统性检索与文献综述（GLOBAL VIDEO DATASET LITERATURE SURVEY）

> **调研日期**：2026-11-25｜**调研性质**：独立科研设计审查（零 GPU 运行、零完整视频下载、零正式代码修改）  
> **证据分级标记**：`[论文原文]` = 直接核验自 arXiv/会议原文；`[官方源码实查]` = 直接读取 GitHub 源码；`[公开元数据实测]` = 直接下载并统计公开 JSON/Parquet/HF 文件树；`[科研推断]` = 基于证据的科学判断。

---

## 一、围绕 UARCap 核心问题的 18 个系统检索方向

本研究并非单纯追求更高 CIDEr 或开发新视频字幕模型，而是诊断 **“外部检索证据预算为何在短视频描述中产生非单调效用（A0 → A1 增益、A1 → A7 下降）及冲突外部文本是否定向诱发幻觉”**（RQ1/RQ2/RQ3，承接 UARCap v0.8 在 MSR-VTT、MSVD 与百万级 VideoUFO 上的既有成果）。据此，本轮覆盖了 18 个国际公开文献与基准方向（检索源涵盖 CVPR、ICCV、ECCV、NeurIPS、ICML、ICLR、ACL、EMNLP、AAAI、ACM MM、TMLR、TPAMI/IJCV、arXiv、GitHub 与 Hugging Face）：

1. Dense video captioning evaluation（密集视频字幕评测）
2. Detailed video description benchmarks（细粒度视频长描述基准）
3. Factual video captioning evaluation（视频字幕事实一致性评测）
4. Video caption hallucination benchmarks（视频字幕幻觉基准）
5. Video hallucination detection（视频幻觉检测）
6. Video factuality verification（视频事实核验）
7. Fine-grained video understanding（细粒度视频理解）
8. Object-centric video annotation（以对象与属性为中心的视频标注）
9. Event-centric video annotation（以事件与动作为中心的视频标注）
10. Spatial-temporal video understanding（时空分离理解基准）
11. Video captioning human evaluation（视频字幕人工评测与元评价数据）
12. Video caption entailment（视频-文本蕴含推理）
13. Video factual grounding（视频证据定位与事实支撑）
14. Video retrieval hard negatives（视频检索困难负例）
15. Video counterfactual reasoning（视频反事实推理）
16. Human-annotated video proposition datasets（人工核验视频原子命题数据）
17. Retrieval-augmented video generation evaluation（检索增强视频生成与冲突证据评测）
18. Video QA benchmarks with exhaustive/fine-grained annotations（细粒度视频问答基准）

---

## 二、数据集五大功能类型划分（严格按实验角色归类）

| 类型 | 定义与核心资产形态 | 适配 UARCap 的实验角色 | 代表候选 |
|---|---|---|---|
| **类型 A** | 丰富的人类视频参考描述（纯人工撰写的细粒度段落/分离时空描述） | **E2 自然检索主数据集**（支撑三值参考事实评价与覆盖分析） | CaReBench、DREAM-1K、ARGUS、Shot2Story、ActivityNet-Captions |
| **类型 B** | 结构化事实标注（对象、属性、数量、动作、时空关系、事件列表或场景图） | E2 结构化拆解辅助 / 静态-动态归因（**须严查是否大模型提取**） | CaReBench（`events`/`objects` 为 GPT-4 提取）、Action Genome、VidOR/VidSTG、VDC（已淘汰） |
| **类型 C** | 真实视频 + 人工/半自动验证的真假陈述对或最小语义替换反事实对 | **E3 受控冲突干预**（构造视觉锁定的 Base/Sup/Con/Neu） | VidHalluc（ACH/TSH/STH）、VidHal、TempCompass、VideoCon、VITATECS、VIOLIN |
| **类型 D** | 细粒度视频问答（二值/三值/多选 QA） | E2/E3 辅助验证或敏感性分析（**不能直接把封闭 QA 准确率等同于开放字幕事实率**） | VCapsBench、VidCapBench、CapQuiz（未释放）、VideoHallucer、EventHallusion、COVER、EG-VQA |
| **类型 E** | 已有旧模型生成字幕的人类错误/幻觉/遗漏评分数据 | **仅适用于自动评价器元验证**（不能直接评价我们新生成的 A0/A1/A7 字幕） | FactVC（ActivityNet-Fact / YouCook2-Fact）、MHaluBench、M-HalDetect（图像为主） |

---

## 三、全球主要候选数据集全量盘点与初筛裁决（22 个代表性基准）

| # | 数据集 / 基准 | 发表年份与出处 | 功能类型 | 规模与体量（实测/论文） | 公开可获得性 | 拟承担角色 | 初筛裁决与核心淘汰/保留理由 |
|---:|---|---|---|---|---|---|---|
| 1 | **CaReBench** | arXiv:2501.00513v2 (2025, NJU/Shanghai AI Lab) | A (+B 缓存) | 1,000 clips（913 独立 FineAction 父视频），均值 14.35s，227.95 词；`3.66 GiB` `[公开元数据实测]` | **`gated=False`, MIT**，HF 直下 | **E2 强首选候选** | 双人独立+专家复核时空分离人工描述，静态对象+动态动作覆盖最密，94% 含单句总览；但来源集中于 FineAction 且 `events`/`objects` 为 GPT-4 提取。 |
| 2 | **DREAM-1K** | arXiv:2407.00634 / 2501.07888 (2024–2025, ByteDance) | A | 1,000 clips（5 领域），均值 8.9s，59.3 词；`3.59 GiB` `[论文原文/HF实测]` | `gated: auto`，metadata `apache-2.0` / 视频限学术 | **E2 强并列候选** | 5 领域开放多样性强、8.9s/59词与旧模型短字幕粒度最贴合、工程已就绪；但静态对象偏少（高 `Unmentioned` 风险）、ID 匿名、需 HF gated 登录。 |
| 3 | **VidHalluc** | CVPR 2025 (`chaoyuli/VidHalluc`) | C (+D) | ACH 3,897 独立片段（1,586 独立 11 位 YouTube ID）；ACH `15.60 GiB` `[公开元数据实测]` | **`gated=False`**, `cc-by-nc-nd-4.0` | **E3 首选候选** | 唯一兼具大规模自然动作视觉相似干扰项、真假长度高度平衡（4.23 vs 4.25 词）与 100% 精确 YouTube ID 去重的 E3 数据集；局限为仅限动作维且体量偏大。 |
| 4 | **VidHal** | arXiv:2411.16771 / TMLR 2025 (`anon-axolotl/VidHal`) | C | 1,000 clips（5 维度：属性/对象/动作/顺序/方向），3 级渐进幻觉字幕；`5.08 GiB` `[公开元数据实测]` | **`gated=False`**, `cc-by-4.0` | **E3 强备选/多维补充** | 最小语义编辑极干净（组内词数差中位数 `0.0`），许可宽松且小巧；但由 GPT-4o 生成（仅 10% 人工验证）、含 26.1% 合成 CLEVRER 球体、无 YouTube ID。 |
| 5 | **ARGUS** | ICCV 2025 / arXiv:2506.07371 (`tomg-group-umd/argus`) | A | 500 clips（15–90s），均值 ~485 词（实测前 25 条）；`15.60 GiB` `[公开元数据实测]` | **`gated=False`**, `cc-by-nc-sa-4.0` | E2 备选（次优） | 仅 500 视频、时长偏长（15–90s 不利于 16 帧 Video-LLaMA）、485 词文学描述与短字幕粒度悬殊、`ArgusCost-H` 将 `underdetermined` 合并计错。 |
| 6 | **VCapsBench** | AAAI / arXiv:2505.23484 (`somos99/VCapsBench`) | D | 5,677 clips（6–15s 为主），109,796 Yes/No QA，21 维；`22.20 GiB` `[公开元数据实测]` | **`gated=False`**, `odbl` | E2 QA 备选（次优） | 面向 T2V（含大量模糊/调色/运镜维），QA 由 VLM 合成，封闭 QA 无法捕捉检索注入的列表外幻觉且短字幕大部分呈 `Unanswerable`。 |
| 7 | **VidCapBench** | Findings of ACL 2025 (`VidCapBench/VidCapBench`) | D | 643 clips，9,494 AE QA + HE QA；`3.59 GiB` `[公开元数据实测]` | **`gated=False`**（无标准开源许可，严格禁分发） | 淘汰（不作主集） | 面向 T2V（28% 为美学/摄影维），73 条视频直接名为 `Tarsier_*`，QA 为 GPT-4o 初生成+人工修订，封闭 QA 同样无法检测列表外检索幻觉。 |
| 8 | **VideoHallucer** | arXiv:2406.16338 (`bigai-nlco/VideoHallucer`) | D (+C) | 907 独立视频（150 条 YouTube 样式 ID），1,800 对抗二值 QA；`~14 GiB` `[公开元数据实测]` | **`gated=False`**, `mit` | E3 备选（偏弱） | `external_factual` 偏教学步骤/外部知识，非纯视觉短动作；`object_relation` 仅 183 视频；二值 QA 难直接构造长度平衡的 Sup/Con 叙述文本。 |
| 9 | **TempCompass** | ACL 2024 (`lmms-lab/TempCompass`) | C/D | 410 视频，500 冲突对（动作/方向/速度/顺序/属性变化）；`0.39 GiB` `[公开元数据实测]` | **`gated=False`** | E3 轻量备选 | 极轻量（0.39 GB），但视频总量仅 410 条且偏向速度/方向/反转等强时序变化，16 帧旧模型对速度与细微方向辨别力偏弱，且缺乏 YouTube ID 去重信息。 |
| 10 | **CapQuiz** | ACL 2026 (Apple) | D | 1,204 视频，19.6 人工核验 MCQ/视频 `[论文原文]` | **未释放（不可获取）** | **硬准入淘汰** | 理论契合度高，但官方页面与 ACL Anthology 均无任何公开数据或代码下载链接。 |
| 11 | **VDC (AuroraCap)** | ICLR 2025 (`wchai/Video-Detailed-Caption`) | A/D | 1,000 clips，均值 500.91 词；`74.61 GiB` `[公开元数据实测]` | `gated=False`, `apache-2.0` | **硬准入淘汰** | 500 词高度 GPT-4o 合成化，实测 `VDCScore_qa/short.jsonl` 存在未清洗类别占位词污染，无事实真值公信力。 |
| 12 | **EventHallusion** | arXiv:2409.16597 (`Stevetich/EventHallusion`) | D | 400 视频，711 问答 `[论文/检索]` | 部分开源（GitHub README 404/散落链接） | 淘汰 | 样本量偏小（400 视频），侧重语言先验诱导的二值/开放问答而非检索文本干预。 |
| 13 | **VideoCon** | ICLR 2024 (`Hritikbansal/videocon`) | C | 基于 ActivityNet/MSR-VTT/SSv2 等的 LLM 对比字幕 `[论文原文]` | HF 数据集返回 `401 Unauthorized` | 淘汰 | 原始对比字幕由 LLM 生成且上游大量重叠 MSR-VTT/ActivityNet，HF 托管视频不可直接公开获取。 |
| 14 | **VITATECS** | ECCV 2024 | C | 时序概念反事实对比描述 `[论文原文]` | HF 仓库返回 `401 Unauthorized` | 淘汰 | 侧重时序概念对比检索而非开放字幕事实评价，公开视频打包获取链路不畅。 |
| 15 | **FactVC** | EMNLP 2023 (`NiteRainHL/FactVC`) | E | ActivityNet-Fact & YouCook2-Fact 历史模型输出人工评分 `[官方源码实查]` | GitHub 公开标注 | **仅可作评价器元校验参考** | 仅有旧模型（COOT/MART/PDVC）字幕评分，无面向新模型输出的可复用事实表。 |
| 16 | **VIOLIN** | CVPR 2020 | C | 15,887 影视片段 + 人工真假陈述 `[论文原文]` | 视频受影视版权/音频字幕依赖限制 | 淘汰 | 强依赖影视对白/字幕与剧情常识，与静音 16 帧视觉字幕任务错配。 |
| 17 | **Shot2Story** | ICLR / arXiv:2312.06183 (`bytedance/Shot2Story`) | A | 20K 多镜头视频 + 镜头级字幕与总结 `[论文原文]` | 部分需 YouTube 自行爬取 | 淘汰 | 主打多镜头切分与音频旁白融合，不符合 16 帧静音单片段场景。 |
| 18 | **Action Genome** | CVPR 2020 | B | Charades 时空场景图（对象+关系） `[论文原文]` | 公开标注+Charades 视频 | 淘汰 | 仅限室内居家 Charades 动作且为离散关系三元组（缺自然语言描述），与开放检索描述映射难度大。 |
| 19 | **FineVideo** | HuggingFaceFV (2024) | A/D | 43K 长 YouTube 视频（~3,400 小时），GPT-4o 标注 `[公开元数据实测]` | `gated=False` | 淘汰 | 长视频 + 大模型自动标注（非人工 Gold 评测集）。 |
| 20 | **COVER / ACQUIRED / EG-VQA** | ACL/CVPR 2024–2026 | D | 反事实因果/社会推理/带时间戳问答 `[检索]` | 部分开源或未完整释放 | 淘汰 | 考查“如果……将会怎样”的高阶因果或长视频定位，而非视觉可见事实描述。 |
| 21 | **MVBench / Video-MME** | CVPR 2024 / 2025 | D | 综合多选 VideoQA（含合成数据与长视频） `[公开元数据实测]` | `gated=False` | 淘汰 | 通用多选 QA，非字幕事实性基准（其中 MVBench 子集已被 VidHal 吸收重构）。 |
| 22 | **PE-Video** | Meta (2025) | A（预训练语料） | 百万级 WebDataset `.tar`（2,000 分片） `[公开元数据实测]` | `cc-by-nc-4.0` | 淘汰 | 预训练语料而非精标评测基准。 |

---

## 四、检索增强视频字幕事实性与创新性查重（Section XV）

针对“是否已有研究完成与 UARCap 高度重叠的实验”，本轮专项检索了 2024–2026 年检索增强视频描述与冲突证据文献：

1. **CaReBench（Xu et al., arXiv:2501.00513v2）与 UARCap 的本质区别**：
   - CaReBench 研究的是两个**独立任务**：(a) **视频-文本检索**（Video-to-Text / Text-to-Video Retrieval，用 `ReBias` 衡量时空检索偏置）；(b) **普通无检索视频字幕生成**（用 `CapST` 衡量 MLLM 的对象与事件覆盖），并训练统一两阶段 SFT 模型 CaRe-7B。
   - **CaReBench 完全没有研究**：检索增强视频字幕（RAVC）、外部检索证据预算（$k=0 \to 1 \to 7$）的非单调事实变化、错误检索文本的受控定向干预（Base/Sup/Con/Neu），以及参考不完整导致的 `Unmentioned` 评价偏置。**因此复用 CaReBench 作为评测数据与原论文零贡献重叠。**
2. **近期视频 RAG 与冲突证据研究（EgoInstructor CVPR 2024、CM² / HiCM²、MAD-RAG 2026、Video-DPL、EAEV）**：
   - EgoInstructor 仅研究跨视角（第三人称→第一人称）检索增强，未拆分正确事实/错误事实/遗漏三态，也未做同视频视觉锁定下的冲突文本干预；
   - MAD-RAG 与文本 RAG 冲突研究聚焦问答（QA）而非开放式视频字幕生成；
   - **结论**：目前全球文献中**尚未出现**在固定视觉输入下联合诊断“自然检索预算（A0/A1/A7）引起的正确事实 vs 错误事实 vs 核心覆盖变化”与“受控最小冲突干预（Base/Sup/Con/Neu）行为因果”的跨代际视频字幕诊断研究。UARCap Q2 的科学创新边界依然清晰稳固。
