# POST_FIX 全量实验最终科研报告与论文科学性终审

日期：2026-11-24  
分支：`fix/val497-frame-loader-postfix-rerun-20261123`  
基线 commit（本报告前）：`92a605588f7da70f627e53caed853bdbef95734d`

---

## 0. 结论摘要

1. val497 的视频帧读取存在已证实的 bug：评测分支把 `start,end` 设为 `0,0`，
   使 PyAV 只保留 1–2 帧，16 帧采样退化为“frame0×15 + frame1×1”。修复后
   val497 绝对 CIDEr 整体上移 **+17~19 点**，说明旧 val 数值不可用于论文。
2. 修复后 **大上下文预算仍一致次优**：12 个 scale×track 设置中 k=7 **0 次最优**，
   k=1 在 11/12 中最好；k7 penalty 均值 D 2.86 / CF 2.56。
3. **Oracle 诊断仍是核心证据**：Oracle3−D3 = +3.04（p=0.0096），
   且 Oracle7−Oracle3 = −2.78（p=0.0076）——在 GT 辅助选择下，7 条文本证据
   仍显著差于 3 条，说明“选择瓶颈”和“多文本整合瓶颈”同时存在。
4. 修复显著削弱了若干旧结论：Text main −3.24→−1.40（ns）；external
   D3−D7 +3.11→+1.44（ns）；GoodTail/HardWrong 旧显著效应未复现。
5. Adaptive selector 是在**有 bug 的 val497** 上确定并冻结的；POST_FIX 显示
   val 最优预算为 k=1，因此 adaptive 的选择依据失效，不能宣称其优于最佳固定 k。
6. 与已发表论文的关系：其 in-domain 设定下 k̃=7 最优（论文 Table 4），
   POST_FIX in-domain 同样 D7=73.76 最高，而 external 在 k1 最优——这一
   **in-domain vs external 的反向预算响应**是本研究最稳健、最有论文价值的方向性发现。
7. 当前证据不足以支撑“文本证据数量显著有害”“外部过载显著”这类旧强主张；
   可以作为方向性/描述性证据，但不能作为已证实的因果机制。
8. k=0 matched 无检索基线缺失，标记 `BLOCKED_REQUIRES_GENERATION`。

---

## 1. 资产盘点与真实性审计

### 1.1 审计范围
- `reports/post_fix/` 全部报告、JSON/CSV、manifests、frame sanity、LOO；
- 48 个主网格 + 4 in-domain + 2 Text×Visual + 3 Oracle + 2 position +
  5 tail/distractor + 7 LOO = **71 个正式配置**，另有 1 个 val20 smoke；
- 逐项检查 generated captions 行数、唯一 ID、非空预测、metrics 存在性、
  ID 是否与 val497/val150 清单一致。

### 1.2 审计结果
```
expected configs: 71
verified complete: 71
bad: []
```
- 主网格 48/48：497 行、ID 完全对齐；
- in-domain 4/4、T7V3/T3V7 2/2、Oracle 3/3、position 2/2、
  tail/distractor 5/5、LOO 7/7（每组 150 行、val150 ID 对齐）；
- `POST_FIX_SCALE_BUDGET_RESULTS.json` 中 FULL CF1=71.17、500K D1=70.59、
  10K CF1=70.43 与原始 metrics 文件一致；
- 72 个 eval 目录中多出的 1 个是 `smoke_val20`，标记 EXPLORATORY_ONLY。

### 1.3 证据链核对方式
对每个核心数字：原始 `*_generated_captions.json` → `cocoeval.py` → `metrics.json`
→ 汇总脚本（`postfix_priority1_stats.py`、`postfix_mechanism_stats.py`）→ 报告。
本报告不依赖聊天记录；PRE_FIX 数字仅作为历史对照并显式标注。

---

## 2. 帧读取 bug 的科学审计

### 2.1 位置与机制
`evcap_external/dataset/msrvtt/vision_language_tsv.py:338`：
```python
start, end = 0, 0   # PRE_FIX
```
- `get_visual_data` 对视频路径行走 `decode_and_get_frames(..., start, end)`；
- `pyav_decode` 在 `start=end=0` 时设置 `video_start_pts=video_end_pts=0`，
  `pyav_decode_stream` 只保留 pts<=0 的首帧与一个缓冲帧；
- `temporal_sampling` 对 2 帧池执行 `linspace(0,1,16)` → 15×frame0 + 1×frame1。

### 2.2 影响范围
- 受影响：所有使用 `val497/val16_video.yaml`、`val150/val150_video.yaml`
  （raw video path）的生成，即全部 PRE_FIX val497/val150 结果。
- 不受影响：test2690/test2990、MSVD670、dev300，它们使用预提取 frame TSV，
  `get_frames_from_tsv` 忽略 start/end。

### 2.3 修复与验证
改为 `start, end = None, None`（整段视频）。23 个 val497 视频 sanity：
**16/16 唯一帧**、与 frame0 相同仅 1 帧、静态 gate 0 失败；video6518
（26.03 s）采样 0→26.03 s，montage 中 frame0 为银行 logo、后续为摔跤画面。
报告：`POST_FIX_FRAME_SANITY_REPORT.md`。

### 2.4 对结论的影响
- 绝对分整体上移 +17~19；val 与 test2690 恢复到同一量级；
- 不能因为“组间共用同一错误输入”就继承旧对照结论：错误输入改变了模型对
  外部文本证据的依赖方式，修复后多个组间效应量显著缩小。
- 所有 PRE_FIX val 结果永久标记 `PRE_FIX_FRAME_BUG`。

---

## 3. 六尺度 × 两排序 × 四预算主实验

### 3.1 D track（CIDEr points）
| scale | k1 | k3 | k5 | k7 | best | k7 penalty |
|---|---|---|---|---|---|---|
| 10K | 69.21 | 67.80 | 67.67 | 67.33 | 1 | 1.88 |
| 50K | 68.89 | 67.84 | 66.80 | 66.12 | 1 | 2.77 |
| 100K | 69.61 | 68.68 | 66.69 | 66.15 | 1 | 3.46 |
| 250K | 70.29 | 69.29 | 68.75 | 66.99 | 1 | 3.30 |
| 500K | 70.59 | 70.08 | 68.25 | 67.62 | 1 | 2.97 |
| FULL | 69.97 | 68.63 | 67.31 | 67.22 | 1 | 2.75 |

### 3.2 CF track
| scale | k1 | k3 | k5 | k7 | best | k7 penalty |
|---|---|---|---|---|---|---|
| 10K | 70.43 | 70.13 | 69.30 | 68.73 | 1 | 1.70 |
| 50K | 69.98 | 68.47 | 68.08 | 67.09 | 1 | 2.89 |
| 100K | 69.89 | 70.10 | 69.32 | 67.57 | 3 | 2.53 |
| 250K | 70.50 | 69.09 | 67.80 | 68.77 | 1 | 1.73 |
| 500K | 70.50 | 68.87 | 68.89 | 67.23 | 1 | 3.27 |
| FULL | 71.17 | 68.88 | 68.84 | 67.95 | 1 | 3.22 |

### 3.3 解读
- k=7 在 12 个设置中 **0 次最优**；k=1 11/12；k=5 0/12。
- k7 penalty 稳定存在（1.70–3.46），说明大预算次优是稳健的描述性现象。
- 规模与 k1 生成不单调：CF1 69.89–71.17；D1 68.89–70.59。
- CF 与 D 互有胜负，差异随 scale/k 改变，没有“某一排序全面胜出”的证据。
- 这只能支持“大预算普遍不优”，不能外推为“k1 对所有模型/任务普遍最优”。

### 3.4 候选 churn 与相似度-效用解耦（CPU 分析）
10K→50K/100K：top1 被替换 80.7%/88.7%；替换者 X-CLIP 相似度几乎总是更高
（400/401、440/441）；但在被替换 query 上 CF1 CIDEr 变化 **−0.56/−0.61**，
D1 为 +0.26/+0.86。即 raw retrieval 上“相似度升、生成降”的直接证据存在，
但 D 轨道不成立。10K vs 50K/100K 的 k1 配对差均不显著。

---

## 4. In-domain 对照与交互

| k | External FULL D | In-domain D |
|---|---|---|
| 1 | 69.97 | 71.93 |
| 3 | 68.63 | 72.42 |
| 5 | 67.31 | 71.07 |
| 7 | 67.22 | **73.76** |

- External D3−D7 = **+1.44**（CI [−1.20,+4.13]，p=0.29）
- In-domain D3−D7 = **−1.33**（CI [−3.57,+0.85]，p=0.24）
- DiD = **+2.77**（CI [−0.61,+6.39]，p=0.13）

方向与论文的 in-domain k̃=7 最优一致（论文 Table 4），而 external 呈相反
方向；但 POST_FIX 下 external 的 k7 损失与 DiD 均不显著，只能作为方向性证据。
二者仍存在语料规模（6,513 vs 1.09M）、来源、风格与训练匹配的混杂。

---

## 5. Text × Visual 2×2

T3V3 = 68.63；T7V3 = 67.18；T3V7 = 68.51；T7V7 = 67.22。

| effect | Δ | CI | p |
|---|---|---|---|
| Text 3→7 @V3 | −1.48 | [−4.12,+1.13] | 0.28 |
| Text 3→7 @V7 | −1.32 | [−4.01,+1.28] | 0.33 |
| Visual 3→7 @T3 | −0.12 | [−0.78,+0.53] | 0.74 |
| Visual 3→7 @T7 | +0.04 | [−0.36,+0.49] | 0.85 |
| Text main | −1.40 | [−4.05,+1.25] | 0.31 |
| Visual main | −0.04 | [−0.43,+0.36] | 0.86 |
| Interaction | +0.16 | [−0.59,+0.94] | 0.69 |

方向保留（text 负、visual 零），但 POST_FIX 下 text main 不显著；
不能把不显著解释为“零作用”，也不能再宣称“文本增加显著有害”。

---

## 6. Oracle 诊断

| condition | CIDEr | vs D | CI | p |
|---|---|---|---|---|
| D1 / Oracle1 | 69.97 / 71.64 | +1.67 | [−0.55,+3.85] | 0.13 |
| D3 / Oracle3 | 68.63 / 71.67 | **+3.04** | [+0.74,+5.37] | **0.0096** |
| D7 / Oracle7 | 67.22 / 68.94 | +1.70 | [−0.73,+4.13] | 0.17 |
| Oracle7 − Oracle3 | — | **−2.78** | [−4.92,−0.77] | **0.0076** |

解释边界：
- Oracle 使用 query 的真实参考 caption 计算候选兼容度，**不是可部署系统**；
  它证明“top20 内存在被 ranking 浪费的更好证据”，不能证明存在可学习的排序器。
- Oracle3 显著高于 D3 支持 selection headroom；Oracle7 显著低于 Oracle3 支持
  “即使证据由 GT 选出，7 条文本仍存在整合过载”。
- Oracle7−Oracle3 不等于已经定位到某个神经网络内部故障；它是行为层面的证据。

---

## 7. Position、Tail、Distractor

| 条件 | Δ vs D7 | CI | p |
|---|---|---|---|
| reverse | −0.24 | [−2.08,+1.71] | 0.82 |
| top3-last | +0.02 | [−1.79,+2.02] | 0.99 |
| MidTail7 | +1.23 | [−0.77,+3.45] | 0.25 |
| LowTail7 | −0.61 | [−2.67,+1.39] | 0.55 |
| GoodTail7 | −0.56 | [−3.30,+2.06] | 0.69 |
| HardWrong7 | −1.42 | [−3.44,+0.60] | 0.18 |
| EasyWrong7 | −0.37 | [−2.36,+1.71] | 0.72 |
| Hard − Easy | −1.05 | [−3.36,+1.27] | 0.38 |

- 位置/顺序解释：**已排除**（≈0）。
- Tail 质量剂量：**不支持**“低排名尾部证据特别有害”。MidTail 甚至略升，
  LowTail 略降，均不显著。
- Distractor：Hard 方向为负、Easy 接近 0，但 hard−easy 不显著；
  只能保留为探索性方向，不能写成“plausible distractor 确定更有害”。
- PRE_FIX 的 GoodTail +2.64、HardWrong −2.02 在修复后均未复现。

---

## 8. LOO Stage 1

Baseline = POST_FIX FULL D7 在同一 150 个 query 上的预测。

| 删除 rank | mean effect | median | P(harmful) |
|---|---|---|---|
| 1 | +2.00 | 0.00 | 0.200 |
| 2 | −0.49 | 0.00 | 0.167 |
| 3 | +0.29 | 0.00 | 0.200 |
| 4 | +0.69 | 0.00 | 0.167 |
| 5 | −0.78 | 0.00 | 0.147 |
| 6 | −0.13 | 0.00 | 0.167 |
| 7 | −1.42 | 0.00 | 0.167 |

Head (1–3) +0.60，P(harmful) 0.189；Tail (4–7) −0.41，P(harmful) 0.162；
各 rank 中位数 0。结论：**不支持固定 rank/尾部存在特别有害的证据**；
rank7 删除反而略微降低质量，说明它通常是正向证据。样本仅 150、功效有限。

---

## 9. 跨基准结果（不受 val497 bug 影响）

MSR-VTT test2690：B 73.09；D1 71.21；D3 70.08；D5 69.48；D7 68.27；
Adaptive 70.67。
MSVD670：in-domain 158.32；D1 156.39；D3 151.45；D5 144.39；D7 141.59；
Adaptive 151.94。

- MSR-VTT 论文工作预算 k=7，但 test2690 上外部 D1>D3>D5>D7；
- MSVD 论文预算 k=3，外部 D1>D3>D5>D7；k7 应视为压力测试。
- 两套 test 输入未受帧 bug 影响，统计结论仍有效；但其 D/Adaptive 的预算选择
  参照来自 PRE_FIX val，这一方法学问题需在论文中说明（见 §11）。

---

## 10. 生成文本行为

已有长度/复制率审计（`reports/cpu_audit/STYLE_LENGTH_COPYING_AUDIT.md`）显示：
k 增大时输出长度与对外部 caption 的词组复制倾向变化，但复制率与 CIDEr 下降
只是相关性，不能写成因果。SPICE 与 CIDEr 在部分条件下不同向，说明 n-gram
与语义指标侧重点不同；不能用 SPICE 单独证明“语义更正确”。

---

## 11. Adaptive-k 方法学复审

- `frozen_method_config.json` 的 selector 规则本身无拟合参数，但其**部署决策、
  k 映射与相对 D3 的收益判断**是在 PRE_FIX val497 上做出的。
- POST_FIX val497 显示最佳固定预算为 k=1，而 frozen 对照/测试使用 D3 作为
  参照；因此 adaptive 的选择依据与“D3 是合理参照”的前提均被削弱。
- 现有证据**不能**支持 adaptive 优于最佳固定 k；test2690 上 Adaptive−D3 仅
  +0.58（p=0.046），且低于 D1；MSVD 上与 D3 持平。
- 结论：adaptive 只能作为探索性方法学案例，不得作为论文主方法贡献。

---

## 12. k=0 缺失

当前没有与 POST_FIX 完全匹配的 k=0（无检索）生成。
旧 dev300 64.18 与论文 70.8 均不能替代。
状态：`BLOCKED_REQUIRES_GENERATION`。
影响：本研究可以比较不同外部检索预算，但**不能证明外部检索优于不使用检索**。

---

## 13. 科学性终审

### A. 科学问题
问题是清晰的：大规模外部检索下，增加检索证据为何不提升（甚至损害）冻结生成器。
但 POST_FIX 后，核心效应显著性与幅度都明显减弱，问题的“危害”强度需要重新表述为
“预算敏感性/次优性”，而不是强“过载伤害”。

### B. 创新性
- 前作已提出 X-CLIP 检索 + quality rerank + cross-gating fusion，并已做过
  in-domain k̃ 消融（k̃=7 最优）。
- 本研究新增的是：百万级外部语料、六尺度、跨排序 CF/D、test2690/MSVD
  外部验证、Oracle/2×2/position/tail/LOO 诊断，以及 in-domain vs external
  的预算响应反转。
- 风险：若只表述为“k 小一点更好”，会被视为重复前作超参数消融。必须把
  **in-domain k7 最优 vs external k1 最优的反转**和 **Oracle 下 k7<k3** 作为
  核心新现象，并明确其边界。

### C. 证据闭环
- 主实验：大预算描述性次优（强）。
- In-domain：方向反转（中；不显著）。
- Oracle：selection headroom + integration overload（强，k3/k7 显著）。
- 2×2：text 方向负、visual 零（弱，均不显著）。
- Position：排除（强，null）。
- Tail/LOO：不支持单条毒证据（中）。
- 跨基准：外部大规模下 k7 压力测试下降（强，但非本 bug 影响）。
整体闭环只能支持“预算选择与整合瓶颈”的行为学结论，不能支持精细内部机制。

### D. 替代解释
- Prompt 排列：已排除。
- 视觉通道：基本排除独立负效应（不显著）。
- 复制倾向/长度：部分排除（相关但未因果）。
- 检索 caption 机器生成质量：尚未排除（VideoUFO brief captions 为自动生成）。
- 训练/测试预算不匹配：尚未排除（MSR-VTT 训练 k7，外部却 k1 最优）。
- 排序器跨语料分布不适配：尚未排除（D 实现与论文 pseudo-label 定义不同）。
- 指标对措辞敏感：尚未排除（CIDEr/SPICE 不同向）。

### E. SCI 竞争力
- **Q4：GO**。问题清晰、实验量充足、可复现。
- **Q3：CONDITIONAL GO**。需要：固定帧 bug 的透明说明；把主张降级为
  方向性；补 k=0；解决 adaptive 失效；明确 D 实现偏差。
- **Q2：暂不达**。缺少：多个独立验证集/更大 n、与论文 pseudo-label reranker
  一致的实现、可部署的 selection 改进（非 oracle）、以及显著的核心效应。

---

## 14. PRE_FIX → POST_FIX 主张变化清单

| 主张 | PRE_FIX | POST_FIX | 方向 | 处理 |
|---|---|---|---|---|
| Text main | −3.24, p=.007 | −1.40, p=.31 | 保留 | 降级为方向性 |
| External D3−D7 | +3.11, p=.012 | +1.44, p=.29 | 保留 | 降级 |
| In-domain interaction | 显著（旧分析） | DiD +2.77, p=.13 | 保留 | 探索性 |
| GoodTail | +2.64, p=.006 | −0.56, p=.69 | **反转** | 撤回 |
| HardWrong | −2.02, p=.022 | −1.42, p=.18 | 保留 | 降级 |
| Oracle1 | +3.27, p=.0008 | +1.67, p=.13 | 保留 | 降级 |
| Oracle3 | +2.05, p=.053 | +3.04, p=.0096 | 保留/增强 | 保留 |
| Oracle7−Oracle3 | −2.03, p=.033 | −2.78, p=.0076 | 保留/增强 | 保留 |
| 10K 最优 k=5 | 是 | 否（k1） | **反转** | 撤回 |
| k7 从未最优 | 是 | 是 | 保留 | 保留 |

---

## 15. 论文可写与不可写

### 可以写
- 百万级外部检索下，k=7 在 12 个设置中从未最优，k7 penalty 约 2.6–2.9 CIDEr。
- 规模提升不自动转化为下游收益；候选 churn 大而生成变化小。
- 在 GT 辅助选择下，k=3 显著优于 D3，且 k=7 显著差于 k=3（行为学层面的
  selection + integration 双瓶颈）。
- In-domain 与 external 的预算响应方向相反（in-domain k7 最高，external k1）。
- 位置/顺序不是解释；单一毒尾证据不被支持。

### 不能写
- “文本证据增加显著有害”“外部过载显著”作为已证实结论。
- “GoodTail 改善、HardWrong 恶化”作为稳定机制。
- adaptive 优于最佳 fixed budget。
- k=0 结论、外部检索优于无检索。
- 与已发表 pseudo-label reranker 完全等价（实现不同）。
- 任何基于 PRE_FIX val 数值的显著性主张。

---

## 16. 可复现性与数据来源

- 原始预测：`evcap_external/outputs/eval_postfix_*/...generated_captions.json`
- 指标：同目录 `metrics.json`；汇总：`reports/post_fix/POST_FIX_*`
- 统计脚本：`scripts/postfix_priority1_stats.py`、`postfix_mechanism_stats.py`、
  `postfix_frame_sanity.py`、`loo_analysis.py`、`nested_scale_survival.py`
- manifests：`reports/post_fix/manifests/*.json`（71 个）
- 帧 sanity：`reports/post_fix/val497_frame_sanity.{json,csv}`、`frames/*.png`
- 修复代码：`evcap_external/dataset/msrvtt/vision_language_tsv.py:338`
  （commit `2b3da9d`）
- 历史：`reports/cpu_audit/VAL_FRAME_INPUT_BUG_CODE_TRACE.md`、`PRE_FIX_*`

所有 POST_FIX 数字均可由上述原始预测经 `cocoeval.py` 重算复现。
