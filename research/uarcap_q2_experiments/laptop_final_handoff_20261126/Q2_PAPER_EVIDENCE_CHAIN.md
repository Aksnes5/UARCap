# Q2 论文证据链

| 科研问题 | 已有结果与质量层级 | 可接受表述 | 核验或限制 |
| --- | --- | --- | --- |
| 外库规模与可获得性 | 原冻结百万级六尺度研究。选中相似度与候选更替已有证据。 | 在固定继承系统中，检索访问改善未带来稳定的固定预算生成增益。 | 集合组成与规模共变，使用原冻结研究和原评估口径。 |
| 自然检索预算的质量响应 | 原论文两基准与 CaReBench 两系统改变了同一查询的预算。CaReBench R-Short Video-LLaMA CIDEr points 为 31.03/35.34/36.64，Qwen 为 33.93/34.23/34.19。对真实 3,600 条重新执行了有正确键顺序的评估。 | 外部证据响应随系统和测试基准变化。 | R-Full 是平均 228.09 词的长参考，R-Short 为平均 14.53 词人写片段。不可混合单参考与多参考绝对分值。A3/A5 仍待执行。 |
| 传统指标与事实端点如何共同变化 | 冻结 3,600 条机器事实标注，逐查询指标键修复后，Video-LLaMA R-Short 中 A7−A0 的 CIDEr 差与支持事实数差 Spearman .2149，与核心覆盖差 .2978。Qwen 对应 −.0530、.1954。 | 同一预算干预下可观察到随指标与系统变化的关联。 | 旧逐视频桥接存在结果错位，弃用旧配对关联。Gemma/Llama 是两个自动评价器，均缺人类 Gold。 |
| 效用来自何种自然外部证据 | 已有逐查询 Top-20 与实际文本和视觉记录。语义配对与溯源执行入口已就绪，真实判定 PENDING。 | 可定位哪些变化与真实候选语义一致、冲突或未知。 | 关联不能识别内部因果来源。当前图文键不可直接提供人类语义标签。 |
| 人为冲突能否诱导错误或抑制正确事实 | Pilot 20 候选、双模型的描述性证据。旧正式 E3 有 159 个固定 fallback Neu，整个旧四臂对照失效。 | 修复后可报告固定视觉的单命题干预与效应大小。 | V2 需要真实相关性、盲法 16/64 帧、来源检查。候选和三项后续评价均 PENDING。 |
| 自然冲突与干预如何连接 | CaReBench Bridge 60 条待核验候选，冻结门槛 30。 | 如门槛满足，可测试真实候选冲突文本对同查询字幕事实的作用。 | 零外部视觉的文本注入设计。允许无充分候选、净效应不显著或与 VidHalluc 方向不同。 |
| 文本、视觉与长度 | 旧系统同时接收外库视觉和文本，Qwen 主实验仅注入外部文字。V2 已补齐 Qwen 统一目标长度（13–15 词 IQR，120×3）、Qwen 外部帧模态 2×2 因子（40×4，16 查询帧 + 8 Top-1 外部帧）与 16/32/64 帧 × A0/A7 交互（150 A7 + 嵌套 60 A0）执行与分析链路。 | 报告完整模型系统差分，并在同系统内拆解长度、外部视觉与采样密度对预算响应的影响（后设探索性）。 | Qwen 外部帧与 32/64 帧结果均待 BN 实机验收；64 帧仍是采样代理而非全视频真值，且 64 帧失败与 16/32 帧隔离。 |
| 成本与推荐预算 | 已有两模型运行时回执及本轮 0/1/7 成本与质量汇总。 | 在同系统、同环境下展示实测时间与事实端点的 Pareto 关系。 | 缺少 A3、A5 真实输出和独立缓存、载模及特征成本，不主张普遍最优 k。 |

主要端点维持每字幕参考冲突事实数与核心覆盖。实际完成的人类审核数量为 0。探索性新增端点、方差方法和报告都不能改变确认性门禁。[修复后的配对表](metric_bridge_repaired/E2_METRIC_BRIDGE_PAIRED.json) · [真实预算汇总](current_budget/Q2_BUDGET_REPORT.json)

## V2 report mapping

Use the versioned final matrix to map accepted module results to paper sections. The archived v0.8 history and real metric-join receipt are separate entries. The Qwen length, retrieved-frame modalities and dense-frame budget interactions are post-hoc robustness modules with their own inputs. Natural provenance, rank relevance and historical reverse bridge are descriptive, with independent second-judge scope marked in their receipts. Join each E3 or Bridge result only to its own candidate freeze and completed model/evaluator panel. If any is unavailable, preserve the denominator and narrow the text. [Manuscript scaffold](FINAL_PAPER_MANUSCRIPT_SCAFFOLD.md)
