# 科学协议到代码与测试逐条映射矩阵（SCIENTIFIC PROTOCOL IMPLEMENTATION MATRIX）

基准科学协议：`761dea2c51ab07c083b88805168ad1757db53d67`（`UARCap_Q2_PILOT_A_PROTOCOL.md`、`UARCap_Q2_EVALUATOR_VALIDATION_PROTOCOL.md`、`UARCap_Q2_PILOT_RESOURCE_AND_STOP_RULES.md`、`UARCap_Q2_v1.2_FINAL_SCIENTIFIC_REVIEW.md`、`PROPOSED_EXPERIMENT_MATRIX_V12.md`）

| 科学协议要求 | 对应实现代码 | 对应测试 | 当前验证状态 | 需 BN 真实测试 | 需人工核验 | 需试点后科学裁决 |
|---|---|---|---|---|---|---|
| **P0-1：参考代理与视频真值非等价边界 + 3×4 交叉矩阵 + `A7` 差分偏置门控** | `evaluators/reference_fact_eval.py`, `evaluators/calibration.py` | `test_04` | CPU 单元测试通过（含 `A7` 漏检偏置触发 `FAIL` 与零错误触发 `INCONCLUSIVE`） | 是（A2 真实裁判） | 是（A2 30 条 + `E2-Val` 50–80 条） | 是（判定代理 PASS/FAIL/回退） |
| **P0-2：三层评价口径分离（Layer 3 绝不冒充修正版 AutoDQ）** | `evaluators/reference_fact_eval.py` (`PROTOCOL_LAYER`), `configs/pilot_a23_config.yaml` | `test_04`, `test_09` | CPU 验证通过 | 否（命名契约已锁） | 否 | 否（可选付费 Layer 1 敏感性） |
| **P0-3：DREAM-1K 与 VidHalluc 同队列衔接（`E3-Bridge` $\ge 25$ 干净对门槛或收缩主张）** | `interventions/e3_bridge.py` | `test_03` | CPU 验证通过（`<25` 自动禁用并输出主张收缩指令） | 是（若启用） | 是（筛选自然冲突） | 是（视自然冲突单靶率决定启停） |
| **P0-4：VidHalluc E3 双独立二元终点 $(Y^+, Y^-)$ 允许共存 + 否定/对冲/泛化 + 双盲单目标语义裁判** | `common/schemas.py`, `interventions/vidhalluc_e3.py`, `evaluators/e3_semantic_eval.py` | `test_03`, `test_04` | CPU 验证通过（含 `(1,1)`、否定、泛化、防循环） | 是（A3 真实裁判） | 是（A3 160 条全量对账） | 是（核对 $\kappa \ge 0.80$ 与偏置 $\le 0.03$） |
| **P0-4：VidHalluc 5 项素材适用性审核与 `CANDIDATE_PENDING_VERIFICATION` 硬锁** | `datasets/vidhalluc.py`, `interventions/vidhalluc_e3.py` | `test_01`, `test_03` | CPU 验证通过（未审批候选不可入池） | 是（A3 真实视频） | 是（A3 30 候选 + 正式 150–180） | 是（通过率 $r_{\text{pass}} \ge 50\%$ 门控） |
| **P0-5：DREAM-1K 无源 ID 诚实声明 + 两阶段近重复保守排除** | `datasets/dream1k.py`, `retrieval/dedup.py` | `test_01`, `test_02` | CPU 验证通过（严禁伪造 source ID、`exact_source_zero_overlap_claimed=False`） | 是（百万级 VideoUFO X-CLIP） | 是（人工复核高相似尾部） | 否（保守排除） |
| **P0-5：VidHalluc 精确 YouTube ID 零重叠 + 每源 1 片去重 + `CC-BY-NC-ND` 合规拦截** | `datasets/vidhalluc.py`, `common/gates.py`, `DATA_AND_MODEL_ASSET_MANIFEST.json` | `test_01`, `test_08` | CPU 验证通过（实测 `15.60 GiB` 已修正） | 是（对照 BN VideoUFO 源表） | 否 | 否 |
| **历史防回归：16 帧均匀采样 + 重复首帧/坏片/VFR/OOB 检测 + EVCap TSV 打包** | `datasets/video_decoder.py`, `generators/videollama_adapter.py` | `test_01`, `test_03`, `test_07` | CPU 真实 PyAV MP4 测试通过 | 是（A1 真实 DREAM-1K） | 否 | 否 |
| **多臂构造：`ARM_A0` 严格空证据、`A1⊂A3⊂A7` 前缀一致、E3 `BASE_A0`/`BASE_B3` 视觉锁** | `retrieval/arm_builder.py`, `generators/base.py` | `test_02`, `test_08` | CPU + 主动故障注入全部拦截通过 | 是（A1/A3 真实运行 trace） | 否 | 否 |
| **低成本本地网页盲法人工核验工具（Phase 1 核心事件锁定 + 四分类 + E3 双目标 + 续标 + 计时）** | `annotation/package_builder.py`, `annotation/web_app.py`, `annotation/adjudicator.py`, `scripts/launch_annotation_ui.py` | `test_05`, `test_07` | Windows 本地 HTTP 实测通过（盲法零泄密、自动保存、续标、CSV 导出、403 防护） | 可选（Ubuntu 同代码） | 是（A2/A3 真实双人标注） | 否 |
| **现代开源 VLM 独立环境适配（16 帧 + 8–18 词短字幕 + 离线 revision + 显存遥测）** | `generators/modern_vlm_adapter.py`, `configs/envs/env_modern_vlm.yaml` | `test_03`, `test_09` | CPU 配置/提示/trace 验证通过 | **是（A3 RTX A6000 timed smoke 必测）** | 是（A3 15 条跨模型抽核） | 是（锁定唯一现代 VLM） |
| **统计套件（E1 探索、视频/源聚类自助法、Holm、McNemar、四象限、DiD、两阶段 IPW、方差定容）** | `statistics/e1_historical.py`, `paired_stats.py`, `two_phase_correction.py`, `power_planner.py` | `test_06`, `test_07` | CPU 全边界测试通过 | 否（纯 CPU 可复现） | 否 | 是（两阶段校正主/次地位与正式定容） |
| **严格阶段授权、Pilot-Formal 永久排除、协同 GPU 锁与一键部署脚本** | `common/gates.py`, `common/gpu_lock.py`, `deployment/*.py`, `scripts/*.sh` | `test_07`, `test_08`, `test_09` | CPU 锁竞争 + 10 类故障注入 + `bash -n` 全部通过 | 是（BN `fcntl.flock` + `nvidia-smi`） | 否 | 是（用户逐级签发授权收据） |
