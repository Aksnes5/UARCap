# 主动故障注入测试报告（FAULT INJECTION REPORT）

本报告记录针对 UARCap Q2 离线工程实施的 **主动故障注入验证**，以及测试过程中**真实发现并即时修复的 2 处底层缺陷**。

---

## 一、10 类强制故障注入验证矩阵（全部被系统成功拦截）

| 编号 | 注入故障类型 | 对应测试 | 系统拦截行为 | 结果 |
|---:|---|---|---|---|
| 1 | **错误检查点哈希（`wrong_checkpoint`）** | `test_08` | `verify_actual_input_trace` 抛出 `TraceVerificationError`，运行器置 `all_traces_verified=False` 与 `verdict=FAIL` | **BLOCKED (PASS)** |
| 2 | **错误查询对应关系（`Query mismatch`）** | `test_08` | `verify_actual_input_trace` 立即拒绝张冠李戴的 `InputTrace` | **BLOCKED (PASS)** |
| 3 | **ARM_A0 注入非空外部证据（`non_empty_a0`）** | `test_08` | 拦截非空 `text_evidence`，置 `a0_zero_evidence_verified=False` 与 `verdict=FAIL` | **BLOCKED (PASS)** |
| 4 | **A1/A7 候选前缀不一致** | `test_08` | `verify_natural_prefix_consistency` 抛出 `ValueError("Prefix consistency violation")` | **BLOCKED (PASS)** |
| 5 | **E3 四臂视觉输入合同漂移** | `test_08` | `verify_e3_visual_lock` 抛出 `ValueError("E3 visual lock violation")`；单步运行器亦拒绝 `visual_drift` | **BLOCKED (PASS)** |
| 6 | **正式集混入 Pilot A 样本** | `test_08` | `enforce_pilot_formal_exclusion` 抛出 `ProtocolIsolationError` | **BLOCKED (PASS)** |
| 7 | **重复来源（VidHalluc 同 YouTube ID）与伪造未知 source ID** | `test_01` | 同源去重仅留 1 片；`UNKNOWN_OPAQUE` 填入非空 `source_id` 立即被 `QueryRecord.validate` 拒绝 | **BLOCKED (PASS)** |
| 8 | **裁判 9 类非法结构化输出**（非法 JSON/空/拒绝/截断/非法标签/缺字段/无证据定论/命题错配/重复） | `test_04` | 全部转为 `parse_status="PARSE_FAILED"` 且指标保持 `None`，**零静默默认判对或判错** | **BLOCKED (PASS)** |
| 9 | **缺失/空输出（`empty_output`）与模拟 CUDA OOM（`oom`）** | `test_08` | 空输出率 $\ge 3\%$ 判 `FAIL`；OOM 立即安全停机（`oom_halted=True`），写入失败收据且不损坏已完成 JSONL | **BLOCKED (PASS)** |
| 10 | **未授权启动 Formal B（含缺少 A3 PASS 前置收据）** | `test_08`, `test_09` | `verify_stage_authorization` 与 `authorization_gate.py` 返回退出码 3（`AUTHORIZATION_BLOCKED`），即使开启 `--allow-cpu-mock` 亦硬拦截 `FORMAL_B` | **BLOCKED (PASS)** |

---

## 二、测试期间真实发现并即时修复的 2 处工程缺陷

1. **有损 MPEG4 压缩下的“重复首帧”漏检修复（`uarcap_q2/datasets/video_decoder.py`）**：
   - *发现*：初次运行 `test_01` 时，对 20 帧完全相同像素编码生成的 MPEG4 测试视频，GOP 关键帧量化微噪使 `unique_frame_hashes=3`、`mean_consecutive_l1_diff=0.1555`，绕过了最初仅检测 `unique_hashes == 1 and mean_diff < 1e-6` 的严格条件。
   - *修复*：升级为 `(unique_hashes == 1) or (mean_diff < 0.5 and unique_hashes <= 4)`，现已能同时拦截无损逐字节重复帧与有损压缩冻结流，而正常动态视频（`mean_l1_diff > 1.0`, `unique_hashes >= 12`）稳定放行。
2. **视觉哈希漂移异常的 `all_traces_verified` 状态位漏置修复（`uarcap_q2/runner/orchestrator.py`）**：
   - *发现*：初次运行 `test_08` 时，运行器依靠错误消息字符串匹配 `"trace"` 来翻转 `all_traces_ok`，导致 `TraceVerificationError("Visual input hash drift...")` 虽使整体判 `FAIL`，但 `all_traces_verified` 仍误报 `True`。
   - *修复*：改为严格按异常类型 `isinstance(exc, TraceVerificationError)` 判定，彻底消除脆弱的字符串匹配。
