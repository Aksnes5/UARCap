# Windows CPU 全量测试结果报告（CPU TEST RESULTS）

- **执行环境**：Windows 11，Python `3.11.9`，`pytest 9.1.1`，`numpy 2.4.6`，`scipy 1.17.1`，`PyAV (av) 18.1.0`，`Pillow 12.3.0`，`PyYAML 6.0.3`，Git Bash (`bash -n`)
- **执行命令**：`pytest -v research/uarcap_q2_experiments/tests`
- **执行结果**：**`43 passed, 0 failed, 0 skipped`（耗时 `4.37 s`）**
- **重要声明**：所有测试均在 Windows 本地 CPU 与人工构造小型动态 fixture 上完成，**不等同于 BN 服务器 RTX A6000 48GB 真实 GPU 验证**。

## 分模块测试清单

| 测试文件 | 通过项数 | 覆盖要点 |
|---|---:|---|
| `test_01_datasets_and_video_decoder.py` | 5 | 严禁伪造 source ID；DREAM-1K 分层抽样与永久排除；VidHalluc YouTube ID 去重与 `CANDIDATE_PENDING_VERIFICATION`；PyAV 正常动态/短片/单帧/VFR/重复首帧/损坏/OOB 窗口 + EVCap 16 帧 TSV 打包；Zip-Slip 阻断与 Route 2 分卷校验重组 |
| `test_02_dedup_and_retrieval_arms.py` | 4 | DREAM-1K 两阶段近重复筛查（`exact_source_zero_overlap_claimed=False`）；`.strip()` 分组与 D 共识 ROUGE-L 重排；`ARM_A0` 空证据与 `A1⊂A3⊂A7` 前缀一致；E3 `BASE_A0`/`BASE_B3` 视觉锁定 |
| `test_03_generators_and_e3_interventions.py` | 4 | 冻结 Video-LLaMA dry-run 与 `A0` 线格式；Modern VLM 配置/提示/trace 校验；VidHalluc E3 5 项人工审批状态机；`E3-Bridge` `<25` 对自动禁用与 $\ge 25$ 对启用 |
| `test_04_evaluators_and_calibration.py` | 13 | 跨家族裁判强制隔离；9 类裁判非法输出（非法 JSON/空/拒绝/截断/非法标签/缺字段/无证据定论/命题错配/重复）零静默默认；原子漏抽错误率审计；Blind E3 共存 `(1,1)`/否定/泛化/防循环；3×4 校准在零错误时返回 `INCONCLUSIVE`、在 `A7` 漏检偏置时返回 `FAIL`、无偏时返回 `PASS` |
| `test_05_annotation_tool_and_blindness.py` | 3 | 盲法包零泄密递归验证与 E3 随机目标准确解盲；本地 HTTP 网页标注服务器实时保存、CSV 导出、403 越权媒体阻断与重启续标；双人 $\kappa$/AC1 与双人合计总人时核算 |
| `test_06_statistics_and_two_phase.py` | 5 | E1 跨参考选择与缺臂处理；全零差/单向差/重复来源聚类/Holm 校正；精确 McNemar/E3 四象限/跨模型 DiD；两阶段 IPW 偏差校正与校准-验证隔离拦截；试点方差定容与 A6000 吞吐外推 |
| `test_07_runner_lock_gates_and_e2e.py` | 2 | **全链路端到端 CPU 模拟**（加载→16 帧解码→检索→D 重排→三臂构造→Mock 生成→三值评价→盲法打包→配对统计→自动三份 Pilot A1 报告）+ 断点续跑全量跳过；协同 GPU 锁并发冲突拒绝且不删锁文件 |
| `test_08_fault_injection_and_security.py` | 4 | 全部 10 类主动故障注入拒绝验证（详见 `FAULT_INJECTION_REPORT.md`）；凭证/Token 扫描（全仓库 0 命中）；受限资产公开发布合规拦截 |
| `test_09_shell_scripts_and_manifests.py` | 3 | 全部 8 个 Linux `.sh` 脚本 Unix LF 校验 + `bash -n` 语法通过；全部 YAML 配置解析通过；CLI `authorization_gate.py` 硬拦截 `FORMAL_B` 与 `bn_preflight_check.py` 干跑通过 |
