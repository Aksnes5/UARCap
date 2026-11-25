# 阶段授权与防误触门控矩阵（AUTHORIZATION GATE MATRIX）

本工程通过 `uarcap_q2/common/gates.py`、`deployment/authorization_gate.py` 与 `scripts/run_*.sh` 实现硬性分层防误触保护：

| 阶段（`PilotStage`） | 启动脚本 | 所需授权收据 | 所需前置 `PASS` 收据 | 默认状态（无收据时） | 允许下载的资产上限 |
|---|---|---|---|---|---|
| **`PILOT_A0`** | `bash scripts/run_preflight.sh` | 无需 GPU 授权（纯只读预检） | 无 | **允许只读预检** | `< 5 MB` 元信息 |
| **`PILOT_A1`** | `bash scripts/run_pilot_a01.sh` | `pilot_a01_auth.json`（含 `PILOT_A1` & `approved: true`） | A0 预检通过 | **硬阻断（退出码 3）** | 仅 DREAM-1K `video.zip`（`3.59 GiB`） |
| **`PILOT_A2`** | `bash scripts/run_pilot_a2.sh` | `pilot_a2_auth.json`（含 `PILOT_A2`） | `PILOT_A1` `run_receipt.json` | **硬阻断（退出码 3）** | 仅 1 款跨家族本地文本裁判 |
| **`PILOT_A3`** | `bash scripts/run_pilot_a3.sh` | `pilot_a3_auth.json`（含 `PILOT_A3`） | `PILOT_A2` `calibration_receipt.json` | **硬阻断（退出码 3）** | 1 款现代 VLM + VidHalluc `ACH_videos.zip`（`15.60 GiB`） |
| **`FORMAL_B`** | `bash scripts/run_formal_b.sh` | `formal_b_auth.json`（须含 `stage: FORMAL_B`、`approved: true`、`approval_id`、`frozen_pilot_params_confirmed: true`） | **`PILOT_A1`、`PILOT_A2`、`PILOT_A3` 三份均为 `PASS`** | **绝对硬阻断（即使开启 `--allow-cpu-mock` 仍被拦截！）** | 无新增大数据集 |

> **关键防误触保障**：在 Pilot A0–A3 全部完成并由用户明确签发 `formal_b_auth.json` 之前，任何命令都无法意外启动正式方案 B。
