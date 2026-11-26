# BN 回传部署与执行

在 BN 上使用 Git 和现有 Python 环境执行。分支为 `antigravity/uarcap-q2-laptop-final-development-20261126`。把 `expected` 设成最终交付回复中核对过的完整 SHA。新工作目录存在时核对当前 SHA 和工作区状态，复用同一副本。

```bash
branch=antigravity/uarcap-q2-laptop-final-development-20261126
release=/home/bn/uarcap_q2_runtime/deploy/laptop-20261126-v1
expected='<交付完整 SHA>'
mkdir -p /home/bn/uarcap_q2_runtime/deploy
git clone --filter=blob:none --no-checkout --single-branch --branch "$branch" \
  ssh://git@ssh.github.com:443/Aksnes5/UARCap.git "$release"
git -C "$release" sparse-checkout init --cone
git -C "$release" sparse-checkout set research/uarcap_q2_experiments \
  research/uarcap_v06_followup_deployment/bn_v06_full_ab manifests
git -C "$release" checkout --detach "$expected"
test "$(git -C "$release" rev-parse HEAD)" = "$expected"
cd "$release/research/uarcap_q2_experiments"
cpu=/home/bn/anaconda3/envs/videollama2/bin/python
"$cpu" -m pytest -q -rs
"$cpu" deployment/q2_laptop_queue.py --commit "$expected" --dry-run
```

逐项核对输出中的实时编排器、巡检器、报告进程和待退役任务。确认项目锁路径、原预测、帧、版本化工作区可写。若 vLLM 连续在 64 图像 smoke 中不能稳定工作，保留失败原文并暂停相应科学分支。Qwen 的 32/64 帧从 CPU 环境解码后进入共享 vLLM。

历史反向桥接在 `/home/bn/uarcap_q2_runtime/data/source_witness/MSRVTT_data.json` 读取公开来源元数据，SHA 在脚本固定。无需下载视频或模型。视觉近重复入口读取两个有序签名账本，若 BN 不具备可说明对齐的签名，保留 UNVERIFIED。VidHalluc E3 的历史来源交集使用随版本交付的小型公开 MSR-VTT 来源见证。CPU 运行可由普通 SSH 会话执行；安装队列后沿用 BN 编排器。

GPU 本身须有 BN 操作员批准记录，文件含 `authorized=true`、最终完整 `commit`、`approved_by`。已有有效批准可直接作为输入。实际启动前留存审批文件和干跑输出。

```bash
"$cpu" deployment/q2_laptop_queue.py --commit "$expected" --install \
  --gpu-approval /home/bn/uarcap_q2_runtime/deploy/laptop-v1-gpu-approval.json
```

队列按依赖推进。每个新增任务都由版本化命令验证脚本、输入及上游收据，完成后核对实际行数和格式。既存运行正常时沿用，停机时同一安装入口恢复运行器、1800 秒巡检和 60 秒心跳。巡检器最长 48 小时，若延长运行在相同 checkout 再次执行安装入口，可仅补拉已退出服务。发生真实 RUNNING 状态冲突时先核查现有进程与输出，避免重复调度或抢 GPU。返回上游时包含任务状态、失败日志和科研收据。

独立文字标注依次使用 [人工核验手册](HUMAN_VALIDATION_RUNBOOK.md)。成果状态及硬阻塞见 [验收契约](EXPECTED_OUTPUT_ARTIFACTS.json) · [已编译队列](REMAINING_GPU_TASK_QUEUE.json) · [回滚](BN_DEPLOYMENT_ROLLBACK.md)
