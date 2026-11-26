# BN V2 执行验收

1. 使用独立 V2 checkout 和输出路径 `/home/bn/uarcap_q2_runtime/outputs/laptop-20261126-v2`。核对提交、Git 工作树、CPU 通道和旧 scheduler state。V1 若存在运行痕迹，先用回执对账。旧工作和其他项目进程继续由原所有者管理。
2. 确认原 16 帧、CaReBench 真正的冻结检索行、参考锁、预测及官方 VideoUFO TSV。使用签名手册构建或验证 Top-7 元数据中的 clip→文件/分片映射。
3. 核对原 E3 路径、正式与 Pilot 全部排除清单。储备需要原 ACH 元数据、完整调试来源记录和独立抽样修订。两个授权的作用不同。
4. 安装队列后确认原调度器、GPU 共享锁、60 秒心跳及 1,800 秒巡检。发生资源冲突按共享锁等待。超时、查无真实 trace 或解析不完整分别登记，继续独立分支。
5. Qwen 先测固定两条查询，验收 16 个输入 query frame、额外八帧、真实 vLLM processor image markers/tokens及同一模型摘要。32、64 各有小规模试跑。无法取得实际消费证据就阻断对应模态结论。
6. 预算、长度、32/64、模态、E3、Bridge、自然溯源、相关性、历史复用、压力测试和来源筛查按队列推进。人审门户另需授权人员。CPU 与 GPU 计算都使用独立记录。
7. 收集 `final/FINAL_EXPERIMENT_MATRIX.json`、`final/ARCHIVE_MANIFEST.json`、报告图表和人审科学回执。如有后续真实授权或人工导入，重新运行同一 `scripts/q2_release_report.py`。只回传允许分享的结果及哈希，私有键、评审导出和视频仍由管理员保管。

预检与执行入口见 [BN 部署](BN_DEPLOYMENT_AND_EXECUTION_PROMPT.md)。
