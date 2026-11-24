# 历史 D1 / D3 / D7 可比性审查

**结论：三组均为 `CONDITIONALLY_REUSABLE`，严格复用 0 组。** 新的 BN 回执补上了日志参数与当前物理文件的联系。历史进程实际使用的代码、依赖和张量仍缺少强绑定。

| 核对层 | 当前证据 | 判定 |
| --- | --- | --- |
| 历史产物 | BN 六个预测文件 SHA-256 与本地归档一致。全量 2,990 条，排除固定 300 pilot 后精确得到归档 2,690 条记录。均配 20 个有序参考，digest 一致。 | `VERIFIED`（产物） |
| 执行意图 | `logs/gen_test_D{1,3,7}.log` 的回执给出各自 topn、共同 test2990 帧 YAML、FULL D CAP/SIM、16 帧、seed 42、checkpoint slot 与 Vicuna 路径。归档 runner 与 BN 当前哈希一致。归档源码用默认 beam 5，根据 outer k 回退至两种证据预算。 | `VERIFIED`（已记录参数及设计） |
| 当前物理文件 | 三个历史 checkpoint slot 现在解析到同一已哈希目标。BN 核实当前数据文件的顺序及偏移，并验证六个核心源码和九个依赖哈希。当前 image TSV 只记录大小、mtime、偏移哈希、行 ID 与样本，不含全文件字节哈希。 | 当前资产核对已通过其回执覆盖范围 |
| 历史实际执行 | 缺少逐进程源码/依赖快照，以及 frame、缓存张量、最终渲染 prompt、权重加载绑定。日志参数加当前文件哈希还原不了这些观测。 | `UNVERIFIED` |

## 已知 frame-loader Bug 的范围

- PRE_FIX val497 使用只存视频路径的 TSV，触发 `row[0] == row[-1]` 分支和 PyAV 实时解码。旧 eval 传 `start=end=0`，把解码窗口压到开头。旧 CPU 审计在目标 val 视频观察到 16 个位置中有 15 个与首帧相同。POST_FIX 把时间窗口改成 `None,None`，修复整段采样。两版对比只改了该 loader 的 eval 时间窗，相关 decoder、test2990 YAML 与 runner 字节相同。
- D1/D3/D7 的日志指向预提取 test2990 TSV。其 18 字段是键、元数据及 16 个帧字段，走 `get_frames_from_tsv(row[2:])`，该分支不接收 start/end。用两版冻结方法的 CPU 测试确认了两种路由。旧 CPU 审计还对 test2990 的 `video7010`、`video7011` 实测，两者各只有首帧本身与首帧相同。BN 回执有 106 个抽样条目（105 个不同索引），字段数和帧 payload 非空均通过。该抽样没有验证真实像素或逐查询张量。
- 所以已知 0/0 解码缺陷与 D 臂记录的 test2990 路由不匹配。历史逐进程是否确实使用该 loader、所读帧字节和预处理结果仍 `UNVERIFIED`。持有完全相同的预测 JSON 也无法解决这部分缺口。框架另有 JPEG/base64 解码与转换，包版本和样本解码问题应在未来正式 smoke 的帧签名中单独核对。

## 对 v0.6 的直接影响

D1→A1、D7→A7、D3→B3 的输入设计相容。B3 与 D3 的有效前三条 CAP/SIM/视觉 ID 对齐，即使外层 topn 设置不同。历史记录只能用作配置与结论的参照。正式矩阵仍保留六个新 arm。新增针对当前 TSV 的内容哈希和未来 smoke 能绑定新运行，不能认证早期历史输入。

依据来自 `bn_readonly_receipts/BN_HISTORICAL_RUNTIME_RECEIPTS.md`、`BN_DATASET_ASSET_RECEIPTS.json`、`BN_DEPLOYMENT_PATH_BINDINGS.json`、归档 `reports/cpu_audit/VAL_FRAME_INPUT_BUG_CODE_TRACE.md` 和 `reports/post_fix/POST_FIX_FRAME_SANITY_REPORT.md`。未运行历史评测、指标或新字幕生成。
