# GitHub 输入交付核验

**TRANSFER_VERIFIED = YES。** 从 Windows 的两个新建空仓库分别通过 GitHub Git over SSH 与 Git over HTTPS 取得固定提交中的 ZIP。HTTPS 取包未使用凭据。对从 SSH 独立下载的包使用同一远端提交取得的接收脚本解压，八份 JSON、六份视觉 ID 清单与原始输入回放收据均和 Windows 原始文件逐字节相同。HTTPS 再次获得相同 ZIP、manifest 与验证脚本。

| 校验对象 | 实测结果 |
| --- | --- |
| 本地交接基线 | `56c908034c0125604e6089a0e6db8d0232c3ff10` 已原样推送到独立分支，并按远端 ref 验证 |
| 输入传输提交 | `bc737176ea700492402be63480a3c6e76f05be2e` |
| ZIP Git blob ID | `3fedb0e531930a7e4fe5c6a6241271046f5ee859`，暂存值与未过滤本地值一致 |
| ZIP 远端实测字节 | **2,861,489** |
| ZIP SHA-256 | `93289136546fc2d96d2f457cc9dec57c50fde1b2c517f735f23df0a1dfc63128` |
| ZIP 内部 | 八份实际输入，六份视觉缓存 ID 清单，原始冻结摘要与回放收据，内部成员清单、验证说明和许可署名 |
| 下载后样本 | 2,990 全量、2,690 untouched、2,618 B eligible，三个队列和各个对象查询顺序都通过验证 |
| 二进制属性 | scoped `.gitattributes` 对 ZIP 禁用文本换行转换 |
| BN 站点接收 | **UNVERIFIED**。Windows 独立取回证明 GitHub 可提供完整对象。BN 的网络出口和落盘需由 BN 按接收指令确认。 |

`raw.githubusercontent.com` 在本次 Windows 检查中 DNS 失败，另一次 REST API 尝试中途断开。已经完成验证的路径是访问 `github.com` 并使用 Git 读取指定提交的 blob。BN 接收命令直接采用这一路径。 [ZIP 下载页](https://github.com/Aksnes5/UARCap/blob/bc737176ea700492402be63480a3c6e76f05be2e/research/uarcap_v06_followup_deployment/local_cpu_handoff/transfer/UARCap_V06_FOLLOWUP_BN_INPUTS.zip) · [外部 SHA 和逐文件清单](https://github.com/Aksnes5/UARCap/blob/bc737176ea700492402be63480a3c6e76f05be2e/research/uarcap_v06_followup_deployment/local_cpu_handoff/GITHUB_INPUT_TRANSFER_MANIFEST.json)

BN 指令中的接收段已在 Windows Git Bash 用隔离项目路径完整演练，三项下载件 SHA 与解包检查均通过。另将临时 ZIP 的一个字节改变，接收脚本在创建解包目录前拒绝了该包。

## 公开交付的依据

GitHub 仓库元数据标示 `PUBLIC`。本次逐查询核对了三个 B CAP 和共享 SIM 中的文本全部属于同一查询的冻结 VideoUFO 候选，没有基准参考描述。ZIP 仅包含规定的派生文本、分数、查询和 clip ID、哈希和说明。禁止把其它日志、凭据、服务器配置、权重或媒体加入这个传输包。VideoUFO 作者在 NeurIPS 发布说明声明数据集采用 CC BY 4.0，数据卡亦以该许可证标记 text / image 数据集。ZIP 内保留作者、来源、许可链接及 UARCap 的选择、组合和排序说明。 [VideoUFO 发布声明](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1e6057620ed314b0020b3a30284b0f83-Abstract-Datasets_and_Benchmarks_Track.html) · [数据卡](https://huggingface.co/datasets/WenhaoWang/VideoUFO) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
