# 本地独立科研终审与复现

从 [终审裁决](LOCAL_UARCAP_FINAL_SCIENTIFIC_REVIEW.md) 进入。审计针对基线 `52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec`，只做 Windows CPU 复算及代码调查。正式数据留在冻结提交、冻结 ZIP 或本地原文件。`results/` 中只有衍生数值、来源/查询标识、哈希和审计统计，无视频、模型权重或参考文本副本。

## 复现指令

用 Python 3.11、NumPy 2.4.6、SciPy 1.17.1、Git 和 Java。PTB JAR 取自前作本地 `evaluation/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar`。JAR、前作 PDF、审阅草稿以及公开来源镜像的 SHA 均在 [清单](DATA_SHA256_MANIFEST.json)。镜像可仅下载 raw metadata JSON，和 Zip 内 references 不互代。来源标识只用于聚类、污染比对；文件 SHA 为 `3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014`。[镜像数据文件](https://huggingface.co/datasets/friedrichor/MSR-VTT/blob/main/raw_data/MSRVTT_data.json)

从本目录运行以下 PowerShell。`$repo` 是包含冻结提交的任一 checkout，`$data` 和 `$out` 使用独立的本地计算目录。Java、JAR、PDF、draft 和 metadata 均传本地路径。

```powershell
$repo = '<repo containing baseline>'
$data = '<scratch directory>'
$out = '<audit output directory>'
$java = '<java executable>'
$jar = '<exact Stanford PTB jar>'
$meta = '<hash-verified MSRVTT_data.json>'
$pdf = '<published predecessor PDF>'
$draft = '<audited manuscript Markdown>'
New-Item -ItemType Directory -Force $out | Out-Null
python -B audit_inputs.py --repo $repo --data $data --manifest "$out/DATA_SHA256_MANIFEST.json" --jar $jar --metadata $meta --predecessor $pdf --draft $draft
python -B audit_statistics.py --bundle "$data/extracted" --repo-data "$data/git" --output $out --java $java --jar $jar --vendor "$data/git/evcap_external/evaluation" --metadata $meta
python -B audit_retrieval.py --bundle "$data/extracted" --repo-data "$data/git" --results $out --output $out --metadata $meta
python -B audit_refit.py --bundle "$data/extracted" --results $out --output "$out/refit_idf_sensitivity.json" --java $java --jar $jar --metadata $meta
python -B audit_outputs.py --bundle "$data/extracted" --repo-data "$data/git" --results $out
python -B audit_corpus.py --repo $repo --metadata $meta --results $out
```

核心点估计以全体冻结样本计算。所有独立重复采样以 seed `20261124` 执行；不同统计家族使用相同 seed 的独立调用，家族内各对比共享抽样，以保留依赖。Oracle/refit 的 sparse 结果先对正式逐查询 PTB CIDEr 做精确交叉验证。`grid_plus_oracle76` 是多重比较范围的敏感性。`historical_whitespace` 与 `bh_frozen_p_values.csv` 分别记录对旧方法的独立重算，以及对旧打印 p 的精确复验。

全样本分数与校正见 `paired_contrasts.csv`，实际回放排序见 `oracle_replayed_top20.csv`，输入核查见 `oracle_query_integrity.csv`。测试集使用固定参考的差值与区间，正式 CIDEr 统一为 points。
