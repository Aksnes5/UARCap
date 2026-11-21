# Final reproducibility snapshot

- Repository: `/home/bn/videoufo_research` (UARCap)
- Branch: `main`
- Remote: `ssh://git@ssh.github.com:443/Aksnes5/UARCap.git`
- Snapshot base commit: `701827f3b66d1c5e5acb96f243b604b87db97ca0`
  (the handoff-package commit is the next one; see remote HEAD)
- Environment: Linux, Python 3.10 (`videollama2` env for extraction,
  `repro_evcap/venv` for generation/eval); offline TORCH_HOME/HF cache.
- GPU used: NVIDIA RTX A6000 (48 GB); no training, inference only.

## Models and checkpoints
- MSR-VTT EVCap checkpoint:
  `train_evcap_top7_bs1_..._review.pt`, sha256
  `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`.
- MSVD EVCap checkpoint: `train_evcap_top3_bs1_..._msvd.pt`, sha256
  `93e3665b3bf1a0db...` (see `frozen_method_config.json` family).
- Vicuna-7B-v1.3 local; X-CLIP `k600_16_16.pth`; Video-LLaMA Q-Former weights
  from local cache.

## Data manifests
- clips_10k 141e775b; clips_50k b89fe74a; clips_100k 1436e5ff;
  clips_250k 77ad4376; clips_500k aac9401d; clips_full f2bacf23 (short SHA).
- untouched2690 fce00da1; test2990 d826647b.
- Replicate seeds/config: `configs/subset_replication_seeds.json`.
- MSVD test captions sha256 `4e1dd9ae…` (see analysis config).

## Frozen configs
- `frozen_method_config.json` (adaptive selector, preprocessing, hashes).
- `configs/msvd_confirmatory_analysis.json` (MSVD analysis plan).
- `configs/subset_replication_seeds.json`.

## Canonical paper package
- reports/MASTER_PAPER_HANDOFF.md
- reports/FINAL_RESULTS_LEDGER.csv
- reports/FINAL_CLAIM_MATRIX.md
- reports/FINAL_NUMBERS_FOR_PAPER.md
- reports/FINAL_INVALID_RESULT_REGISTRY.md
- reports/PAPER_FIGURE_TABLE_PLAN.md
- reports/PAPER_SAFE_CLAIMS_AND_PROHIBITED_CLAIMS.md
- reports/PAPER_NARRATIVE.md
- reports/RELATED_WORK_SEARCH_TERMS.md

## Primary scripts
build_manifests.py; extract_text_embeddings.py; extract_query_features_v2.py;
make_external_retrieval_val.py; make_external_retrieval_val_scale.py;
build_val_grid.py; extract_external_features_v2.py; run_val.sh;
run_val_scale.sh; run_test.sh; run_msvd.sh; eval_*_prompt5*.py;
cocoeval.py; phase3/v2/scale/val/test statistics scripts;
oracle_fixeddf.py; factorial_stats.py; loo_analysis.py;
nested_scale_survival.py; source_cluster_bootstrap.py.
8ccdd4570fdd31b4d0653318be229716dce25f5dd93b5e6aa7fc2c2fd35f43ff frozen_method_config.json
54ef93b46c0b2d3a0c816d5838dcb9293e198989d3f5cf8051f899fbd9c7b12c reports/FINAL_NUMBERS_FOR_PAPER.md
82d0c69cb1de69b1ac8f7cc2c124a5928594e8f5bfd17561c1012bedcf7c79b4 reports/FINAL_CLAIM_MATRIX.md
