# N1 — checkpoint / val usage audit

Checkpoint: `train_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review.pt`
(sha256 `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`,
147,337,150 bytes, mtime 2026-05-27).

## Evidence

1. Training script `evcap_external/train_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py`
   contains **no validation dataloader and no metric-based checkpoint selection**.
   It trains for `--epochs` (default 1) and saves the final model:
   - `epochs == 1` → `ckpt_name` directly;
   - otherwise `ckpt_name_epoch{N}.pt` per epoch.
2. `outputs/train_evcap_msrvtt_quality_text/` contains five sibling checkpoints
   (concat / gate / self_attention / cross_fusion / cross_fusion_video_as_kv),
   each a single final file — consistent with one-epoch training, not with
   best-on-val selection.
3. No training log, tensorboard event, or config snapshot records a val CIDEr
   used for early stopping; the `_review` suffix indicates a manual review
   checkpoint, not a metric-selected one.
4. Our own val497 pipeline was created 2026-11-19, months after the checkpoint
   (2026-05-27), so it cannot have influenced training or checkpoint choice.

## Verdict

- val497 was **not** used for training, early stopping, or checkpoint selection
  by this project.
- Residual uncertainty: the checkpoint was produced by the original authors'
  run whose exact CLI/logs are not preserved. Based on the available source,
  there is no selection-leakage mechanism.
- No severe method-selection leakage found → test evaluation may proceed.
  This limitation will be stated in the paper's protocol section.
