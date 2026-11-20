# MSVD checkpoint / split usage audit

Checkpoint: `train_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd.pt`
(147,336,938 B), reproducing MSVD test CIDEr **1.583** (B1 91.5 / B4 69.7 /
M 47.0 / R 84.0 / SPICE 8.8), matching paper 158.32.

## Evidence

1. `train_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5_msvd.py` has no
   validation dataloader, no metric-based checkpoint selection, no early
   stopping; `--epochs` defaults to 1 and the final model is saved
   (`save_checkpoint` once per epoch; single epoch → one file).
2. Training config consumes `datasets/MSVD/train_16frames.yaml` (1,200 videos)
   and in-domain retrieval `ext_data/msvd_train_allGT_avg_frames_xclip_retrieve_20_captions_no_duplicat_sim_scores.json`.
3. Evaluation script reads `datasets/MSVD/test_16frames.yaml` (670 videos) only.
4. No training log records val100 metrics or checkpoint selection.

## Answers

1. Training on train1200 only: **yes** (per config).
2. val100 used for monitoring: **no evidence**.
3. val100 used to select checkpoint: **no** (single final save).
4. test670 used for tuning: **no evidence**.
5. Test leakage into training/selection: **none found**.

Residual uncertainty: original training logs are not preserved; this is a
source-code + artifact audit, same limitation as the MSR-VTT audit. val100 will
still be reported as unused, not silently repurposed.
