#!/bin/bash
cd /home/bn/videoufo_research
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd_generated_captions.json
for K in 1 3 5 7; do
  while :; do
    F="evcap_external/outputs/eval_msvd_D${K}/${GEN}"
    [ -f "$F" ] && [ "$($PY -c "import json;print(len(json.load(open('$F'))))" 2>/dev/null)" = "670" ] && break
    sleep 20
  done
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_msvd_D${K}/${GEN}" --eval_file "outputs/eval_msvd_D${K}/metrics.json" > ../logs/metrics_msvd_D${K}.log 2>&1 )
  echo "FINAL_METRICS_D${K}_DONE"
done
$PY scripts/msvd_confirmatory_prepare.py > logs/msvd_prepare.log 2>&1 && echo "MSVD_PREPARE_DONE"
( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_msvd_Adaptive/${GEN}" --eval_file "outputs/eval_msvd_Adaptive/metrics.json" > ../logs/metrics_msvd_Adaptive.log 2>&1 )
echo "FINAL_METRICS_ADAPTIVE_DONE"
$PY scripts/msvd_stats.py > logs/msvd_stats.log 2>&1
echo "MSVD_STATS_DONE"
