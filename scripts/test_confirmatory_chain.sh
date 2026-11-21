#!/bin/bash
cd /home/bn/videoufo_research
while ! grep -q "TEST_GRID_DONE_ALL" logs/test_finish_chain.log 2>/dev/null; do sleep 60; done
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
$PY scripts/test_confirmatory_prepare.py > logs/test_prepare.log 2>&1 || { echo PREPARE_FAILED; exit 1; }
echo "TEST_PREPARE_DONE"
cd evcap_external
for TAG in test2690 test2990; do
  for V in B D1 D3 D5 D7 Adaptive; do
    $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_${TAG}_${V}/${GEN}" --eval_file "outputs/eval_${TAG}_${V}/metrics.json" > ../logs/metrics_${TAG}_${V}.log 2>&1
    echo "METRICS_${TAG}_${V}_DONE"
  done
done
cd /home/bn/videoufo_research
$PY scripts/test_confirmatory_stats.py > logs/test_confirmatory_stats.log 2>&1
echo "TEST_STATS_DONE"
