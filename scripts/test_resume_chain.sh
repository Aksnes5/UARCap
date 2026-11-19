#!/bin/bash
cd /home/bn/videoufo_research
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
while pgrep -f "out_path outputs/eval_test2990_D7" >/dev/null; do sleep 30; done
echo "D7_EXITED"
for K in 1 3 5; do
  V="D${K}"
  OUT="evcap_external/outputs/eval_test2990_${V}/${GEN}"
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "2990" ]; then echo "SKIP_GEN_${V}"; else
    T0=$(date +%s)
    bash scripts/run_test.sh D $K > logs/gen_test_${V}.log 2>&1 || { echo "GEN_${V}_FAILED"; continue; }
    T1=$(date +%s)
    echo "GEN_${V}_DONE"
  fi
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_test2990_${V}/${GEN}" --eval_file "outputs/eval_test2990_${V}/metrics.json" > ../logs/metrics_test_${V}.log 2>&1 )
  echo "METRICS_${V}_DONE"
done
echo "TEST_GRID_DONE_ALL"
