#!/bin/bash
# Sequential val497 scale x budget grid with resume + ledger.
cd /home/bn/videoufo_research
LEDGER=reports/runtime_ledger.csv
[ -f "$LEDGER" ] || echo "experiment,scale,track,k,n_queries,load_seconds,generation_seconds,eval_seconds,gpu_hours,status" > "$LEDGER"
PY=/home/bn/anaconda3/envs/videollama2/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
for S in 10k 100k 500k; do
  for K in 1 3 5 7; do
    V="${S}_D${K}"
    OUT="evcap_external/outputs/eval_val497_${V}/${GEN}"
    if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "497" ]; then
      echo "SKIP_GEN_${V}"
      T0=$(date +%s)
    else
      T0=$(date +%s)
      bash scripts/run_val_scale.sh $S D $K > logs/gen_val_${V}.log 2>&1 || { echo "GEN_${V}_FAILED"; tail -3 logs/gen_val_${V}.log; continue; }
      echo "GEN_${V}_DONE"
    fi
    T1=$(date +%s)
    M="evcap_external/outputs/eval_val497_${V}/metrics.json"
    if [ ! -f "$M" ]; then
      cd evcap_external
      $PY evaluation/cocoeval.py --result_file_path "outputs/eval_val497_${V}/${GEN}" --eval_file "outputs/eval_val497_${V}/metrics.json" > ../logs/metrics_val_${V}.log 2>&1
      cd ..
    fi
    T2=$(date +%s)
    N=$($PY -c "import json;print(len(json.load(open('$OUT'))))")
    echo "val497_${V},${S},D,${K},${N},,$((T1-T0)),$((T2-T1)),$(echo "scale=3;($T2-$T0)/3600"|bc),done" >> "$LEDGER"
    echo "METRICS_${V}_DONE"
  done
  $PY scripts/val_retrieval_metrics_scale.py $S > logs/retrieval_metrics_${S}.log 2>&1
  echo "RETRIEVAL_METRICS_${S}_DONE"
done
echo "SCALE_GRID_DONE"
