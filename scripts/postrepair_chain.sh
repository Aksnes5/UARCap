#!/bin/bash
cd /home/bn/videoufo_research
while ! grep -q "REPAIR_DONE" logs/repair_chain.log 2>/dev/null; do sleep 30; done
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
runv () {
  N="$1"; K="$2"
  OUT="evcap_external/outputs/eval_val497_${N}/${GEN}"
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "497" ]; then echo "SKIP_${N}"; else
    bash scripts/run_val_variant.sh "$N" "$K" 0 0 > logs/gen_${N}.log 2>&1 && echo "GEN_${N}_DONE"
  fi
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val497_${N}/${GEN}" --eval_file "outputs/eval_val497_${N}/metrics.json" > ../logs/metrics_${N}.log 2>&1 )
  echo "METRICS_${N}_DONE"
}
runv oracle1 1
runv goodtail7 7
runv hardwrong7 7
runv easywrong7 7
echo "POSTREPAIR_DONE"
