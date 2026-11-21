#!/bin/bash
source /home/bn/videoufo_research/scripts/sched/common.sh
cd $ROOT
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
while ! grep -q "ROBUST_QUEUE_DONE" logs/robust_queue4.log 2>/dev/null; do sleep 30; done
run_stage(){
  id="$1"; name="$2"; k="$3"
  stage_done "$id" && return 0
  stage_start "$id"
  if ! preflight_eval val497/val16_video.yaml external/external_val_${name}_cap.json external/external_val_${name}_sims.json > logs/pf_$id.log 2>&1; then stage_fail "$id"; return 0; fi
  out="evcap_external/outputs/eval_val497_${name}/${GEN}"
  if [ ! -f "$out" ] || [ "$($PY -c "import json;print(len(json.load(open('$out'))))")" != "497" ]; then
    bash scripts/run_named_variant.sh "$name" "$k" > logs/gen_${name}.log 2>&1 || { stage_fail "$id"; return 0; }
  fi
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val497_${name}/${GEN}" --eval_file "outputs/eval_val497_${name}/metrics.json" > ../logs/metrics_${name}.log 2>&1 )
  stage_finish "$id"
}
run_stage oracle1 oracle1 1
run_stage oracle3 oracle3 3
run_stage oracle7 oracle7 7
run_stage goodtail7 goodtail7 7
run_stage hardwrong7 hardwrong7 7
run_stage easywrong7 easywrong7 7
echo "NAMED_FIX_DONE"
