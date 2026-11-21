#!/bin/bash
source /home/bn/videoufo_research/scripts/sched/common.sh
cd $ROOT
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
DRYRUN=${DRYRUN:-0}

eval_stage(){
  local id="$1" name="$2" k="$3" run="$4" yaml="$5" cap="$6" sims="$7" exp="$8"
  stage_done "$id" && { echo "SKIP $id"; return 0; }
  stage_start "$id"
  if ! preflight_eval "$yaml" "$cap" "$sims" >/tmp/pf_$id.log 2>&1; then
    cat /tmp/pf_$id.log; stage_fail "$id"; return 0
  fi
  if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_PASS $id"; rm -f "$ST/$id.running"; return 0; fi
  local out="evcap_external/outputs/eval_${name}/${GEN}"
  if [ -f "$out" ] && [ "$($PY -c "import json;print(len(json.load(open('$out'))))")" = "$exp" ]; then
    : # generation already complete
  else
    if ! gpu_run bash "$run" $k > "logs/gen_$name.log" 2>&1; then
      stage_fail "$id"; return 0
    fi
  fi
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_${name}/${GEN}" --eval_file "outputs/eval_${name}/metrics.json" > "../logs/metrics_$name.log" 2>&1 )
  stage_finish "$id"
}

stage_done dryrun_done && exit 0
if [ "$DRYRUN" = 0 ]; then
  while ! grep -q "REPAIR_DONE" logs/repair_chain.log 2>/dev/null; do sleep 30; done
fi
echo "ROBUST_QUEUE_START (dryrun=$DRYRUN)"
# invalidate oracle-dependent outputs from the pre-fixed-DF scorer
for n in oracle1 oracle3 oracle7 goodtail7 hardwrong7 easywrong7; do
  rm -rf "evcap_external/outputs/eval_val497_${n}" "evcap_external/outputs/train_val497_${n}"
done

# A. in-domain features
if ! stage_done indomain_features; then
  stage_start indomain_features
  NM=$(grep -c . /tmp/indomain_missing.txt || true)
  echo "indomain_missing $NM"
  if [ "$NM" -gt 0 ]; then
    if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_PASS indomain_features"; rm -f "$ST/indomain_features.running"; else
      TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache \
      HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache \
      TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
      PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_msrvtt_train_features.py /tmp/indomain_missing.txt > logs/indomain_features.log 2>&1
      if [ $? -eq 0 ]; then stage_finish indomain_features; else stage_fail indomain_features; fi
    fi
  else stage_finish indomain_features; fi
fi

# B. in-domain D grid
for K in 1 3 5 7; do
  eval_stage indomain_D$K val497_indomain_D$K $K scripts/run_indomain.sh val497/val16_video.yaml external/external_val_indomain_d7_cap.json external/external_val_indomain_d7_sims.json 497
done
# C/D. oracle and distractor conditions
eval_stage oracle1 val497_oracle1 1 scripts/run_val_variant.sh val497/val16_video.yaml external/external_val_oracle1_cap.json external/external_val_oracle1_sims.json 497
eval_stage oracle3 val497_oracle3 3 scripts/run_val_variant.sh val497/val16_video.yaml external/external_val_oracle3_cap.json external/external_val_oracle3_sims.json 497
eval_stage oracle7 val497_oracle7 7 scripts/run_val_variant.sh val497/val16_video.yaml external/external_val_oracle7_cap.json external/external_val_oracle7_sims.json 497
eval_stage goodtail7 val497_goodtail7 7 scripts/run_val_variant.sh val497/val16_video.yaml external/external_val_goodtail7_cap.json external/external_val_goodtail7_sims.json 497
eval_stage hardwrong7 val497_hardwrong7 7 scripts/run_val_variant.sh val497/val16_video.yaml external/external_val_hardwrong7_cap.json external/external_val_hardwrong7_sims.json 497
eval_stage easywrong7 val497_easywrong7 7 scripts/run_val_variant.sh val497/val16_video.yaml external/external_val_easywrong7_cap.json external/external_val_easywrong7_sims.json 497

# E. LOO (lineidx auto-fix first)
$PY scripts/fix_lineidx.py evcap_external/val150/*.tsv > logs/fix_lineidx.log 2>&1
for I in 0 1 2 3 4 5 6; do
  eval_stage loo$I val150_loo$I $I scripts/run_loo.sh val150/val150_video.yaml external/external_val_loo_rm${I}_cap.json external/external_val_loo_rm${I}_sims.json 150
done

# F. replicates (optional, retry once then skip)
for S in repB50k repC50k repD250k; do
  id=rep_$S; stage_done $id && continue; stage_start $id
  ok=0
  for try in 1 2; do
    if [ "$DRYRUN" = 1 ]; then echo "DRYRUN_PASS $id"; ok=1; break; fi
    if bash scripts/scale_track_chain.sh $S D 3 7 > logs/rep_$S.log 2>&1; then ok=1; break; fi
  done
  [ "$ok" = 1 ] && stage_finish $id || stage_fail $id
done
stage_finish robust_queue_done
echo "ROBUST_QUEUE_DONE"
