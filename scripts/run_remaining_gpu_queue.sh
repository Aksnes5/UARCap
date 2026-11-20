#!/bin/bash
# Remaining GPU queue: D5 validation -> MSVD -> 50K D -> 250K D -> CF scale grids.
cd /home/bn/videoufo_research
PY=/home/bn/anaconda3/envs/videollama2/bin/python
Q=logs/queue_state; mkdir -p $Q
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json

stage_done() { [ -f "$Q/$1.done" ]; }
mark() { echo "$(date -Is) $1" >> $Q/stages.log; touch "$Q/$1.done"; }

# Stage 0: validate D5 generation (2990 rows, unique, nonempty)
if ! stage_done d5_validated; then
  while :; do
    F="evcap_external/outputs/eval_test2990_D5/${GEN}"
    if [ -f "$F" ]; then
      if $PY - <<PYEOF
import json,sys
d=json.load(open("$F"))
ids=[x["image_name"] for x in d]
ok=len(d)==2990 and len(set(ids))==2990 and all(x.get("prediction","").strip() for x in d)
sys.exit(0 if ok else 1)
PYEOF
      then mark d5_validated; break; else echo "D5_INTEGRITY_FAILED"; sleep 60; fi
    fi
    sleep 30
  done
fi
touch $Q/d5_ok
echo "D5_VALIDATED"

# Stage 1: MSVD frozen cross-dataset validation
if ! stage_done msvd; then
  bash scripts/msvd_chain.sh
  mark msvd
fi
echo "STAGE_MSVD_DONE"

# Stage 2/3: 50K / 250K D grids
for S in 50k 250k; do
  if ! stage_done scale_${S}_D; then
    bash scripts/scale_track_chain.sh $S D 1 3 5 7
    mark scale_${S}_D
  fi
  echo "STAGE_${S}_D_DONE"
done

# Stage 4: CF grids at 10k/100k/500k (FULL CF already done on val497)
for S in 10k 100k 500k; do
  if ! stage_done cf_${S}; then
    bash scripts/scale_track_chain.sh $S CF 1 3 5 7
    mark cf_${S}
  fi
  echo "STAGE_CF_${S}_DONE"
done

# Stage 5: 50K / 250K CF if GPU still available
for S in 50k 250k; do
  if ! stage_done cf_${S}; then
    bash scripts/scale_track_chain.sh $S CF 1 3 5 7
    mark cf_${S}
  fi
  echo "STAGE_CF_${S}_DONE"
done
echo "QUEUE_DONE"
