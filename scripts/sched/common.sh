#!/bin/bash
ROOT=/home/bn/videoufo_research
ST=$ROOT/logs/stages2
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
mark() { echo "$(date -Is) $1 $2" >> $ST/events.log; }
stage_done(){ [ -f "$ST/$1.done" ]; }
stage_start(){ rm -f "$ST/$1.failed"; touch "$ST/$1.running"; }
stage_finish(){ rm -f "$ST/$1.running"; touch "$ST/$1.done"; mark "$1" done; }
stage_fail(){ rm -f "$ST/$1.running"; touch "$ST/$1.failed"; mark "$1" failed; }

# wait for a free GPU lock; stale holders cannot exist because flock dies with process,
# but a live non-GPU holder is bounded by timeout
gpu_run(){
  # runners own the GPU lock; do not double-lock here (nested flock deadlocks)
  "$@"
}

preflight_eval(){
  local yaml="$1" cap="$2" sims="$3"
  case "$yaml" in /*) ;; *) yaml="/home/bn/videoufo_research/evcap_external/$yaml";; esac
  [ -f "$yaml" ] || { echo "missing yaml $yaml" >&2; return 1; }
  /home/bn/anaconda3/envs/videollama2/bin/python - "$yaml" <<'PYEOF'
import sys,yaml,os
y=yaml.safe_load(open(sys.argv[1]))
for k in ('img','label','caption'):
    p=y.get(k)
    if not p: print('missing key',k); sys.exit(1)
    if not os.path.isabs(p): p=os.path.join(os.path.dirname(sys.argv[1]),p)
    if not os.path.exists(p): print('missing',p); sys.exit(1)
    if not os.path.exists(p+'.lineidx'): print('NO_LINEIDX',p)
PYEOF
  [ -f "$cap" ] || { echo "missing cap $cap" >&2; return 1; }
  [ -f "$sims" ] || { echo "missing sims $sims" >&2; return 1; }
  return 0
}
