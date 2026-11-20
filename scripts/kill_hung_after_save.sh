#!/bin/bash
# Kill a generation python only if its output file is complete (670) and older than 180 s.
cd /home/bn/videoufo_research
while true; do
  for K in 1 3 5 7; do
    F="evcap_external/outputs/eval_msvd_D${K}/eval_evcap_top3_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_msvd_generated_captions.json"
    [ -f "$F" ] || continue
    AGE=$(( $(date +%s) - $(stat -c %Y "$F") ))
    [ "$AGE" -lt 180 ] && continue
    N=$(/home/bn/anaconda3/envs/videollama2/bin/python -c "import json;print(len(json.load(open('$F'))))" 2>/dev/null)
    [ "$N" = "670" ] || continue
    PIDS=$(ps -eo pid,args | grep "[e]val_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5_msvd.py" | grep "eval_msvd_D${K}" | awk '{print $1}')
    for p in $PIDS; do echo "$(date -Is) killing hung PID $p (D${K} saved, age ${AGE}s)" >> logs/kill_hung.log; kill $p; done
  done
  sleep 60
done
