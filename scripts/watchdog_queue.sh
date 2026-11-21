#!/bin/bash
# Stage-stall watchdog: 8 consecutive checks (~16 min) with GPU util <5%
# while a stage is marked running => kill its generation process, fail stage,
# let the queue continue.
cd /home/bn/videoufo_research
idle=0
while true; do
  running=$(ls logs/stages2/*.running 2>/dev/null | head -1)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  if [ -n "$running" ] && [ "${util:-0}" -lt 5 ]; then idle=$((idle+1)); else idle=0; fi
  if [ "$idle" -ge 8 ]; then
    echo "$(date -Is) STALL_HUNG $running gpu_util=$util" >> logs/queue_stalls.log
    pkill -f "eval_evcap_fix_video_llama" 2>/dev/null
    pkill -f "extract_msrvtt_train_features" 2>/dev/null
    stage=$(basename "$running" .running)
    rm -f "$running"; touch "logs/stages2/$stage.failed"
    idle=0
  fi
  sleep 120
done
