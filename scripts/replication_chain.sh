#!/bin/bash
cd /home/bn/videoufo_research
while ! grep -q "PRIORITY_DONE" logs/priority_chain.log 2>/dev/null; do sleep 30; done
echo "REPLICATION_START"
for S in repB50k repC50k; do
  bash scripts/scale_track_chain.sh $S D 3 7
  echo "STAGE_${S}_DONE"
done
bash scripts/scale_track_chain.sh repD250k D 3 7
echo "STAGE_repD250k_DONE"
echo "REPLICATION_DONE"
