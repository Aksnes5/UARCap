#!/bin/bash
cd /home/bn/videoufo_research
while kill -0 1492718 2>/dev/null; do sleep 30; done
echo "SCALE_FEATURES_DONE"
bash scripts/scale_grid_chain.sh
echo "SCALE_DONE_ALL"
bash scripts/test_grid_chain.sh
echo "MASTER_DONE"
