#!/bin/bash
cd /home/bn/videoufo_research
while kill -0 1757445 2>/dev/null; do sleep 20; done
echo "D3_EVAL_EXITED"
bash scripts/scale_grid_chain.sh
echo "SCALE_DONE_ALL"
bash scripts/test_grid_chain.sh
echo "MASTER_DONE"
