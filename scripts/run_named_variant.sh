#!/bin/bash
set -e
N="$1"; K="$2"
exec bash /home/bn/videoufo_research/scripts/run_val_variant.sh "$N" "$K" 0 0
