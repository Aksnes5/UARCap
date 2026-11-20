#!/bin/bash
cd /home/bn/videoufo_research
while ! grep -q "QUEUE_DONE" logs/run_remaining_gpu_queue.log 2>/dev/null; do sleep 30; done
echo "PRIORITY_START"
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache \
HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false \
PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_external_features_v2.py /tmp/missing_union.txt > logs/features_priority.log 2>&1
echo "PRIORITY_FEATURES_DONE"

variant () {
  N="$1"; K="$2"; TT="${3:-0}"; VV="${4:-0}"
  OUT="evcap_external/outputs/eval_val497_${N}/${GEN}"
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "497" ]; then echo "SKIP_${N}"; else
    bash scripts/run_val_variant.sh "$N" "$K" "$TT" "$VV" > logs/gen_${N}.log 2>&1 || { echo "GEN_${N}_FAILED"; return; }
    echo "GEN_${N}_DONE"
  fi
  if [ ! -f "evcap_external/outputs/eval_val497_${N}/metrics.json" ]; then
    ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val497_${N}/${GEN}" --eval_file "outputs/eval_val497_${N}/metrics.json" > ../logs/metrics_${N}.log 2>&1 )
  fi
  echo "METRICS_${N}_DONE"
}

cp external/external_val_d7_cap.json external/external_val_T7V3_cap.json
cp external/external_val_d7_sims.json external/external_val_T7V3_sims.json
cp external/external_val_d7_cap.json external/external_val_T3V7_cap.json
cp external/external_val_d7_sims.json external/external_val_T3V7_sims.json
variant T7V3 7 7 3
variant T3V7 7 3 7

# Matched in-domain grid with FAITHFUL_V2 candidate features
if [ ! -f external/external_val_indomain_reranked.json ]; then
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/make_indomain_retrieval_val.py > logs/indomain_retrieval.log 2>&1 && echo "INDOMAIN_RETRIEVAL_DONE"
fi
$PY scripts/build_val_grid_scale.py indomain > logs/indomain_grid.log 2>&1
$PY - <<'PYEOF' > /tmp/indomain_missing.txt
import json,os
ROOT='/home/bn/videoufo_research'
C=json.load(open(f'{ROOT}/external/external_val_indomain_d7_cap.json'))
sel={l[0] for v in C.values() for l in v[2] if l}
have=set()
for root,_,files in os.walk(f'{ROOT}/features/videollama_faithful_v2'):
    for f in files:
        if f.endswith('.pt'):
            have.add(os.path.relpath(os.path.join(root,f),f'{ROOT}/features/videollama_faithful_v2')[:-3])
print('\n'.join(sorted(sel-have)))
PYEOF
NM=$(grep -c . /tmp/indomain_missing.txt || true); echo "INDOMAIN_FEATURES_MISSING $NM"
if [ "$NM" -gt 0 ]; then
  TORCH_HOME=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/torch_cache \
  HUGGINGFACE_HUB_CACHE=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/hf_cache \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false \
  PYTHONDONTWRITEBYTECODE=1 $PY -u scripts/extract_msrvtt_train_features.py /tmp/indomain_missing.txt > logs/indomain_features.log 2>&1
  echo "INDOMAIN_FEATURES_DONE"
fi
for K in 1 3 5 7; do
  OUT="evcap_external/outputs/eval_val497_indomain_D${K}/${GEN}"
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "497" ]; then echo "SKIP_indomain_D${K}"; else
    bash scripts/run_indomain.sh $K > logs/gen_indomain_D${K}.log 2>&1 && echo "GEN_indomain_D${K}_DONE"
  fi
  ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val497_indomain_D${K}/${GEN}" --eval_file "outputs/eval_val497_indomain_D${K}/metrics.json" > ../logs/metrics_indomain_D${K}.log 2>&1 )
  echo "METRICS_indomain_D${K}_DONE"
done

for K in 1 3 7; do variant oracle7 $K; done
variant posrev 7 0 0
variant postop3last 7 0 0
variant midtail7 7 0 0
variant lowtail7 7 0 0
variant goodtail7 7 0 0
variant hardwrong7 7 0 0
variant easywrong7 7 0 0

$PY scripts/verify_loo_assertions.py > logs/loo_assertions.log 2>&1
if grep -q PASS logs/loo_assertions.log; then
  for I in 0 1 2 3 4 5 6; do
    OUT="evcap_external/outputs/eval_val150_loo${I}/${GEN}"
    if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "150" ]; then echo "SKIP_loo${I}"; else
      bash scripts/run_loo.sh $I > logs/gen_loo${I}.log 2>&1 && echo "GEN_loo${I}_DONE"
    fi
    ( cd evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_val150_loo${I}/${GEN}" --eval_file "outputs/eval_val150_loo${I}/metrics.json" > ../logs/metrics_loo${I}.log 2>&1 )
    echo "METRICS_loo${I}_DONE"
  done
else
  echo "LOO_BLOCKED_ASSERTIONS"
fi
echo "PRIORITY_DONE"
