#!/bin/bash
cd /home/bn/videoufo_research
while [ ! -f external/external_val_indomain_reranked.json ]; do sleep 30; done
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
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
NM=$(grep -c . /tmp/indomain_missing.txt || true); echo "INDOMAIN_MISSING $NM"
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
echo "INDOMAIN_DONE"
