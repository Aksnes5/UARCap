#!/bin/bash
cd /home/bn/videoufo_research
S=/home/bn/videoufo_research
PY=/home/bn/anaconda3/envs/videollama2/bin/python
VENV=/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python
GEN=eval_evcap_top7_bs1_load_video_feature_sim_score_cross_fusion_gate_not_share_linear_prompt5_review_generated_captions.json
MAN=$S/reports/post_fix/manifests; mkdir -p $MAN $S/reports/post_fix
COMMIT=$(git -C $S rev-parse HEAD)
run_cfg(){
  N="$1"; K="$2"; CAP="$3"; SIM="$4"; TT="${5:-0}"; VV="${6:-0}"; SCALE="$7"; TRACK="$8"
  OUT="$S/evcap_external/outputs/eval_postfix_${N}/${GEN}"
  STATUS=Skipped
  if [ -f "$OUT" ] && [ "$($PY -c "import json;print(len(json.load(open('$OUT'))))")" = "497" ] && [ -f "$S/evcap_external/outputs/eval_postfix_${N}/metrics.json" ]; then
    STATUS=Complete
  else
    if bash $S/scripts/run_postfix.sh "$N" "$K" val497/val16_video.yaml "$CAP" "$SIM" "$TT" "$VV" > $S/logs/postfix_${N}.log 2>&1; then
      ( cd $S/evcap_external && $VENV evaluation/cocoeval.py --result_file_path "outputs/eval_postfix_${N}/${GEN}" --eval_file "outputs/eval_postfix_${N}/metrics.json" > ../logs/postfix_metrics_${N}.log 2>&1 )
      STATUS=Complete
    else STATUS=Failed; fi
  fi
  $PY - "$N" "$K" "$SCALE" "$TRACK" "$STATUS" "$CAP" "$SIM" "$COMMIT" <<'PYEOF'
import json,sys,os,datetime
n,k,scale,track,status,cap,sim,commit=sys.argv[1:9]
root='/home/bn/videoufo_research'; mp=root+'/evcap_external/outputs/eval_postfix_'+n+'/metrics.json'
m=json.load(open(mp)) if os.path.exists(mp) else {}
json.dump({'name':n,'k':int(k),'scale':scale,'track':track,'status':status,
 'queries':497 if status=='Complete' else None,'cap':cap,'sims':sim,'commit':commit,
 'result_dir':'evcap_external/outputs/eval_postfix_'+n,
 'metrics':{kk:round(vv*100,4) for kk,vv in m.items()} if m else {},
 'finished_at':datetime.datetime.now().isoformat(timespec='seconds')},
 open(f'{root}/reports/post_fix/manifests/{n}.json','w'),indent=2)
PYEOF
  echo "$(date -Is) $N $STATUS"
}
echo "POSTFIX_P1_START commit=$COMMIT"
for SCL in 10k 50k 100k 250k 500k full; do
  if [ "$SCL" = full ]; then CFCAP=$S/external/external_val_cf7_cap.json; CFSIM=$S/external/external_val_cf7_sims.json; DCAP=$S/external/external_val_d7_cap.json; DSIM=$S/external/external_val_d7_sims.json
  else CFCAP=$S/external/external_val_${SCL}_cf7_cap.json; CFSIM=$S/external/external_val_${SCL}_cf7_sims.json; DCAP=$S/external/external_val_${SCL}_d7_cap.json; DSIM=$S/external/external_val_${SCL}_d7_sims.json; fi
  for K in 1 3 5 7; do
    run_cfg "postfix_${SCL}_CF${K}" $K "$CFCAP" "$CFSIM" 0 0 $SCL CF
    run_cfg "postfix_${SCL}_D${K}" $K "$DCAP" "$DSIM" 0 0 $SCL D
  done
  echo "SCALE_${SCL}_DONE"
done
# matched in-domain
for K in 1 3 5 7; do
  run_cfg "postfix_indomain_D${K}" $K $S/external/external_val_indomain_d7_cap.json $S/external/external_val_indomain_d7_sims.json 0 0 in-domain D
done
# Text x Visual (T3V3/T7V7 reused from FULL D3/D7)
run_cfg postfix_T7V3 7 $S/external/external_val_d7_cap.json $S/external/external_val_d7_sims.json 7 3 FULL D
run_cfg postfix_T3V7 7 $S/external/external_val_d7_cap.json $S/external/external_val_d7_sims.json 3 7 FULL D
echo "POSTFIX_P1_DONE"
