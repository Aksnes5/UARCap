#!/usr/bin/env python3
import json,sys
sys.path.insert(0,"/home/bn/uarcap_q2_carebench/research/uarcap_q2_experiments")
from uarcap_q2.final_analysis.pipelines import prepare_e3_bridge_approved
r=prepare_e3_bridge_approved()
print(json.dumps({k:r[k] for k in ("status","n_approved","min_required","eligible")},indent=2))
raise SystemExit(0 if r["eligible"] else 3)
