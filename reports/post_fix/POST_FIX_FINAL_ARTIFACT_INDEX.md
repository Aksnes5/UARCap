# POST_FIX artifact index

## Primary report
- reports/post_fix/POST_FIX_FINAL_COMPREHENSIVE_RESEARCH_REPORT.md

## Auxiliary
- POST_FIX_FINAL_EXECUTIVE_SUMMARY.md
- POST_FIX_CLAIM_EVIDENCE_MATRIX.md
- POST_FIX_PREFX_TO_POSTFIX_CLAIM_AUDIT.md
- POST_FIX_REVIEWER_CRITICAL_ISSUES.md
- POST_FIX_PAPER_REVISION_ROADMAP.md
- POST_FIX_FINAL_ARTIFACT_INDEX.md (this file)

## Results and statistics
- POST_FIX_SCALE_BUDGET_RESULTS.{md,csv,json}
- POST_FIX_PRIORITY1_RESULTS.md / POST_FIX_PRIORITY1_STATS.json
- POST_FIX_TEXT_VISUAL.md
- POST_FIX_INDOMAIN_INTERACTION.md
- POST_FIX_ORACLE.md
- POST_FIX_MECHANISM_RESULTS.md / POST_FIX_MECHANISM_STATS.json
- POST_FIX_LOO.md / POST_FIX_LOO.json
- CANDIDATE_CHURN_10K_50K_100K.md / candidate_churn_10k_vs_50k_100k.json
- PRE_FIX_VS_POST_FIX_COMPARISON.md
- POST_FIX_FINAL_RESULTS_LEDGER.csv
- POST_FIX_FRAME_SANITY_REPORT.md / val497_frame_sanity.{json,csv} / frames/*.png

## Manifests
- reports/post_fix/manifests/*.json (71 formal configs; smoke excluded)

## Scripts
- scripts/postfix_frame_sanity.py
- scripts/run_postfix.sh
- scripts/postfix_priority1_queue.sh
- scripts/postfix_priority23_queue.sh
- scripts/postfix_priority1_stats.py
- scripts/postfix_mechanism_stats.py

## Raw outputs
- evcap_external/outputs/eval_postfix_*/...generated_captions.json
- evcap_external/outputs/eval_postfix_*/metrics.json

## Fix commit
- `2b3da9d` fix: val497 eval frame loader decodes whole video (POST_FIX)
