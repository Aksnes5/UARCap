# PRE_FIX_DIAGNOSTIC — superseded by FAITHFUL_V2

All results here were produced BEFORE the two confirmed preprocessing fixes:
1. MSR-VTT X-CLIP query preprocessing used cv2.INTER_LINEAR (official X-CLIP uses nearest).
2. VideoUFO Video-LLaMA fusion features used center-of-bin sampling (16 center frames),
   not the EVCap ffmpeg fps=16/duration or linspace(0,N-1,16) endpoint protocol.

Do NOT mix these numbers with FAITHFUL_V2 runs. Kept for provenance only.

Artifacts in place (paths relative to /home/bn/videoufo_research):
- embeddings/msrvtt_test_video_fp16.npy (pre-fix query video), msrvtt_test_imgfeat_fp16.npy
- external/external_sims_full.json, external_reranked_full.json, external_top20_full.json
- external/external_uar_{E,F,Fv,G,H,H1,H3,H5,H7,Ip,Ia}_cap/sims.json
- features/external_top7/ (7431 Video-LLaMA features, center sampling)
- evcap_external/outputs/eval_dev300_{CF,D,D10k,D100k,D500k,UAR*,A,B,no_retrieval}
- reports/phase2_scale_final.md, phase3_final.md, phase3_h5h7.md, oracle_diagnostic_table.md

Headline pre-fix numbers: CF7 64.38 | D7 63.25 | 100K 64.98 | 500K 63.31 | FULL 63.25 |
B 66.51 | A 64.18 | H1 65.12 | H3 67.63 | H5 65.41 | H7 64.01 (dev300 CIDEr x100).
