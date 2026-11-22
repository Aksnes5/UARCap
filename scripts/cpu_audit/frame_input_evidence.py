import sys, argparse, json, os, numpy as np
ROOT="/home/bn/videoufo_research"; sys.path.insert(0,os.path.join(ROOT,"evcap_external"))
os.chdir(os.path.join(ROOT,"evcap_external"))
from dataset.msrvtt.vl_dataloader import build_dataset
args=argparse.Namespace(img_res=224, patch_size=16, img_feature_dim=512, max_num_frames=16,
    use_asr=False, decoder_sampling_strategy='uniform', debug_speed=False,
    effective_batch_size=1, add_od_labels=False, on_memory=False)
out={"code_path":{"eval_start_end":"evcap_external/dataset/msrvtt/vision_language_tsv.py:338 start,end = 0, 0",
                  "decoder":"evcap_external/dataset/msrvtt/data_utils/video_decoder.py: pyav_decode uses start/end to set video_end_pts; temporal_sampling replicates the single decoded frame",
                  "frame_tsv_path":"get_frames_from_tsv ignores start/end (test/MSVD/dev300 use pre-extracted frame TSVs)"},"splits":{}}
for name,yaml in [("MSR-VTT val497","val497/val16_video.yaml"),
                  ("MSR-VTT test2990","test2990/test2990_16frames.yaml"),
                  ("MSVD test670","datasets/MSVD/test_16frames.yaml")]:
    ds=build_dataset(args,yaml,None,is_train=False)
    rows=[]
    for idx in [0,1,10]:
        it=ds[idx]; img=it["image"].numpy()
        d=[float(np.abs(img[i]-img[0]).mean()) for i in range(img.shape[0])]
        rows.append(dict(index=idx,image_id=it["image_id"],n_identical_to_frame0=int(sum(x<1e-9 for x in d)),
                         max_abs_diff=float(max(d)),mean_abs_diff=float(np.mean(d)),per_frame_diff_vs_frame0=[round(x,6) for x in d]))
    out["splits"][name]={"yaml":yaml,"img_field":ds.cfg.get("img"),"rows":rows}
json.dump(out,open(os.path.join(ROOT,"reports/val_frame_input_bug_evidence.json"),"w"),indent=2)
print(json.dumps(out,indent=2))
