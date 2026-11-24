"""Drive all six deployment plans with real frozen data and no runtime credentials."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from audit_archives import CHECKPOINT, DESIGN, INPUTS, read, verify
from runtime_trace import frozen


def check(scratch):
    frozen_cpu = read(DESIGN/'CPU_PREFLIGHT_SUMMARY.json')
    files = {name:str((scratch/Path(path).name).resolve()) for name,path in INPUTS.items()}
    verify(all(frozen.digest(files[k]) == digest for k,digest in frozen_cpu['source_sha256'].items()),
           'Wrong data bytes for dry run')
    artifacts = {name:str((scratch/'prepared'/f'{name}.json').resolve()) for name in frozen_cpu['inputs_fingerprints']}
    for name,path in artifacts.items():
        verify(frozen.primitive_hash(read(Path(path))) == frozen_cpu['inputs_fingerprints'][name], 'Wrong prepared input')
    artifacts.update(frozen_D_cap=files['cap'],frozen_D_sims=files['sims'])
    with tempfile.TemporaryDirectory(prefix='uarcap_dry_') as tmp:
        root = Path(tmp)/'nonexistent_checkout'
        deploy = dict(run_id='archive_preflight', attempt=1, checkpoint_name=CHECKPOINT, device='cuda:0',
                      llm_model='UNVERIFIED', video_feature_dir='UNVERIFIED', server_data_ready=False,
                      test_yaml=str((scratch/'archived.yaml').resolve()), artifacts=artifacts)
        verify(frozen.digest(deploy['test_yaml'])==read(DESIGN/'FROZEN_SOURCE_BINDINGS.json')['archived_yaml_sha256'],
               'Archived YAML mismatch')
        cfg = Path(tmp)/'unapproved.json'
        cfg.write_text(json.dumps(deploy),encoding='utf-8')
        plans = {}
        for arm,(t,v,cap,sim) in frozen.ARMS.items():
            cmd = [sys.executable,'-B',str(Path(__file__).with_name('deploy_guard.py')),
                   '--evcap-root',str(root),'--deployment',str(cfg),'--archive-scratch',str(scratch),'--arm',arm]
            plan = json.loads(subprocess.check_output(cmd))
            verify(plan['mode']=='dry-run' and not plan['loads_model'], 'Unsafe launcher plan')
            flags = dict(zip(plan['argv'][::2],plan['argv'][1::2]))
            want = (7 if arm.startswith('B') else max(t,v), t, v)
            got = tuple(int(flags['--'+n]) for n in ('topn','text_topn','visual_topn'))
            verify(got == want and flags['--ext_path'] == artifacts[cap] and flags['--sim_scores_path'] == artifacts[sim],
                   'Unexpected arm command')
            plans[arm] = {'outer_text_visual_k':got,'artifact_keys':[cap,sim],
                          'output':Path(plan['out']).name,'checkpoint_slot':Path(plan['checkpoint_slot']).parent.name}
        verify(not root.exists(), 'Dry run created execution outputs')
        return {'mode':'dry-run','approval':'NO','loaded_model':False,'created_execution_outputs':False,'arms':plans}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-scratch',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(check(args.archive_scratch),indent=2))


if __name__=='__main__':
    main()
