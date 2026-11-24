"""Fail closed on trace and artifact mismatches before emitting a gate or scorer manifest."""

import argparse
import json
import math
from pathlib import Path

from audit_archives import DESIGN, HERE, read, sha, verify
from deploy_guard import CP_HASH, EXECUTION_KEYS, SCOPE, SMOKE_LIMIT, implementation_hashes
from runtime_trace import A_EXTENDED_PINS, EXTENDED_PINS, frozen, vision_py39
from evaluate_followup import FAMILIES, check_trace

ZERO_AGGREGATE = {'shape': [1, 32, 768], 'dtype': 'torch.float32', 'sha256': sha(bytes(32 * 768 * 4))}


def check_rows(traces, order, family, caps, sims, eligible, cuda_expected=False):
    check_trace(traces, order, family)
    base = FAMILIES[family]['arms'][0]
    pins = EXTENDED_PINS if family == 'B' else A_EXTENDED_PINS
    for arm, rows in traces.items():
        for i, row in enumerate(rows):
            q = order[i]
            t, v, _, _ = frozen.ARMS[arm]
            if arm in ('B7HQ', 'B7LQ') and q not in eligible:
                t = 3
            tag = f'{arm}:{q}'
            verify(row['schema_version'] == 'uarcap-v06-deploy-trace-1' and type(row['training']) is bool,
                   f'{tag}: wrong trace schema')
            rng = row['encoder_rng']
            verify(isinstance(rng, dict) and isinstance(rng.get('cpu'), dict) and 'cuda' in rng and
                   (isinstance(rng['cuda'], dict) if cuda_expected else rng['cuda'] is None),
                   f'{tag}: missing device RNG signature')
            verify(all(row[k] == traces[base][i][k] for k in pins), f'{tag}: extended tensor parity')
            verify(row['text_count'] == t and row['consumed_text_sha256'] == frozen.primitive_hash(caps[arm][q][0][:t]),
                   f'{tag}: wrong prompt evidence')
            verify(row['vision'] == vision_py39(caps[arm][q], sims[arm][q], v), f'{tag}: wrong clips or raw visual scores')
            verify(len(row['feature_tensors']) == v and all(x['shape'] == [32, 768] for x in row['feature_tensors']),
                   f'{tag}: wrong feature load count or shape')
            verify(row['attn_key'] == row['attn_value'] == row['aggregate'] and row['aggregate']['shape'] == [1, 32, 768],
                   f'{tag}: wrong attention K/V')
            if v == 0:
                verify(row['aggregate'] == ZERO_AGGREGATE and row['softmax_weights'] is None, f'{tag}: A0 is not zero')
            else:
                raw = row['vision']['similarities']
                exp = [math.exp(x - max(raw)) for x in raw]
                weights = row['softmax_weights']
                total = sum(exp)
                verify(len(weights) == v and all(abs(w - e / total) < 1e-6 for w, e in zip(weights, exp)),
                       f'{tag}: unexpected consumed softmax weights')
            verify(row['visual_mask']['shape'] == [1, 32] and row['visual_mask']['dtype'] == 'torch.int64',
                   f'{tag}: visual mask differs')
            verify(row['prompt_mask']['shape'] == [1, row['prompt_tokens']] and row['prompt_mask']['dtype'] == 'torch.int32'
                   and row['wrapped_prompt']['shape'][:2] == row['prompt_mask']['shape'], f'{tag}: invalid prompt schema')


def inspect_bundle(family, mode, arm_dirs, scratch):
    pre = read(DESIGN/'CPU_PREFLIGHT_SUMMARY.json')
    frozen_code = read(DESIGN/'FROZEN_SOURCE_BINDINGS.json')['code_sha256']
    archive = read(HERE/'ARCHIVE_EVIDENCE.json')
    ids = scratch/'msrvtt_test2990.tsv'
    verify(frozen.digest(ids) == pre['source_sha256']['full_ids'], 'Wrong full query list')
    full = ids.read_text(encoding='utf-8').splitlines()
    order = full[:SMOKE_LIMIT] if mode == 'smoke' else full
    refs = read(scratch/'reference_witness.json')
    verify(frozen.primitive_hash(refs) == archive['baseline_predictions']['1']['full_ref_lists_canonical_sha256'],
           'Archived references differ')
    eligible = read(scratch/'prepared'/'B_eligible_queries.json')
    verify(frozen.primitive_hash(eligible) == pre['eligible_ids_sha256'], 'Wrong B eligibility')
    traces, caps, sims, files = {}, {}, {}, {}
    first = None
    verify(set(arm_dirs) == set(FAMILIES[family]['arms']), 'Wrong arm directory set')
    for arm, folder in arm_dirs.items():
        folder = Path(folder).resolve()
        seal = read(folder/'arm_completed.json')
        verify(seal['scope'] == SCOPE and seal['mode'] == mode and seal['arm'] == arm and seal['rows'] == len(order),
               f'{arm}: invalid completion marker')
        verify(Path(seal['predictions']).name == seal['predictions'], 'Invalid prediction filename')
        paths = {'predictions': folder/seal['predictions'], 'trace': folder/'input_trace.jsonl',
                 'receipt': folder/'run_receipt.json'}
        for kind, digest in (('receipt', 'receipt_sha256'), ('trace', 'trace_sha256'), ('predictions', 'prediction_sha256')):
            verify(frozen.digest(paths[kind]) == seal[digest], f'{arm}: {kind} changed after completion')
        rec = read(paths['receipt'])
        verify(rec['scope'] == SCOPE and rec['mode'] == mode and rec['arm'] == arm and rec['code_sha256'] == frozen_code
               and rec['engine_args'] == frozen.BASE_ARGS, f'{arm}: runtime receipt differs from frozen code or arguments')
        common = rec['common']
        verify(common['run_id'].startswith(mode + '_') and common['checkpoint_sha256'] == CP_HASH
               and rec['execution_inputs'] == frozen.primitive_hash({k: common[k] for k in EXECUTION_KEYS}),
               f'{arm}: incorrect runtime input binding')
        verify(rec['deployment_code_sha256'] == implementation_hashes(), f'{arm}: different instrumentation version')
        first = rec if first is None else first
        verify(all(first[k] == rec[k] for k in ('common', 'runtime', 'execution_inputs', 'deployment_code_sha256')),
               f'{arm}: mixed runtime bundle')
        _, _, cap_key, sim_key = frozen.ARMS[arm]
        for name, target in ((cap_key, caps), (sim_key, sims)):
            if name.startswith('frozen_D'):
                key = 'cap' if name.endswith('cap') else 'sims'
                path = scratch/f'external_test_d7_{key}.json'
                verify(frozen.digest(path) == pre['source_sha256'][key], f'{arm}: wrong archived input')
            else:
                path = scratch/'prepared'/f'{name}.json'
                verify(frozen.primitive_hash(read(path)) == pre['inputs_fingerprints'][name], f'{arm}: wrong prepared input')
            verify(frozen.digest(path) == rec['arm_artifact_sha256'][name], f'{arm}: deployed input bytes differ')
            target[arm] = read(path)
        preds = read(paths['predictions'])
        verify([r['image_name'] for r in preds] == order and [r['captions'] for r in preds] == refs[:len(order)]
               and all(isinstance(r['prediction'], str) and r['prediction'].strip() for r in preds),
               f'{arm}: invalid prediction order, content or references')
        traces[arm] = [json.loads(line) for line in paths['trace'].read_text(encoding='utf-8').splitlines()]
        files[arm] = {k:str(v) for k,v in paths.items()}
    check_rows(traces, order, family, caps, sims, set(eligible), cuda_expected=True)
    return {'scope':SCOPE, 'mode':mode, 'family':family, 'trace_parity':True, 'run_id':first['common']['run_id'],
            'full_ids_sha256':pre['source_sha256']['full_ids'], 'execution_inputs':first['execution_inputs'],
            'runtime':first['runtime'], 'deployment_code_sha256':first['deployment_code_sha256'],
            'rows_per_arm':len(order),
            'validated_files_sha256':{arm:{k:frozen.digest(path) for k,path in group.items()} for arm,group in files.items()}}, files


def write_new(path, obj):
    with path.open('x', encoding='utf-8') as out:
        json.dump(obj, out, indent=2, ensure_ascii=False)
        out.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=FAMILIES, required=True)
    p.add_argument('--mode', choices=('smoke','full'), required=True)
    for key in ('arm-directories','archive-scratch','gate-out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--scorer-manifest', type=Path)
    a = p.parse_args()
    verify(not a.scorer_manifest or a.mode == 'full', 'Smoke may never produce a scorer manifest')
    report, files = inspect_bundle(a.family, a.mode, read(a.arm_directories), a.archive_scratch)
    write_new(a.gate_out, report)
    if a.scorer_manifest:
        write_new(a.scorer_manifest, files)


if __name__ == '__main__':
    main()
