"""Build frozen intervention inputs locally, or copy and verify an existing complete replay."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile

DEPLOYMENT = Path(__file__).resolve().parent.parent
DESIGN = DEPLOYMENT.parent / 'uarcap_v06_followup_experiments'
sys.path.insert(0, str(DESIGN))
from prepare_inputs import primitive_hash, run

ARCHIVE = '52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec'
BN_RECEIPTS = 'd184360cb1cda832dbaa2d6ed9f46fb4aaa186e6'
SOURCE_PATHS = {
    'top20': 'external/external_test_top20.json',
    'cap': 'external/external_test_d7_cap.json',
    'sims': 'external/external_test_d7_sims.json',
    'full_ids': 'manifests/msrvtt_test2990.tsv',
    'test_ids': 'manifests/msrvtt_untouched2690.tsv',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return sha256(raw).hexdigest()


def archive_sources(repo, fingerprints):
    require(set(fingerprints) == set(SOURCE_PATHS), 'Unexpected source list')
    source = {}
    for name, path in SOURCE_PATHS.items():
        raw = subprocess.check_output(['git','--no-lazy-fetch','-C',str(repo),'cat-file','blob',f'{ARCHIVE}:{path}'])
        require(sha(raw) == fingerprints[name], f'Raw archive bytes changed: {path}')
        source[name] = raw
    return source


def manifest(raw):
    return raw.decode('utf-8-sig').splitlines()


def verify_prepared(prepared, expected_hashes, full, untouched):
    payload, records = {}, {}
    eligible_name = 'B_eligible_queries'
    for name, digest in expected_hashes.items():
        relative = f'{name}.json'
        raw = (prepared/relative).read_bytes()
        obj = json.loads(raw)
        require(primitive_hash(obj) == digest, f'Frozen content differs: {relative}')
        if name == eligible_name:
            allowed, present = set(untouched), set(obj)
            require(obj == [q for q in full if q in present] and present <= allowed,
                    'B eligibility order or membership differs')
        else:
            require(list(obj) == full, f'Prepared object query order differs: {relative}')
        payload[relative] = raw
        records[relative] = {'bytes':len(raw),'sha256':sha(raw),'canonical_sha256':digest}
    return payload, records


def required_ids(d7_cap, visual, full):
    require(all(visual[q]['representatives'] == [group[0] for group in d7_cap[q][2][:3]] for q in full),
            'Prepared B visual contract differs from the archived D prefix')
    d_ids = {k:sorted({group[0] for q in full for group in d7_cap[q][2][:k]}) for k in (1,7)}
    b_ids = sorted({clip for q in full for clip in visual[q]['representatives']})
    require(set(b_ids) <= set(d_ids[7]), 'B introduces a cache path outside the A7 inventory')
    return {'A0':[], 'A1':d_ids[1], 'A7':d_ids[7], 'B3':b_ids, 'B7HQ':b_ids, 'B7LQ':b_ids}


def write_bundle(repo, out, already_prepared):
    require(not out.exists() and not out.is_symlink(), 'Choose a new local output directory')
    frozen = json.loads((DESIGN/'CPU_PREFLIGHT_SUMMARY.json').read_bytes())
    archive = json.loads((DEPLOYMENT/'ARCHIVE_EVIDENCE.json').read_bytes())
    raw = archive_sources(repo, frozen['source_sha256'])
    full, untouched = (manifest(raw[name]) for name in ('full_ids','test_ids'))
    d7_cap = json.loads(raw['cap'])
    if already_prepared is None:
        require(sys.version_info >= (3,10), 'Run the frozen CPU builder under Python 3.10 or newer')
        with tempfile.TemporaryDirectory(prefix='uarcap_cpu_build_') as temp:
            work = Path(temp)
            summary = run(json.loads(raw['top20']),d7_cap,json.loads(raw['sims']),full,untouched,work)
            require(summary == {k:v for k,v in frozen.items() if k!='source_sha256'}, 'Frozen builder summary differs')
            payload, records = verify_prepared(work, frozen['inputs_fingerprints'], full, untouched)
    else:
        payload, records = verify_prepared(already_prepared, frozen['inputs_fingerprints'], full, untouched)

    visual = json.loads(payload['B_visual_contract.json'])
    feature_lists = required_ids(d7_cap, visual, full)
    features = {}
    for arm, ids in feature_lists.items():
        witness = archive['required_cache_by_arm'][arm]
        digest = primitive_hash(ids)
        require(len(ids) == witness['unique_cache_files'] and digest == witness['sorted_feature_ids_sha256'],
                f'Frozen expected feature IDs differ for {arm}')
        name = f'expected_feature_ids_{arm}.txt'
        body = ''.join(f'{clip}\n' for clip in ids).encode('utf-8')
        payload[name] = body
        features[name] = {'count':len(ids),'sha256':sha(body),'canonical_ids_sha256':digest}

    receipt = {
        'archive_commit':ARCHIVE,
        'bn_receipt_commit':BN_RECEIPTS,
        'prepared_mode':'EXISTING_REPLAY_VERIFIED_AND_COPIED' if already_prepared else 'FROZEN_CPU_BUILDER',
        'scope':'WINDOWS_LOCAL_ONLY',
        'source_files':{SOURCE_PATHS[k]:{'bytes':len(data),'sha256':sha(data)} for k,data in raw.items()},
        'prepared_files':records,
        'feature_lists':features,
        'run_queries':len(full),
        'A_scored_per_arm':len(untouched),
        'B_eligible':len(json.loads(payload['B_eligible_queries.json'])),
        'B_eligible_canonical_sha256':records['B_eligible_queries.json']['canonical_sha256'],
        'B_feature_ids_subset_of_A7':True,
        'reads_references_predictions_media_or_weights':False,
        'server_staged':False,
        'gpu_execution_approved':False,
    }
    out.mkdir()
    for name, data in payload.items():
        dst = out/name
        with dst.open('xb') as handle:
            handle.write(data)
        require(sha(dst.read_bytes()) == sha(data), f'Copy verification failed: {name}')
    (out/'PREPARED_INPUT_REPLAY_RECEIPT.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True, help='Local Git checkout with the pinned archive blobs')
    p.add_argument('--out', type=Path, required=True, help='Fresh local directory with an existing parent')
    p.add_argument('--reuse-prepared', type=Path, help='Previously generated local files to verify and copy')
    args = p.parse_args()
    receipt = write_bundle(args.repo, args.out, args.reuse_prepared)
    print(f"{receipt['prepared_mode']}: {len(receipt['prepared_files'])} JSON files, B eligible={receipt['B_eligible']}")


if __name__ == '__main__':
    main()
