"""Verify the frozen transfer ZIP, optionally unpacking it into a new directory."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import stat
from zipfile import ZipFile


INPUTS = (
    'A_k0_cap.json','A_k0_sims.json','B_text3_cap.json','B_text7_hq_cap.json','B_text7_lq_cap.json',
    'B_shared_visual_sims.json','B_visual_contract.json','B_eligible_queries.json',
)
ARMS = ('A0','A1','A7','B3','B7HQ','B7LQ')
SUPPORT = ('CPU_PREFLIGHT_SUMMARY.json','PREPARED_INPUT_REPLAY_RECEIPT.json',
           'INPUT_FILE_MANIFEST.json','README_TRANSFER.md')
FEATURES = tuple(f'expected_feature_ids_{arm}.txt' for arm in ARMS)
ZIP_LIMIT = 16 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def canonical(data):
    return digest(json.dumps(data,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode('utf-8'))


def verify(archive, manifest, full_ids, test_ids):
    bundle = manifest['bundle']
    size = archive.stat().st_size
    require(0 < size <= ZIP_LIMIT and size == bundle['bytes'], 'ZIP byte size differs')
    # Check and unpack the same bytes, even if the original path changes during verification.
    from io import BytesIO
    raw_archive = archive.read_bytes()
    require(len(raw_archive) == size and digest(raw_archive) == bundle['sha256'], 'ZIP SHA-256 differs')
    contents = manifest['members']
    names = set(INPUTS + SUPPORT + FEATURES)
    require(set(contents) == names, 'Unexpected manifest membership')
    data = {}
    with ZipFile(BytesIO(raw_archive)) as z:
        items = z.infolist()
        require(len(items) == len(names) and {i.filename for i in items} == names, 'ZIP has missing, duplicate or unexpected members')
        require(sum(i.file_size for i in items) <= ZIP_LIMIT, 'Unexpected uncompressed ZIP size')
        for entry in items:
            name = entry.filename
            fmt = stat.S_IFMT(entry.external_attr >> 16)
            require(not entry.is_dir() and not entry.flag_bits & 1 and fmt in (0, stat.S_IFREG),
                    f'Unexpected member type: {name}')
            record = contents[name]
            require(entry.file_size == record['bytes'], f'Wrong member size: {name}')
            raw = z.read(entry)
            require(digest(raw) == record['sha256'], f'Wrong member SHA-256: {name}')
            data[name] = raw

    frozen = json.loads(data['CPU_PREFLIGHT_SUMMARY.json'])
    replay = json.loads(data['PREPARED_INPUT_REPLAY_RECEIPT.json'])
    embedded = json.loads(data['INPUT_FILE_MANIFEST.json'])
    require(embedded == {k:v for k,v in contents.items() if k!='INPUT_FILE_MANIFEST.json'},
            'Embedded member manifest differs')
    require(replay['prepared_mode']=='EXISTING_REPLAY_VERIFIED_AND_COPIED', 'Wrong preparation provenance')
    ids, untouched = full_ids.read_bytes(), test_ids.read_bytes()
    for key, raw in (('full_ids',ids), ('test_ids',untouched)):
        require(digest(raw) == frozen['source_sha256'][key] == manifest['frozen_source_sha256'][key],
                f'Frozen query file differs: {key}')
    full = ids.decode('utf-8-sig').splitlines()
    test = untouched.decode('utf-8-sig').splitlines()
    test_set = set(test)
    require(len(full)==len(set(full))==2990 and len(test)==len(test_set)==2690 and [q for q in full if q in test_set]==test,
            'Unexpected query cohorts or order')
    for name in INPUTS:
        meta = replay['prepared_files'][name]
        value = json.loads(data[name])
        require((len(data[name]), digest(data[name])) == (meta['bytes'], meta['sha256']), f'Replay bytes disagree: {name}')
        require(canonical(value) == meta['canonical_sha256'] == frozen['inputs_fingerprints'][Path(name).stem],
                f'Frozen object digest differs: {name}')
        if name == 'B_eligible_queries.json':
            eligible = set(value)
            require(len(value)==len(eligible)==2618 and value==[q for q in test if q in eligible]
                    and canonical(value)==frozen['eligible_ids_sha256'], 'Frozen B eligibility differs')
        else:
            require(list(value)==full, f'Incorrect per-query order: {name}')
    for name in FEATURES:
        items = data[name].decode('utf-8').splitlines()
        meta = replay['feature_lists'][name]
        require(items == sorted(set(items)) and
                (digest(data[name]), canonical(items), len(items)) ==
                (meta['sha256'], meta['canonical_ids_sha256'], meta['count']), f'Wrong feature ID list: {name}')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('zip','manifest','full-ids','test-ids'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--out',type=Path,help='Fresh output directory. Omit to verify only.')
    a = p.parse_args()
    data = verify(a.zip,json.loads(a.manifest.read_bytes()),a.full_ids,a.test_ids)
    if a.out:
        a.out.mkdir(exist_ok=False)
        for name, raw in data.items():
            with (a.out/name).open('xb') as handle:
                handle.write(raw)
            require(digest((a.out/name).read_bytes()) == digest(raw), f'Written bytes differ: {name}')
    print('Verified ZIP, all members and frozen query order: 2990 full, 2690 untouched, 2618 B-eligible.')


if __name__ == '__main__':
    main()
