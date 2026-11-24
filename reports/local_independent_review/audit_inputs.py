"""Verify and stage exact frozen inputs; never change the Git checkout or its artifacts."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
from zipfile import ZipFile

from audit_statistics import dump, read


REF = '52b5ed3a69e4b84a8e16ed747f141eefb5b1e2ec'
ZIP_HASH = '5f5eca2b28aff270cfdf99cb7b473515c546ad40018aa931e16c2334357450ff'
ARCHIVE = 'reports/post_fix/independent_audit_handoff/'
JAR = 'stanford-corenlp-3.4.1.jar'


def digest(b):
    return hashlib.sha256(b).hexdigest()


def get(repo, args):
    return subprocess.check_output(['git','-C',str(repo),*args])


def export(repo, target):
    out = target/'git'
    paths = get(repo,['ls-tree','-r','--name-only',REF]).decode().splitlines()
    chosen = [p for p in paths if ('pycocoevalcap/cider/' in p or p.endswith('/ptbtokenizer.py')
              or p.startswith('external/external_val_') and p.endswith(('_cap.json','_sims.json','_top20.json'))
              or p in ('external/external_test_d7_cap.json','external/external_msvd_d7_cap.json',
                       'reports/post_fix/AUDIT_MAIN_GRID_STATS.json','reports/post_fix/POST_FIX_FINAL_RESULTS_LEDGER.csv',
                       'reports/post_fix/POST_FIX_SCALE_BUDGET_RESULTS.json','reports/post_fix/POST_FIX_MECHANISM_STATS.json',
                       'reports/post_fix/POST_FIX_PRIORITY1_STATS.json','reports/post_fix/POST_FIX_LOO.json','reports/post_fix/AUDIT_ORACLE_ANALYSIS.json',
                       'reports/test_confirmatory.json','reports/msvd_stats.json')
              or ('/eval_test2690_' in p or '/eval_msvd_' in p) and p.endswith('.json'))]
    checks={}
    for p in chosen:
        data = get(repo,['cat-file','blob',f'{REF}:{p}'])
        dest = out/p
        dest.parent.mkdir(exist_ok=True,parents=True)
        dest.write_bytes(data)
        checks[p]={'bytes':len(data),'sha256':digest(data)}
    return checks


def unpack(path, dest):
    raw=path.read_bytes()
    if len(raw)!=11553117 or digest(raw)!=ZIP_HASH:
        raise ValueError('Unexpected ZIP identity')
    with ZipFile(io.BytesIO(raw)) as z:
        if z.testzip() is not None:
            raise ValueError('Failed ZIP CRC')
        files = {}
        for item in z.infolist():
            if item.is_dir():
                continue
            if not item.filename.startswith(ARCHIVE):
                raise ValueError('Unexpected ZIP prefix')
            path = item.filename.removeprefix(ARCHIVE)
            if path in files or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
                raise ValueError('Invalid ZIP membership')
            files[path]=item
        manifest='06_MANIFEST_AND_PROVENANCE/MANIFEST.json'
        manifest_csv=manifest.replace('.json','.csv')
        m = json.loads(z.read(files[manifest]))
        rows = {row['path']:row for row in m['rows']}
        csv_rows = {row['path']:row for row in csv.DictReader(io.StringIO(z.read(files[manifest_csv]).decode()))}
        expected = m['files']
        if len(rows)!=expected or len(m['rows'])!=expected or set(files)-set(rows)!={manifest,manifest_csv} or set(csv_rows)!=set(rows):
            raise ValueError('Manifest membership mismatch')
        if sum(x['size_bytes'] for x in rows.values())!=m['total_bytes']:
            raise ValueError('Manifest total bytes mismatch')
        for name,item in files.items():
            b=z.read(item)
            if name in rows:
                r=rows[name]
                if (len(b),digest(b))!=(r['size_bytes'],r['sha256']) or r['sha256']!=csv_rows[name]['sha256'] or len(b)!=int(csv_rows[name]['size_bytes']):
                    raise ValueError(f'Manifest mismatch: {name}')
            p=dest/name
            if not p.resolve().is_relative_to(dest.resolve()):
                raise ValueError('Target outside extraction folder')
            p.parent.mkdir(exist_ok=True,parents=True)
            p.write_bytes(b)
    return {'sha256':ZIP_HASH,'bytes':len(raw),'archive_members':len(z.infolist()),'files':len(files),'verified_manifest_records':expected,
            'manifest_total_bytes':m['total_bytes'],'manifest_files':{k:r['sha256'] for k,r in rows.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','data','manifest','jar','metadata','predecessor','draft'):
        parser.add_argument('--'+name,type=Path,required=name!='metadata')
    args=parser.parse_args()
    args.data.mkdir(exist_ok=True,parents=True)
    archive=args.data/'handoff.zip'
    if not archive.is_file():
        archive.write_bytes(get(args.repo,['cat-file','blob',f'{REF}:{ARCHIVE}UARCap_POSTFIX_INDEPENDENT_AUDIT_HANDOFF.zip']))
    result={'base_commit':REF,'archive':unpack(archive,args.data/'extracted'),'git_objects':export(args.repo,args.data)}
    result['local_dependencies']={'ptb_jar':{'name':JAR,'sha256':digest(args.jar.read_bytes())},
                                  'predecessor':{'name':args.predecessor.name,'sha256':digest(args.predecessor.read_bytes())},
                                  'manuscript':{'name':args.draft.name,'sha256':digest(args.draft.read_bytes())}}
    if args.metadata:
        source='https://huggingface.co/datasets/friedrichor/MSR-VTT/blob/main/raw_data/MSRVTT_data.json'
        result['local_dependencies']['msrvtt_source_metadata']={'source':source,'sha256':digest(args.metadata.read_bytes())}
    dump(args.manifest,result)


if __name__=='__main__':
    main()
