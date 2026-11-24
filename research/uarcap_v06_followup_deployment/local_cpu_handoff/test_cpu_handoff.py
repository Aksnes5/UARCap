"""Verify reuse guards and frozen frame routing without importing Torch or opening media."""

import ast
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from build_prepared_inputs import ARCHIVE, DESIGN, archive_sources, primitive_hash, required_ids, verify_prepared

REPO = Path(__file__).resolve().parents[3]
PREF = json.loads((DESIGN/'CPU_PREFLIGHT_SUMMARY.json').read_bytes())
LOADER = 'evcap_external/dataset/msrvtt/vision_language_tsv.py'
PRE_FIX = '7ecc6c5'


class HandoffTests(unittest.TestCase):
    def test_wrong_raw_source_hash_is_rejected(self):
        expected = dict(PREF['source_sha256'],top20='0'*64)
        with self.assertRaisesRegex(ValueError, 'Raw archive bytes changed'):
            archive_sources(REPO, expected)

    def test_reuse_preserves_bytes_and_rejects_digest_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            obj = {'q':[[],[],[]]}
            raw = b'{"q": [[], [], []]}\r\n'
            (path/'A_k0_cap.json').write_bytes(raw)
            payload, _ = verify_prepared(path, {'A_k0_cap':primitive_hash(obj)}, ['q'], ['q'])
            self.assertEqual(payload['A_k0_cap.json'], raw)
            with self.assertRaisesRegex(ValueError, 'Frozen content differs'):
                verify_prepared(path, {'A_k0_cap':'0'*64}, ['q'], ['q'])

    def test_reuse_rejects_nonfrozen_order_even_with_same_object_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            obj = {'q2':[[],[]], 'q1':[[],[]]}
            (path/'A_k0_sims.json').write_text(json.dumps(obj),encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'query order differs'):
                verify_prepared(path, {'A_k0_sims':primitive_hash(obj)}, ['q1','q2'], ['q1'])

    def test_feature_contract_pins_representatives_and_subset(self):
        caps = {'q':[['a','b','c','d'],[],[['A','alternate'],['B'],['C'],['D']]]}
        out = required_ids(caps, {'q':{'representatives':['A','B','C']}}, ['q'])
        self.assertEqual(out['A0'], [])
        self.assertEqual(out['B3'], out['B7HQ'])
        self.assertLess(set(out['B3']), set(out['A7']))
        with self.assertRaisesRegex(ValueError, 'visual contract differs'):
            required_ids(caps, {'q':{'representatives':['alternate','B','C']}}, ['q'])

    def test_preextracted_rows_bypass_raw_decoder_before_and_after_fix(self):
        for revision, window in ((PRE_FIX, (0,0)), (ARCHIVE, (None,None))):
            raw = subprocess.check_output(['git','--no-lazy-fetch','-C',str(REPO),'cat-file','blob',f'{revision}:{LOADER}'])
            tree = ast.parse(raw)
            cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VisionLanguageTSVDataset')
            method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_visual_data')
            scope = {}
            exec(compile(ast.Module(body=[method],type_ignores=[]),'<pinned-source>','exec'),scope)
            visual = scope['get_visual_data']
            frames = [f'frame-{n}' for n in range(16)]
            q = 'query.mp4'
            for row, expected in (([q], ('decoder', (q,*window))), ([q,'16',*frames], ('stored',frames))):
                dataset = SimpleNamespace(
                    visual_tsv={0:row}, decoder_num_frames=16, get_row_from_tsv=lambda t,i:t[i],
                    decode_and_get_frames=lambda *args:('decoder',args), get_frames_from_tsv=lambda p:('stored',p),
                )
                with self.subTest(revision=revision, row_width=len(row)):
                    self.assertEqual(visual(dataset,0,*window), (expected,True))


if __name__ == '__main__':
    unittest.main()
