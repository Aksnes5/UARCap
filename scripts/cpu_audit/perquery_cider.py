import json, sys, os, itertools, statistics
sys.path.insert(0, "/home/bn/videoufo_research/evcap_external/evaluation")
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider

def load(path):
    return json.load(open(path))

def scores_for(data):
    gts = {it['image_name']: [{'caption': c} for c in it['captions']] for it in data}
    res = {it['image_name']: [{'caption': it['prediction']}] for it in data}
    tok = PTBTokenizer()
    gts = tok.tokenize(gts); res = tok.tokenize(res)
    _, per = Cider().compute_score(gts, res)
    return gts, res, per

if __name__ == '__main__':
    import glob
    for path in sys.argv[1:]:
        data = load(path)
        gts,res,per = scores_for(data)
        per = [float(x) for x in per]
        predlens = [len(it['prediction'].split()) for it in data]
        reflens = [statistics.mean(len(c.split()) for c in it['captions']) for it in data]
        print(os.path.basename(os.path.dirname(path)))
        print(' n=%d CIDEr_mean=%.4f median=%.4f sd=%.4f q10=%.4f q25=%.4f q75=%.4f q90=%.4f min=%.4f max=%.4f' % (
            len(data), statistics.mean(per), statistics.median(per), statistics.pstdev(per),
            sorted(per)[int(.1*len(per))], sorted(per)[int(.25*len(per))],
            sorted(per)[int(.75*len(per))], sorted(per)[int(.9*len(per))], min(per), max(per)))
        print(' pred_len mean=%.3f med=%.1f  ref_len mean=%.3f' % (
            statistics.mean(predlens), statistics.median(predlens), statistics.mean(reflens)))
        print(' zero_frac=%.4f  <0.1=%.4f  >1.0=%.4f' % (
            sum(x<1e-9 for x in per)/len(per), sum(x<0.1 for x in per)/len(per), sum(x>1 for x in per)/len(per)))
