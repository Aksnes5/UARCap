from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file_path", type=str, help="the result file to evaluate")
    #by lijian
    parser.add_argument('--eval_file', default = 'outputs/eval_evcap/000.josn')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    cocoEval = COCOEvalCap(args.result_file_path)
    cocoEval.evaluate()
            
    for metric, score in cocoEval.eval.items():
        print ('%s: %.3f'%(metric, score))
        
    # by lijian
    outfile = args.eval_file
    print(f'cocoeval outfile: {outfile}')
    with open(outfile, 'w') as fp:
        json.dump(cocoEval.eval, fp, indent=4)