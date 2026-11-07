"""Phasera CLI: validates and launches the frozen, offline inference runtime."""
from pathlib import Path
import argparse,subprocess,sys

def main():
    root=Path(sys.argv[0]).resolve().parent
    p=argparse.ArgumentParser(description='Phasera KoMaP submission (Python + Linux CUDA for inference)')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('validate',help='Verify frozen weights, configuration and runtime hashes')
    pred=sub.add_parser('predict',help='Infer original PNG images with frozen settings')
    pred.add_argument('--input',required=True);pred.add_argument('--output',required=True)
    pred.add_argument('--threads',type=int,default=4)
    ev=sub.add_parser('evaluate',help='GT-present image-macro IoU; no fitting')
    ev.add_argument('--predictions',required=True);ev.add_argument('--masks',required=True);ev.add_argument('--output')
    a=p.parse_args()
    if a.command=='evaluate':
        cmd=[sys.executable,'-B',str(root/'code/evaluate.py'),'--predictions',a.predictions,'--masks',a.masks]
        if a.output:cmd+=['--output',a.output]
    else:
        if not (root/'model_bundle/manifest.json').is_file():
            p.error('Model bundle is missing. Extract A901005_weights.zip into the same folder as A901005.zip.')
        cmd=[sys.executable,'-B',str(root/'code/inference/predict.py'),'--model-dir',str(root/'model_bundle')]
        if a.command=='validate':cmd+=['--validate-only']
        else:cmd+=['--input',a.input,'--output',a.output,'--threads',str(a.threads)]
    raise SystemExit(subprocess.call(cmd))
if __name__=='__main__':main()
