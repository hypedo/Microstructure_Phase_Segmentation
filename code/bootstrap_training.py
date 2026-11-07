"""Explicitly prepare Oct09 warm start and data layout; does not train."""
import argparse,json,shutil,hashlib
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,help='Directory with train/images, train/masks, valid/images, valid/masks')
    ap.add_argument('--bundle',default=str(Path(__file__).resolve().parents[1]/'model_bundle'));a=ap.parse_args()
    root=Path(__file__).resolve().parent;bundle=Path(a.bundle).resolve();data=Path(a.data).resolve()
    for split,n in [('train',70),('valid',20)]:
        assert len(list((data/split/'images').glob('*_image.png')))==n
        assert len(list((data/split/'masks').glob('*_mask.png')))==n
    dest=root/'data'
    if dest.exists():assert dest.resolve()==data,'Existing data path differs'
    else:dest.symlink_to(data,target_is_directory=True)
    manifest=json.loads((bundle/'manifest.json').read_text())
    model=next(m for m in manifest['models']if m['name']=='native_whole_semantic_base_d4')
    src=bundle/model['checkpoint'];assert hashlib.sha256(src.read_bytes()).hexdigest()==model['checkpoint_sha256']
    out=root/'runs/oct04_collected/native_whole_semantic_base';out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(src,out/'best.pt');(out/'config.json').write_text(json.dumps(model['config'],indent=2))
    print('Ready. Run python code/experiments/oct09_official/train_aligned.py --mode aligned')

if __name__=='__main__':main()
