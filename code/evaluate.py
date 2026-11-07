# 공식 수정 평가 안내의 독립 구현 (결과서 3-2, 3-3).
# 정답에 있는 상만 평균한다. GT와 예측 모두에 없는 상에 0 또는 1을 넣지 않으며,
# GT에는 없고 예측에만 있는 상도 class 평균에서는 제외한다.
# 다만 그 오분류 픽셀은 실제 정답 상의 FN에 포함되므로 점수 손실은 남는다.
# 이미지 크기에 관계없이 각 이미지 점수에 같은 가중치를 준다.
# 데이터셋 전체 confusion을 합쳐 계산하거나 상별 평균을 다시 단순 평균하면 공식식과 다르다.

"""Official GT-present image-macro evaluation; no model fitting."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
PALETTE=np.array([[153,127,76],[76,178,76],[25,76,153],[204,204,204]],np.uint8)

def encode(path):
    image=np.asarray(Image.open(path).convert('RGB'))
    matches=(image[:,:,None,:]==PALETTE[None,None]).all(-1)
    if not np.all(matches.sum(-1)==1):raise ValueError(f'Unexpected mask palette: {path}')
    return matches.argmax(-1)

def evaluate(predictions,masks):
    files=sorted(Path(masks).glob('*_mask.png'))
    if not files:raise ValueError('No *_mask.png ground truths')
    rows=[]
    for path in files:
        iid=path.stem.removesuffix('_mask');target=encode(path);pred=encode(Path(predictions)/f'{iid}_prediction.png')
        if pred.shape!=target.shape:raise ValueError(f'Native shape mismatch: {iid}')
        cm=np.bincount((target*4+pred).ravel(),minlength=16).reshape(4,4)
        tp=np.diag(cm).astype(np.float64);truth=cm.sum(1);union=truth+cm.sum(0)-tp
        # present는 예측 union>0이 아니라 GT pixel count>0으로 정의한다.
        # 이 차이는 초정 Si가 부재한 영상의 점수에 직접 영향을 준다.
        present=truth>0;iou=np.divide(tp,union,out=np.zeros(4),where=union>0)
        rows.append({'id':iid,'score':float(iou[present].mean()),'gt_present_classes':np.flatnonzero(present).tolist(),
            'class_iou':[float(iou[c])if present[c]else None for c in range(4)],'confusion':cm.tolist()})
    return {'metric':'image_macro_gt_present_iou','score':float(np.mean([r['score']for r in rows])),'n_images':len(rows),'per_image':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--masks',required=True);p.add_argument('--output');a=p.parse_args()
    result=evaluate(a.predictions,a.masks)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
