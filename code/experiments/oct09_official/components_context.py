# 검색 단계의 해석과 연구 전략.
# 영상 예측 조성의 연속 특징을 사용해 4상 경쟁을 보정한다.
# 검증 GT는 목적함수 계산/계수 적합에만 사용하며 최종 추론 특징에 GT 존재 여부를 넣지 않는다.
# 차분진화는 비미분 argmax IoU의 낮은 차원 탐색에 사용한 휴리스틱이며 전역 최적성을 보장하지 않는다.
# 보정 단계 그룹 교차검증은 이미 선택한 기본 모델을 조건으로 한 값이다. 앞선 모델 선택에서
# valid20을 사용했으므로 전체 파이프라인의 독립 nested CV나 숨은 test 성능으로 해석하면 안 된다.
# 이 폴더에는 채택되지 않은 entropy/contrast/범위 실험도 포함된다. 최종 활성 규칙은
# model_bundle/manifest.json의 postprocess이며, 모든 실험 파일이 최종 추론에 쓰이는 것은 아니다.

"""Fold-own threshold selection after frozen calibration; original-size masks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from metrics import confusion,PALETTE
from official_metric import per_image_score
from calibrate_context import features
from PIL import Image
GRID=[0,4,16,64,128,256]

def outputs(p,parameters,kind):
    b=np.r_[0,np.asarray(parameters).reshape(3,-1)@features(p,kind)]
    log=np.log(np.maximum(p,1e-30).astype(np.float64))+b[:,None,None]
    pred=log.argmax(0).astype(np.uint8)
    lab,n=ndimage.label(pred==1,structure=np.ones((3,3),bool));sizes=np.bincount(lab.ravel(),minlength=n+1)
    keep=np.zeros(n+1,bool);keep[np.unique(np.r_[lab[0],lab[-1],lab[:,0],lab[:,-1]])]=True;keep[0]=True
    log[1]=-np.inf;replacement=log.argmax(0).astype(np.uint8)
    for t in GRID:
        mask=((sizes<=t)&~keep)[lab];q=pred.copy();q[mask]=replacement[mask];yield q

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True);a=ap.parse_args()
    parent=ROOT/a.parent;meta=json.loads((parent/'protocol.json').read_text());kind=meta['kind']
    out=Path(str(parent)+'_components');out.mkdir(exist_ok=False);cross=[]
    (out/'protocol.json').write_text(json.dumps({'parent':a.parent,'grid':GRID,'metric':'image_macro_gt_present_iou','test_used':False,
        'warning':meta['warning'],'border_touching_components_preserved':True,'replacement':'highest calibrated non-Primary class'},indent=2))
    proto=json.loads((ROOT/'runs/oct04_expanded_ensemble_v4/protocol.json').read_text());ids=proto['catalog']['image_ids']
    for j,task in enumerate(['all','0','1','2','3','4']):
        r=json.loads((parent/f'result_{task}.json').read_text());cms=np.zeros((len(GRID),20,4,4),np.int64)
        for i,iid in enumerate(ids):
            with np.load(ROOT/f'runs/oct09_official/mixtures/{iid}.npz')as z:p=z['probabilities'][j];y=z['target']
            for k,q in enumerate(outputs(p,r['parameters'],kind)):cms[k,i]=confusion(y,q)
        scores=per_image_score(cms);selected=int(scores[:,r['fit_ids']].mean(1).argmax());ev=r['evaluation_ids']
        rows=[{'id':ids[i],'confusion':cms[selected,i].tolist(),'score':float(scores[selected,i])}for i in ev]
        result={'task':task,'threshold':GRID[selected],'fit_curve':scores[:,r['fit_ids']].mean(1).tolist(),
            'evaluation_score':float(scores[selected,ev].mean()),'evaluation_rows':rows,'full_pixel_verified':True,'parameters':r['parameters']}
        (out/f'result_{task}.json').write_text(json.dumps(result,indent=2))
        if task!='all':cross.extend(rows)
        print(json.dumps({'task':task,'threshold':GRID[selected],'score':result['evaluation_score']}),flush=True)
    (out/'VERIFIED.json').write_text(json.dumps({'all_fit':json.loads((out/'result_all.json').read_text()),
        'crossfit_score':float(np.mean([r['score']for r in cross])),'crossfit_rows':cross,'test_used':False,'warning':meta['warning']},indent=2))

if __name__=='__main__':main()
