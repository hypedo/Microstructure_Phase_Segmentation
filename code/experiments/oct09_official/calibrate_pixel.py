"""Class-score calibration for mutually exclusive material phases.

Global offsets and prediction-derived image-composition conditional offsets.
No ground-truth presence, filenames, test images or validation labels are inputs
to the prediction rule. Fit labels are used only to optimize declared parameters.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from numba import njit,prange,set_num_threads
from scipy.optimize import differential_evolution
from PIL import Image
from official_metric import per_image_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from metrics import confusion


def features(prob,kind):
    mass=prob.mean((1,2),dtype=np.float64)
    if kind=='global':return np.ones(1)
    # Bounded, prediction-only context; fixed constants, no image-ID lookup.
    ratio=2*mass[2]/max(mass[2]+mass[3],1e-8)-1
    primary=np.clip((np.log(max(mass[1],1e-6))-np.log(.05))/4,-1,1)
    return np.array([1.,ratio,primary])


@njit(parallel=True,cache=True)
def evaluate_parameters(parameters,logs,code,idx,fixed,feat,contrast):
    n=len(parameters);out=np.empty((n,20,4,4),np.int64)
    for k in prange(n):
        cm=fixed.copy();bias=np.zeros((20,4),np.float64)
        for i in range(20):
            for c in range(1,4):
                for f in range(feat.shape[1]):bias[i,c]+=parameters[k,(c-1)*feat.shape[1]+f]*feat[i,f]
        for j in range(len(idx)):
            i=idx[j];win=0;best=logs[0,j]
            for c in range(1,4):
                val=logs[c,j]+bias[i,c]+parameters[k,9+c-1]*contrast[j]
                if val>best:win=c;best=val
            cm[i,code[j],win]+=1
        out[k]=cm
    return out


def contrast_feature(image):
    x=np.asarray(image,np.float64)/255
    lo,hi=np.quantile(x,[.1,.9])
    return np.clip((2*x-hi-lo)/max(hi-lo,1/255),-1,1)


def predict(prob,parameters,kind,contrast):
    f=features(prob,kind);b=np.r_[0,np.asarray(parameters[:9]).reshape(3,-1)@f]
    return (np.log(np.maximum(prob,1e-30).astype(np.float64))+b[:,None,None]+np.r_[0,parameters[9:]][:,None,None]*contrast[None]).argmax(0).astype(np.uint8)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['contrast'],required=True);a=ap.parse_args()
    set_num_threads(4);kind=a.kind;out=ROOT/f'runs/oct09_official/calibration_{kind}_long';out.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'runs/oct04_expanded_ensemble_v4/protocol.json').read_text());ids=meta['catalog']['image_ids']
    bounds=[(-.5,.5),(-.35,.35),(-.35,.35)]*3+[(-.35,.35)]*3
    limit=1.55
    protocol={'kind':kind,'metric':'image_macro_gt_present_iou','bounds':bounds,'optimizer':'DE vectorized, seed20261009, popsize8, maxiter200, zero start; exact full-pixel final verification',
        'features':'constant only'if kind=='global'else 'constant, predicted Ni/(Ni+Eut) signed ratio, clipped log predicted Primary fraction plus per-pixel robust grayscale contrast',
        'test_used':False,'warning':'Calibration-stage group cross-fit only; base checkpoints and candidate selection reused valid20.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));cross=[]
    for task_index,task in enumerate(['all','0','1','2','3','4']):
        path=out/f'result_{task}.json'
        if path.exists():
            old=json.loads(path.read_text())
            if task!='all':cross.extend(old['evaluation_rows'])
            continue
        start=time.time();held=[]if task=='all'else meta['heldout_folds'][int(task)];fit=[i for i in range(20)if i not in held]
        full=[];logs=[];targets=[];indices=[];fixed=[];feats=[];contrasts=[]
        for i,iid in enumerate(ids):
            with np.load(ROOT/f'runs/oct09_official/mixtures/{iid}.npz')as z:p=z['probabilities'][task_index];y=z['target']
            contrast=contrast_feature(Image.open(ROOT/f'data/valid/images/{iid}_image.png').convert('L'))
            log=np.log(np.maximum(p,1e-30).astype(np.float64));top=np.sort(log,axis=0)[-2:]
            static=(top[1]-top[0])>2*limit+1e-5
            fixed.append(confusion(y[static],log.argmax(0)[static]));logs.append(log[:,~static]);targets.append(y[~static]);indices.append(np.full((~static).sum(),i,np.int64));feats.append(features(p,kind));full.append((p,y,contrast));contrasts.append(contrast[~static])
        x=np.ascontiguousarray(np.concatenate(logs,1));y=np.concatenate(targets);idx=np.concatenate(indices);fixed=np.stack(fixed);feat=np.stack(feats);contrast_flat=np.concatenate(contrasts)
        def objective(parameters):
            if parameters.ndim==1:parameters=parameters[:,None]
            cm=evaluate_parameters(np.ascontiguousarray(parameters.T),x,y,idx,fixed,feat,contrast_flat)
            return -per_image_score(cm)[:,fit].mean(1)
        zero=np.zeros(len(bounds));initial=float(-objective(zero)[0]);gen=[0]
        def cb(xk,convergence):
            gen[0]+=1
            if gen[0]%10==0:print(json.dumps({'task':task,'kind':kind,'generation':gen[0],'seconds':round(time.time()-start,1)}),flush=True)
        r=differential_evolution(objective,bounds,seed=20261009,popsize=8,maxiter=200,polish=False,updating='deferred',vectorized=True,x0=zero,tol=0,atol=0,callback=cb)
        params=r.x if -r.fun>initial else zero
        cm=evaluate_parameters(params[None],x,y,idx,fixed,feat,contrast_flat)[0]
        rows=[]
        for i,(p,yfull,contrast)in enumerate(full):
            direct=confusion(yfull,predict(p,params,kind,contrast));assert np.array_equal(direct,cm[i]),(task,i)
            rows.append({'id':ids[i],'confusion':direct.tolist(),'score':float(per_image_score(direct))})
        evaluation=[rows[i]for i in held or range(20)];score=float(np.mean([r['score']for r in evaluation]))
        result={'task':task,'parameters':params.tolist(),'fit_ids':fit,'evaluation_ids':held or list(range(20)),
            'fit_score':float(-objective(params)[0]),'initial_fit_score':initial,'evaluation_score':score,'evaluation_rows':evaluation,
            'full_pixel_verified':True,'seconds':time.time()-start}
        path.write_text(json.dumps(result,indent=2))
        if task!='all':cross.extend(evaluation)
        print(json.dumps({'completed':task,'kind':kind,'score':score,'seconds':result['seconds']}),flush=True)
    (out/'VERIFIED.json').write_text(json.dumps({'all_fit':json.loads((out/'result_all.json').read_text()),
        'crossfit_score':float(np.mean([r['score']for r in cross])),'crossfit_rows':cross,**protocol},indent=2))


if __name__=='__main__':main()
