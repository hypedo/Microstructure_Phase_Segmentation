"""Official-objective conditional-gradient ensemble refinement on CPU.

All 44 archived native probability candidates; nonnegative simplex only.
Each group fold starts from its OWN historical fit, not all-validation weights.
Incremental scores screen proposals; every accepted move is recomputed in the
fixed FP32 model order. Final full-image verification is a separate required step.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from numba import njit, prange, set_num_threads
from official_metric import per_image_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from optimize_ensemble_weights import digest


@njit(parallel=True, cache=True)
def screen(p, current, candidates, alpha, labels, indices, fixed):
    cms = np.empty((len(candidates), 20, 4, 4), np.int64)
    for k in prange(len(candidates)):
        cm = fixed.copy(); m = candidates[k]; a = alpha[k]
        for j in range(len(labels)):
            best = 0
            value = np.float32(current[0,j]*np.float32(1-a) + p[m,0,j]*a)
            for c in range(1,4):
                v = np.float32(current[c,j]*np.float32(1-a) + p[m,c,j]*a)
                if v > value:
                    value = v; best = c
            cm[indices[j],labels[j],best] += 1
        cms[k] = cm
    return cms


def mixture(p, w):
    total = np.zeros(p.shape[1:], np.float32)
    for j in range(len(w)):
        total += p[j]*np.float32(w[j])
    return total


def main():
    set_num_threads(4)
    parent=ROOT/'runs/oct04_expanded_ensemble_v4'
    old=ROOT/'runs/oct04_greedy_v4'; out=ROOT/'runs/oct09_official/weights';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((parent/'protocol.json').read_text())
    assert digest(parent/'data.npz') == meta['data_sha256']
    protocol={'metric':'image_macro_gt_present_iou','parent_protocol_sha256':digest(parent/'protocol.json'),
        'data_sha256':meta['data_sha256'],'model_names':meta['model_names'],'heldout_folds':meta['heldout_folds'],
        'source_sha256':digest(__file__),'metric_source_sha256':digest(Path(__file__).with_name('official_metric.py')),
        'steps':[.2,.1,.05,.02,.01,.005,.002], 'max_sweeps':3,
        'warning':'Development and ensemble-stage group cross-fit only; checkpoints/candidate selection already reused valid20.',
        'test_used':False,'full_pixel_verification_required':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with np.load(parent/'data.npz',allow_pickle=False) as z:
        p=z['probabilities']; y=z['labels']; idx=z['image_index'];fixed=z['fixed_confusions']
    for task in ['all','0','1','2','3','4']:
        path=out/f'result_{task}.json'
        if path.exists(): continue
        start=time.time(); prev=json.loads((old/f'result_{task}.json').read_text())
        held=[] if task=='all' else meta['heldout_folds'][int(task)]
        fit=[i for i in range(20) if i not in held];assert fit==prev['fit_ids']
        w=np.array(prev['weights'],np.float64); w/=w.sum()
        total=mixture(p,w)
        def exact(total):
            pred=total.argmax(0)
            cm=np.bincount(idx.astype(np.int64)*16+y.astype(np.int64)*4+pred,minlength=320).reshape(20,4,4)+fixed
            return per_image_score(cm),cm
        values,cm=exact(total); score=values[fit].mean();initial=float(score);trace=[]
        for step in protocol['steps']:
            for sweep in range(protocol['max_sweeps']):
                candidates=[];alphas=[]
                for m in range(len(w)):
                    candidates.append(m);alphas.append(step)
                    if w[m]>0 and w[m]<1:
                        candidates.append(m);alphas.append(-min(step,w[m]/(1-w[m])))
                candidates=np.array(candidates);alphas=np.array(alphas,np.float32)
                cms=screen(p,total,candidates,alphas,y,idx,fixed)
                scores=per_image_score(cms)[:,fit].mean(1)
                accepted=False
                for k in np.argsort(-scores)[:5]:
                    if scores[k]<=score:break
                    proposal=w*(1-float(alphas[k]));proposal[candidates[k]]+=float(alphas[k])
                    proposal=np.maximum(proposal,0);proposal/=proposal.sum()
                    newtotal=mixture(p,proposal);vs,newcm=exact(newtotal);s=vs[fit].mean()
                    if s>score:
                        w,total,values,cm,score=proposal,newtotal,vs,newcm,s;accepted=True;break
                trace.append({'step':step,'sweep':sweep,'fit_score':float(score),'accepted':accepted})
                print(json.dumps({'task':task,**trace[-1],'seconds':round(time.time()-start,1)}),flush=True)
                if not accepted:break
        eval_ids=held or list(range(20))
        result={'task':task,'weights':w.tolist(),'fit_ids':fit,'evaluation_ids':eval_ids,
            'initial_fit_score':initial,'fit_score':float(score),'evaluation_score':float(values[eval_ids].mean()),
            'confusions':cm[eval_ids].tolist(),'per_image_scores':values[eval_ids].tolist(),
            'trace':trace,'seconds':time.time()-start,'full_pixel_verified':False}
        path.write_text(json.dumps(result,indent=2))
        print(json.dumps({'completed':task,'score':result['evaluation_score']}),flush=True)


if __name__=='__main__':main()
