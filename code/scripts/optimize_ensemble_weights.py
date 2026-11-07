"""Fixed-budget simplex search of exact image-macro hard IoU.

The five validation-group cross-fits assess weight fitting only. Base model and
candidate selection already reused these images; this is not an independent
test estimate. Final all20 weights are explicitly a development-set fit.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import time
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def normalize_weights(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim==1:x=x[None]
    if not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid simplex coordinates')
    den=x.sum(1,keepdims=True)
    return np.divide(x,den,out=np.full_like(x,1/x.shape[1]),where=den>0)

def prepare(args):
    from PIL import Image
    root=Path(args.root).resolve();cat=json.loads(Path(args.catalog).read_text())
    models=cat['models'];ids=cat['image_ids'];names=[m['name'] for m in models]
    assert len(ids)==20 and len(set(ids))==20 and len(names)==6
    palette=np.array([[153,127,76],[76,178,76],[25,76,153],[204,204,204]],dtype=np.uint8)
    probs=[];ys=[];imids=[];fixed=[];stats=[]
    files=[{f['id']:f for f in m['files']} for m in models]
    for j,iid in enumerate(ids):
        maskpath=root/'data/valid/masks'/f'{iid}_mask.png'
        rgb=np.asarray(Image.open(maskpath).convert('RGB'))
        matches=(rgb[...,None,:]==palette[None,None]).all(-1)
        assert matches.sum(-1).min()==1 and matches.sum(-1).max()==1
        y=matches.argmax(-1).astype(np.uint8);ps=[]
        for m,table in zip(models,files):
            f=table[iid];p=root/f['path'];assert digest(p)==f['sha256']
            with np.load(p,allow_pickle=False) as z:a=z[f.get('key','probabilities')].astype(np.float32)
            assert a.shape==(4,*y.shape) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1.001
            den=a.sum(0);assert np.allclose(den,1,atol=.005,rtol=0)
            a/=den[None];ps.append(a)
        stack=np.stack(ps);pred=stack.argmax(1)
        agrees=(pred==pred[0]).all(0)
        # A strict FP32 margin prevents unanimous predictions becoming a tie
        # after weighted accumulation; <=1e-5 stays in the dynamic path.
        second=np.partition(stack,-2,axis=1)[:,-2]
        margin=(stack.max(1)-second).min(0)
        unanimous=agrees & (margin>1e-5);vary=~unanimous
        cm=np.bincount((y[unanimous].astype(np.int64)*4+pred[0,unanimous]).ravel(),minlength=16).reshape(4,4)
        fixed.append(cm);probs.append(stack[:,:,vary]);ys.append(y[vary]);imids.append(np.full(vary.sum(),j,np.uint8))
        stats.append({'id':iid,'shape':list(y.shape),'pixels':int(y.size),'safe_unanimous_pixels':int(unanimous.sum()),'unanimous_near_ties_kept_dynamic':int((agrees & ~unanimous).sum()),'disagreement_pixels':int(vary.sum()),'mask_sha256':digest(maskpath)})
    groups={}
    for j,iid in enumerate(ids):groups.setdefault(re.match(r'^A\d+',iid).group(),[]).append(j)
    rng=np.random.default_rng(20260930);keys=list(groups);rng.shuffle(keys)
    keys.sort(key=lambda k:-len(groups[k]));folds=[[] for _ in range(5)];fold_groups=[[] for _ in range(5)]
    for key in keys:
        k=min(range(5),key=lambda i:(len(folds[i]),i));folds[k].extend(groups[key]);fold_groups[k].append(key)
    assert sorted(sum(folds,[]))==list(range(20))
    baseline=cat['current_best_reference']['weights_by_name']
    if isinstance(baseline,dict):baseline=[baseline.get(n,0) for n in names]
    baseline=normalize_weights(baseline)[0].tolist()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'data.npz',probabilities=np.concatenate(probs,axis=2),labels=np.concatenate(ys),
             image_index=np.concatenate(imids),fixed_confusions=np.stack(fixed))
    metadata={'image_ids':ids,'model_names':names,'candidate_catalog':cat,'candidate_catalog_sha256':digest(args.catalog),
              'baseline_weights':baseline,'image_stats':stats,'heldout_folds':[sorted(x) for x in folds],
              'heldout_fold_groups':fold_groups,'n_groups':len(groups),'data_sha256':digest(out/'data.npz'),
              'fold_assignment_seed':20260930,'method':'Two-seed differential evolution on nonnegative simplex, followed by pairwise mass-transfer coordinate refinement.',
              'objective':'Equal mean across images of four class hard IoUs, empty union zero; no pooled-pixel surrogate.',
              'search_budget':{'seeds':[20260930,20260931],'population':48,'de_generations':40,'coordinate_steps':[.1,.03,.01,.003,.001],'coordinate_sweeps_per_step':2},
              'validation_warning':'Cross-fitting excludes held-out labels from weight optimization only; prior checkpoint and candidate selection reused valid20. All20-fit score is an optimized development score, not an independent estimate.',
              'unanimity_acceleration':'Only all-model same argmax pixels with each model top-two margin >1e-5 are fixed; numerical near-ties remain dynamic. All final weights require full-image CPU re-evaluation.',
              'test_images_used':False,'source_sha256':digest(__file__)}
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'prepared':str(out),'n_models':len(names),'disagreement_pixels':sum(s['disagreement_pixels'] for s in stats),'n_groups':len(groups),'fold_sizes':[len(x) for x in folds],'bytes':(out/'data.npz').stat().st_size}),flush=True)

class Scorer:
    def __init__(self,data,indices,device,batch=8):
        import torch
        self.torch=torch;self.device=device;self.batch=batch;self.indices=list(indices)
        mask=np.isin(data['image_index'],indices);remap=np.full(20,-1,np.int64);remap[indices]=np.arange(len(indices))
        self.p=torch.as_tensor(data['probabilities'][:,:,mask].copy(),device=device)
        code=remap[data['image_index'][mask]]*16+data['labels'][mask].astype(np.int64)*4
        self.code=torch.as_tensor(code,device=device);self.fixed=torch.as_tensor(data['fixed_confusions'][indices],device=device)
        self.n=len(indices);self.evaluations=0

    def evaluate(self,weights,return_confusions=False):
        torch=self.torch;weights=normalize_weights(weights);result=[];all_cm=[]
        with torch.inference_mode():
            for start in range(0,len(weights),self.batch):
                w=torch.as_tensor(weights[start:start+self.batch].astype(np.float32),device=self.device)
                # Separate multiply/add kernels match NumPy's FP32 accumulation.
                output=torch.zeros((len(w),4,self.p.shape[-1]),device=self.device,dtype=torch.float32)
                for m in range(self.p.shape[0]):output.add_(self.p[m][None]*w[:,m,None,None])
                pred=output.argmax(1)
                cm=torch.stack([torch.bincount(self.code+row,minlength=self.n*16).reshape(self.n,4,4) for row in pred])+self.fixed[None]
                cmf=cm.to(torch.float64);tp=cmf.diagonal(dim1=-2,dim2=-1);union=cmf.sum(-1)+cmf.sum(-2)-tp
                values=torch.where(union>0,tp/union,0.).mean(-1)
                result.append(values.cpu().numpy())
                if return_confusions:all_cm.append(cm.cpu().numpy())
                self.evaluations+=len(w)
        values=np.concatenate(result)
        return (values,np.concatenate(all_cm)) if return_confusions else values

def fit(scorer,baseline,budget,label):
    from scipy.optimize import differential_evolution
    m=len(baseline);candidates=[];started=time.time()
    for seed in budget['seeds']:
        rng=np.random.default_rng(seed);pop=rng.dirichlet(np.full(m,.6),budget['population'])
        pop[0]=baseline;pop[1]=np.ones(m)/m;pop[2:2+m]=np.eye(m)
        def objective(x):return -scorer.evaluate(x.T).mean(1)
        generation=[0]
        def callback(xk,convergence):
            generation[0]+=1
            if generation[0]%10==0:print(json.dumps({'task':label,'seed':seed,'generation':generation[0],'elapsed_seconds':round(time.time()-started,1)}),flush=True)
        r=differential_evolution(objective,[(0,1)]*m,init=pop,seed=seed,maxiter=budget['de_generations'],
            tol=0,atol=0,polish=False,updating='deferred',vectorized=True,callback=callback)
        candidates.append(normalize_weights(r.x)[0])
    candidates=np.stack([*candidates,np.array(baseline),np.ones(m)/m,*np.eye(m)])
    scores=scorer.evaluate(candidates).mean(1);best=candidates[scores.argmax()];score=float(scores.max())
    for step in budget['coordinate_steps']:
        for sweep in range(budget['coordinate_sweeps_per_step']):
            neighbors=[best]
            for i in range(m):
                for j in range(m):
                    if i==j or best[i]==0:continue
                    w=best.copy();delta=min(step,w[i]);w[i]-=delta;w[j]+=delta;neighbors.append(w)
            scores=scorer.evaluate(neighbors).mean(1);index=int(scores.argmax())
            if scores[index]>score:best=normalize_weights(neighbors[index])[0];score=float(scores[index])
    return {'weights':best.tolist(),'fit_miou':score,'fitness_evaluations':scorer.evaluations,'elapsed_seconds':time.time()-started}

def run(args):
    import torch,scipy
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    folder=Path(args.input);meta=json.loads((folder/'protocol.json').read_text());assert digest(folder/'data.npz')==meta['data_sha256']
    assert digest(__file__)==meta['source_sha256']
    with np.load(folder/'data.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    all_indices=list(range(20));held=[] if args.fold=='all' else meta['heldout_folds'][int(args.fold)]
    train=[i for i in all_indices if i not in held]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    scorer=Scorer(data,train,device,args.batch)
    baseline=scorer.evaluate(meta['baseline_weights'])[0]
    # The previous best mixture was selected using all20 validation images.
    # Keep it only as a comparator, never as a cross-fit search starting point.
    start_weights=np.full(len(meta['model_names']),1/len(meta['model_names']))
    result=fit(scorer,start_weights,meta['search_budget'],args.fold)
    result.update(task=args.fold,fit_ids=[meta['image_ids'][i] for i in train],fit_baseline_miou=float(baseline.mean()),
                  protocol_sha256=digest(folder/'protocol.json'),torch=torch.__version__,scipy=scipy.__version__,
                  source_sha256=digest(__file__),device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None)
    del scorer
    if device=='cuda':torch.cuda.empty_cache()
    evaluation_indices=held if held else all_indices
    evaluation=Scorer(data,evaluation_indices,device,args.batch)
    per_image,cm=evaluation.evaluate(result['weights'],return_confusions=True)
    ref=evaluation.evaluate(meta['baseline_weights'])[0]
    result.update(evaluation_scope='weight_heldout_fold' if held else 'all20_weight_fit_development',
                  evaluation_ids=[meta['image_ids'][i] for i in evaluation_indices],evaluation_miou=float(per_image[0].mean()),
                  per_image_miou=per_image[0].tolist(),confusions=cm[0].tolist(),baseline_per_image_miou=ref.tolist(),
                  validation_warning=meta['validation_warning'])
    output=folder/f'result_{args.fold}.json';assert not output.exists();output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'complete':args.fold,'weights':result['weights'],'fit_miou':result['fit_miou'],'evaluation_miou':result['evaluation_miou'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',required=True);p.add_argument('--catalog',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('run');p.add_argument('--input',required=True);p.add_argument('--fold',choices=['all','0','1','2','3','4'],required=True);p.add_argument('--batch',type=int,default=8)
    a=ap.parse_args();(prepare if a.command=='prepare' else run)(a)

if __name__=='__main__':main()
