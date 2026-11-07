"""Phase-specific convex probability mixtures with bounded global phase gains.

Search optimizes exact image-macro four-class hard IoU. Fold label arrays are
physically separated. This diagnoses only ensemble fitting, because all base
checkpoints were already selected on valid20. Test is never accessed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def parameters(x,m):
    x=np.asarray(x,np.float64)
    if x.ndim==1:x=x[None]
    w=x[:,:m*4].reshape(-1,m,4);s=w.sum(1,keepdims=True)
    w=np.divide(w,s,out=np.full_like(w,1/m),where=s>0).astype(np.float32)
    gain=np.exp(np.concatenate([np.zeros((len(x),1)),x[:,m*4:]],axis=1)).astype(np.float32)
    return w,gain

def load_image(cat,i):
    from PIL import Image
    from metrics import encode_mask
    r=cat['images'][i];path=ROOT/r['mask_path'];assert sha(path)==r['mask_sha256']
    y=encode_mask(np.asarray(Image.open(path).convert('RGB')));ps=[]
    for model in cat['models']:
        f=next(f for f in model['files'] if f['id']==r['id']);p=ROOT/f['path'];assert sha(p)==f['sha256']
        with np.load(p,allow_pickle=False) as z:a=z[f.get('key','probabilities')].astype(np.float32)
        assert a.shape==(4,*y.shape) and np.isfinite(a).all() and a.min()>=0
        a/=a.sum(0,keepdims=True);ps.append(a)
    return np.stack(ps),y

def prepare(args):
    from metrics import confusion,score_confusion,summarize
    out=Path(args.folder);out.mkdir(parents=True,exist_ok=False)
    cat=json.loads((ROOT/'reports/ensemble_weight_candidates.json').read_text())
    run=ROOT/'runs/oct04_collected/native_halo448'
    assert (run/'LOCAL_VERIFIED.json').is_file()
    cat['models'].append({'name':'native_halo448_d4','files':[{'id':iid,
        'path':str((run/'valid_d4'/(iid+'.npz')).relative_to(ROOT)),
        'sha256':sha(run/'valid_d4'/(iid+'.npz'))}for iid in cat['image_ids']]})
    m=len(cat['models']);old=json.loads((ROOT/'configs/current_best.json').read_text())['weights']
    old=np.array([*old,0.]);old/=old.sum()
    base=np.r_[np.repeat(old[:,None],4,axis=1).ravel(),np.zeros(3)]
    probs=[];ys=[];indices=[];fixed=[];mixrows=[[]for _ in range(21)];stats=[]
    for i in range(20):
        ps,y=load_image(cat,i)
        # For any class-wise convex weights, each score lies between the
        # minimum and maximum model probability times its bounded gain.
        # This stronger condition is necessary: same argmax alone is UNSAFE.
        lower=ps.min(0)*np.exp(-.35);upper=ps.max(0)*np.exp(.35)
        winner=lower.argmax(0);others=upper.copy()
        np.put_along_axis(others,winner[None],-np.inf,axis=0)
        static=(lower.max(0)>others.max(0)+1e-5);dynamic=~static
        fixed.append(confusion(y[static],winner[static]));probs.append(ps[:,:,dynamic])
        ys.append(y[dynamic]);indices.append(np.full(dynamic.sum(),i,np.uint8))
        baseline=np.zeros((4,*y.shape),np.float32)
        for p,w in zip(ps,old.astype(np.float32)):baseline+=p*w
        for j,a in enumerate(np.linspace(0,1,21)):
            pred=(baseline*np.float32(1-a)+ps[-1]*np.float32(a)).argmax(0)
            mixrows[j].append({'id':cat['image_ids'][i],**score_confusion(confusion(y,pred))})
        stats.append({'id':cat['image_ids'][i],'dynamic_pixels':int(dynamic.sum()),'total_pixels':int(y.size)})
    np.savez_compressed(out/'data.npz',probabilities=np.concatenate(probs,2),labels=np.concatenate(ys),
        image_index=np.concatenate(indices),fixed_confusions=np.stack(fixed))
    (out/'mixture_screen.json').write_text(json.dumps([{'halo_weight':float(a),**summarize(rows)}
        for a,rows in zip(np.linspace(0,1,21),mixrows)],indent=2))
    protocol={'catalog':cat,'baseline':base.tolist(),'n_models':m,'stats':stats,
        'heldout_folds':json.loads((ROOT/'runs/ensemble_weight_search_20260930/protocol.json').read_text())['heldout_folds'],
        'source_sha256':sha(__file__),'data_sha256':sha(out/'data.npz'),'seed':20261004,
        'budget':{'population':64,'generations':50,'coordinate_steps':[.1,.03,.01],'coordinate_sweeps':2},
        'warning':'Ensemble-stage group cross-fit only; checkpoint and candidate selection reused valid20. No independent accuracy claim.',
        'test_used':False,'class_order':['Al','Primary','Al3Ni','Eut'],'gain_bounds':[-.35,.35],
        'method':'Separate nonnegative convex model weights per class; Al gain fixed1, three exp(loggain) factors. Exact native hard IoU.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'dynamic_pixels':sum(x['dynamic_pixels']for x in stats),'bytes':(out/'data.npz').stat().st_size,
        'mixture_screen':[(round(j*.05,2),summarize(rows)['miou_absent_zero'])for j,rows in enumerate(mixrows)]}),flush=True)

class Scorer:
    def __init__(self,data,ids,device,m):
        import torch
        self.t=torch;self.m=m;self.n=len(ids);self.device=device;mask=np.isin(data['image_index'],ids)
        self.p=torch.as_tensor(data['probabilities'][:,:,mask].copy(),device=device)
        mapping=np.full(20,-1,np.int64);mapping[ids]=np.arange(len(ids))
        self.code=torch.as_tensor(mapping[data['image_index'][mask]]*16+data['labels'][mask].astype(np.int64)*4,device=device)
        self.fixed=torch.as_tensor(data['fixed_confusions'][ids],device=device);self.evaluations=0
    def evaluate(self,x,cm=False):
        t=self.t;w,g=parameters(x,self.m);values=[];cms=[]
        with t.inference_mode():
            for start in range(0,len(w),4):
                wt=t.as_tensor(w[start:start+4],device=self.device);gt=t.as_tensor(g[start:start+4],device=self.device)
                out=t.zeros((len(wt),4,self.p.shape[-1]),device=self.device)
                for j in range(self.m):out.add_(self.p[j][None]*wt[:,j,:,None])
                out.mul_(gt[:,:,None]);pred=out.argmax(1)
                c=t.stack([t.bincount(self.code+r,minlength=self.n*16).reshape(self.n,4,4)for r in pred])+self.fixed[None]
                cf=c.to(t.float64);tp=cf.diagonal(dim1=-2,dim2=-1);union=cf.sum(-1)+cf.sum(-2)-tp
                values.append(t.where(union>0,tp/union,0.).mean(-1).cpu().numpy())
                if cm:cms.append(c.cpu().numpy())
        self.evaluations+=len(w)
        return (np.concatenate(values),np.concatenate(cms))if cm else np.concatenate(values)

def run(args):
    import torch
    from scipy.optimize import differential_evolution
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    folder=Path(args.folder);meta=json.loads((folder/'protocol.json').read_text())
    assert sha(__file__)==meta['source_sha256'];assert sha(folder/'data.npz')==meta['data_sha256']
    with np.load(folder/'data.npz',allow_pickle=False)as z:data={k:z[k]for k in z.files}
    m=meta['n_models'];held=[]if args.fold=='all'else meta['heldout_folds'][int(args.fold)]
    fit=[i for i in range(20)if i not in held];device='cuda'if torch.cuda.is_available()else'cpu'
    scorer=Scorer(data,fit,device,m);rng=np.random.default_rng(meta['seed']);budget=meta['budget'];started=time.time()
    pop=np.concatenate([rng.dirichlet(np.full(m,.6),size=(budget['population'],4)).transpose(0,2,1).reshape(-1,m*4),
        rng.uniform(-.35,.35,(budget['population'],3))],1)
    pop[0]=np.r_[np.full(m*4,1/m),np.zeros(3)]
    for j in range(m):pop[j+1]=np.r_[np.repeat(np.eye(m)[j,:,None],4,axis=1).ravel(),np.zeros(3)]
    # All20-selected previous weights are only a comparator for held-out runs.
    if not held:pop[m+1]=meta['baseline']
    generation=[0]
    def callback(x,convergence):
        generation[0]+=1
        if generation[0]%5==0:print(json.dumps({'fold':args.fold,'generation':generation[0],'seconds':time.time()-started}),flush=True)
    result=differential_evolution(lambda x:-scorer.evaluate(x.T).mean(1),[(0,1)]*(m*4)+[(-.35,.35)]*3,
        init=pop,seed=meta['seed'],maxiter=budget['generations'],polish=False,vectorized=True,updating='deferred',
        tol=0,atol=0,callback=callback)
    w,g=parameters(result.x,m);best=np.r_[w[0].ravel(),np.log(g[0,1:])];bestscore=float(scorer.evaluate(best).mean())
    for step in budget['coordinate_steps']:
        for sweep in range(budget['coordinate_sweeps']):
            candidates=[best]
            for c in range(4):
                for i in range(m):
                    for j in range(m):
                        if i==j or best[i*4+c]==0:continue
                        candidate=best.copy();delta=min(step,candidate[i*4+c]);candidate[i*4+c]-=delta;candidate[j*4+c]+=delta;candidates.append(candidate)
            for c in range(3):
                for sign in [-1,1]:
                    candidate=best.copy();candidate[m*4+c]=np.clip(candidate[m*4+c]+sign*step,-.35,.35);candidates.append(candidate)
            scores=scorer.evaluate(candidates).mean(1);j=int(scores.argmax())
            if scores[j]>bestscore:best=candidates[j];bestscore=float(scores[j])
    nfev=scorer.evaluations;del scorer
    if device=='cuda':torch.cuda.empty_cache()
    ids=held or list(range(20));evaluation=Scorer(data,ids,device,m)
    values,cms=evaluation.evaluate(best,cm=True);baseline=evaluation.evaluate(meta['baseline'])
    record={'fold':args.fold,'parameters':best.tolist(),'fit_score':bestscore,'evaluation_score':float(values.mean()),
        'per_image':values[0].tolist(),'image_indices':ids,'confusions':cms[0].tolist(),'baseline_per_image':baseline[0].tolist(),
        'fit_ids':fit,'evaluations':nfev,'seconds':time.time()-started,'device':device,
        'source_sha256':sha(__file__),'protocol_sha256':sha(folder/'protocol.json'),'warning':meta['warning']}
    path=folder/f'result_{args.fold}.json';assert not path.exists();path.write_text(json.dumps(record,indent=2))
    print(json.dumps({k:record[k]for k in ['fold','fit_score','evaluation_score','seconds']}),flush=True)

def verify(args):
    from metrics import confusion,score_confusion
    folder=Path(args.folder);meta=json.loads((folder/'protocol.json').read_text());rows=[]
    for fold in ['all','0','1','2','3','4']:
        r=json.loads((folder/f'result_{fold}.json').read_text());w,g=parameters(r['parameters'],meta['n_models']);scores=[]
        for j,i in enumerate(r['image_indices']):
            ps,y=load_image(meta['catalog'],i);total=np.zeros((4,*y.shape),np.float32)
            for p,weight in zip(ps,w[0]):total+=p*weight[:,None,None]
            total*=g[0,:,None,None];cm=confusion(y,total.argmax(0));assert np.array_equal(cm,np.array(r['confusions'][j]))
            score=score_confusion(cm)['miou_absent_zero'];scores.append(score)
            if fold!='all':rows.append((i,score))
        assert abs(np.mean(scores)-r['evaluation_score'])<1e-12
    assert sorted(i for i,s in rows)==list(range(20))
    report={'all_fit_score':json.loads((folder/'result_all.json').read_text())['evaluation_score'],
        'crossfit_score':float(np.mean([s for i,s in rows])),'full_pixel_CPU_GPU_confusion_parity':True,'warning':meta['warning']}
    (folder/'VERIFIED.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','verify']);p.add_argument('--folder',required=True)
    p.add_argument('--fold',default='all',choices=['all','0','1','2','3','4']);a=p.parse_args();globals()[a.action](a)
