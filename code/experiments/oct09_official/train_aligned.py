"""Matched warm-start ablations: R022, official soft-IoU, mixed whole/patch.

All three start from the same archived base-DINO checkpoint. Only train70 labels
enter gradients. Valid20 selects checkpoints using explicit GT-present mIoU.
Mixed patch steps are a surrogate, not claimed equal to full-image metric loss.
"""
import argparse,json,sys,time,random,math,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/oct04_context'))
from train_whole_semantic_base import Model,Dataset,predict,amp,R022Loss
from metrics import PALETTE,encode_mask,confusion
from official_metric import per_image_score,aligned_loss,self_test


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


@torch.inference_mode()
def evaluate(model,config,out=None,d4=False):
    rows=[]
    if out:out.mkdir(exist_ok=False)
    for ip in sorted((ROOT/'data/valid/images').glob('*_image.png')):
        iid=ip.stem.removesuffix('_image');image=np.asarray(Image.open(ip).convert('L'))
        p=predict(model,image,config,d4).astype(np.float16)
        q=p.astype(np.float32);q/=q.sum(0,keepdims=True);pred=q.argmax(0)
        y=encode_mask(np.asarray(Image.open(ROOT/f'data/valid/masks/{iid}_mask.png').convert('RGB')))
        cm=confusion(y,pred);rows.append({'id':iid,'score':float(per_image_score(cm)),'confusion':cm.tolist()})
        if out:
            np.savez_compressed(out/f'{iid}.npz',probabilities=p)
            Image.fromarray(PALETTE[pred]).save(out/f'{iid}_prediction.png')
    return {'metric':'image_macro_gt_present_iou','score':float(np.mean([r['score']for r in rows])),'per_image':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['control','aligned','mixed'],required=True)
    ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--seed',type=int,default=20261009);a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);self_test()
    initial=ROOT/'runs/oct04_collected/native_whole_semantic_base/best.pt'
    config=json.loads((initial.parent/'config.json').read_text())
    config.update(name='oct09_'+a.mode,mode=a.mode,steps=a.steps,eval_every=300,seed=a.seed,lr=1e-5,
        warm_checkpoint_sha256=sha(initial),loss='R022'if a.mode=='control'else '.8 GT-present image soft-IoU + .2 image CE',
        sampler='uniform full images'if a.mode!='mixed'else '50% whole image; 50% native crop512, half uniform and half class-balanced center',
        metric='image_macro_gt_present_iou',test_used=False)
    out=ROOT/f'runs/oct09_official/train_{a.mode}';out.mkdir(exist_ok=False);write(out/'config.json',config)
    for p in [Path(__file__),Path(__file__).with_name('official_metric.py')]:shutil.copy2(p,out/p.name)
    model=Model(config,initialize=False).cuda()
    cp=torch.load(initial,map_location='cpu',weights_only=True);model.load_state_dict(cp['model'],strict=True);del cp
    dataset=Dataset(ROOT/'data',samples_per_epoch=a.steps)
    optimizer=torch.optim.AdamW([{'params':model.network.parameters(),'lr':config['lr']},
        {'params':model.semantic_projection.parameters(),'lr':config['lr']*10}],weight_decay=.0001)
    control=R022Loss(class_weights=(1,1,1,2)).cuda();start=time.time()
    score=evaluate(model,config);best=score['score'];best_step=0
    torch.save({'model':model.state_dict(),'config':config,'step':0},out/'best.pt');write(out/'initial.json',score)
    print(json.dumps({'step':0,'score':best,'mode':a.mode}),flush=True)
    # Dedicated crop RNG: treatment does not perturb photometric/image RNG.
    rng=np.random.default_rng(a.seed+1)
    for step in range(1,a.steps+1):
        x,y=dataset[step]
        if a.mode=='mixed' and rng.random()<.5:
            h,w=y.shape;size=min(512,h,w)
            if rng.random()<.5:
                c=int(rng.choice(torch.unique(y).numpy()));ys,xs=np.where(y.numpy()==c);j=int(rng.integers(len(ys)))
                top=int(np.clip(ys[j]-size//2,0,h-size));left=int(np.clip(xs[j]-size//2,0,w-size))
            else:top=int(rng.integers(h-size+1));left=int(rng.integers(w-size+1))
            x=x[:,top:top+size,left:left+size];y=y[top:top+size,left:left+size]
        model.train();x=x[None].cuda();y=y[None].cuda();optimizer.zero_grad(set_to_none=True)
        lr=config['lr']*(.1+.9*(1+math.cos(math.pi*(step-1)/a.steps))/2)
        for j,g in enumerate(optimizer.param_groups):g['lr']=lr*(10 if j else 1)
        with amp(config):logits=model(x)
        loss=control(logits.float(),y)if a.mode=='control'else aligned_loss(logits,y)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%50==0:
            status={'step':step,'loss':float(loss.detach()),'best':best,'seconds':time.time()-start,'mode':a.mode}
            write(out/'status.json',status);print(json.dumps(status),flush=True)
        if step%config['eval_every']==0 or step==a.steps:
            score=evaluate(model,config);write(out/f'validation_{step}.json',score)
            if score['score']>best:
                best=score['score'];best_step=step;torch.save({'model':model.state_dict(),'config':config,'step':step},out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
            print(json.dumps({'validation_step':step,'score':score['score'],'best':best,'mode':a.mode}),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    for d4 in [False,True]:
        tag='d4'if d4 else 'single';write(out/f'final_{tag}.json',evaluate(model,config,out/f'valid_{tag}',d4))
    write(out/'COMPLETE.json',{'best_step':best_step,'best_score':best,'completed_steps':a.steps,'seconds':time.time()-start,'test_used':False,
        'checkpoint_sha256':sha(out/'best.pt'),'source_sha256':sha(__file__),'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__)})


if __name__=='__main__':main()
