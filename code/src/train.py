"""Reproducible patch training; validation always at native mask resolution."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from metrics import PALETTE, CLASS_NAMES, encode_mask, confusion, score_confusion, summarize

cv2.setNumThreads(0)

def read_image(path):
    return np.asarray(Image.open(path).convert('RGB'))

def image_id(path):
    return Path(path).stem.removesuffix('_image')

def pairs(data, split):
    result=[]
    for p in sorted((Path(data)/split/'images').glob('*_image.png')):
        m=Path(data)/split/'masks'/f'{image_id(p)}_mask.png'
        if not m.exists(): raise FileNotFoundError(m)
        result.append((p,m))
    if not result: raise ValueError(f'No data found: {data}/{split}')
    return result

def normalize(x):
    return (x-torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]

class Patches(Dataset):
    def __init__(self, data, crop, length, rare_prob, scale_min, scale_max, canonical_width=0,
                 photometric_strength=1., blur_prob=.12, noise_prob=.2, full_image=False,
                 mask_resize_mode='nearest'):
        self.crop,self.length,self.rare_prob=crop,length,rare_prob
        self.scale_min,self.scale_max=scale_min,scale_max
        self.photometric_strength,self.blur_prob,self.noise_prob=photometric_strength,blur_prob,noise_prob
        self.full_image=full_image
        if mask_resize_mode not in ('nearest','nearest_exact'):raise ValueError('Unknown mask resize mode')
        self.mask_interpolation=cv2.INTER_NEAREST_EXACT if mask_resize_mode=='nearest_exact' else cv2.INTER_NEAREST
        self.samples=[]
        for ip,mp in pairs(data,'train'):
            x,y=read_image(ip),encode_mask(read_image(mp))
            if x.shape[:2]!=y.shape: raise ValueError(f'Image/mask mismatch: {ip}')
            if canonical_width and x.shape[1]!=canonical_width:
                size=(canonical_width,round(x.shape[0]*canonical_width/x.shape[1]))
                x=cv2.resize(x,size,interpolation=cv2.INTER_AREA if size[0]<x.shape[1] else cv2.INTER_LINEAR)
                y=cv2.resize(y,size,interpolation=self.mask_interpolation)
            locations=[np.argwhere(y==c)[::16] for c in range(1,4)]
            self.samples.append((x,y,locations))
    def __len__(self): return self.length
    def __getitem__(self,index):
        x,y,locations=self.samples[np.random.randint(len(self.samples))]
        if self.full_image:
            x,y=x.copy(),y.copy()
        else:
            x,y=self.sample_crop(x,y,locations)
        k=np.random.randint(4);x=np.rot90(x,k);y=np.rot90(y,k)
        if np.random.rand()<.5:x=np.fliplr(x);y=np.fliplr(y)
        x=x.astype(np.float32)/255
        if np.random.rand()<.8:
            strength=self.photometric_strength
            contrast=np.random.uniform(1-.15*strength,1+.15*strength);bright=np.random.uniform(-.07*strength,.07*strength)
            x=np.clip((x-.5)*contrast+.5+bright,0,1)
        if np.random.rand()<.4:x=np.power(x,np.random.uniform(1-.15*self.photometric_strength,1+.15*self.photometric_strength))
        if np.random.rand()<self.noise_prob:
            x=np.clip(x+np.random.normal(0,np.random.uniform(0,.015),(*x.shape[:2],1)).astype(np.float32),0,1)
        if np.random.rand()<self.blur_prob:x=cv2.GaussianBlur(x,(3,3),np.random.uniform(.2,.7))
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))),torch.from_numpy(np.ascontiguousarray(y)).long()

    def sample_crop(self,x,y,locations):
        scale=np.exp(np.random.uniform(np.log(self.scale_min),np.log(self.scale_max)))
        raw=max(32,round(self.crop/scale))
        h,w=y.shape
        if np.random.rand()<self.rare_prob:
            available=[p for p in locations if len(p)]
            center=available[np.random.randint(len(available))]
            cy,cx=center[np.random.randint(len(center))]
            cy+=np.random.randint(-raw//4,raw//4+1);cx+=np.random.randint(-raw//4,raw//4+1)
            top=int(np.clip(cy-raw//2,0,max(0,h-raw)));left=int(np.clip(cx-raw//2,0,max(0,w-raw)))
        else:
            top=np.random.randint(max(1,h-raw+1));left=np.random.randint(max(1,w-raw+1))
        x=x[top:top+raw,left:left+raw].copy();y=y[top:top+raw,left:left+raw].copy()
        ph,pw=max(0,raw-x.shape[0]),max(0,raw-x.shape[1])
        if ph or pw:
            x=cv2.copyMakeBorder(x,0,ph,0,pw,cv2.BORDER_REFLECT_101)
            y=cv2.copyMakeBorder(y,0,ph,0,pw,cv2.BORDER_REFLECT_101)
        if raw!=self.crop:
            x=cv2.resize(x,(self.crop,self.crop),interpolation=cv2.INTER_LINEAR)
            y=cv2.resize(y,(self.crop,self.crop),interpolation=self.mask_interpolation)
        return x,y

def worker_seed(worker_id):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

def build_model(config,pretrained=True):
    if config['encoder']=='native_unet_nested':
        from native_nested_unet import NativeNestedUNet
        if config.get('micronet') or config.get('encoder_state') or config.get('detail_refine'):
            raise ValueError('Nested Native U-Net uses its own supervised full-resolution heads')
        return NativeNestedUNet(classes=5 if config.get('morphology') else 4,base_channels=config.get('base_channels',32))
    if config['encoder'] in ('native_unet','native_unet_bn','native_unet_context'):
        from native_unet import NativeResolutionUNet
        if config.get('micronet') or config.get('encoder_state'):raise ValueError('Native U-Net uses no external encoder weights')
        model=NativeResolutionUNet(classes=5 if config.get('morphology') else 4,
            base_channels=config.get('base_channels',32),
            normalization='batch' if config['encoder']=='native_unet_bn' else 'group',
            context='nonlocal' if config['encoder']=='native_unet_context' else 'none')
        if config.get('detail_refine'):
            from detail_refinement import ResidualDetailRefiner
            model=ResidualDetailRefiner(model,classes=5 if config.get('morphology') else 4)
        return model
    if config['encoder']=='dino_vits14':
        from dino_model import DinoPhaseSegmenter
        if config.get('micronet') or config.get('encoder_state') or config.get('detail_refine'):raise ValueError('DINO has its own full-resolution decoder; use external encoder/detail wrapper with SMP backbones')
        return DinoPhaseSegmenter(classes=5 if config.get('morphology') else 4,
            pretrained=pretrained and config.get('weights','imagenet')!='none',
            freeze_encoder=config.get('freeze_encoder',False),
            gradient_checkpointing=config.get('gradient_checkpointing',False))
    weights=config.get('weights','imagenet') if pretrained else None
    if weights=='none':weights=None
    if config.get('micronet') and config.get('encoder_state'):raise ValueError('Choose one external encoder initialization')
    external_encoder_state=config.get('micronet') or config.get('encoder_state')
    model=smp.Unet(encoder_name=config['encoder'],encoder_weights=weights if not external_encoder_state else None,
        in_channels=3,classes=5 if config.get('morphology') else 4)
    if pretrained and external_encoder_state:
        state=torch.load(external_encoder_state,map_location='cpu',weights_only=True)
        if 'state_dict' in state:state=state['state_dict']
        # SMP encoder removes its pretrained classifier; all backbone keys must match.
        model.encoder.load_state_dict(dict(state),strict=True)
        print(json.dumps({'external_encoder_load':{'strict':True,'loaded':True,
            'kind':'micronet' if config.get('micronet') else 'exported_encoder_state'}}),flush=True)
    if config.get('detail_refine'):
        from detail_refinement import ResidualDetailRefiner
        model=ResidualDetailRefiner(model,classes=5 if config.get('morphology') else 4)
    return model

def segmentation_loss(output,target,lovasz,morphology,step,total,
                      ce_weight=1.,dice_weight=.5,lovasz_weight=.4,lovasz_start=.6,thin_ce_weight=1.,auxiliary_weight=1.,cldice_weight=0.,ce_class_weights=None):
    if output.ndim==5:
        # Equal deep-supervision losses, not a loss on already averaged logits.
        return torch.stack([segmentation_loss(output[:,head],target,lovasz,morphology,step,total,
            ce_weight,dice_weight,lovasz_weight,lovasz_start,thin_ce_weight,auxiliary_weight,cldice_weight,ce_class_weights)
            for head in range(output.shape[1])]).mean()
    logits=output[:,:4]
    if ce_class_weights is not None:
        if thin_ce_weight!=1.:raise ValueError('Class-weighted and thin-weighted CE are separate ablations')
        weights=torch.as_tensor(ce_class_weights,device=logits.device,dtype=logits.dtype)
        if weights.shape!=(4,) or not torch.isfinite(weights).all() or not (weights>0).all():
            raise ValueError('CE class weights must contain four finite positive values')
        ce=F.cross_entropy(logits,target,weight=weights)
    elif thin_ce_weight!=1.:
        from phase_losses import phase_cross_entropy
        ce=phase_cross_entropy(logits,target,thin_weight=thin_ce_weight,radius=2)
    else:ce=F.cross_entropy(logits,target)
    probs=logits.softmax(1)
    onehot=F.one_hot(target,4).permute(0,3,1,2).float()
    inter=(probs*onehot).sum((2,3));denom=(probs+onehot).sum((2,3))
    present=onehot.sum((2,3))>0
    dice=(1-(2*inter+1)/(denom+1))[present].mean()
    lov=lovasz(logits,target) if lovasz_weight and step>=total*lovasz_start else logits.sum()*0
    loss=ce_weight*ce+dice_weight*dice+lovasz_weight*lov
    if morphology and auxiliary_weight:
        edge=torch.zeros_like(target,dtype=torch.bool)
        edge[:,1:,:]|=target[:,1:,:]!=target[:,:-1,:]
        edge[:,:-1,:]|=target[:,1:,:]!=target[:,:-1,:]
        edge[:,:,1:]|=target[:,:,1:]!=target[:,:,:-1]
        edge[:,:,:-1]|=target[:,:,1:]!=target[:,:,:-1]
        boundary=F.max_pool2d(edge[:,None].float(),3,1,1)
        boundary_loss=F.binary_cross_entropy_with_logits(output[:,4:5],boundary,pos_weight=output.new_tensor(3.))
        # Local phase fractions supervise spatial distribution at two morphology scales.
        fraction_loss=sum(F.l1_loss(F.avg_pool2d(probs,s,s),F.avg_pool2d(onehot,s,s)) for s in (16,64))
        loss=loss+(.15*auxiliary_weight)*boundary_loss+(.2*auxiliary_weight)*fraction_loss
    if cldice_weight:
        from topology_loss import eutectic_cldice
        loss=loss+cldice_weight*eutectic_cldice(probs,target,iterations=5)
    return loss

@torch.inference_mode()
def tiled_probabilities(model,image,device,tile=512,overlap=128,batch=8,tta=False,canonical_width=0,whole_image=False):
    if whole_image:
        from full_image import full_image_probabilities
        return full_image_probabilities(model,image,device,tta=tta,canonical_width=canonical_width)
    if tile<=overlap or overlap<0 or tile%32 or batch<1:
        raise ValueError('Require tile divisible by 32, 0 <= overlap < tile, batch >= 1')
    original_shape=image.shape[:2]
    if canonical_width and image.shape[1]!=canonical_width:
        size=(canonical_width,round(image.shape[0]*canonical_width/image.shape[1]))
        image=cv2.resize(image,size,interpolation=cv2.INTER_AREA if size[0]<image.shape[1] else cv2.INTER_LINEAR)
    h,w=image.shape[:2];ph,pw=max(h,tile),max(w,tile)
    image=cv2.copyMakeBorder(image,0,ph-h,0,pw-w,cv2.BORDER_REFLECT_101)
    def starts(n):return sorted(set(list(range(0,max(1,n-tile+1),tile-overlap))+[n-tile]))
    coords=[(y,x) for y in starts(ph) for x in starts(pw)]
    win=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.02).astype(np.float32)
    sums=np.zeros((4,ph,pw),np.float32);counts=np.zeros((ph,pw),np.float32)
    for start in range(0,len(coords),batch):
        group=coords[start:start+batch]
        crops=np.stack([image[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in group]).copy()
        z=torch.from_numpy(crops).to(device,dtype=torch.float32)/255
        ps=[]
        for k,flip in ([(0,False),(1,False),(2,False),(3,False),(0,True),(1,True),(2,True),(3,True)] if tta else [(0,False)]):
            zi=torch.rot90(z,k,(-2,-1))
            if flip:zi=zi.flip(-1)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                p=model(normalize(zi))[:,:4].float().softmax(1)
            if flip:p=p.flip(-1)
            ps.append(torch.rot90(p,-k,(-2,-1)))
        pred=torch.stack(ps).mean(0).cpu().numpy()
        for (y,x),p in zip(group,pred):sums[:,y:y+tile,x:x+tile]+=p*win;counts[y:y+tile,x:x+tile]+=win
    probs=(sums/counts[None])[:,:h,:w]
    if original_shape!=(h,w):
        probs=np.stack([cv2.resize(c,(original_shape[1],original_shape[0]),interpolation=cv2.INTER_LINEAR) for c in probs])
    return probs

@torch.inference_mode()
def evaluate(model,data,device,out=None,tile=512,batch=8,tta=False,canonical_width=0,ids=None,whole_image=False,evaluation_scope='provided_valid_all'):
    model.eval();rows=[]
    available=pairs(data,'valid')
    known={image_id(p) for p,m in available}
    if ids is not None and (len(set(ids))!=len(ids) or not set(ids).issubset(known)):
        raise ValueError('Validation IDs contain duplicates or IDs not present in valid')
    if out:
        out=Path(out)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Use a fresh prediction export directory to prevent mixed checkpoints')
        out.mkdir(parents=True,exist_ok=True)
    for ip,mp in available:
        iid=image_id(ip)
        if ids is not None and iid not in ids:continue
        image=read_image(ip);target=encode_mask(read_image(mp))
        p=tiled_probabilities(model,image,device,tile=tile,batch=batch,tta=tta,canonical_width=canonical_width,whole_image=whole_image)
        pred=p.argmax(0).astype(np.uint8)
        row=score_confusion(confusion(target,pred));row['id']=iid;rows.append(row)
        if out:
            np.savez_compressed(Path(out)/f'{iid}.npz',probabilities=p.astype(np.float16))
            Image.fromarray(PALETTE[pred]).save(Path(out)/f'{iid}_prediction.png')
    result=summarize(rows)
    result['evaluation_scope']=evaluation_scope if ids is None else evaluation_scope+'/explicit_subset'
    result['all_evaluation_images_evaluated']={r['id'] for r in rows}==known
    result['all_provided_valid_evaluated']=evaluation_scope=='provided_valid_all' and result['all_evaluation_images_evaluated']
    result['provided_valid_count']=len(known) if evaluation_scope=='provided_valid_all' else None
    result['evaluation_available_count']=len(known)
    if out:
        (out/'COMPLETE.json').write_text(json.dumps({'ids':[r['id'] for r in rows], 'n_images':len(rows),
            'tta':tta,'tile':tile,'canonical_width':canonical_width,'whole_image':whole_image},indent=2))
    return result

def atomic_save(state,path):
    path=Path(path);temporary=path.with_suffix('.partial')
    torch.save(state,temporary);temporary.replace(path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data');ap.add_argument('--run',required=True)
    ap.add_argument('--encoder',default='resnet34');ap.add_argument('--weights',default='imagenet');ap.add_argument('--micronet')
    ap.add_argument('--encoder-state',help='Strict SMP encoder state from a pinned external pretrained source')
    ap.add_argument('--morphology',action='store_true');ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--detail-refine',action='store_true')
    ap.add_argument('--base-channels',type=int,default=32)
    ap.add_argument('--ce-weight',type=float,default=1.);ap.add_argument('--dice-weight',type=float,default=.5)
    ap.add_argument('--lovasz-weight',type=float,default=.4);ap.add_argument('--lovasz-start',type=float,default=.6)
    ap.add_argument('--thin-ce-weight',type=float,default=1.)
    ap.add_argument('--ce-class-weights',type=float,nargs=4,help='Fixed train-only phase weights in Al, PrimarySi, Al3Ni, EutecticSi order')
    ap.add_argument('--auxiliary-weight',type=float,default=1.,help='Scale boundary and local phase-fraction losses; zero keeps the same morphology architecture but disables both auxiliary losses')
    ap.add_argument('--full-image',action='store_true',help='Train batch1 on complete native images; evaluate complete images with padding cropped before loss/metrics')
    ap.add_argument('--batch',type=int,default=8);ap.add_argument('--crop',type=int,default=512)
    ap.add_argument('--canonical-width',type=int,default=0);ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--mask-resize-mode',choices=['nearest','nearest_exact'],default='nearest',help='Training label interpolation only; validation targets always stay at native resolution')
    ap.add_argument('--encoder-lr-multiplier',type=float,default=.1,help='DINO encoder LR multiplier; SMP encoders use --backbone-lr-multiplier')
    ap.add_argument('--backbone-lr-multiplier',type=float,default=1.,help='Optional SMP encoder LR multiplier; decoder and detail heads retain base LR')
    ap.add_argument('--freeze-encoder',action='store_true');ap.add_argument('--gradient-checkpointing',action='store_true')
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--eval-every',type=int,default=250);ap.add_argument('--rare-prob',type=float,default=.4)
    ap.add_argument('--scale-min',type=float,default=.75);ap.add_argument('--scale-max',type=float,default=1.25)
    ap.add_argument('--photometric-strength',type=float,default=1.)
    ap.add_argument('--cldice-weight',type=float,default=0.,help='Optional Eutectic-only soft-clDice, five soft erosion iterations, per-image mean')
    ap.add_argument('--blur-prob',type=float,default=.12);ap.add_argument('--noise-prob',type=float,default=.2)
    ap.add_argument('--resume','--init-checkpoint',dest='resume',help='Load model weights only: a warm start, not optimizer/RNG/step continuation')
    ap.add_argument('--eval-only',action='store_true',help='Evaluate --init-checkpoint using its stored model/geometry config and a fresh --run directory; data,batch,TTA,export and validation IDs remain evaluation options')
    ap.add_argument('--tta',action=argparse.BooleanOptionalAction,default=None)
    ap.add_argument('--save-probs',action='store_true');ap.add_argument('--validation-ids',help='JSON list restricting model-selection images')
    ap.add_argument('--evaluation-scope',default='provided_valid_all',help='Explicitly distinguish internal train-group CV from official provided validation')
    args=ap.parse_args()
    checkpoint=None
    if args.eval_only:
        if not args.resume:raise ValueError('--eval-only requires --init-checkpoint')
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        protected={'data','run','resume','eval_only','tta','save_probs','validation_ids','batch','evaluation_scope'}
        for key,value in checkpoint['config'].items():
            if hasattr(args,key) and key not in protected:setattr(args,key,value)
        if args.tta is None:args.tta=bool(checkpoint['config'].get('tta',False))
        args.full_image=bool(checkpoint['config'].get('full_image',False))
        if args.full_image:args.batch=1
        print(json.dumps({'evaluation_protocol_from_checkpoint':{k:getattr(args,k) for k in
            ['encoder','morphology','detail_refine','crop','canonical_width','full_image','tta']}}),flush=True)
    if args.tta is None:args.tta=False
    if min(args.ce_weight,args.dice_weight,args.lovasz_weight)<0 or args.ce_weight+args.dice_weight+args.lovasz_weight<=0 or not 0<=args.lovasz_start<=1:
        raise ValueError('Loss weights must be nonnegative and nonzero in total; Lovasz start in [0,1]')
    if args.encoder in ('native_unet','native_unet_bn','native_unet_context','native_unet_nested'):args.weights='none'
    if args.thin_ce_weight<1:raise ValueError('Thin CE multiplier must be >=1')
    if args.ce_class_weights is not None:
        if any(not math.isfinite(w) or w<=0 for w in args.ce_class_weights):raise ValueError('CE class weights must be finite and positive')
        if args.thin_ce_weight!=1.:raise ValueError('Class-weighted and thin-weighted CE are separate ablations')
    if not math.isfinite(args.auxiliary_weight) or args.auxiliary_weight<0:raise ValueError('Auxiliary loss weight must be finite and nonnegative')
    if not math.isfinite(args.cldice_weight) or args.cldice_weight<0:raise ValueError('clDice weight must be finite and nonnegative')
    if not math.isfinite(args.backbone_lr_multiplier) or args.backbone_lr_multiplier<=0:
        raise ValueError('Backbone LR multiplier must be finite and positive')
    if args.full_image:
        if args.batch!=1 or args.scale_min!=1 or args.scale_max!=1:
            raise ValueError('Full-image training requires batch1 and scale-min=scale-max=1')
        args.rare_prob=0.
    config=vars(args).copy();run=Path(args.run)
    if run.exists() and any(run.iterdir()):raise FileExistsError('Use a fresh run directory; preserve existing configs, checkpoints and provenance')
    run.mkdir(parents=True,exist_ok=True)
    source_directory=run/'source';source_directory.mkdir()
    for source_path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source_path,source_directory/source_path.name)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=not args.full_image
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    (run/'config.json').write_text(json.dumps(config,indent=2))
    provenance={'torch':torch.__version__,'smp':smp.__version__,'numpy':np.__version__,'opencv':cv2.__version__,
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'seed':args.seed,'strict_determinism':False,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'class_names':CLASS_NAMES,'palette':PALETTE.tolist(),'train_ids':[image_id(p) for p,m in pairs(args.data,'train')],
        'data_hashes':{str(p.relative_to(Path(args.data))):hashlib.sha256(p.read_bytes()).hexdigest() for p,m in pairs(args.data,'train')+pairs(args.data,'valid') for p in (p,m)}}
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    ids=json.loads(Path(args.validation_ids).read_text()) if args.validation_ids else None
    model=build_model(config,pretrained=not bool(args.resume)).to(device)
    if not args.resume and (args.encoder_state or args.micronet):
        encoder_path=Path(args.encoder_state or args.micronet)
        provenance['external_encoder_state']={'path':str(encoder_path),
            'sha256':hashlib.sha256(encoder_path.read_bytes()).hexdigest()}
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if args.resume:
        if checkpoint is None:checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if args.detail_refine and not checkpoint['config'].get('detail_refine'):
            model.backbone.load_state_dict(checkpoint['model'],strict=True)
        else:
            model.load_state_dict(checkpoint['model'],strict=True)
        provenance['initialization_checkpoint']={'path':args.resume,
            'sha256':hashlib.sha256(Path(args.resume).read_bytes()).hexdigest(),
            'source_config':checkpoint['config'],'source_step':checkpoint.get('step'),
            'mode':'weights_only_warm_start_with_fresh_optimizer_and_schedule'}
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if args.eval_only:
        result=evaluate(model,args.data,device,run/'predictions' if args.save_probs else None,args.crop,args.batch,args.tta,args.canonical_width,ids,whole_image=args.full_image,evaluation_scope=args.evaluation_scope)
        (run/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    if hasattr(model,'parameter_groups'):
        if args.backbone_lr_multiplier!=1:raise ValueError('DINO uses --encoder-lr-multiplier')
        groups=model.parameter_groups(args.lr,args.encoder_lr_multiplier)
    elif args.backbone_lr_multiplier!=1:
        backbone=getattr(model,'backbone',model)
        if not hasattr(backbone,'encoder'):raise ValueError('Separate backbone LR requires an SMP encoder')
        encoder_parameters=list(backbone.encoder.parameters());encoder_ids={id(p) for p in encoder_parameters}
        groups=[{'params':encoder_parameters,'lr':args.lr*args.backbone_lr_multiplier,
                 'lr_scale':args.backbone_lr_multiplier,'group_name':'pretrained_encoder'},
                {'params':[p for p in model.parameters() if id(p) not in encoder_ids],
                 'lr':args.lr,'lr_scale':1.,'group_name':'decoder_and_detail_heads'}]
    else:groups=model.parameters()
    optimizer=torch.optim.AdamW(groups,lr=args.lr,weight_decay=.01)
    provenance['optimizer_groups']=[{'name':g.get('group_name','default'),
        'base_lr':g['lr'],'lr_scale':g.get('lr_scale',1.),
        'parameter_count':sum(p.numel() for p in g['params'])} for g in optimizer.param_groups]
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    lovasz=smp.losses.LovaszLoss(mode='multiclass',per_image=True)
    ds=Patches(args.data,args.crop,args.steps*args.batch,args.rare_prob,args.scale_min,args.scale_max,args.canonical_width,
        args.photometric_strength,args.blur_prob,args.noise_prob,args.full_image,args.mask_resize_mode)
    loader=DataLoader(ds,batch_size=args.batch,num_workers=args.workers,pin_memory=device.type=='cuda',worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(args.seed),persistent_workers=args.workers>0)
    best=-1;started=time.time();loss_sum=0
    for step,(x,y) in enumerate(loader,1):
        model.train();x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        lr=args.lr*min(1,step/50)*(.05+.95*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:group['lr']=lr*group.get('lr_scale',1.)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
            if args.full_image:
                from full_image import forward_padded
                output=forward_padded(model,normalize(x))
            else:output=model(normalize(x))
            loss=segmentation_loss(output.float(),y,lovasz,args.morphology,step,args.steps,
                args.ce_weight,args.dice_weight,args.lovasz_weight,args.lovasz_start,args.thin_ce_weight,args.auxiliary_weight,args.cldice_weight,args.ce_class_weights)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
        scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(optimizer);scaler.update();loss_sum+=loss.item()
        decay=min(.995,1-1/(step+1))
        with torch.no_grad():
            for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,1-decay)
            for e,b in zip(ema.buffers(),model.buffers()):e.copy_(b)
        if step%50==0:
            row={'step':step,'loss':loss_sum/50,'elapsed_seconds':round(time.time()-started,1),'lr':lr}
            print(json.dumps(row),flush=True);loss_sum=0
        if step%args.eval_every==0 or step==args.steps:
            result=evaluate(ema,args.data,device,tile=args.crop,batch=args.batch,tta=args.tta,canonical_width=args.canonical_width,ids=ids,whole_image=args.full_image,evaluation_scope=args.evaluation_scope)
            score=result['miou_absent_zero']
            result.update(step=step,elapsed_seconds=round(time.time()-started,1),selection_score_convention='absent_zero')
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print(json.dumps({'validation_step':step,'image_mean_miou':score,'per_class':result['mean_class_iou_absent_zero'],'best':max(best,score)}),flush=True)
            state={'model':ema.state_dict(),'config':config,'step':step,'validation':result}
            atomic_save(state,run/'last.pt')
            if score>best:
                best=score;atomic_save(state,run/'best.pt');(run/'best_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'training_complete':True,'best_image_mean_miou':best,'elapsed_seconds':time.time()-started}),flush=True)

if __name__=='__main__':main()
