# 대표 모델의 과제 특화 확장: Native64의 공간 세부와 DINOv2의 넓은 시야 문맥을 결합한다.
# DINOv2 (https://arxiv.org/abs/2304.07193)의 ViT-B/14를 동결하여 70장 학습에서
# 전체 foundation encoder를 다시 적합하는 부담을 줄인다. 자연영상 사전학습이 금속조직에
# 항상 유리하다는 보장은 없으므로 여러 문맥 경로/스케일을 검증에서 비교했다.
# 이 파일의 역사적 score 함수는 당시 실험 기록용이다. 현재 공식 평가는 code/evaluate.py,
# 새 정렬 손실 비교는 oct09_official/official_metric.py를 사용한다.

"""Prospective context-halo experiments; canonical palette and strict image-macro IoU.

Image sampling, scSE features and loss follow the fixed four-class order.
It never trains on valid or reads test. Selection bias from prior valid use remains.
"""
import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'references/KOMP-AI-COMPETITION-2026/experiments/server_architecture_lab'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(REFERENCE))
from metrics import PALETTE, CLASS_NAMES, encode_mask, confusion, score_confusion, summarize
from native_unet import NativeResolutionUNet
from train import normalize
from komap_lab.data import NativePatchDataset, patch_origins, manifest
from komap_lab.models import SegmentationModel
from komap_lab.loss import R022Loss


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2)); temporary.replace(path)


def center(x, size):
    h, w = x.shape[-2:]
    assert h >= size and w >= size and (h-size) % 2 == (w-size) % 2 == 0
    return x[..., (h-size)//2:(h+size)//2, (w-size)//2:(w+size)//2]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, **kwargs):
        self.samples=[];self.length=kwargs['samples_per_epoch'];self.photometric=True
        for ip in sorted((root/'train/images').glob('*_image.png')):
            iid=ip.stem.removesuffix('_image')
            x=np.asarray(Image.open(ip).convert('L')).copy()
            y=encode_mask(np.asarray(Image.open(root/'train/masks'/(iid+'_mask.png')).convert('RGB')))
            self.samples.append((x,y))
        assert len(self.samples)==70
    def __len__(self):return self.length
    def __getitem__(self,index):
        x,y=self.samples[int(torch.randint(len(self.samples),()))]
        x=torch.from_numpy(x.copy()).float()[None]/255;y=torch.from_numpy(y.astype(np.int64,copy=True))
        x=NativePatchDataset._photometric(self,x)
        k=int(torch.randint(4,()));x,y=x.rot90(k,(-2,-1)),y.rot90(k,(-2,-1))
        if bool(torch.randint(2,())):x,y=x.flip(-1),y.flip(-1)
        return x,y


class Model(nn.Module):
    def __init__(self, config, initialize=True):
        super().__init__(); self.config = config
        if config['architecture'] == 'reference_resnet101':
            rc = json.loads((REFERENCE / 'configs/B.json').read_text())
            rc['train']['patch'], rc['train']['target'] = config['context'], config['target']
            self.network = SegmentationModel(rc, pretrained=initialize)
        else:
            self.network = NativeResolutionUNet(classes=5, base_channels=64)
            if initialize:
                path = ROOT / config['initial_checkpoint']
                assert digest(path) == config['initial_checkpoint_sha256']
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                self.network.load_state_dict(checkpoint['model'], strict=True)

        if config.get('semantic', False):
            import timm
            # Preserve the base model/sampler CPU RNG when adding modules.
            with torch.random.fork_rng(devices=[]):
                self.semantic = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                    pretrained=initialize, num_classes=0, global_pool='', img_size=518,
                    dynamic_img_size=True, dynamic_img_pad=False, patch_drop_rate=0.,
                    pretrained_cfg_overlay={'hf_hub_id':
                      'timm/vit_base_patch14_dinov2.lvd142m@4685c99dabffe5affac90bd99dbffd25801ae58d'})
                # ViT의 768차원 특징을 native bottleneck 채널로 사상한다.
                # projection을 0으로 초기화하면 추가 시점에 문맥 잔차가 0이 되어 기존 native 출력을 보존한다.
                # ViT는 동결하지만 projection과 native 분할 경로는 학습할 수 있다.
                self.semantic_projection = nn.Conv2d(768, self.network.channels[-1], 1)
                nn.init.zeros_(self.semantic_projection.weight);nn.init.zeros_(self.semantic_projection.bias)
            self.semantic.requires_grad_(False);self.semantic.eval()

    def forward(self, x):
        if self.config['architecture'] == 'reference_resnet101':
            return self.network(x)[:, [3, 0, 1, 2]]
        from full_image import forward_padded
        rgb=normalize(x.repeat(1,3,1,1))
        if not self.config.get('semantic',False):return forward_padded(self.network,rgb)[:,:4]
        h,w=rgb.shape[-2:];ph,pw=(-h)%32,(-w)%32
        if ph or pw:rgb=F.pad(rgb,(0,pw,0,ph),mode='reflect'if ph<h and pw<w else'replicate')
        features=[self.network.stem(rgb)]
        for block in self.network.down_blocks:features.append(block(self.network.pool(features[-1])))
        deep=self.network.context(features[-1]);self.semantic.eval()
        # 한 장의 전체 시야를 장축 약 448 px로 읽되 patch14 배수로 맞춘다.
        # 원본 경로 전체를 이 크기로 축소하지 않으므로 미세 경계는 native encoder에 남는다.
        hh,ww=rgb.shape[-2:];scale=448/max(hh,ww)
        shape=(max(14,round(hh*scale/14)*14),max(14,round(ww*scale/14)*14))
        with torch.no_grad():
            overview=F.interpolate(rgb,size=shape,mode='bilinear',align_corners=False,antialias=True)
            semantic=self.semantic.forward_intermediates(overview,indices=[11],norm=True,output_fmt='NCHW',intermediates_only=True)[0]
        # 문맥 특징은 학습 projection 후 bottleneck 공간 크기로 bilinear 보간해 합산한다.
        # concat으로 채널을 두 배 늘리는 구조가 아니며 최종 decoder는 원래 네 skip을 사용한다.
        deep=deep+F.interpolate(self.semantic_projection(semantic),size=deep.shape[-2:],mode='bilinear',align_corners=False)
        for block,skip in zip(self.network.up_blocks,reversed(features[:-1])):deep=block(deep,skip)
        return self.network.head(deep)[:,:4,:h,:w]


def amp(config):
    return torch.autocast('cuda', dtype=torch.bfloat16) if config['precision'] == 'bf16' else nullcontext()


@torch.inference_mode()
def predict(model, image, config, d4=False):
    model.eval();x=torch.from_numpy(image.copy()).cuda().float()[None,None]/255
    transforms=[(k,f)for k in range(4)for f in (False,True)]if d4 else [(0,False)]
    total=np.zeros((4,*image.shape),np.float32)
    for k,flip in transforms:
        view=x.rot90(k,(-2,-1));view=view.flip(-1)if flip else view
        with amp(config):logits=model(view)
        prob=logits.float().softmax(1)
        if flip:prob=prob.flip(-1)
        prob=prob.rot90(-k,(-2,-1));total+=prob[0].cpu().numpy()/len(transforms)
    return total


@torch.inference_mode()
def evaluate(model, config, out=None, d4=False):
    rows=[]
    if out:Path(out).mkdir(parents=True,exist_ok=False)
    for ip in sorted((ROOT/'data/valid/images').glob('*_image.png')):
        iid=ip.stem.removesuffix('_image')
        image=np.asarray(Image.open(ip).convert('L'))
        target=encode_mask(np.asarray(Image.open(ROOT/'data/valid/masks'/f'{iid}_mask.png').convert('RGB')))
        probabilities=predict(model,image,config,d4=d4)
        # Same cache numeric protocol as existing ensemble.
        raw_cache=probabilities.astype(np.float16)
        probabilities=raw_cache.astype(np.float32)
        probabilities/=probabilities.sum(0,keepdims=True)
        prediction=probabilities.argmax(0)
        rows.append({'id':iid,**score_confusion(confusion(target,prediction))})
        if out:
            metadata={'image_id':iid,'image_sha256':digest(ip),'shape':list(image.shape),
                'class_names':CLASS_NAMES,'palette':PALETTE.tolist(),'d4':d4,'config':config}
            np.savez_compressed(Path(out)/(iid+'.npz'), probabilities=raw_cache,
                                metadata=json.dumps(metadata))
            Image.fromarray(PALETTE[prediction]).save(Path(out)/(iid+'_prediction.png'))
    assert len(rows)==20
    return summarize(rows)


def train(config):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    run=ROOT/'runs/oct04_context'/config['name'];run.mkdir(parents=True,exist_ok=False)
    write(run/'config.json',config)
    write(run/'data_manifest.json',manifest(ROOT/'data',patch=224))
    source=run/'source';source.mkdir();shutil.copy2(__file__,source/'train_context.py')
    write(run/'source_hashes.json',{str(p.relative_to(ROOT)):digest(p) for p in
        [Path(__file__),*(ROOT/'src').glob('*.py'),*(REFERENCE/'komap_lab').glob('*.py')]})
    write(run/'environment.json',{'torch':str(torch.__version__),'numpy':np.__version__,
          'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda})
    dataset=Dataset(ROOT/'data',context=config['context'],target=config['target'],stride=config['stride'],
        rare_fraction=.1,enriched_probability=.5,rare_classes=(1,2),samples_per_epoch=config['steps']*config['batch'])
    loader=DataLoader(dataset,batch_size=config['batch'],shuffle=False,num_workers=0,pin_memory=True)
    model=Model(config).cuda();criterion=R022Loss(class_weights=(1,1,1,2)).cuda()
    groups=[{'params':model.network.parameters(),'lr_scale':1.}]
    if config.get('semantic',False):groups.append({'params':model.semantic_projection.parameters(),'lr_scale':config['semantic_lr_multiplier']})
    optimizer=torch.optim.AdamW(groups,lr=config['lr'],weight_decay=.0001)
    best=-1.;started=time.time()
    try:
        if config['architecture']=='native64':
            initial=evaluate(model,config);best=initial['miou_absent_zero']
            torch.save({'model':model.state_dict(),'config':config,'step':0},run/'best.pt')
            write(run/'best_metrics.json',{'step':0,**initial})
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':0,**initial})+'\n')
            print(json.dumps({'validation_step':0,'miou_absent_zero':best,'initial_checkpoint_only':True}),flush=True)
        for step,(x,y) in enumerate(loader,1):
            model.train();x,y=x.cuda(),y.cuda();optimizer.zero_grad(set_to_none=True)
            lr=config['lr']*(.05+.95*(1+math.cos(math.pi*(step-1)/config['steps']))/2)
            for group in optimizer.param_groups:group['lr']=lr*group['lr_scale']
            with amp(config):output=model(x)
            loss=criterion(output.float(),y)
            if config['consistency']:
                with amp(config):other=center(model(center(x,config['secondary_context'])),config['target'])
                p,q=output.float().softmax(1),other.float().softmax(1)
                mix=((p+q)/2).detach().clamp_min(1e-7)
                consistency_loss=.5*(F.kl_div(p.clamp_min(1e-7).log(),mix,reduction='none').sum(1).mean()+
                         F.kl_div(q.clamp_min(1e-7).log(),mix,reduction='none').sum(1).mean())
                loss=loss+config['consistency']*consistency_loss
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if step==1 or step%50==0:
                status={'step':step,'loss':float(loss),'gradient_norm':float(grad),'elapsed_seconds':time.time()-started,'best':best}
                write(run/'status.json',status);print(json.dumps(status),flush=True)
            if step%config['eval_every']==0 or step==config['steps']:
                score=evaluate(model,config);value=score['miou_absent_zero']
                checkpoint={'model':model.state_dict(),'config':config,'step':step,'optimizer':optimizer.state_dict(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
                torch.save(checkpoint,run/'last.tmp');(run/'last.tmp').replace(run/'last.pt')
                if value>best:
                    best=value;torch.save({'model':model.state_dict(),'config':config,'step':step},run/'best.tmp')
                    (run/'best.tmp').replace(run/'best.pt');write(run/'best_metrics.json',{'step':step,**score})
                with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**score})+'\n')
                print(json.dumps({'validation_step':step,'miou_absent_zero':value,'best':best}),flush=True)
        cp=torch.load(run/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['model'],strict=True)
        for d4 in (False,True):
            tag='d4' if d4 else 'single';result=evaluate(model,config,run/('valid_'+tag),d4=d4)
            write(run/('final_'+tag+'.json'),{'checkpoint_step':cp['step'],**result})
        write(run/'COMPLETE.json',{'best_step':cp['step'],'completed_steps':config['steps'],'test_used':False})
    except BaseException as exc:
        write(run/'FAILED.json',{'error':repr(exc)});raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    train(json.loads(Path(args.config).read_text()))
