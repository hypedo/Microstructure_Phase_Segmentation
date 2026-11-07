"""Native-resolution, validation-informed error diagnosis; never read test.

Train examples are prespecified morphology representatives. Validation
examples are selected as the four worst checkpoint-selection scores.
These unequal samples do not give an unbiased generalization-gap estimate.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image,ImageDraw

TRAIN_REPRESENTATIVES=['A111_step3_sample4','A130_step1_sample1','A57_step1_sample3','A153_step1_sample4']

def json_write(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')

def group_summary(rows):
    from metrics import summarize
    return summarize(rows)

def analyze_metrics(metrics,data):
    from metrics import CLASS_NAMES
    data=Path(data);rows=metrics['per_image'];train_names=[p.stem.removesuffix('_image') for p in (data/'train/images').glob('*_image.png')]
    train_A={re.match(r'A(\d+)_',n).group(1) for n in train_names}
    groups={'resolution':defaultdict(list),'train_A_overlap':defaultdict(list)}
    for r in rows:
        name=r['id'];image=Image.open(data/'valid/images'/f'{name}_image.png')
        groups['resolution'][f'{image.width}x{image.height}'].append(r)
        A=re.match(r'A(\d+)_',name).group(1)
        groups['train_A_overlap']['seen_A' if A in train_A else 'unseen_A'].append(r)
    cm=np.sum([r['confusion'] for r in rows],axis=0)
    report={'checkpoint_validation_step':metrics.get('step'),'full_validation':metrics,'class_order':CLASS_NAMES,
            'worst_five':sorted(rows,key=lambda x:x['miou_absent_zero'])[:5],
            'groups':{name:{k:group_summary(v) for k,v in d.items()} for name,d in groups.items()},
            'pooled_confusion_row_normalized_diagnostic_only':(cm/cm.sum(1,keepdims=True)).tolist(),
            'interpretation_limits':['Grouping is post-hoc validation analysis, used for development; it is not an independent holdout.','Different resolution and A-overlap groups have different phase/morphology distributions and small sample counts. Association is not causation or proof of leakage.','Pooled confusion is an error diagnostic only; target score remains equal-image mean of four IoUs.']}
    return report

def radius_two_boundary(target):
    import cv2
    p=np.pad(target,1,mode='edge');h,w=target.shape;b=np.zeros((h,w),bool)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:b|=target!=p[1+dy:1+dy+h,1+dx:1+dx+w]
    return cv2.dilate(b.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)

def worst_crop(error,side=320):
    h,w=error.shape;side=min(side,h,w)
    integ=np.pad(error.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    ys=sorted(set(list(range(0,h-side+1,80))+[h-side]));xs=sorted(set(list(range(0,w-side+1,80))+[w-side]))
    best=(-1,0,0)
    for y in ys:
        for x in xs:
            n=int(integ[y+side,x+side]-integ[y,x+side]-integ[y+side,x]+integ[y,x])
            if n>best[0]:best=(n,x,y)
    return [best[1],best[2],side,side],best[0]/(side*side)

def add_visual(sheet_images,image,target,pred,row):
    from metrics import PALETTE
    error=target!=pred;overlay=image.copy();overlay[error]=(.35*image[error]+.65*np.array([255,35,35])).astype(np.uint8)
    rect,fraction=worst_crop(error);row['native_error_crop_xywh']=rect;row['native_crop_error_fraction']=fraction
    sheet_images.append((row,[Image.fromarray(x) for x in [image,PALETTE[target],PALETTE[pred],overlay]]))

def contact(entries,path,native=True):
    side=320;row_h=350 if native else 264;sheet=Image.new('RGB',(4*side,len(entries)*row_h+34),'#161b22');d=ImageDraw.Draw(sheet)
    for col,title in enumerate(['image','ground truth','prediction','red = wrong phase']):d.text((col*side+5,8),title,fill='white')
    for i,(r,images) in enumerate(entries):
        top=34+i*row_h;x,y,w,h=r['native_error_crop_xywh']
        title=f"{r['split']}/{r['id']} mIoU={r['miou_absent_zero']:.4f}"
        if native:title+=f'  crop x={x}, y={y}; error={r["native_crop_error_fraction"]:.1%}'
        d.text((5,top),title,fill='white')
        for j,im in enumerate(images):
            if native:im=im.crop((x,y,x+w,y+h))
            else:
                im=im.copy();im.thumbnail((side,230),Image.Resampling.LANCZOS if j in [0,3] else Image.Resampling.NEAREST)
            sheet.paste(im,(j*side,top+23))
    sheet.save(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--checkpoint',required=True);ap.add_argument('--best-metrics',required=True)
    ap.add_argument('--require-done',required=True);ap.add_argument('--output-prefix',default='error_analysis_imagenet');ap.add_argument('--summary-only',action='store_true');ap.add_argument('--batch',type=int,default=2)
    ap.add_argument('--train-only',action='store_true');ap.add_argument('--gpu-memory-gb',type=float,default=3.5)
    args=ap.parse_args();root=Path.cwd();data=Path(args.data);prefix=args.output_prefix
    best=json.loads(Path(args.best_metrics).read_text());summary=analyze_metrics(best,data)
    json_write(root/'reports'/f'{prefix}_summary.json',summary)
    if args.summary_only:
        print(json.dumps({'summary_only':True,'validation_miou':best['miou_absent_zero']}));return
    if not Path(args.require_done).exists():raise RuntimeError('Training completion marker is absent; refuse concurrent checkpoint inference.')
    import torch
    from train import build_model,tiled_probabilities,read_image
    from metrics import CLASS_NAMES,encode_mask,confusion,score_confusion,summarize,PALETTE
    torch.set_num_threads(2);device=torch.device('cuda')
    if args.gpu_memory_gb<=0:raise ValueError('Positive GPU memory allowance is required')
    torch.cuda.set_per_process_memory_fraction(min(.12,args.gpu_memory_gb*1e9/torch.cuda.get_device_properties(device).total_memory))
    checkpoint_path=Path(args.checkpoint);checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=checkpoint['config'];model=build_model(config,pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    # Select the actual checkpoint's validation records, avoiding snapshot races.
    checkpoint_metrics=checkpoint['validation'];summary=analyze_metrics(checkpoint_metrics,data)
    json_write(root/'reports'/f'{prefix}_summary.json',summary)
    valid_names=[] if args.train_only else [r['id'] for r in sorted(checkpoint_metrics['per_image'],key=lambda x:x['miou_absent_zero'])[:4]]
    rows=[];visuals={'train':[],'valid':[]};artifact_dir=root/'artifacts';artifact_dir.mkdir(parents=True,exist_ok=True)
    start=time.time()
    for split,names in [('train',TRAIN_REPRESENTATIVES),('valid',valid_names)]:
        for name in names:
            image=read_image(data/split/'images'/f'{name}_image.png');target=encode_mask(read_image(data/split/'masks'/f'{name}_mask.png'))
            probs=tiled_probabilities(model,image,device,tile=512,overlap=128,batch=args.batch,tta=False,canonical_width=0)
            pred=probs.argmax(0).astype(np.uint8);row=score_confusion(confusion(target,pred));row.update(id=name,split=split,width=image.shape[1],height=image.shape[0])
            cm=np.array(row['confusion']);row['class_recall']=(np.diag(cm)/np.maximum(cm.sum(1),1)).tolist();row['class_precision']=(np.diag(cm)/np.maximum(cm.sum(0),1)).tolist()
            row['target_phase_fraction']=(cm.sum(1)/cm.sum()).tolist();row['predicted_phase_fraction']=(cm.sum(0)/cm.sum()).tolist()
            boundary=radius_two_boundary(target);correct=pred==target
            row['boundary_diagnostics']={}
            for c,cl in enumerate(CLASS_NAMES):
                member=target==c
                row['boundary_diagnostics'][cl]={}
                for region,mask in [('boundary2',member&boundary),('interior',member&~boundary)]:
                    n=int(mask.sum());row['boundary_diagnostics'][cl][region]={'pixels':n,'recall':float(correct[mask].mean()) if n else None}
            row['mean_confidence_wrong']=float(probs.max(0)[~correct].mean()) if (~correct).any() else None
            if split=='valid':
                original=next(r for r in checkpoint_metrics['per_image'] if r['id']==name)
                row['stored_selection_miou']=original['miou_absent_zero'];row['tile512_minus_stored_selection_miou']=row['miou_absent_zero']-original['miou_absent_zero']
            add_visual(visuals[split],image,target,pred,row)
            Image.fromarray(PALETTE[pred]).save(artifact_dir/f'{prefix}_{split}_{name}_prediction.png')
            rows.append(row);print(json.dumps({'image_done':name,'split':split,'miou':row['miou_absent_zero'],'class_iou':row['class_iou']}),flush=True)
    for split,entries in visuals.items():
        if not entries:continue
        contact(entries,artifact_dir/f'{prefix}_{split}_native.png',True)
        contact(entries,artifact_dir/f'{prefix}_{split}_overview.png',False)
    report={'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),'checkpoint_step':checkpoint['step'],'checkpoint_config':config,
            'inference':{'tile':512,'overlap':128,'batch':args.batch,'tta':False,'canonical_width':0,'device':str(device),'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'max_cuda_reserved_bytes':torch.cuda.max_memory_reserved()},
            'class_order':CLASS_NAMES,'selection':{'train':'Four prespecified visual/morphology representatives; not a random sample.','valid':'Four lowest image-mIoU validation records for this selected checkpoint.'},
            'train_representative_scores':summarize([r for r in rows if r['split']=='train']),
            'valid_worst_four_scores':summarize([r for r in rows if r['split']=='valid']) if valid_names else None,
            'all_twenty_valid_stored_checkpoint_score':checkpoint_metrics['miou_absent_zero'],
            'elapsed_seconds':time.time()-start,'limitations':['This is validation-informed development analysis, not an untouched generalization test.','Training representatives and worst-validation selection have intentionally unequal sampling; their score difference is descriptive and must not be called an unbiased generalization gap.','Inference uses native tile512 without TTA or canonical resize. Compare stored selection score only after checking checkpoint crop/canonical settings.','No test images, test filenames, or test labels are read.']}
    json_write(root/'reports'/f'{prefix}_inference.json',report)
    json_write(root/'reports'/f'{prefix}_COMPLETE.json',{'checkpoint_sha256':report['checkpoint_sha256'],'train_ids':TRAIN_REPRESENTATIVES,'valid_ids':valid_names})
    print(json.dumps({'done':True,'train_representative_miou':report['train_representative_scores']['miou_absent_zero'],'valid_worst_four_miou':report['valid_worst_four_scores']['miou_absent_zero'] if valid_names else None,'gpu_peak_GB':report['inference']['max_cuda_reserved_bytes']/1e9}),flush=True)

if __name__=='__main__':main()
