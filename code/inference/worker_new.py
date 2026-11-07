# Phasera 경로별 추론 — 모델·전처리·정밀도 설정을 checkpoint와 함께 보존한다.
# Native U-Net은 미세 공정 Si의 경계 특징을, DINOv2 문맥 경로는 넓은 조직 형태를 읽는다.
# DINOv2: https://arxiv.org/abs/2304.07193
# 새 다운로드/사전학습 초기화를 피하고 저장된 모델 상태를 strict=True로 복원한다.
# 이 파일은 검증된 모듈을 호출하는 실행 어댑터이며 새 backbone을 구현했다는 뜻이 아니다.

"""Isolated inference of one Oct04 model; no ground truth or network access."""
from pathlib import Path
import argparse,json,sys,importlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_json,sha,require,write_json,resolve_file,verify_release

def main():
    ap=argparse.ArgumentParser()
    for k in ['model-dir','model','images','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();root=Path(a.model_dir).resolve()
    manifest,index=verify_release(root);item=next(m for m in manifest['models']if m['name']==a.model)
    require(item['family']!='old_frozen','Old models must use frozen original worker')
    project=root/'project';sys.path.insert(0,str(project/'src'));sys.path.insert(0,str(project/'experiments/oct04_context'))
    import numpy as np
    from PIL import Image
    import torch
    require(torch.cuda.is_available(),'This verified Oct04 reference implementation requires CUDA')
    require(str(torch.__version__)=='2.8.0+cu128','Use pinned Torch2.8.0+cu128 environment')
    torch.set_num_threads(a.threads);torch.backends.cudnn.benchmark=False
    # Context runners explicitly disabled TF32; old/native exporters used their
    # original backend defaults. Keep that distinction in fresh subprocesses.
    if item['family']!='native':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=torch.load(resolve_file(root,item['checkpoint']),map_location='cpu',weights_only=True)
    require(sha(root/item['checkpoint'])==item['checkpoint_sha256'],'Checkpoint identity mismatch')
    config=item['config'];d4=item['d4'];family=item['family']
    if family=='native':
        from train import build_model,tiled_probabilities,read_image
        model=build_model(cp['config'],pretrained=False)
        def inference(path):
            image=read_image(path)
            return tiled_probabilities(model,image,torch.device('cuda'),tile=512,overlap=128,batch=1,tta=d4,
                canonical_width=config.get('canonical_width',0),whole_image=config.get('full_image',False))
    else:
        modules={'context':'train_context','semantic_context':'train_context_semantic_flexible',
            'material_context':'train_whole_matssl','semantic_multilevel':'train_whole_semantic_multilevel','semantic_whole':'train_whole_semantic','semantic_whole_base':'train_whole_semantic_base',
            'whole_r022':'train_whole_r022','features':'train_context_features_t1'}
        require(family in modules,'Unsupported model family')
        module=importlib.import_module(modules[family]);model=module.Model(config,initialize=False)
        if 'inference_scale'in config:
            from infer_whole_semantic_scales import predict
        else:predict=module.predict
        def inference(path):return predict(model,np.asarray(Image.open(path).convert('L')),config,d4=d4)
    # 각 경로의 class 순서·padding·resize·crop·D4 설정은 동결된 원천 모듈에 있다.
    # 모델 간 서로 다른 수치 backend 설정을 하나로 바꾸면 미세하게 출력이 달라질 수 있어
    # 독립 프로세스에서 원래 설정을 복원한다.
    model.load_state_dict(cp['model'],strict=True);model.cuda().eval();step=cp.get('step');del cp
    out=Path(a.output);require(not out.exists(),'Refuse existing worker output');out.mkdir(parents=True)
    rows=load_json(a.images);records=[]
    with torch.inference_mode():
        for row in rows:
            path=Path(row['path']);require(sha(path)==row['sha256'],'Input changed')
            p=inference(path);require(p.shape==(4,*row['shape']) and np.isfinite(p).all() and (p>=0).all(),'Invalid probability map')
            # Direct FP16 cast is intentional: identical to each Oct04 exporter.
            # 캐시 생성 단계의 직접 FP16 변환까지 기존 검증 프로토콜과 일치시킨다.
            # 그 뒤 predict.py에서 FP32 복원·확률 재정규화를 수행한다. 이 순서를 바꾸지 않는다.
            half=p.astype(np.float16);dest=out/(row['id']+'.npz')
            metadata={'id':row['id'],'shape':row['shape'],'image_sha256':row['sha256'],'model':a.model,'checkpoint_sha256':item['checkpoint_sha256']}
            np.savez_compressed(dest,probabilities=half,metadata=json.dumps(metadata));records.append({'id':row['id'],'cache_sha256':sha(dest)})
            print(json.dumps({'model':a.model,'image':row['id'],'completed':len(records)}),flush=True)
    write_json(out/'worker_report.json',{'model':a.model,'step':step,'files':records,'labels_read':False,'torch':str(torch.__version__)})

if __name__=='__main__':main()
