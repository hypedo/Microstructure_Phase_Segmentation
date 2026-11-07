# Phasera 방법론 안내 — 최종 실행 경로 (결과서 3-1, 3-2).
# 서로 다른 backbone, 해상도, D4 설정은 상의 크기·질감별 상보성을 확보하기 위한 후보였다.
# 최종 구성은 9개 고유 checkpoint를 15개 경로로 실행한다. 같은 checkpoint의 다른
# 추론 스케일/TTA를 별개 학습 모델로 세지 않는다. 실제 구성과 순서는 manifest가 기준이다.
# KoMaP의 상호배타적 4상 확률을 원본 좌표계에서 결합한 뒤
# 예측 조성 기반 조건부 logit 보정을 적용한다 (postprocess.py 참조).
# 전체 검증을 반복 사용해 구성·보정값을 선택했으므로 81.0665%는 개발 점수이다.
# 추론에서는 정답, GT 존재 상, 영상 ID별 lookup을 사용하지 않는다.

"""Verified ordered probability ensemble; image-only inference, frozen weights."""
from pathlib import Path
import argparse,json,os,subprocess,sys,tempfile
from common import verify_release,require,write_json,load_json,sha,input_inventory,PALETTE

from postprocess import apply_postprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--input');p.add_argument('--output');p.add_argument('--threads',type=int,default=4);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    root=Path(a.model_dir).resolve();manifest,index=verify_release(root)
    if a.validate_only:print(json.dumps({'verified':True,'models':len(manifest['models']),'inference_executed':False}));return
    require(a.input and a.output,'Input/output required');rows=input_inventory(a.input);out=Path(a.output).resolve();require(not out.exists(),'Output must be fresh');require(not out.is_relative_to(root) and not out.is_relative_to(Path(a.input).resolve()),'Output cannot be inside model/input')
    require(1<=a.threads<=64,'Threads must be in1..64')
    import numpy as np
    from PIL import Image
    env={k:os.environ[k]for k in ['PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']if k in os.environ};env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads))
    # 가중치 탐색과 실제 추론을 분리한다. 이 함수는 manifest의 고정된 최종값만 읽는다.
    # 44개 후보 재탐색에서 기존 가중치를 대체할 검증된 개선을 얻지 못했으므로 최종 향상은
    # 조건부 보정/성분 정제에서 나온다. 이 실행기는 검증·test를 보고 추가 적합하지 않는다.
    weights=np.asarray(manifest['weights'],np.float64);weights=(weights/weights.sum()).astype(np.float32);active=[(m,w)for m,w in zip(manifest['models'],weights)if w>0]
    out.mkdir(parents=True);reports=[];outputs=[]
    with tempfile.TemporaryDirectory(prefix='.oct09-infer-',dir=out.parent)as temp:
        work=Path(temp);write_json(work/'images.json',rows)
        for item,w in active:
            name=item['name'];dest=work/name
            if item['family']=='old_frozen':
                command=[sys.executable,'-I','-B',str(root/'old_tools/worker.py'),'--model-dir',str(root/'old_bundle'),'--model',name,'--images',str(work/'images.json'),'--output',str(dest),'--device','cuda','--threads',str(a.threads)]
            else:
                command=[sys.executable,'-I','-B',str(Path(__file__).with_name('worker_new.py').resolve()),'--model-dir',str(root),'--model',name,'--images',str(work/'images.json'),'--output',str(dest),'--threads',str(a.threads)]
            subprocess.run(command,check=True,cwd=work,env=env);reports.append(load_json(dest/'worker_report.json'))
        # 모델별 D4 역변환과 원본 H×W 복원은 worker에서 완료된다.
        # FP16 캐시를 FP32로 복원·재정규화하고 고정 순서로 가중합하여 수치 경로를 보존한다.
        # argmax 마스크를 투표하면 불확실성 정보가 사라지므로 확률 공간에서 결합한다.
        for row in rows:
            total=np.zeros((4,*row['shape']),np.float32)
            for (item,w),report in zip(active,reports):
                dest=work/item['name']/(row['id']+'.npz');record=next(x for x in report['files']if x['id']==row['id']);require(sha(dest)==record['cache_sha256'],'Cache changed')
                with np.load(dest,allow_pickle=False)as z:
                    prob=z['probabilities'].astype(np.float32);metadata=json.loads(str(z['metadata'].item()))
                require(metadata['image_sha256']==row['sha256'] and metadata['shape']==row['shape'],'Cache/input mismatch')
                require(prob.shape==total.shape and np.isfinite(prob).all() and (prob>=0).all() and (prob.sum(0)>0).all(),'Invalid cache probabilities')
                prob/=prob.sum(0,keepdims=True);total+=prob*w
            # 후처리 입력은 현재 영상의 결합 확률뿐이다. 설정을 동결한 뒤 적용하며,
            # 4상 RGB palette로만 출력하여 대회 라벨의 색과 공간 크기를 보존한다.
            prediction=apply_postprocess(total,manifest.get('postprocess',{}));dest=out/(row['id']+'_prediction.png');Image.fromarray(np.asarray(PALETTE,np.uint8)[prediction]).save(dest)
            outputs.append({'id':row['id'],'input_sha256':row['sha256'],'shape':row['shape'],'prediction':dest.name,'sha256':sha(dest)})
    verify_release(root)
    for row in rows:require(sha(row['path'])==row['sha256'],'Input changed during inference')
    write_json(out/'RUN_COMPLETE.json',{'manifest_sha256':sha(root/'manifest.json'),'n_images':len(rows),'outputs':outputs,'worker_reports':reports,'weights':manifest['weights'],'postprocess':manifest.get('postprocess',{}),'labels_read':False,'test_metric':None})
    print(json.dumps({'complete':True,'images':len(rows)}))

if __name__=='__main__':main()
