"""One predefined CPU-only multiscale texture / ExtraTrees experiment.

Train-only fitting; native-pixel validation; no parameter search or test use.
Inspired by ilastik and Trainable Weka pixel-classification workflows.
"""
import os

# Set before NumPy/sklearn import. Four parallel trees, one native thread each.
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_name]='1'
if hasattr(os,'sched_getaffinity'):
    _available=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(_available[:4]))

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import cv2
import joblib
import numpy as np
from PIL import Image,ImageDraw
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits,threadpool_info

from metrics import CLASS_NAMES,PALETTE,encode_mask,confusion,score_confusion,summarize

cv2.setNumThreads(1)
SIGMAS=(1,2,4,8,16,32)
FEATURE_NAMES=['raw_gray_0_1','per_image_p01_p99_normalized']+[
    f'{feature}_sigma{sigma}' for sigma in SIGMAS
    for feature in ['gaussian_mean','local_std','gradient_magnitude_scaled',
                    'laplacian_scaled','hessian_eigen_min_scaled',
                    'hessian_eigen_max_scaled','gray_minus_mean']]
SEED=20261001
REFERENCES=[
    {'title':'ilastik: interactive machine learning for (bio)image analysis',
     'year':2019,'url':'https://www.nature.com/articles/s41592-019-0582-9',
     'documentation':'https://www.ilastik.org/documentation/pixelclassification/pixelclassification',
     'relevance':'Multiscale intensity, edge and texture features with a forest pixel classifier.'},
    {'title':'Trainable Weka Segmentation: a machine learning tool for microscopy pixel classification',
     'year':2017,'doi':'10.1093/bioinformatics/btx180',
     'url':'https://pubmed.ncbi.nlm.nih.gov/28369169/',
     'documentation':'https://imagej.net/plugins/tws/index',
     'relevance':'Gaussian scales, Sobel derivatives and Hessian features for microscopy pixel classification.'},
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.partial');partial.write_text(json.dumps(value,indent=2)+'\n');partial.replace(path)


def pairs(data,split):
    if split not in ('train','valid'):raise ValueError('Only train and valid are allowed')
    result=[]
    for ip in sorted((Path(data)/split/'images').glob('*_image.png')):
        name=ip.stem.removesuffix('_image');mp=Path(data)/split/'masks'/f'{name}_mask.png'
        if not mp.exists():raise FileNotFoundError(mp)
        result.append((name,ip,mp))
    expected=70 if split=='train' else 20
    if len(result)!=expected:raise ValueError(f'Expected {expected} {split} images, got {len(result)}')
    return result


def extract_features(gray):
    """44 float32 features, one image at a time (<300 MB at supplied sizes).

    Per-image percentiles use only that input image, equally at train and
    inference. No validation-set aggregate normalization is fitted.
    Derivative eigenvalues/magnitudes are orientation invariant.
    """
    if gray.ndim!=2:raise ValueError('Expected a grayscale image')
    g=gray.astype(np.float32)/255.
    p01,p99=np.percentile(g,[1,99])
    norm=np.clip((g-p01)/max(float(p99-p01),1/255),0,1).astype(np.float32)
    h,w=g.shape;out=np.empty((h,w,len(FEATURE_NAMES)),np.float32)
    if out.nbytes>650_000_000:raise MemoryError('One-image feature allocation exceeds fixed bound')
    out[:,:,0]=g;out[:,:,1]=norm
    squared=norm*norm;k=2
    for sigma in SIGMAS:
        mean=cv2.GaussianBlur(norm,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)
        variance=cv2.GaussianBlur(squared,(0,0),sigma,borderType=cv2.BORDER_REFLECT_101)-mean*mean
        std=np.sqrt(np.maximum(variance,0))
        dx=cv2.Sobel(mean,cv2.CV_32F,1,0,ksize=3,scale=float(sigma)/8,borderType=cv2.BORDER_REFLECT_101)
        dy=cv2.Sobel(mean,cv2.CV_32F,0,1,ksize=3,scale=float(sigma)/8,borderType=cv2.BORDER_REFLECT_101)
        gradient=cv2.magnitude(dx,dy)
        # Finite differences of the Gaussian-smoothed input; scale-normalized.
        dxx=cv2.Sobel(mean,cv2.CV_32F,2,0,ksize=3,scale=float(sigma*sigma)/4,borderType=cv2.BORDER_REFLECT_101)
        dyy=cv2.Sobel(mean,cv2.CV_32F,0,2,ksize=3,scale=float(sigma*sigma)/4,borderType=cv2.BORDER_REFLECT_101)
        dxy=cv2.Sobel(mean,cv2.CV_32F,1,1,ksize=3,scale=float(sigma*sigma)/4,borderType=cv2.BORDER_REFLECT_101)
        trace=dxx+dyy;delta=np.sqrt((dxx-dyy)**2+4*dxy*dxy)
        for feature in (mean,std,gradient,trace,(trace-delta)*.5,(trace+delta)*.5,norm-mean):
            out[:,:,k]=feature;k+=1
    if k!=len(FEATURE_NAMES) or not np.isfinite(out).all():raise ValueError('Invalid features')
    return out,{'p01':float(p01),'p99':float(p99),'feature_bytes':out.nbytes}


def read_gray(path):
    return np.asarray(Image.open(path).convert('L'))


def native_predict(model,features,chunk=131072):
    h,w,n=features.shape;flat=features.reshape(-1,n)
    probabilities=np.empty((4,h*w),np.float16)
    prediction=np.empty(h*w,np.uint8)
    if not np.array_equal(model.classes_,np.arange(4)):raise ValueError('Unexpected class order')
    for first in range(0,len(flat),chunk):
        last=min(first+chunk,len(flat));p=model.predict_proba(flat[first:last])
        if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
        prediction[first:last]=p.argmax(1).astype(np.uint8)
        probabilities[:,first:last]=p.T.astype(np.float16)
    return probabilities.reshape(4,h,w),prediction.reshape(h,w)


def contact_sheet(data,pred_dir,rows,path):
    names=['A11_step1_sample4','A153_step3_sample3','A186_step2_sample2','A71_step2_sample1']
    tile_w,tile_h=340,266;canvas=Image.new('RGB',(3*tile_w,len(names)*tile_h+28),'#171b21');d=ImageDraw.Draw(canvas)
    for j,title in enumerate(['input','ground truth','texture ExtraTrees prediction']):d.text((5+j*tile_w,7),title,fill='white')
    for i,name in enumerate(names):
        row=next(r for r in rows if r['id']==name);top=30+i*tile_h
        d.text((5,top),f'{name} | mIoU={row["miou_absent_zero"]:.4f}',fill='white')
        sources=[Path(data)/'valid/images'/f'{name}_image.png',Path(data)/'valid/masks'/f'{name}_mask.png',pred_dir/f'{name}_prediction.png']
        for j,source in enumerate(sources):
            im=Image.open(source).convert('RGB');im.thumbnail((tile_w,240),Image.Resampling.LANCZOS if j==0 else Image.Resampling.NEAREST)
            canvas.paste(im,(j*tile_w,top+22))
    canvas.save(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data');parser.add_argument('--run',default='runs/texture_extratrees');args=parser.parse_args()
    started=time.time();run=Path(args.run)
    if run.exists() and any(run.iterdir()):raise FileExistsError('Refuse to mix or overwrite texture runs')
    run.mkdir(parents=True,exist_ok=True);pred_dir=run/'predictions';pred_dir.mkdir()
    config={'seed':SEED,'sigmas':SIGMAS,'feature_names':FEATURE_NAMES,'samples_per_train_image':8000,
            'trees':96,'max_depth':22,'min_samples_leaf':5,'max_features':'sqrt','class_weight':None,
            'bootstrap':False,'n_jobs':4,'native_threads':1,'prediction_chunk_pixels':131072,
            'fit_split':'train70 only','validation_tuning':False,'single_predeclared_run':True}
    provenance={'config':config,'references':REFERENCES,'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                'numpy':np.__version__,'sklearn':sklearn.__version__,'opencv':cv2.__version__,
                'threadpools':threadpool_info(),'opencv_threads':cv2.getNumThreads(),'class_names':CLASS_NAMES,'palette':PALETTE.tolist(),
                'source_sha256':sha(__file__),'fit_sources':[]}
    write_json(run/'config.json',config);write_json(run/'provenance.json',provenance)
    rng=np.random.default_rng(SEED);train=pairs(args.data,'train')
    x_train=np.empty((len(train)*8000,len(FEATURE_NAMES)),np.float32);y_train=np.empty(len(train)*8000,np.uint8)
    with threadpool_limits(limits=1):
        for i,(name,ip,mp) in enumerate(train):
            before=time.time();gray=read_gray(ip);target=encode_mask(np.asarray(Image.open(mp).convert('RGB')))
            if gray.shape!=target.shape:raise ValueError('Image/mask mismatch')
            f,info=extract_features(gray);indices=rng.choice(gray.size,size=8000,replace=False)
            x_train[i*8000:(i+1)*8000]=f.reshape(-1,len(FEATURE_NAMES))[indices]
            y_train[i*8000:(i+1)*8000]=target.ravel()[indices]
            provenance['fit_sources'].append({'id':name,'image_sha256':sha(ip),'mask_sha256':sha(mp),'sample_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),**info})
            del f,gray,target;gc.collect()
            print(json.dumps({'stage':'train_features','image':i+1,'total':70,'id':name,'seconds':round(time.time()-before,2),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
        provenance['sample_class_counts']=np.bincount(y_train,minlength=4).tolist();provenance['feature_extraction_seconds']=time.time()-started
        write_json(run/'provenance.json',provenance)
        model=ExtraTreesClassifier(n_estimators=96,max_depth=22,min_samples_leaf=5,max_features='sqrt',
                                   n_jobs=4,random_state=SEED,class_weight=None,bootstrap=False,verbose=0)
        before=time.time();print(json.dumps({'stage':'fit_start','samples':len(y_train),'features':len(FEATURE_NAMES),'n_jobs':4}),flush=True)
        model.fit(x_train,y_train);fit_seconds=time.time()-before
        del x_train,y_train;gc.collect()
        model_path=run/'model.joblib';joblib.dump({'model':model,'config':config,'class_names':CLASS_NAMES,'palette':PALETTE},model_path,compress=3)
        model_hash=sha(model_path)
        print(json.dumps({'stage':'fit_complete','seconds':round(fit_seconds,2),'model_bytes':model_path.stat().st_size,'model_sha256':model_hash}),flush=True)
        rows=[];normalization=[];quantization_changed=0
        for i,(name,ip,mp) in enumerate(pairs(args.data,'valid')):
            before=time.time();gray=read_gray(ip);target=encode_mask(np.asarray(Image.open(mp).convert('RGB')));f,info=extract_features(gray)
            p,pred=native_predict(model,f);quantized=p.argmax(0).astype(np.uint8);quantization_changed+=int((pred!=quantized).sum())
            row=score_confusion(confusion(target,pred));row.update(id=name,width=gray.shape[1],height=gray.shape[0]);rows.append(row)
            normalization.append({'id':name,**info})
            np.savez_compressed(pred_dir/f'{name}.npz',probabilities=p)
            Image.fromarray(PALETTE[pred]).save(pred_dir/f'{name}_prediction.png')
            del f,p,pred,quantized,gray,target;gc.collect()
            print(json.dumps({'stage':'valid_image','image':i+1,'total':20,'id':name,'miou':row['miou_absent_zero'],'class_iou':row['class_iou'],'seconds':round(time.time()-before,2),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
        result=summarize(rows)
        result.update(method='multiscale_grayscale_ExtraTrees',config=config,references=REFERENCES,model_sha256=model_hash,
                      model_path=str(model_path),train_image_count=70,train_sample_count=560000,
                      all_provided_valid_evaluated=True,provided_valid_count=20,evaluation_scope='provided_valid_all',
                      validation_image_normalization=normalization,fit_seconds=fit_seconds,total_seconds=time.time()-started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      probability_export_dtype='float16',fp16_argmax_changed_pixels=quantization_changed,
                      feature_importance=dict(zip(FEATURE_NAMES,map(float,model.feature_importances_))),
                      limitations=['One predefined run; no validation tuning or class-prior balancing.',
                                   'This is still part of a workflow that reuses the same validation set for choosing methods; it is not an independent final-test estimate.',
                                   '8000 uniform pixels per train image gives equal image contribution and retains the resulting natural class prior.',
                                   'Per-image percentile normalization uses only the observed input image at train and inference.',
                                   'Pixels are highly correlated. No pixel-level holdout or OOB score is presented as an independent validation estimate.',
                                   'Native validation scores use full-precision forest probabilities; fp16 export can flip near-tied argmax pixels.',
                                   'No test images or test labels are loaded.'])
        write_json(run/'evaluation.json',result);write_json(Path('reports')/'texture_extratrees.json',result)
        write_json(run/'provenance.json',provenance)
        write_json(pred_dir/'COMPLETE.json',{'ids':[r['id'] for r in rows],'n_images':20,'model_sha256':model_hash,'class_names':CLASS_NAMES,'native_resolution':True,'probability_dtype':'float16'})
        Path('artifacts').mkdir(exist_ok=True);contact_sheet(args.data,pred_dir,rows,Path('artifacts')/'texture_extratrees_contact.png')
        write_json(run/'COMPLETE.json',{'n_train':70,'n_valid':20,'model_sha256':model_hash,'total_seconds':result['total_seconds']})
        print(json.dumps({'stage':'complete','miou_absent_zero':result['miou_absent_zero'],'mean_class_iou':result['mean_class_iou'],'seconds':result['total_seconds'],'peak_rss_GB':result['peak_process_rss_bytes']/1e9}),flush=True)


if __name__=='__main__':main()
