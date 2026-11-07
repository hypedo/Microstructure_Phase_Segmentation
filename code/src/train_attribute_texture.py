"""One fixed CPU forest ablation: local texture plus input component attributes.

The 30 added channels use ONLY raw input gray thresholds and connected regions.
No label, filename, train-set statistics, or validation result enters features.
Related to established morphological attribute-profile ideas; not an exact
reproduction of attribute filtering, and not claimed as a new architecture.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits,threadpool_info
import sklearn
import train_texture as baseline
from metrics import CLASS_NAMES,PALETTE,encode_mask,confusion,score_confusion,summarize

THRESHOLDS=(48,80,112,144,176,208)
ATTRIBUTES=('foreground_indicator','component_log1p_area','component_mean_gray_0_1','component_std_gray_0_1','component_covariance_anisotropy')
FEATURE_NAMES=baseline.FEATURE_NAMES+[f'{a}_threshold{t}' for t in THRESHOLDS for a in ATTRIBUTES]
SEED=baseline.SEED
REFERENCE={'title':'Morphological Attribute Profiles for the Analysis of Very High Resolution Images',
           'authors':'Dalla Mura, Benediktsson, Waske, Bruzzone','year':2010,'doi':'10.1109/TGRS.2010.2048116',
           'url':'https://ieeexplore.ieee.org/document/5482208/',
           'author_pdf':'https://rslab.disi.unitn.it/papers/R89-TGRS-DallaMura.pdf',
           'relevance':'Established connected-region multilevel geometric/intensity attributes. This implementation attaches attributes of threshold components; it is not the paper\'s attribute-filter profile construction.'}
cv2.setNumThreads(1)


def component_attributes(gray):
    """30 channels. Background component excluded, with a separate indicator.

    Covariance eigenvalue anisotropy=(lambda_max-lambda_min)/(lambda_max+
    lambda_min) is translation/rotation invariant in continuous coordinates,
    invariant to exact D4 image transforms; rasterization affects other angles.
    Single pixels have zero coordinate covariance and receive anisotropy zero.
    Bounding-box aspect/fill and absolute coordinates are not model features.
    """
    if gray.ndim!=2 or gray.dtype!=np.uint8:raise ValueError('Expected uint8 grayscale input')
    h,w=gray.shape;g=gray.ravel().astype(np.float64)
    x=np.tile(np.arange(w,dtype=np.float64),h);y=np.repeat(np.arange(h,dtype=np.float64),w)
    out=np.empty((h,w,len(THRESHOLDS)*len(ATTRIBUTES)),np.float32)
    counts=[]
    for k,t in enumerate(THRESHOLDS):
        n,cc=cv2.connectedComponents((gray<t).astype(np.uint8),connectivity=8)
        labels=cc.ravel();area=np.bincount(labels,minlength=n).astype(np.float64);safe=np.maximum(area,1)
        mean=np.bincount(labels,weights=g,minlength=n)/safe
        variance=np.maximum(np.bincount(labels,weights=g*g,minlength=n)/safe-mean*mean,0)
        mx=np.bincount(labels,weights=x,minlength=n)/safe;my=np.bincount(labels,weights=y,minlength=n)/safe
        vx=np.maximum(np.bincount(labels,weights=x*x,minlength=n)/safe-mx*mx,0)
        vy=np.maximum(np.bincount(labels,weights=y*y,minlength=n)/safe-my*my,0)
        cov=np.bincount(labels,weights=x*y,minlength=n)/safe-mx*my
        trace=vx+vy;delta=np.sqrt((vx-vy)**2+4*cov*cov)
        # Suppress subtraction roundoff that could otherwise encode coordinate
        # origin in nearly isotropic components at ~1e-9 rather than shape.
        anisotropy=np.round(np.clip(np.divide(delta,trace,out=np.zeros_like(trace),where=trace>1e-12),0,1),6)
        attrs=np.column_stack((np.ones(n),np.log1p(area),mean/255,np.sqrt(variance)/255,anisotropy)).astype(np.float32)
        attrs[0]=0
        out[:,:,k*5:(k+1)*5]=attrs[labels].reshape(h,w,5)
        counts.append(int(n-1))
    if not np.isfinite(out).all():raise ValueError('Nonfinite component attribute')
    return out,counts


def extract_features(gray):
    local,info=baseline.extract_features(gray)
    out=np.empty((*gray.shape,len(FEATURE_NAMES)),np.float32)
    if out.nbytes>850_000_000:raise MemoryError('Feature image exceeds fixed memory allowance')
    out[:,:,:len(baseline.FEATURE_NAMES)]=local;del local
    attributes,counts=component_attributes(gray)
    out[:,:,len(baseline.FEATURE_NAMES):]=attributes
    info.update(feature_bytes=out.nbytes,threshold_component_counts=counts)
    return out,info


def synthetic_check():
    gray=np.full((25,29),240,np.uint8);gray[3:8,4:15]=96;gray[16,23]=32
    f,_=component_attributes(gray)
    rotated,_=component_attributes(np.rot90(gray).copy())
    if not np.allclose(f,np.rot90(rotated,k=-1),atol=1e-5):raise AssertionError('D4 attribute invariance failed')
    # threshold112 contains the55pixel rectangle; singleton belongs to a separate component.
    j=THRESHOLDS.index(112)*5
    assert np.isclose(f[4,5,j+1],np.log1p(55)) and np.isclose(f[4,5,j+2],96/255)
    assert f[16,23,j+4]==0 and not f[0,0].any()
    for value in [0,255]:
        z,_=component_attributes(np.full((9,11),value,np.uint8));assert np.isfinite(z).all()
    return {'D4_rotation':True,'known_area_intensity_singleton':True,'empty_full_foreground':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--run',default='runs/texture_attribute_extratrees');ap.add_argument('--baseline',default='runs/texture_extratrees');ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    checks=synthetic_check()
    if args.self_test:print(json.dumps(checks));return
    def budget_expired(signum,frame):raise TimeoutError('Predeclared1200second CPU experiment budget exceeded')
    if hasattr(signal,'SIGALRM'):
        signal.signal(signal.SIGALRM,budget_expired);signal.alarm(1200)
    started=time.time();run=Path(args.run)
    if run.exists() and any(run.iterdir()):raise FileExistsError('Refuse overwrite or mixed-model cache')
    old_config=json.loads((Path(args.baseline)/'config.json').read_text());old_provenance=json.loads((Path(args.baseline)/'provenance.json').read_text())
    expected={'seed':SEED,'samples_per_train_image':8000,'trees':96,'max_depth':22,'min_samples_leaf':5,'max_features':'sqrt','class_weight':None,'bootstrap':False,'n_jobs':4}
    for k,v in expected.items():
        if old_config[k]!=v:raise ValueError(f'Baseline config mismatch: {k}')
    old_sources={r['id']:r for r in old_provenance['fit_sources']}
    run.mkdir(parents=True);pred_dir=run/'predictions';pred_dir.mkdir()
    config={**old_config,'feature_names':FEATURE_NAMES,'thresholds':THRESHOLDS,'component_connectivity':8,'component_attributes':ATTRIBUTES,
            'baseline_run':str(args.baseline),'input_only_component_features':True,'no_bbox_features':True}
    references=baseline.REFERENCES+[REFERENCE]
    provenance={'config':config,'references':references,'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                'numpy':np.__version__,'sklearn':sklearn.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads(),
                'threadpools':threadpool_info(),'class_names':CLASS_NAMES,'palette':PALETTE.tolist(),'source_sha256':baseline.sha(__file__),
                'baseline_helper_sha256':baseline.sha(baseline.__file__),'fit_sources':[],'synthetic_checks':checks,'same_train_samples_as_baseline':True}
    baseline.write_json(run/'config.json',config);baseline.write_json(run/'provenance.json',provenance)
    rng=np.random.default_rng(SEED);train=baseline.pairs(args.data,'train')
    x_train=np.empty((70*8000,len(FEATURE_NAMES)),np.float32);y_train=np.empty(70*8000,np.uint8)
    with threadpool_limits(limits=1):
        for i,(name,ip,mp) in enumerate(train):
            gray=baseline.read_gray(ip);features,info=extract_features(gray)
            target=encode_mask(np.asarray(Image.open(mp).convert('RGB')))
            if gray.shape!=target.shape:raise ValueError('Image/mask mismatch')
            indices=rng.choice(gray.size,size=8000,replace=False);sample_hash=hashlib.sha256(indices.tobytes()).hexdigest()
            image_hash=baseline.sha(ip);mask_hash=baseline.sha(mp);old=old_sources[name]
            if sample_hash!=old['sample_indices_sha256'] or image_hash!=old['image_sha256'] or mask_hash!=old['mask_sha256']:raise ValueError('Baseline data/sample mismatch')
            x_train[i*8000:(i+1)*8000]=features.reshape(-1,len(FEATURE_NAMES))[indices];y_train[i*8000:(i+1)*8000]=target.ravel()[indices]
            provenance['fit_sources'].append({'id':name,'image_sha256':image_hash,'mask_sha256':mask_hash,'sample_indices_sha256':sample_hash,**info})
            del features,gray,target;gc.collect()
            print(json.dumps({'stage':'train_features','image':i+1,'total':70,'elapsed_seconds':round(time.time()-started,1)}),flush=True)
        provenance['sample_class_counts']=np.bincount(y_train,minlength=4).tolist();provenance['feature_extraction_seconds']=time.time()-started
        baseline.write_json(run/'provenance.json',provenance)
        model=ExtraTreesClassifier(n_estimators=96,max_depth=22,min_samples_leaf=5,max_features='sqrt',n_jobs=4,random_state=SEED,class_weight=None,bootstrap=False)
        before=time.time();print(json.dumps({'stage':'fit_start','samples':len(y_train),'features':len(FEATURE_NAMES)}),flush=True)
        model.fit(x_train,y_train);fit_seconds=time.time()-before;del x_train,y_train;gc.collect()
        model_path=run/'model.joblib';joblib.dump({'model':model,'config':config,'class_names':CLASS_NAMES,'palette':PALETTE},model_path,compress=3);model_hash=baseline.sha(model_path)
        print(json.dumps({'stage':'fit_complete','fit_seconds':fit_seconds}),flush=True)
        rows=[];image_info=[];changed=0
        for i,(name,ip,mp) in enumerate(baseline.pairs(args.data,'valid')):
            gray=baseline.read_gray(ip);features,info=extract_features(gray);p,pred=baseline.native_predict(model,features)
            target=encode_mask(np.asarray(Image.open(mp).convert('RGB')));changed+=int((p.argmax(0)!=pred).sum())
            row=score_confusion(confusion(target,pred));row.update(id=name,width=gray.shape[1],height=gray.shape[0]);rows.append(row);image_info.append({'id':name,**info})
            np.savez_compressed(pred_dir/f'{name}.npz',probabilities=p)
            Image.fromarray(PALETTE[pred]).save(pred_dir/f'{name}_prediction.png')
            del gray,features,p,pred,target;gc.collect()
            print(json.dumps({'stage':'valid_image','image':i+1,'id':name,'miou':row['miou_absent_zero'],'elapsed_seconds':round(time.time()-started,1)}),flush=True)
        result=summarize(rows)
        for suffix,empty in [('zero',0.),('one',1.)]:
            result[f'mean_class_iou_absent_{suffix}']=np.mean([[empty if z is None else z for z in r['class_iou']] for r in rows],axis=0).tolist()
        assert abs(np.mean(result['mean_class_iou_absent_zero'])-result['miou_absent_zero'])<1e-12
        old_eval=json.loads((Path(args.baseline)/'evaluation.json').read_text());old_rows={r['id']:r for r in old_eval['per_image']}
        comparison={'baseline_miou':old_eval['miou_absent_zero'],'attribute_minus_baseline':result['miou_absent_zero']-old_eval['miou_absent_zero'],
                    'per_image_delta':{r['id']:r['miou_absent_zero']-old_rows[r['id']]['miou_absent_zero'] for r in rows}}
        result.update(method='fixed_threshold_component_attributes_plus_multiscale_ExtraTrees',config=config,references=references,model_sha256=model_hash,
                      model_path=str(model_path),train_image_count=70,train_sample_count=560000,provided_valid_count=20,all_provided_valid_evaluated=True,
                      evaluation_scope='provided_valid_all',fit_seconds=fit_seconds,total_seconds=time.time()-started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,fp16_argmax_changed_pixels=changed,
                      validation_input_statistics=image_info,feature_importance=dict(zip(FEATURE_NAMES,map(float,model.feature_importances_))),baseline_comparison=comparison,
                      limitations=['One predefined run, no validation tuning. Same training pixels, labels, seed and forest settings as baseline; only feature set extended.',
                                   'max_features=sqrt is unchanged but candidate-feature count naturally grows from6 to8 when feature count grows from44 to74.',
                                   'Raw fixed grayscale thresholds, all connected attributes use input only. No mask-derived postprocessing or image identifier feature.',
                                   'Whole-image connected context includes image-border-truncated components. Pixel area is resolution dependent; no ground-truth geometric prior used.',
                                   'Continuous-coordinate covariance attributes are rotation invariant; exact D4 holds numerically, arbitrary-angle rasterization changes components.',
                                   'Repeated valid20 method selection is development, not an independent final-test estimate. No test images or labels read.',
                                   'Full-precision forest probabilities score validation; float16 cached argmax can change ties.'])
        baseline.write_json(run/'evaluation.json',result);baseline.write_json(Path('reports')/'texture_attribute_extratrees.json',result)
        baseline.write_json(run/'provenance.json',provenance)
        baseline.write_json(pred_dir/'COMPLETE.json',{'ids':[r['id'] for r in rows],'n_images':20,'model_sha256':model_hash,'class_names':CLASS_NAMES,'native_resolution':True,'probability_dtype':'float16'})
        Path('artifacts').mkdir(exist_ok=True);baseline.contact_sheet(args.data,pred_dir,rows,Path('artifacts')/'texture_attribute_extratrees_contact.png')
        baseline.write_json(run/'COMPLETE.json',{'n_train':70,'n_valid':20,'model_sha256':model_hash,'total_seconds':result['total_seconds']})
        print(json.dumps({'stage':'complete','miou':result['miou_absent_zero'],'mean_class_iou_absent_zero':result['mean_class_iou_absent_zero'],'delta':comparison['attribute_minus_baseline'],'seconds':result['total_seconds'],'peak_rss_GB':result['peak_process_rss_bytes']/1e9}),flush=True)


if __name__=='__main__':main()
