"""Cache each fold's own FP32 ensemble, retaining full native pixels."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/oct04_context'))
from phase_ensemble import load_image,sha
from metrics import confusion
from official_metric import per_image_score

def main():
    out=ROOT/'runs/oct09_official/mixtures';out.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'runs/oct04_expanded_ensemble_v4/protocol.json').read_text())
    tasks=['all','0','1','2','3','4'];ws=[];expected=[]
    for t in tasks:
        r=json.loads((ROOT/f'runs/oct09_official/weights/result_{t}.json').read_text())
        w=np.array(r['weights']);ws.append((w/w.sum()).astype(np.float32));expected.append(r)
    ws=np.stack(ws);use=np.flatnonzero(ws.any(0));cat=meta['catalog'].copy()
    cat['models']=[cat['models'][i]for i in use];ws=ws[:,use];rows={t:[]for t in tasks};started=time.time()
    for i,iid in enumerate(cat['image_ids']):
        path=out/f'{iid}.npz'
        if path.exists():
            with np.load(path)as z:mixtures=z['probabilities'];y=z['target']
        else:
            ps,y=load_image(cat,i);mixtures=np.zeros((6,*ps.shape[1:]),np.float32)
            for j in range(len(use)):mixtures+=ws[:,j,None,None,None]*ps[j][None]
            np.savez(path,probabilities=mixtures,target=y)
        for j,t in enumerate(tasks):
            cm=confusion(y,mixtures[j].argmax(0));rows[t].append({'id':iid,'confusion':cm.tolist(),'score':float(per_image_score(cm))})
        print(json.dumps({'cached':i+1,'seconds':round(time.time()-started,1)}),flush=True)
    for j,t in enumerate(tasks):
        ids=expected[j]['evaluation_ids'];cms=np.array([rows[t][i]['confusion']for i in ids])
        assert np.array_equal(cms,np.array(expected[j]['confusions'])),t
    (out/'VERIFIED.json').write_text(json.dumps({'tasks':tasks,'rows':rows,'full_pixel_confusion_parity':True,
        'source_sha256':sha(__file__),'files_sha256':{p.name:sha(p)for p in out.glob('*.npz')}},indent=2))

if __name__=='__main__':main()
