from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.font_manager import fontManager
import argparse
ap=argparse.ArgumentParser(description='Reproduce report figures from provided competition data')
ap.add_argument('--data',required=True,help='Folder containing train and valid splits')
ap.add_argument('--output',required=True,help='Fresh output directory')
a=ap.parse_args()
R=Path(__file__).resolve().parents[1]; D=Path(a.data).resolve()
T=R.parent/'01_Proposal'; O=Path(a.output).resolve();O.mkdir(parents=True,exist_ok=False)
fontManager.addfont(str(T/'fonts/NanumGothic-Regular.ttf'));fontManager.addfont(str(T/'fonts/NanumGothic-Bold.ttf'))
plt.rcParams.update({'font.family':['NanumGothic','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,'savefig.facecolor':'white'})
colors=['#997f4c','#4cb24c','#194c99','#cccccc']; names=['Al','Primary Si','Al₃Ni','Eutectic Si']
def save(fig,name):
 fig.savefig(O/(name+'.png'),dpi=260,bbox_inches='tight',pad_inches=.09);fig.savefig(O/(name+'.pdf'),bbox_inches='tight',pad_inches=.09);plt.close(fig)
def read(split,i,mask=False):return np.asarray(Image.open(D/split/('masks'if mask else'images')/(i+('_mask.png'if mask else'_image.png'))).convert('RGB'))
# Empirical phase area means, not pooled pixel counts.
d=json.loads((R/'reports/data_audit.json').read_text())
fig,ax=plt.subplots(figsize=(7.3,2.0));x=np.arange(4)
for j,sp in enumerate(['train','valid']):
 a=np.array(d['summary'][sp]['class_image_mean_proportions'])[[3,0,1,2]]*100
 bars=ax.bar(x+(j-.5)*.33,a,.30,label=sp+(' 70장'if j==0 else' 20장'),color=colors,alpha=1 if j==0 else .48,edgecolor='#444',linewidth=.4)
 for b,v in zip(bars,a):ax.text(b.get_x()+b.get_width()/2,v+.9,f'{v:.1f}',ha='center',fontsize=9)
ax.set_xticks(x,names);ax.set_ylabel('이미지별 면적분율 평균 (%)');ax.set_ylim(0,72);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncol=2,fontsize=9,loc='upper right');save(fig,'phase_distribution')
fig,axs=plt.subplots(2,3,figsize=(7.4,3.8))
for row,(sp,i)in enumerate([('train','A111_step3_sample4'),('valid','A117_step4_sample2')]):
 a=read(sp,i);b=read(sp,i,True);h,w=a.shape[:2];crop=(slice(h//3,h//3+280),slice(w//3,w//3+360))
 for c,img in enumerate([a,b,a[crop]]):axs[row,c].imshow(img);axs[row,c].axis('off')
 axs[row,0].set_title(sp+' '+i,fontsize=8);axs[row,1].set_title('정답 4상 마스크',fontsize=9);axs[row,2].set_title('원본 국소 영역 확대',fontsize=9)
fig.tight_layout(pad=.7);save(fig,'morphology')
# The revised diagram is generated separately by report_architecture.py.
# Representative comparisons include one difficult case and one good case.
fig,axs=plt.subplots(2,4,figsize=(8,3.35))
r=json.loads((R/'reports/oct09_fresh_preview_evaluation.json').read_text());scores={i['id']:i['score'] for i in r['per_image']}
for row,i in enumerate(['A71_step2_sample1','A117_step4_sample2']):
 a=read('valid',i);g=read('valid',i,True);p=np.asarray(Image.open(R/'validation_predictions'/(i+'_prediction.png')).convert('RGB'));e=np.any(g!=p,axis=-1)
 error=(a*.45).astype(np.uint8);error[e]=[211,55,73]
 for c,img in enumerate([a,g,p,error]):axs[row,c].imshow(img);axs[row,c].axis('off')
 axs[row,0].set_title(i,fontsize=7.5);axs[row,1].set_title('정답',fontsize=9);axs[row,2].set_title(f'예측 {scores[i]*100:.2f}%',fontsize=9);axs[row,3].set_title('오류 픽셀 표시',fontsize=9)
fig.tight_layout(pad=.6);save(fig,'qualitative')
# Score comparison, show full scale enough to avoid overstating tiny gains.
fig,ax=plt.subplots(figsize=(7.3,1.9));labels=['기존 최종','조건부 보정','보정 + 성분 제거','보정 단계 교차검증'];vals=[80.5276637,81.0151361,81.0664748,80.6409542]
ax.barh(labels[::-1],vals[::-1],color=['#849aa9','#297d81','#87b6aa','#b5c4cc'],height=.6);ax.set_xlim(75,83);ax.axvline(81,ls='--',lw=.8,color='#666');ax.set_xlabel('공식 GT-present image-macro IoU (%)');ax.spines[['top','right']].set_visible(False)
for k,v in enumerate(vals[::-1]):ax.text(v+.08,k,f'{v:.4f}',va='center',fontsize=9)
fig.tight_layout();save(fig,'performance')
print(O)
# Typeset metric and calibration without ambiguous Unicode superscripts.
fig,ax=plt.subplots(figsize=(8.8,.65));ax.axis('off');ax.text(.5,.52,r'$S=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|C_i|}\sum_{c\in C_i}\frac{TP_{ic}}{TP_{ic}+FP_{ic}+FN_{ic}}$',ha='center',va='center',fontsize=17);save(fig,'metric_formula')
fig,ax=plt.subplots(figsize=(8.8,1.5));ax.axis('off')
lines=[r'$m_c=\mathrm{mean}_x\,q_c(x),\quad r=2m_{\mathrm{Ni}}/\max(m_{\mathrm{Ni}}+m_{\mathrm{Eut}},10^{-8})-1$',r'$h=\mathrm{clip}\!\left(\frac{\log\max(m_{\mathrm{Primary}},10^{-6})-\log 0.05}{4},-1,1\right)$',r'$f=[1,r,h],\quad b_{\mathrm{Al}}=0,\quad b_c=\beta_c^\top f,\quad \hat y(x)=\arg\max_c[\log\max(q_c(x),10^{-30})+b_c]$']
for y,t in zip([.85,.49,.13],lines):ax.text(.5,y,t,ha='center',va='center',fontsize=12)
save(fig,'calibration_formula')
