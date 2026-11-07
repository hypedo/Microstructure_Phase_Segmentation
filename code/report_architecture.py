"""Paper-style, rectangular scientific diagram of the implemented inference graph."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.font_manager import fontManager
import argparse
p=argparse.ArgumentParser(description='Recreate the final paper-style architecture figure')
p.add_argument('--output',required=True);a=p.parse_args()
O=Path(a.output).resolve();O.mkdir(parents=True,exist_ok=True)
R=Path(__file__).resolve().parents[1];T=R.parent/'01_Proposal'
fontManager.addfont(str(T/'fonts/NanumGothic-Regular.ttf'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(10.5,6.2));ax=fig.add_axes([.01,.01,.98,.98]);ax.set_xlim(0,120);ax.set_ylim(0,75);ax.axis('off')
edge='#334454';blue='#eaf0f7';green='#e7f1ed';sand='#f5eee2'
labels=[]
def text(x,y,t,fs=10,weight='normal',**kwargs):
 ob=ax.text(x,y,t,fontsize=fs,fontweight=weight,ha='center',va='center',color='#17242f',**kwargs);labels.append(ob);return ob
def box(x,y,w,h,title,detail='',fill=blue,fs=10):
 ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,lw=.85,joinstyle='miter'))
 text(x+w/2,y+h*(.70 if detail else .5),title,fs,'bold')
 if detail:text(x+w/2,y+h*.30,detail,fs-1,linespacing=1.25)
def line(points,color=edge,dashed=False):
 ax.plot(*zip(*points),color=color,lw=.85,ls='--'if dashed else'-',solid_joinstyle='miter',zorder=1)
def arrow(a,b,color=edge,dashed=False):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.85,color=color,linestyle='--'if dashed else'-',shrinkA=0,shrinkB=1,zorder=2))
ax.text(1,73,'a  Phasera inference pipeline',fontsize=12,fontweight='bold',va='center',color='#17242f')
box(1,49.5,12,12,'OM image','Native H × W\nGrayscale',fs=9.4)
box(19,61,30,9,'Native U-Net','Base channels: 32 / 64 / 128',fs=10)
box(19,50,30,9,'DINOv2 + Native64','Frozen ViT-B / ViT-S context',fill=green,fs=10)
box(19,39,30,9,'Complementary backbones','ResNet101 scSE / MiT / DINO',fs=10)
line([(13,55.5),(16,55.5)]);line([(16,43.5),(16,65.5)])
for y in [65.5,54.5,43.5]:arrow((16,y),(19,y));line([(49,y),(52,y)])
line([(52,43.5),(52,65.5)]);arrow((52,55.5),(55,55.5))
box(55,49,16,13,'Probability fusion','Inverse TTA\nFP32 weighted sum',fs=9.4)
arrow((71,55.5),(77,55.5))
box(77,49,21,13,'Conditional calibration','Predicted phase context\nPrimary-Si component filter',fill=sand,fs=9.1)
arrow((98,55.5),(104,55.5))
box(104,49,15,13,'Phase mask','4-class RGB\nNative H × W',fill=green,fs=9.4)
text(86,44,'log q + βf(q) → argmax',10)
text(60,35.8,'9 unique checkpoints · 15 inference routes · All probability maps aligned to the input coordinates',9)
line([(1,33.8),(119,33.8)],color='#cbd3da')
ax.text(1,31.6,'b  Native64 with frozen DINOv2-Base context',fontsize=11.5,fontweight='bold',va='center',color='#17242f')
box(1,9.5,12,10,'Input','H × W',fs=10)
box(19,24,24,6,'Global overview','Long side ≈ 448; 14-px grid',fill=green,fs=9.5)
box(49,24,18,6,'Frozen DINOv2','ViT-B/14 · 768 channels',fill=green,fs=9.4)
box(73,24,25,6,'Trainable 1×1 projection','Zero-initialized · 768 → 768',fill=green,fs=9.4)
line([(7,19.5),(7,27)]);arrow((7,27),(19,27));arrow((43,27),(49,27));arrow((67,27),(73,27))
box(19,9,24,11,'Native64 encoder','64 → 128 → 256 → 512 → 768',fs=9.8)
box(49,9,18,11,'Bottleneck sum','Native + resized context\n1/16 spatial resolution',fill=green,fs=9.4)
box(73,9,25,11,'Four decoder stages','512 → 256 → 128 → 64\nUpsample + concatenate',fs=9.8)
box(104,9,15,11,'1×1 head','4 phase logits\nCrop to H × W',fill=sand,fs=9.3)
arrow((13,14.5),(19,14.5));arrow((43,14.5),(49,14.5));arrow((67,14.5),(73,14.5));arrow((98,14.5),(104,14.5))
line([(85.5,24),(85.5,22),(58,22)],color='#26705b');arrow((58,22),(58,20),color='#26705b')
line([(31,9),(31,4),(85.5,4)],dashed=True);arrow((85.5,4),(85.5,9),dashed=True)
text(58,1.7,'Four encoder–decoder skip connections',9)
# Label bounding boxes must not overlap. Panel titles are outside all nodes.
fig.canvas.draw();ren=fig.canvas.get_renderer()
for i,a in enumerate(labels):
 for b in labels[i+1:]:
  assert not a.get_window_extent(ren).overlaps(b.get_window_extent(ren)),(a.get_text(),b.get_text())
for ext in ['png','pdf','svg']:
 fig.savefig(O/('architecture_v2.'+ext),dpi=300,bbox_inches='tight',pad_inches=.06)
print('Rendered rectangular architecture; label-overlap check passed')
