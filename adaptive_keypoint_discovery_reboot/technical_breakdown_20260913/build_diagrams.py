from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'assets'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
colors={'frozen':'#f8e8b4','train':'#f7d4cc','geo':'#d6e8d5','data':'#d7e8f2','note':'#e6def0'}
def start(w=14,h=6):
 f,a=plt.subplots(figsize=(w,h));a.set_xlim(0,14);a.set_ylim(0,h);a.axis('off');return f,a
def box(a,x,y,w,h,t,c='data',fs=11):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.08',lw=1,ec='#333333',fc=colors[c]));a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(a,x,y,u,v,dash=False,col='#333333'):
 a.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=13,lw=1.3,linestyle='--' if dash else '-',color=col))
def save(f,n):
 f.savefig(OUT/(n+'.png'),dpi=300,bbox_inches='tight');f.savefig(OUT/(n+'.svg'),bbox_inches='tight');plt.close(f)
f,a=start(h=6)
box(a,.15,2.5,1.6,1,'Real plant RGB\n+ automatic ROI')
box(a,2.3,4.5,2.4,1,'G1-prime Teacher\ngeometry + DINOv2','geo')
box(a,5.2,4.5,2.2,1,'9-view consensus\n(x, y, confidence)','geo')
box(a,8,4.5,2.4,1,'Gaussian targets\n129 x 129','data')
box(a,11,4.5,2.5,1,'Weighted BCE\n+ 0.02 count loss','train')
box(a,2.3,2.5,2.4,1,'DINOv2 ViT-S/14-Reg\nALL backbone frozen','frozen')
box(a,5.2,2.5,2.2,1,'Conv heatmap head\n830,401 parameters','train')
box(a,8,2.5,2.4,1,'Sigmoid + peak NMS\nvariable point count','data')
box(a,11,2.5,2.5,1,'Accepted projections\npoint-conditioned MST','geo')
box(a,8,.4,2.4,1,'Local-width decoder\nlearned basal anchor','geo')
box(a,11,.4,2.5,1,'PCHIP + geometry\ncandidate phenotypes','geo')
arr(a,1.75,3,2.3,3);arr(a,1.75,3.5,2.3,5);arr(a,4.7,5,5.2,5);arr(a,7.4,5,8,5);arr(a,10.4,5,11,5)
arr(a,4.7,3,5.2,3);arr(a,7.4,3,8,3);arr(a,10.4,3,11,3);arr(a,12.25,2.5,9.2,1.4);arr(a,10.4,.9,11,.9)
arr(a,6.3,3.5,12.2,4.5,True);arr(a,11.3,4.5,6.4,3.5,True,'#a82c2c')
a.text(.15,.65,'Solid arrows: inference / data flow\nDashed arrows: training only\nGreen: deterministic geometry   Yellow: frozen   Pink: trainable',fontsize=10)
save(f,'06_overview')
f,a=start(h=6)
box(a,.1,4.45,2,1,'RGB 518 x 518\n14 x 14 patch embed','frozen')
box(a,2.65,4.45,3.4,1,'1 CLS + 4 registers + 1369 patches\n1374 tokens x 384 dimensions','frozen')
box(a,6.65,4.45,2.2,1,'Transformer blocks\n12 layers / 6 heads','frozen')
box(a,9.45,4.45,1.7,1,'Final LayerNorm','frozen')
box(a,11.7,4.45,2.1,1,'Patch tokens only\n1369 x 384','data')
for x,u in [(2.1,2.65),(6.05,6.65),(8.85,9.45),(11.15,11.7)]:arr(a,x,4.95,u,4.95)
box(a,.3,1.8,1.3,.8,'Input Z','data');box(a,2.1,1.8,1.4,.8,'LayerNorm','frozen');box(a,4,1.8,2.4,.8,'Q K V -> attention\nsoftmax(QK^T / sqrt(64))','frozen',9)
box(a,7,1.8,1,.8,'+ Z','frozen');box(a,8.55,1.8,1.4,.8,'LayerNorm','frozen');box(a,10.5,1.8,1.6,.8,'MLP\n384-1536-384','frozen',10);box(a,12.6,1.8,1,.8,'+ residual','frozen',8)
for x,u in [(1.6,2.1),(3.5,4),(6.4,7),(8,8.55),(9.95,10.5),(12.1,12.6)]:arr(a,x,2.2,u,2.2)
arr(a,1,2.6,7.5,3.35);arr(a,7.5,3.35,7.5,2.6);arr(a,7.7,1.8,13.1,1);arr(a,13.1,1,13.1,1.8)
a.text(.3,3.65,'One pre-norm Transformer block (LayerScale residual factors included in implementation)',fontsize=11)
a.text(.3,.3,'Register values change during each forward pass; their learned parameters stay fixed. Registers have no image coordinates.',fontsize=11)
save(f,'07_backbone')
f,a=start(h=4.6)
labels=[('Patch map\n384 x 37 x 37',2.1,'data'),('3 x 3 Conv\n384 -> 192',1.6,'train'),('GN 8 + GELU',1.5,'train'),('3 x 3 Conv\n192 -> 96',1.6,'train'),('GN 8 + GELU',1.5,'train'),('1 x 1 Conv\n96 -> 1',1.4,'train'),('Bilinear resize\n37 -> 129',1.7,'data')]
x=.1
for i,(t,w,c) in enumerate(labels):
 box(a,x,2.4,w,1.2,t,c,10)
 if i:arr(a,x-.22,3,x,3)
 x+=w+.23
a.text(.2,1.65,'One channel is a shared point-objectness field. It does not encode fixed leaf identities or fixed keypoint slots.',fontsize=11)
a.text(.2,.95,'No U-Net skip pyramid, no six-stage residual heatmap cascade, no Transformer query decoder in this implementation.',fontsize=11)
a.text(.2,.25,'The geometric path decoder operates after detected points; it is not a layer inside this neural head.',fontsize=11)
save(f,'08_head')
hist=pd.read_csv(ROOT.parent/'experiment/training_outputs/core_dinov2_v4_phenotype_roi/history.csv')
f,a=plt.subplots(figsize=(10,3.3));a.plot(hist.epoch,hist.train_loss,label='Train loss',color='#3274a1');a.plot(hist.epoch,hist.validation_loss,label='Internal validation loss',color='#c85d3a');k=hist.validation_loss.idxmin();a.scatter(hist.loc[k,'epoch'],hist.loc[k,'validation_loss'],c='black',zorder=3);a.annotate(f"Best epoch {int(hist.loc[k,'epoch'])}: {hist.loc[k,'validation_loss']:.4f}",(hist.loc[k,'epoch'],hist.loc[k,'validation_loss']),xytext=(34,.65),arrowprops={'arrowstyle':'->'});a.set(xlabel='Epoch',ylabel='Loss',title='Route B Student | archived 80-epoch training log');a.legend();a.spines[['top','right']].set_visible(False);f.tight_layout();save(f,'09_training')
