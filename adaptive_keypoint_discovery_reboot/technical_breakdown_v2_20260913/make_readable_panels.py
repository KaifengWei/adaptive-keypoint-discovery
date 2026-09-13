from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;E=R.parent/'experiment';A=R/'assets'
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none'})
def save(f,n):f.savefig(A/(n+'.png'),dpi=260,bbox_inches='tight',facecolor='white');f.savefig(A/(n+'.svg'),bbox_inches='tight');plt.close(f)
def trim(im):
 ar=np.array(im);mask=(ar.max(2)-ar.min(2)>30)&(ar.min(2)<210);ys,xs=np.where(mask)
 return im.crop((max(0,xs.min()-25),max(0,ys.min()-25),min(im.width,xs.max()+26),min(im.height,ys.max()+26)))
for sid in ['0004','0034']:
 f,axs=plt.subplots(2,2,figsize=(12,4.5))
 dirs=['point_conditioned_organ_paths_v2_phenotype_roi_val','point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val','point_conditioned_organ_paths_v3_structure_coverage_global_decoder_val','point_conditioned_organ_paths_v3_structure_coverage_val']
 for ax,d,lab in zip(axs.flat,dirs,['A 原教师 + 全局尺度','B 原教师 + 局部叶宽','C 增强教师 + 全局尺度','D 增强教师 + 局部叶宽']):
  im=Image.open(E/'evaluation_outputs'/d/'overlays'/f'v4_val_{sid}.png').convert('RGB');im=im.crop((im.width//2,0,im.width,im.height));ax.imshow(trim(im));ax.set_title(lab,pad=12);ax.axis('off')
 f.subplots_adjust(hspace=.55,wspace=.15);save(f,'factor_zoom_'+sid)
W=json.loads((R/'walkthrough.json').read_text(encoding='utf-8'));ar=np.load(A/'nine_view_arrays.npz');base=ar['base'];pts=W['accepted']+W['rejected']
f,ax=plt.subplots(figsize=(12,3));ax.imshow(base)
for group,c,m in [(W['accepted'],'#15815b','o'),(W['rejected'],'#c63339','x')]:
 for p in group:
  ax.scatter(p['x_model'],p['y_model'],c=c,marker=m,s=70)
  ax.annotate(str(p['reference_index']),(p['x_model'],p['y_model']),xytext=(2,12 if p['reference_index']%2 else -17),textcoords='offset points',fontsize=10,color=c)
ax.set_xlim(55,450);ax.set_ylim(302,208);ax.axis('off');ax.set_title('共识去留局部放大  ·  绿色保留 / 红色拒绝  ·  编号对应逐点表');save(f,'consensus_zoom')
