from pathlib import Path
import sys,json,argparse,math
import numpy as np,pandas as pd,torch,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;E=R.parent/'experiment';A=R/'assets';O=R.parent/'technical_breakdown_20260913/assets';sys.path.insert(0,str(E))
import g1_dinov2_feasibility as g1
import g1_prime_structural_support as gp
import g1_prime_phenotype_bridge as bridge
from phenotype_roi_basal_anchor import load_phenotype_input
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
def save(f,n):f.savefig(A/(n+'.png'),dpi=250,bbox_inches='tight',facecolor='white');f.savefig(A/(n+'.svg'),bbox_inches='tight');plt.close(f)
def show(ax,im,t):ax.imshow(im);ax.set_title(t,fontsize=11);ax.axis('off')
def trim(im):
 ar=np.array(im);mask=(ar.max(2)-ar.min(2)>30)&(ar.min(2)<210);ys,xs=np.where(mask)
 if len(xs):return im.crop((max(0,xs.min()-30),max(0,ys.min()-40),min(im.width,xs.max()+31),min(im.height,ys.max()+41)))
 return im
# Archived result crops remove only surrounding whitespace; maintain each panel's aspect ratio.
for sid in ['0004','0034']:
 f,axs=plt.subplots(4,1,figsize=(12,7))
 dirs=['point_conditioned_organ_paths_v2_phenotype_roi_val','point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val','point_conditioned_organ_paths_v3_structure_coverage_global_decoder_val','point_conditioned_organ_paths_v3_structure_coverage_val']
 for ax,d,lab in zip(axs,dirs,['A  原教师 + 全局阈值','B  原教师 + 局部叶宽','C  增强教师 + 全局阈值','D  增强教师 + 局部叶宽']):
  im=Image.open(E/'evaluation_outputs'/d/'overlays'/f'v4_val_{sid}.png').convert('RGB');im=im.crop((im.width//2,0,im.width,im.height));show(ax,trim(im),lab)
 save(f,'factor_zoom_'+sid)
for n in ['old_root','failure_0002','failure_0023','failure_0039']:
 im=Image.open(A/(n+'.png'));trim(im).save(A/(n+'_zoom.png'))
# Training curves: actual saved histories, first run and later domain correction.
f,axs=plt.subplots(1,2,figsize=(12,4))
for ax,run,title in zip(axs,['core_dinov2','core_dinov2_v4_phenotype_roi'],['V3 首轮：87 张自动教师','地上部重训：216 张自动教师']):
 df=pd.read_csv(E/'training_outputs'/run/'history.csv');ax.plot(df.epoch,df.train_loss,label='训练损失',color='#b85548');ax.plot(df.epoch,df.validation_loss,label='内部验证损失',color='#286990');ax.set_xlabel('epoch');ax.set_ylabel('loss');ax.set_title(title);ax.legend();ax.grid(alpha=.2);ax.set_ylim(0,min(1.5,df.train_loss.max()))
save(f,'training_curves')
# Genuine internal feature maps for each early method on clean_020.
torch.set_num_threads(4);model=g1.load_official_model(argparse.Namespace(local_repo=E/'third_party/dinov2_git',model='dinov2_vits14_reg',weights=E/'third_party/checkpoints/dinov2_vits14_reg4_pretrain.pth'),torch.device('cpu'));model.eval()
row=pd.read_csv(E/'data_clean_core20/smoke6_manifest.csv').set_index('clean_id').loc['clean_020'];maps={}
for size in [518,728]:
 base,_=g1.letterbox_rgb(Path(row.clean_image_path),size)
 with torch.inference_mode():reps,att,_=g1.extract_representations(model,base,torch.device('cpu'))
 maps[size]=(base,reps,att)
for method,name,tau,size,layer in [('cls_to_patch_attention','attention_detail',.996,518,'last4avg'),('feature_local_contrast','contrast_detail',6.,518,'last4avg'),('feature_hdbscan_medoid','cluster_detail',0.,728,'last')]:
 base,reps,att=maps[size];f,axs=plt.subplots(1,3,figsize=(12,4));show(axs[0],base,f'clean_020 / {size}²');points=g1.candidates_for(method,reps[layer],att,tau,size,20260716)
 if 'attention' in method:score=att;show(axs[1],plt.cm.viridis((score-score.min())/(np.ptp(score)+1e-9)),'真实 CLS attention / 37²')
 elif 'contrast' in method:score=g1.feature_local_contrast(reps[layer]);show(axs[1],plt.cm.magma((score-score.min())/(np.ptp(score)+1e-9)),'真实局部余弦差 / 37²')
 else:
  feat=reps[layer];c,h,w=feat.shape;data=feat.reshape(c,-1).T.astype(np.float64);data/=np.linalg.norm(data,axis=1,keepdims=True)+1e-9;data=g1.PCA(n_components=16,random_state=20260716).fit_transform(data);data/=np.std(data,axis=0,keepdims=True)+1e-6;yy,xx=np.meshgrid(np.linspace(-1,1,h),np.linspace(-1,1,w),indexing='ij');data=np.concatenate([data,.30*np.stack([xx.ravel(),yy.ravel()],axis=1)],axis=1);mc=max(8,round(.012*len(data)));labels=g1.SklearnHDBSCAN(min_cluster_size=mc,min_samples=max(3,mc//3),metric='euclidean',cluster_selection_method='eom',allow_single_cluster=False,copy=True).fit_predict(data);rgb=plt.cm.tab20((labels.reshape(h,w)%20)/19);rgb[labels.reshape(h,w)<0]=[.95,.95,.95,1];show(axs[1],rgb,'真实密度簇 / 52²（浅灰为噪声）')
 show(axs[2],base,f'该方法输出：{len(points)} 个点');axs[2].scatter(*points.T,c='#d84c39',s=28,edgecolors='black');save(f,name);print(name,len(points),flush=True)
# G1 geometric intermediate on same real early sample.
base,reps,att=maps[518];pts,recs,support,skel,diag=gp.structural_candidates(base,reps['last4avg'],att,20)
f,axs=plt.subplots(1,3,figsize=(12,4));show(axs[0],base,'clean_020 / 真实 RGB');show(axs[1],skel,'自动前景骨架');show(axs[2],base,'结构候选 / 类型着色')
for p,r in zip(pts,recs):axs[2].scatter(*p,s=30,c={'endpoint':'#c94d38','junction':'#5f4aa0','shape_corner':'#df982e','dino_distinctive':'#2886a2'}.get(r['kind'],'#2886a2'))
save(f,'g1_detail')
# Teacher enhancement: actual executed identity proposals, not a fabricated historical label set.
W=json.loads((R/'walkthrough.json').read_text(encoding='utf-8'));sid=W['sample'];ds=E/'data_stage_clean_v4_fullplant_candidate';row=pd.read_csv(ds/'manifests/train.csv').set_index('dataset_id').loc[sid].to_dict();row['dataset_id']=sid;base,mp,focus,roi=load_phenotype_input(ds,row,518)
with torch.inference_mode():reps,att,_=g1.extract_representations(model,base,torch.device('cpu'))
f,axs=plt.subplots(1,3,figsize=(12,4));show(axs[0],base,'同一张训练 ROI');rows=[]
for ax,flag,t in [(axs[1],False,'原教师：通用候选去重'),(axs[2],True,'增强教师：终端 / 基部覆盖优先')]:
 pts,recs,sup,sk,diag=gp.structural_candidates(base,reps['last4avg'],att,30,evidence_mode='full',structure_coverage=flag,basal_transition_mask=roi['basal_transition_model'] if flag else None);show(ax,base,t+f'\n单视图候选 {len(pts)}')
 for p,r in zip(pts,recs):ax.scatter(*p,s=30,c='#ba4b36' if r.get('coverage_roles') else '#207493',edgecolors='white')
 rows.append({'enhanced':flag,'records':recs,'points':pts.tolist()})
save(f,'teacher_coverage_detail');(R/'teacher_identity_walkthrough.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
# Local width measured on an actual current ROI; plot distance transform and crop branch.
row=pd.read_csv(ds/'manifests/val.csv').set_index('dataset_id').loc['v4_val_0004'].to_dict();row['dataset_id']='v4_val_0004';base,mp,focus,roi=load_phenotype_input(ds,row,518);sup,sk,_=gp.automatic_structural_support(base);dt=cv2.distanceTransform(sup.astype(np.uint8),cv2.DIST_L2,5)
f,axs=plt.subplots(2,1,figsize=(12,5));ys,xs=np.where(sup);lim=(xs.min()-10,xs.max()+10,ys.max()+15,ys.min()-15)
show(axs[0],base,'v4_val_0004：真实短叶位于主轴下侧');axs[0].set_xlim(lim[:2]);axs[0].set_ylim(lim[2:]);im=axs[1].imshow(dt,cmap='magma');axs[1].set_xlim(lim[:2]);axs[1].set_ylim(lim[2:]);axs[1].set_title('支持域距离变换：骨架处半径 → 远端半段中位叶宽');axs[1].axis('off');f.colorbar(im,ax=axs[1],fraction=.02,pad=.02,label='距边界 / px');save(f,'local_width_detail')
