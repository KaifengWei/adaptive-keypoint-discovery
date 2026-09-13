from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parent
EXP=ROOT.parent/'experiment'
sys.path.insert(0,str(EXP))
import g1_dinov2_feasibility as g1
import g1_prime_structural_support as gp
from phenotype_roi_basal_anchor import load_phenotype_input
from adaptive_point_model import AdaptivePointDetector
from train_adaptive_point_detector import gaussian_heatmap
OUT=ROOT/'assets'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=260,bbox_inches='tight',facecolor='white');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def panel(ax,arr,title,cmap=None):
 ax.imshow(arr,cmap=cmap);ax.set_title(title,fontsize=11,pad=10);ax.axis('off')
args=argparse.Namespace(local_repo=EXP/'third_party/dinov2_git',model='dinov2_vits14_reg',weights=EXP/'third_party/checkpoints/dinov2_vits14_reg4_pretrain.pth')
backbone=g1.load_official_model(args,torch.device('cpu'))
model=AdaptivePointDetector(backbone)
cp=EXP/'training_outputs/core_dinov2_v4_phenotype_roi/best.pt'
state=torch.load(cp,map_location='cpu',weights_only=False)
print('checkpoint keys',list(state),flush=True)
model.load_state_dict(state['model_state']);model.eval()
ds=EXP/'data_stage_clean_v4_fullplant_candidate'
val=pd.read_csv(ds/'manifests/val.csv').set_index('dataset_id')
old=pd.read_csv(EXP/'evaluation_outputs/core_dinov2_v4_phenotype_roi_val/points.csv')
meta={'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'samples':{},'test_images_read':0,'training':False,'parameter_counts':{'backbone':sum(p.numel() for p in backbone.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}}
for sid in ['v4_val_0004','v4_val_0002','v4_val_0034']:
 row=val.loc[sid].to_dict();row['dataset_id']=sid
 canvas,mapping,focused,roi=load_phenotype_input(ds,row,518)
 with torch.inference_mode():
  logits=model(g1.normalize_tensor(canvas,torch.device('cpu')))
 prob=logits.sigmoid()[0,0].numpy();decoded=model.decode(logits,(518,518))[0]
 ref=old[old.dataset_id==sid]
 savedxy=np.array([[r.x_source*mapping['scale']+mapping['pad_x'],r.y_source*mapping['scale']+mapping['pad_y']] for r in ref.itertuples()])
 xy=np.array([[p.x,p.y] for p in decoded])
 meta['samples'][sid]={'mapping':mapping,'cpu_decoded_count':len(decoded),'saved_count':len(ref),'max_ranked_coordinate_difference_model_px':float(np.max(np.abs(xy-savedxy))) if xy.shape==savedxy.shape else None,'probability_max':float(prob.max()),'points':[vars(p) for p in decoded]}
 np.savez_compressed(OUT/(sid+'_tensors.npz'),probability=prob,logits=logits.numpy(),input_rgb=canvas)
 Image.fromarray(canvas).save(OUT/(sid+'_canvas.png'))
 if sid=='v4_val_0004':
  reps,attention,amode=g1.extract_representations(backbone,canvas,torch.device('cpu'))
  contrast=g1.feature_local_contrast(reps['last4avg'])
  f=reps['last'].reshape(384,-1).T;u,s,v=np.linalg.svd(f-f.mean(0),full_matrices=False);rgb=u[:,:3]*s[:3];rgb=(rgb-rgb.min(0))/(np.ptp(rgb,axis=0)+1e-9);rgb=rgb.reshape(37,37,3)
  np.savez_compressed(OUT/'dinov2_real_features.npz',last=reps['last'],last4avg=reps['last4avg'],attention=attention,contrast=contrast,pca=rgb)
  fig,axes=plt.subplots(1,4,figsize=(13,3.3));panel(axes[0],canvas,'Real ROI input | 518 x 518');panel(axes[1],rgb,'Last-layer patch features\nPCA display | 37 x 37');panel(axes[2],attention,'CLS-to-patch attention\n'+amode.replace('exact_last_block_',''),'viridis');panel(axes[3],contrast,'Last-4 local contrast\n1 - cosine similarity','magma');save(fig,'02_features')
  fig,axes=plt.subplots(1,3,figsize=(11,3.6));panel(axes[0],canvas,'Real model input');panel(axes[1],prob,'Actual Student probability\n129 x 129','viridis');panel(axes[2],canvas,'Threshold 0.35 + 5 x 5 NMS\n'+str(len(decoded))+' detected points');axes[2].scatter(xy[:,0],xy[:,1],s=45,c='#d62728',edgecolors='white');save(fig,'04_heatmap')
  raw=ds/Path(str(row['raw_review_relative_path']).replace('\\','/'));full=ds/Path(str(row['relative_path']).replace('\\','/'))
  fig,axes=plt.subplots(3,1,figsize=(11,6));panel(axes[0],np.array(Image.open(raw)),'Real scan crop | before background standardization');panel(axes[1],np.array(Image.open(full)),'Standardized full plant | seed and roots preserved');panel(axes[2],focused,'Phenotype ROI input | seed and roots excluded');save(fig,'01_input')
 print(sid,meta['samples'][sid],flush=True)
# Historical train pseudo-labels: render their exact targets, never generate val teachers.
pseudo=[json.loads(l) for l in (EXP/'pseudo_labels_g1prime_v4_phenotype_roi/pseudo_labels.jsonl').read_text(encoding='utf-8').splitlines()]
pr=next(r for r in pseudo if r['dataset_id'].startswith('v4_train_new_') and 6<=len(r['points'])<=10)
train=pd.read_csv(ds/'manifests/train.csv').set_index('dataset_id');r=train.loc[pr['dataset_id']].to_dict();r['dataset_id']=pr['dataset_id']
canvas,m,focused,roi=load_phenotype_input(ds,r,518)
pts=[(p['x_source']*m['scale']+m['pad_x'],p['y_source']*m['scale']+m['pad_y'],p['consensus_confidence']) for p in pr['points']]
target=gaussian_heatmap(pts,(129,129),(518,518),1.6)
support,skel,_=gp.automatic_structural_support(canvas)
fig,axes=plt.subplots(1,4,figsize=(13,3.3));panel(axes[0],canvas,pr['dataset_id']+'\nReal training ROI');panel(axes[1],skel,'Automatic skeleton','gray');panel(axes[2],canvas,'Archived consensus coordinates\n'+str(len(pts))+' pseudo-labels');axes[2].scatter(*np.array(pts)[:,:2].T,c=np.array(pts)[:,2],cmap='viridis',vmin=.35,vmax=1,edgecolors='black',s=30);panel(axes[3],target,'Exact Gaussian training target\nsigma = 1.6 output pixels','viridis');save(fig,'03_teacher')
meta['teacher_sample']=pr;np.savez_compressed(OUT/'teacher_target.npz',target=target,input_rgb=canvas,skeleton=skel)
# Reuse stored spatial paths, preserving their model-canvas coordinates.
pathdir=EXP/'evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val'
paths=[json.loads(l) for l in (pathdir/'paths.jsonl').read_text().splitlines()]
nodes=pd.read_csv(EXP/'evaluation_outputs/point_conditioned_graph_v2_phenotype_roi_val/nodes.csv')
fig,axes=plt.subplots(3,3,figsize=(12,10))
for j,sid in enumerate(meta['samples']):
 canvas=np.array(Image.open(OUT/(sid+'_canvas.png')))
 panel(axes[j,0],canvas,sid+' | input')
 panel(axes[j,1],canvas,'Saved accepted graph nodes')
 ns=nodes[nodes.dataset_id==sid];axes[j,1].scatter(ns.projected_x,ns.projected_y,c='#d62728',s=35,edgecolors='white')
 panel(axes[j,2],canvas,'Frozen B | stored candidate paths')
 for i,p in enumerate([p for p in paths if p['dataset_id']==sid]):
  xy=np.array(p['full_base_to_tip_path']);axes[j,2].plot(xy[:,0],xy[:,1],lw=2,color=['#1976d2','#e67e22','#8e44ad','#198754'][i%4])
 if not any(p['dataset_id']==sid for p in paths):axes[j,2].text(259,330,'NO ELIGIBLE LEARNED BASE',ha='center',color='#b22222',fontsize=9)
save(fig,'05_cases')
hist=pd.read_csv(EXP/'training_outputs/core_dinov2_v4_phenotype_roi/history.csv');print(hist.columns.tolist(),flush=True)
meta['history_columns']=hist.columns.tolist()
(ROOT/'evidence.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print('evidence complete',flush=True)
