from pathlib import Path
import sys,json,argparse,math,hashlib
import numpy as np,pandas as pd,torch,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent; E=R.parent/'experiment'; A=R/'assets';A.mkdir(exist_ok=True)
sys.path.insert(0,str(E))
import g1_dinov2_feasibility as g1
import g1_prime_structural_support as gp
import generate_g1prime_pseudolabels as pl
from phenotype_roi_basal_anchor import load_phenotype_input
from point_conditioned_graph import build_point_conditioned_graph
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
manifest=[]
def save(fig,name):
 fig.savefig(A/(name+'.png'),dpi=240,bbox_inches='tight',facecolor='white');fig.savefig(A/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def axim(ax,im,t):ax.imshow(im);ax.set_title(t,fontsize=11,pad=8);ax.axis('off')
def archived(path,name,crop=None):
 im=Image.open(E/path).convert('RGB')
 if crop: im=im.crop(crop(im.size))
 im.save(A/(name+'.png'));manifest.append({'asset':name,'source':str(E/path),'operation':'crop' if crop else 'unchanged'})
for i,n in enumerate(['attention_archive','contrast_archive','cluster_archive']):
 archived('outputs_clean_smoke6_baselines/overlays/clean_020.png',n,lambda s,i=i:(s[0]*i//3,0,s[0]*(i+1)//3,s[1]))
archived('outputs_clean_smoke6_g1prime/overlays/clean_020.png','g1_archive')
for src,name in [('core_dinov2/overlays/stagev3_0003.png','first_student'),('point_conditioned_organ_paths_v1_val/overlays/v4_val_0004.png','old_root'),('phenotype_roi_basal_anchor_v1_val/overlays/v4_val_0004.png','roi_audit')]:archived('evaluation_outputs/'+src,name)
for sid in ['0002','0023','0039']:
 archived(f'evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_val/overlays/v4_val_{sid}.png','failure_'+sid)
old=R.parent/'technical_breakdown_20260913'
for n in ['01_input','02_features','03_teacher','04_heatmap','05_cases']:
 Image.open(old/'assets'/(n+'.png')).save(A/(n+'.png'))
 manifest.append({'asset':n,'source':str(old/'assets'/(n+'.png')),'operation':'unchanged'})
# Real archived comparison, one row per sample, same crop and original marks.
for sid in ['0004','0034']:
 fig,axs=plt.subplots(2,2,figsize=(13,9))
 dirs=['point_conditioned_organ_paths_v2_phenotype_roi_val','point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val','point_conditioned_organ_paths_v3_structure_coverage_global_decoder_val','point_conditioned_organ_paths_v3_structure_coverage_val']
 for ax,d,lab in zip(axs.flat,dirs,['A 原教师 / 全局尺度','B 原教师 / 局部叶宽','C 增强教师 / 全局尺度','D 增强教师 / 局部叶宽']):
  fp=E/'evaluation_outputs'/d/'overlays'/f'v4_val_{sid}.png';im=Image.open(fp);axim(ax,im,lab);manifest.append({'asset':'factor_'+sid,'source':str(fp)})
 save(fig,'factor_'+sid)
# Early data contact sheet: retained as an actual historical page.
q=list((E/'outputs_g1/pilot/human_review').glob('*local*worst*'))
if q: archived(str(q[0].relative_to(E)),'background_failures')
# Nine-view reproduction on one archived train sample; model and teacher untouched.
torch.set_num_threads(4)
model=g1.load_official_model(argparse.Namespace(local_repo=E/'third_party/dinov2_git',model='dinov2_vits14_reg',weights=E/'third_party/checkpoints/dinov2_vits14_reg4_pretrain.pth'),torch.device('cpu'));model.eval()
ev=json.loads((old/'evidence.json').read_text(encoding='utf-8')); sid=ev['teacher_sample']['dataset_id']
ds=E/'data_stage_clean_v4_fullplant_candidate';row=pd.read_csv(ds/'manifests/train.csv').set_index('dataset_id').loc[sid].to_dict();row['dataset_id']=sid
base,mapping,focused,roi=load_phenotype_input(ds,row,518)
views=g1.make_transforms(base,True);outputs=[]
for v in views:
 with torch.inference_mode(): reps,att,_=g1.extract_representations(model,v['image'],torch.device('cpu'))
 pts,recs,support,skel,diag=gp.structural_candidates(v['image'],reps['last4avg'],att,30,evidence_mode='full',structure_coverage=False,basal_transition_mask=None)
 outputs.append(dict(points=pts,records=recs,support=support,skeleton=skel,diagnostics=diag));print('view',v['name'],len(pts),flush=True)
b=gp.bbox_from_mask(outputs[0]['support']);D=max(1,math.hypot(b[2]-b[0],b[3]-b[1]))
accepted,rejected=pl.consensus(outputs,views,D,argparse.Namespace(size=518,no_consistency_filter=False,min_presence=.75,max_localization_error=.025))
fig,axs=plt.subplots(3,3,figsize=(11,10))
names=['原图','水平翻转','旋转 +10°','亮度 ×0.75','旋转 −10°','尺度 ×0.90','尺度 ×1.10','平移 (+5%, −4%)','对比度 ×1.25']
for ax,v,o,t in zip(axs.flat,views,outputs,names):
 axim(ax,v['image'],t+f"  |  {len(o['points'])} 个候选")
 if len(o['points']): ax.scatter(*o['points'].T,s=24,facecolors='none',edgecolors='#d84735',linewidths=1.1)
save(fig,'nine_views')
fig,axs=plt.subplots(1,2,figsize=(11,5))
axim(axs[0],base,'九视图坐标逆变换回原图');axim(axs[1],base,f'共识筛选：保留 {len(accepted)} / 拒绝 {len(rejected)}')
for j,(v,o) in enumerate(zip(views,outputs)):
 p,_=pl.inverse_mapped_records(o['points'],o['records'],v['matrix'],518)
 if len(p):axs[0].scatter(*p.T,s=15,color=plt.cm.tab10(j),alpha=.65)
for group,c,marker in [(accepted,'#13875b','o'),(rejected,'#c33039','x')]:
 for p in group:
  axs[1].scatter(p['x_model'],p['y_model'],c=c,marker=marker,s=42)
  axs[1].annotate(str(p['reference_index']),(p['x_model'],p['y_model']),xytext=(4,3),textcoords='offset points',fontsize=8)
save(fig,'consensus_result')
pd.DataFrame(accepted+rejected).sort_values('reference_index').to_csv(R/'consensus_walkthrough.csv',index=False,encoding='utf-8-sig')
np.savez_compressed(A/'nine_view_arrays.npz',images=np.stack([v['image'] for v in views]),matrices=np.stack([v['matrix'] for v in views]),base=base)
arch=ev['teacher_sample']['points']; archival_xy=np.array([[p['x_source']*mapping['scale']+mapping['pad_x'],p['y_source']*mapping['scale']+mapping['pad_y']] for p in arch]);newxy=np.array([[p['x_model'],p['y_model']] for p in accepted])
match={'archived_count':len(arch),'reproduced_count':len(accepted),'max_coordinate_difference':float(np.max(np.abs(archival_xy-newxy))) if archival_xy.shape==newxy.shape else None}
# Point removal on actual frozen B predictions and actual ROI skeleton.
nodeframe=pd.read_csv(E/'evaluation_outputs/point_conditioned_graph_v2_phenotype_roi_val/nodes.csv')
valrow=pd.read_csv(ds/'manifests/val.csv').set_index('dataset_id').loc['v4_val_0004'].to_dict();valrow['dataset_id']='v4_val_0004'
canvas,mp,focused,roi=load_phenotype_input(ds,valrow,518);support,skel,_=gp.automatic_structural_support(canvas);b=gp.bbox_from_mask(support);d=math.hypot(b[2]-b[0],b[3]-b[1])
points=ev['samples']['v4_val_0004']['points'];graphs=[build_point_conditioned_graph(skel,points,d,.025),build_point_conditioned_graph(skel,[],d,.025)]
effects=[]
for i in range(len(points)):
 g=build_point_conditioned_graph(skel,points[:i]+points[i+1:],d,.025);effects.append((int(graphs[0]['edge_union'].sum())-int(g['edge_union'].sum()),i,g))
loss,removed,g=max(effects,key=lambda t:t[0]);graphs.append(g)
fig,axs=plt.subplots(1,3,figsize=(12,4.2))
for ax,g,t in zip(axs,graphs,['完整预测点','移除全部预测点',f'移除点 #{removed}（影响最大）']):
 axim(ax,canvas,t+f"\n{len(g['nodes'])} 节点 / {len(g['edges'])} 边 / {int(g['edge_union'].sum())} 路径像素")
 yy,xx=np.where(g['edge_union']);ax.scatter(xx,yy,s=2,c='#1674a4')
 for n in g['nodes']:ax.scatter(*n['projected_xy'],s=38,c='#dd463b',edgecolors='white')
save(fig,'graph_removal')
out={'sample':sid,'bbox_diag':D,'accepted':accepted,'rejected':rejected,'archive_comparison':match,'graph_sample':'v4_val_0004','graph_removed_input_index':removed,'graph_lost_path_pixels':loss,'graph_diagnostics':[g['diagnostics'] for g in graphs],'source_images':manifest}
(R/'walkthrough.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(match,flush=True)
