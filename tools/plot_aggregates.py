import json,csv,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
def main(B, output):
 B=Path(B);x=json.loads((B/'data/aggregates/analysis_results.json').read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False)
 order=['physical_full_fov','native_nominal_control','enhancement_balanced'];names=['Physical full field','Native control','Balanced enhancement'];colors=['#6E8193','#0072B2','#009E73'];markers=['s','o','^'];classes=['Basal-like','HER2-enriched','Luminal A','Luminal B'];short=['B','H','L-A','L-B']
 assert x['patients']==743 and x['original_round']=={'planned':6,'successful':5,'failed':1,'failed_stage':'enhancement_weighted_outer2','failure_preserved':True,'all_six_complete':False}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8C99A5','text.color':'#183246','axes.labelcolor':'#183246','xtick.color':'#425363','ytick.color':'#425363','savefig.dpi':300})
 cmap=LinearSegmentedColormap.from_list('result_blues',['#f1f5f9','#c0dae8','#69a8c9','#2175a4','#0a405c'])
 pooled=[x['arms'][k]['pooled_selected_development'] for k in order];cms=[np.array(z['confusion'],dtype=int) for z in pooled];truth=np.array(pooled[0]['true_class_counts']);assert list(truth)==[318,108,138,179]
 for k,z,c in zip(order,pooled,cms):
  assert c.shape==(4,4) and c.sum()==743 and list(c.sum(1))==list(truth) and list(c.sum(0))==z['prediction_counts']
  assert np.array_equal(c,sum(np.array(f['confusion']) for f in x['arms'][k]['folds']))
  assert [f['patients'] for f in x['arms'][k]['folds']]==[247,248,248]
  assert np.allclose(c.diagonal()/truth,z['recalls'],rtol=0,atol=1e-12)
 def save(fig,stem):
  if stem in ['result_fold_profiles_20261106','result_paired_errors_20261106']:
   stem=stem.replace('_20261106','_v2_20261106')
  fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',pad_inches=.08,metadata={'Creator':'Matplotlib; saved aggregate results only','Title':stem});fig.savefig(out/(stem+'.png'),bbox_inches='tight',pad_inches=.08);plt.close(fig)
 # Shared-scale row-normalized coloring; every original integer is displayed.
 fig,axs=plt.subplots(1,3,figsize=(6.7,3.35));fig.subplots_adjust(left=.12,right=.99,top=.81,bottom=.25,wspace=.24)
 for j,(ax,c,name,col) in enumerate(zip(axs,cms,names,colors)):
  r=c/truth[:,None];im=ax.imshow(r,vmin=0,vmax=1,cmap=cmap)
  ax.set_title(name,color=col,pad=10,fontsize=9);ax.set_xticks(range(4),short);ax.set_yticks(range(4),short);ax.set_xlabel('Predicted subtype');
  if j==0:ax.set_ylabel('True subtype')
  ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True);ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='both',length=0)
  for i in range(4):
   for q in range(4):
    textcol='white' if r[i,q]>.50 else '#183246';ax.text(q,i-.10,str(c[i,q]),ha='center',va='center',fontsize=9,fontweight='bold',color=textcol);ax.text(q,i+.20,f'{r[i,q]*100:.1f}%',ha='center',va='center',fontsize=6.8,color=textcol)
  for spine in ax.spines.values():spine.set_visible(False)
 cax=fig.add_axes([.34,.16,.40,.025]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.25,.5,.75,1]);cb.outline.set_visible(False);cb.ax.tick_params(length=2,labelsize=7);cb.set_label('Fraction of the true subtype',fontsize=8,labelpad=2)
 fig.text(.5,.96,'Same 743 development patients · cell labels: count and row percentage',ha='center',fontsize=9)
 fig.text(.5,.01,'B: basal-like   H: HER2-enriched   L-A: luminal A   L-B: luminal B',ha='center',fontsize=7.5)
 save(fig,'result_confusion_20261106')
 # Direct class-count contrast, not calibration or prevalence estimation.
 fig,ax=plt.subplots(figsize=(6.7,4.25));fig.subplots_adjust(left=.20,right=.93,top=.84,bottom=.14)
 vals=[truth]+[np.array(p['prediction_counts']) for p in pooled];labs=['True class count']+names;cs=['#FFFFFF']+colors
 y=np.arange(4)*1.35;h=.23
 for j,(v,lab,col) in enumerate(zip(vals,labs,cs)):
  pos=y+(j-1.5)*h;bars=ax.barh(pos,v,height=h*.88,label=lab,color=col,edgecolor='#4A5563' if j==0 else 'none',linewidth=.9,zorder=3)
  for b,num in zip(bars,v):ax.text(num+4,b.get_y()+b.get_height()/2,str(num),va='center',fontsize=8,color='#243746')
 ax.set_yticks(y,classes);ax.invert_yaxis();ax.set_xlim(0,420);ax.set_xlabel('Patients assigned to each subtype (n = 743 per row series)');ax.grid(axis='x',color='#dce4eb',linewidth=.7,zorder=0);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
 fig.legend(ncol=2,loc='upper center',bbox_to_anchor=(.55,1.01),columnspacing=1.7,handlelength=1.3)
 fig.text(.55,.015,'Published true labels and model argmax counts; totals are equal, class allocations differ.',ha='center',fontsize=8)
 save(fig,'result_class_counts_20261106')
 # All five saved metrics for every fold; no fold dispersion treated as uncertainty.
 fig=plt.figure(figsize=(6.7,6.4));gs=fig.add_gridspec(3,2,left=.10,right=.98,top=.86,bottom=.10,hspace=.66,wspace=.32)
 specs=[('macro_auc','a  Macro-AUC',(.45,.65)),('macro_f1','b  Macro-F1',(.05,.35)),('balanced_accuracy','c  Balanced accuracy',(.20,.35)),('accuracy','d  Accuracy',(.15,.45)),('log_loss','e  Log loss (lower is better)',(1.30,1.46))]
 for j,(key,title,ylim) in enumerate(specs):
  ax=fig.add_subplot(gs[j//2,j%2] if j<4 else gs[2,:]);ax.set_title(title,loc='left',fontsize=9)
  for k,col,m,name in zip(order,colors,markers,names):ax.plot(range(3),[f[key] for f in x['arms'][k]['folds']],color=col,marker=m,markersize=5,lw=1.4,label=name)
  ax.set_xticks(range(3),['Fold 0\nn=247','Fold 1\nn=248','Fold 2\nn=248']);ax.set_xlim(-.12,2.12);ax.set_ylim(*ylim);ax.grid(axis='y',color='#dce4eb',lw=.7);ax.set_axisbelow(True)
 fig.legend(handles=[Line2D([0],[0],color=c,marker=m,label=n,lw=1.4) for c,m,n in zip(colors,markers,names)],ncol=3,loc='upper center',bbox_to_anchor=(.54,.98),columnspacing=1.2,fontsize=8)
 save(fig,'result_fold_profiles_20261106')
 # Two archived adaptive paired tables plus the original paired table transcribed in main.tex.
 pairs=[('Native control','Physical full field',[[64,151],[166,362]]),('Balanced enhancement','Physical full field',[[74,143],[156,370]]),('Balanced enhancement','Native control',[[72,145],[143,383]])]
 src=(B/'provenance/paired_correctness_statement.txt').read_text(encoding='utf8');assert 'both branches were correct in 64, only complete-field was correct in 166, only control in 151, and both were wrong in 362' in src
 for pair,k in zip(pairs[1:],['physical_full_fov','native_nominal_control']):
  q=x['contrasts'][k]['paired_correctness'];assert pair[2]==[[q['both_correct'],q['only_balanced']],[q['only_control'],q['both_wrong']]]
 for first,second,c in pairs:assert np.array(c).sum()==743
 fig,axs=plt.subplots(1,3,figsize=(6.7,3.15));fig.subplots_adjust(left=.10,right=.98,top=.76,bottom=.29,wspace=.40)
 for ax,(first,second,cc) in zip(axs,pairs):
  c=np.array(cc);im=ax.imshow(c/743,cmap=cmap,vmin=0,vmax=.55);ax.set_title('A: '+first+'\nB: '+second,fontsize=8.3,pad=10)
  ax.set_xticks([0,1],['B correct','B wrong'],fontsize=7);ax.set_yticks([0,1],['A correct','A wrong'],fontsize=7);ax.tick_params(length=0)
  ax.set_xticks([-.5,.5,1.5],minor=True);ax.set_yticks([-.5,.5,1.5],minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
  for i in range(2):
   for j in range(2):ax.text(j,i,str(c[i,j]),ha='center',va='center',fontsize=15,fontweight='bold',color='white' if c[i,j]/743>.30 else '#183246')
  for sp in ax.spines.values():sp.set_visible(False)
 cax=fig.add_axes([.33,.15,.40,.025]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.25,.5]);cb.outline.set_visible(False);cb.set_label('Fraction of the same 743 patients',fontsize=8,labelpad=2);cb.ax.tick_params(labelsize=7,length=2)
 fig.text(.54,.96,'Paired correctness across input strategies',ha='center',fontsize=10,fontweight='bold')
 save(fig,'result_paired_errors_20261106')
 # Exact raw plot inputs plus tidy tables, for transparent data-driven reproduction.
 data={'source':'reports/model_improvement_balanced_analysis_20261027/RESULTS.json','source_sha256':hashlib.sha256((B/'data/aggregates/analysis_results.json').read_bytes()).hexdigest(),'arm_order':order,'arm_names':names,'class_order':classes,'patients':743,'arms':x['arms'],'paired_correctness_panels':pairs,'original_round':x['original_round'],'scope':'Saved development summaries only. No new prediction, fitted statistic, test, interval or resampling.'}
 (out/'PLOT_DATA.json').write_text(json.dumps(data,indent=2),encoding='utf8')
 with (out/'FOLD_METRICS.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.writer(f);w.writerow(['arm','fold','patients']+[a[0] for a in specs])
  for k in order:
   for z in x['arms'][k]['folds']:w.writerow([k,z['fold'],z['patients']]+[z[a[0]] for a in specs])
 assert json.loads(json.dumps(data))==json.loads((B/'data/aggregates/plot_data.json').read_text(encoding='utf8')), 'Saved plot inputs changed'
 (out/'RENDER_RECEIPT.json').write_text(json.dumps({'status':'PASS','source':'saved aggregate summaries','new_patient_predictions':False,'new_statistical_tests':False,'font':'DejaVu Sans'}),encoding='utf8')

if __name__ == "__main__":
 import argparse
 parser=argparse.ArgumentParser(description='Render saved aggregate comparisons; no patient inputs or training.')
 parser.add_argument('--output',type=Path,default=Path('generated/figures'))
 args=parser.parse_args()
 main(Path(__file__).resolve().parents[1], args.output)
