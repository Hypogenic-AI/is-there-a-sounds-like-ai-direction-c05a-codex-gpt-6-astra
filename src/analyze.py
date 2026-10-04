"""Prompt-cluster analysis with fixed seeds, paired bootstrap and sign flips.
Preserves all generated conditions; no selection on detector or quality outcomes.
"""
from common import *
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METRICS=['encoder','ai_likeness','coherence','content','unsupported','anchor_recall','tokens','truncated','repetition']
def stats(d):
 d=np.asarray(d,dtype=float);d=d[np.isfinite(d)];rng=np.random.default_rng(77)
 if not len(d):return dict(n=0,delta=None,lo=None,hi=None,p=None,dz=None)
 boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1);obs=d.mean()
 perm=(rng.choice([-1,1],size=(10000,len(d)))*d).mean(1)
 return dict(n=len(d),delta=float(obs),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),p=float((1+(np.abs(perm)>=abs(obs)-1e-12).sum())/10001),dz=float(obs/d.std(ddof=1)) if d.std(ddof=1)>0 else None)
def holm(p):
 order=np.argsort(p);out=np.empty(len(p));prev=0
 for rank,i in enumerate(order):prev=max(prev,(len(p)-rank)*p[i]);out[i]=min(1,prev)
 return out

def repeat(text):
 w=text.lower().split();tri=[tuple(w[i:i+3]) for i in range(len(w)-2)]
 return 1-len(set(tri))/max(1,len(tri))

def main():
 from score import key
 gen=lines('generations.jsonl');df=pd.DataFrame(gen);df['key']=[key(r) for r in gen];assert df.key.is_unique
 enc={r['id']:r['encoder'] for r in lines('detector_scores.jsonl')};jud={r['id']:r for c in lines('judge_calls.jsonl') for r in c.get('ratings',[])}
 df['encoder']=df.key.map(enc)
 for m in ['ai_likeness','coherence','content','unsupported']:df[m]=[jud.get(k,{}).get(m,np.nan) for k in df.key]
 df['repetition']=df.text.map(repeat);df.to_csv(R/'scored_generations.csv',index=False)
 df.groupby(['model','condition'])[METRICS].agg(['mean','std','min','max','count']).to_csv(R/'descriptives.csv')
 # Average seeds inside each source prompt before resampling prompts.
 comparisons=[]
 for model in df.model.unique():
  sub=df[df.model==model];base=sub[sub.condition=='zero'].set_index(['id','seed'])
  for condition in sorted(set(sub.condition)-{'zero'}):
   treatment=sub[sub.condition==condition].set_index(['id','seed']);idx=treatment.index.intersection(base.index)
   for metric in METRICS:
    dif=(treatment.loc[idx,metric].astype(float)-base.loc[idx,metric].astype(float)).groupby('id').mean()
    comparisons.append(dict(model=model,condition=condition,metric=metric,**stats(dif.values)))
 cp=pd.DataFrame(comparisons);cp['p_holm_primary']=np.nan
 primary=(cp.model=='instruct')&(cp.condition=='mean_m2')&cp.metric.isin(['encoder','ai_likeness'])
 cp.loc[primary,'p_holm_primary']=holm(cp.loc[primary,'p'].values);cp.to_csv(R/'paired_comparisons.csv',index=False)
 # Direct equal-dose controls and signed contrast, independently of zero effects.
 control=[]
 sub=df[(df.model=='instruct')&(df.seed==42)]
 for a,b in [('mean_p2','mean_m2'),('mean_m2','formality_m2'),('mean_m2','assistant_m2'),('mean_m2','human_prompt'),('residual_m2','mean_m2')]+[('mean_m2',f'random{i}') for i in range(5)]:
  da=sub[sub.condition==a].set_index('id');db=sub[sub.condition==b].set_index('id');ix=da.index.intersection(db.index)
  for m in ['encoder','ai_likeness','coherence','content']:control.append(dict(a=a,b=b,metric=m,**stats(da.loc[ix,m]-db.loc[ix,m])))
 pd.DataFrame(control).to_csv(R/'control_contrasts.csv',index=False)
 # Conservative mean-level quality gates, not filtering away damaged outputs.
 gates=[]
 for model,condition in cp[['model','condition']].drop_duplicates().itertuples(index=False):
  cur=cp[(cp.model==model)&(cp.condition==condition)].set_index('metric')
  gates.append(dict(model=model,condition=condition,coherence_noninferior=bool(cur.loc['coherence','lo']>-.25),content_noninferior=bool(cur.loc['content','lo']>-.05)))
 save('quality_gates.json',gates)
 # Coherence matching is explicitly descriptive post-treatment conditioning.
 matched=[]
 for model in df.model.unique():
  sub=df[(df.model==model)&(df.seed==42)];z=sub[sub.condition=='zero'].set_index('id')
  for condition in sorted(set(sub.condition)-{'zero'}):
   t=sub[sub.condition==condition].set_index('id');ix=z.index.intersection(t.index)
   equal=ix[(z.loc[ix,'coherence']>=4)&(t.loc[ix,'coherence']>=4)&((z.loc[ix,'coherence']-t.loc[ix,'coherence']).abs()<=.25)&(z.loc[ix,'content']>=.95)&(t.loc[ix,'content']>=.95)]
   for m in ['encoder','ai_likeness']:matched.append(dict(model=model,condition=condition,metric=m,**stats(t.loc[equal,m]-z.loc[equal,m])))
 pd.DataFrame(matched).to_csv(R/'quality_matched.csv',index=False)
 # Independent judge/encoder validity on known-provenance HAP-E test pairs.
 cal=[];gpt=read('data_audit.json')['training_generator']
 for r in read('corpus.json'):
  k='cal:'+r['generator']+':'+r['id']
  if k in enc and k in jud:cal.append(dict(id=r['id'],label=r['label'],encoder=enc[k],judge=jud[k]['ai_likeness']))
 c=pd.DataFrame(cal);c.to_csv(R/'detector_calibration.csv',index=False);valid=[]
 for m in ['encoder','judge']:
  # Paired groups guarantee both labels per bootstrap resample.
  piv=c.pivot(index='id',columns='label',values=m).dropna();arr=piv[[0,1]].values;labels=np.tile([0,1],len(arr));rng=np.random.default_rng(72)
  bs=[roc_auc_score(labels,arr[rng.integers(len(arr),size=len(arr))].ravel()) for _ in range(2000)]
  valid.append(dict(metric=m,n_pairs=len(arr),auc=roc_auc_score(labels,arr.ravel()),lo=np.quantile(bs,.025),hi=np.quantile(bs,.975),human_mean=arr[:,0].mean(),ai_mean=arr[:,1].mean()))
 save('detector_validity.json',valid)
 save('score_coverage.json',{'generations':len(df),'encoder_missing':int(df.encoder.isna().sum()),'judge_missing':int(df.ai_likeness.isna().sum()),'judge_encoder_spearman':float(spearmanr(df.encoder,df.ai_likeness,nan_policy='omit').statistic)})
 # API metadata, actual returned usage and model/provider identifiers.
 calls=lines('judge_calls.jsonl')+lines('judge_reliability_calls.jsonl');usage=[c.get('raw',{}).get('usage',{}) for c in calls];meta=read('judge_model.json')
 cost=sum(float(u.get('cost',0) or 0) for u in usage);pt=sum(u.get('prompt_tokens',0) for u in usage);ct=sum(u.get('completion_tokens',0) for u in usage)
 save('api_usage.json',{'calls':len(calls),'failed_calls':sum('failed_ids' in c for c in calls),'prompt_tokens':pt,'completion_tokens':ct,'returned_cost_usd':cost,'catalog_estimated_cost_usd':pt*float(meta['pricing']['prompt'])+ct*float(meta['pricing']['completion']),'returned_models':sorted(set(str(c.get('raw',{}).get('model')) for c in calls)),'providers':sorted(set(str(c.get('raw',{}).get('provider')) for c in calls))})
 # Approximate condition costs split a mixed batch equally among its scored items.
 allocated={}
 for call in calls:
  ratings=call.get('ratings',[]);amount=float(call.get('raw',{}).get('usage',{}).get('cost',0) or 0)/max(1,len(ratings))
  for r in ratings:
   parts=r['id'].split(':');condition='calibration' if parts[0]=='cal' else parts[0]+':'+parts[2]
   allocated[condition]=allocated.get(condition,0)+amount
 save('api_cost_allocation.json',{'method':'equal fraction of actual mixed-batch API cost per scored item, including reliability reratings','usd':allocated})
 # Plots: full signed score/quality curve and all intervention contrasts.
 plt.rcParams.update({'font.size':10})
 fig,axes=plt.subplots(1,4,figsize=(16,4));order=['mean_m2','mean_m1','zero','mean_p1','mean_p2'];doses=[-2,-1,0,1,2]
 for ax,m in zip(axes,['encoder','ai_likeness','coherence','content']):
  for model,color in [('instruct','tab:blue'),('base','tab:orange')]:
   sub=df[(df.model==model)&(df.seed==42)];means=[];errs=[];xs=[]
   for dose,cond in zip(doses,order):
    vals=sub[sub.condition==cond][m].dropna().values
    if not len(vals):continue
    s=stats(vals);xs.append(dose);means.append(s['delta']);errs.append([max(0,s['delta']-s['lo']),max(0,s['hi']-s['delta'])])
   ax.errorbar(xs,means,yerr=np.array(errs).T,label=model,color=color,marker='o',capsize=3)
  ax.set(xlabel='Signed dose (training projection SD)',ylabel=m.replace('_',' '));ax.grid(alpha=.2)
 axes[0].legend();fig.tight_layout();fig.savefig(ROOT/'figures/dose_response.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,7))
 for ax,m in zip(axes,['encoder','ai_likeness']):
  d=cp[(cp.model=='instruct')&(cp.metric==m)].sort_values('condition');yy=np.arange(len(d));ax.errorbar(d.delta,yy,xerr=np.maximum(0,np.vstack([d.delta-d.lo,d.hi-d.delta])),fmt='o',capsize=3);ax.set_yticks(yy,d.condition);ax.axvline(0,color='gray');ax.set_xlabel('Paired change vs zero: '+m);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'figures/controls.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));readouts=pd.DataFrame(read('readout_instruct.json')+read('readout_base.json'))
 selected=[]
 for kind in ['instruct','base']:
  layer=read(f'geometry_{kind}.json')['layer'];d=readouts[(readouts.model==kind)&(readouts.layer==layer)&(readouts.method=='mean')&(readouts.split!='val')].copy();d['setting']=d.split+':'+d.generator.str.replace('hape-text_','');selected.append(d)
 rd=pd.concat(selected);rd.to_csv(R/'selected_readouts.csv',index=False);rd.pivot(index='setting',columns='model',values='auc').plot.barh(ax=ax,color=['tab:orange','tab:blue']);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2);ax.set(xlim=(0,1),xlabel='Readout AUROC',ylabel='Held-out setting');ax.axvline(.5,color='gray',ls='--');fig.tight_layout();fig.savefig(ROOT/'figures/readout.png',dpi=180);plt.close(fig)
 # Transparent automatic error inventory, representative full outputs, not just favorable cases.
 worst=df.sort_values(['content','coherence']).head(12);worst[['key','source','text','coherence','content','unsupported','encoder','ai_likeness']].to_json(R/'failure_examples.json',orient='records',indent=2)
 examples=df[(df.id=='task_0_0')&(df.seed==42)];examples[['model','condition','source','text','coherence','content','encoder','ai_likeness']].to_json(R/'paired_examples.json',orient='records',indent=2)
 print(cp[primary].to_string(index=False));print(valid);print('complete',len(df))
 state('Phase 5 analysis complete: results/paired_comparisons.csv, control_contrasts.csv, quality_gates.json, detector_validity.json and figures/*.png. Source-cluster paired bootstrap, sign flips and Holm for two primary outcomes; all interventions retained. Next Phase 6: interpret detector validity/quality gates, write REPORT.md/README.md, audit leakage and deterministic reproduction. Claims remain conditional on these checks.')
if __name__=='__main__':main()
