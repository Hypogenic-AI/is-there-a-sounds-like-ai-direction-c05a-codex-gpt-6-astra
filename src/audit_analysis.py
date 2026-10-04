"""Leakage audit and predeclared readout-confound/transfer diagnostics."""
from common import *
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

def auc_ci(y,score,ids):
 rng=np.random.default_rng(918);unique=np.unique(ids);groups=[np.flatnonzero(ids==g) for g in unique];bs=[]
 for _ in range(2000):
  ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]);bs.append(roc_auc_score(y[ix],score[ix]))
 return dict(auc=float(roc_auc_score(y,score)),lo=float(np.quantile(bs,.025)),hi=float(np.quantile(bs,.975)))

def main():
 df=pd.DataFrame(read('corpus.json'));tr=df.split=='train';test=df.split.isin(['test','ood','hc3']);val=df.split=='val'
 sets={s:set(df[df.split==s].id) for s in df.split.unique()};assert all(not sets[a]&sets[b] for a in sets for b in sets if a<b)
 sha={s:set(df[df.split==s].sha) for s in df.split.unique()};assert all(not sha[a]&sha[b] for a in sha for b in sha if a<b)
 # Fit only training vocabulary, diagnostic near-duplicate threshold not sample selection.
 tf=TfidfVectorizer(analyzer='char_wb',ngram_range=(4,4),max_features=40000,dtype=np.float32)
 xtr=tf.fit_transform(df.loc[tr,'text']);xt=tf.transform(df.loc[test,'text']);sim=(xt@xtr.T).toarray();maxsim=sim.max(1);bad=df.loc[test].iloc[np.flatnonzero(maxsim>.9)][['id','split','generator']].to_dict('records')
 save('leakage_audit.json',{'source_disjoint':True,'normalized_text_disjoint':True,'near_duplicate_definition':'training-fitted character 4gram TF-IDF cosine > 0.90','test_max_similarity':float(maxsim.max()),'above_threshold':bad,'note':'Cannot exclude shared pretraining or less literal semantic duplicates.'})
 conf=[];boot=[];model_scores={};gpt=read('data_audit.json')['training_generator']
 for kind in ['instruct','base']:
  data=np.load(R/f'activations_{kind}.npz');vec=np.load(R/f'vectors_{kind}.npz');l=int(vec['layer']);h=data[f'h{l}'];proj=h@vec['mean'];res=h@vec['residualized'];model_scores[kind]=proj
  for split in ['test','ood','hc3']:
   for gen in sorted(set(df[df.split==split].generator)-{'human'}):
    mask=((df.split==split)&df.generator.isin(['human',gen])).values;y=df.loc[mask,'label'].values;ids=df.loc[mask,'id'].values
    for name,scores in [('mean',proj),('residualized',res),('length',data['length']),('negative_nll',-data['nll'])]:
     record=dict(model=kind,split=split,generator=gen,metric=name,**auc_ci(y,scores[mask],ids));conf.append(record)
  # Length matched test sensitivity retains only paired GPT4o/human texts within 10 tokens.
  for split in ['test','ood']:
   sub=df[(df.split==split)&df.generator.isin(['human',gpt])].copy();sub['length']=data['length'][sub.index];sub['score']=proj[sub.index];piv=sub.pivot(index='id',columns='label',values='length');ids=piv.index[(piv[1]-piv[0]).abs()<=10];ss=sub[sub.id.isin(ids)]
   if len(ids)>5:boot.append(dict(model=kind,split=split,n_pairs=len(ids),**auc_ci(ss.label.values,ss.score.values,ss.id.values)))
 pd.DataFrame(conf).to_csv(R/'readout_diagnostics.csv',index=False);save('length_matched_readout.json',boot)
 # Paired base-instruct AUC difference on exactly identical text sets.
 diffs=[]
 for split in ['test','ood','hc3']:
  mask=((df.split==split)&df.generator.isin(['human',gpt if split!='hc3' else 'chatgpt'])).values;y=df.loc[mask,'label'].values;ids=df.loc[mask,'id'].values;groups=[np.flatnonzero(ids==i) for i in np.unique(ids)];rng=np.random.default_rng(98);a=model_scores['instruct'][mask];b=model_scores['base'][mask]
  bs=[]
  for _ in range(2000):
   ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]);bs.append(roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix]))
  diffs.append(dict(split=split,delta=roc_auc_score(y,a)-roc_auc_score(y,b),lo=np.quantile(bs,.025),hi=np.quantile(bs,.975)))
 save('base_instruct_auc_difference.json',diffs)
 # Domain clustered sensitivity (only 8 templates, so interval is exploratory).
 if (R/'scored_generations.csv').exists():
  from analyze import stats
  d=pd.read_csv(R/'scored_generations.csv');out=[]
  for m in ['encoder','ai_likeness','coherence','content']:
   sub=d[d.model=='instruct'];p=sub.pivot_table(index=['domain','id','seed'],columns='condition',values=m);delta=(p.mean_m2-p.zero).groupby('domain').mean();out.append(dict(metric=m,**stats(delta)))
  save('template_cluster_sensitivity.json',out)
 print('leakage, readout and confound diagnostics complete',flush=True)
if __name__=='__main__':main()
