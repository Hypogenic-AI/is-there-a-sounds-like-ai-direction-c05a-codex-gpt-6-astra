"""Exploratory seed/template robustness and generation failure inventories."""
from common import *
from analyze import stats
import numpy as np,pandas as pd,re

d=pd.read_csv(R/'scored_generations.csv');out=[]
for model in ['instruct','base']:
 for sd in d[d.model==model].seed.unique():
  sub=d[(d.model==model)&(d.seed==sd)]
  for m in ['encoder','ai_likeness','coherence','content']:
   p=sub.pivot(index='id',columns='condition',values=m)
   for cond in ['mean_m2','mean_p2']:out.append(dict(model=model,seed=int(sd),condition=cond,metric=m,**stats(p[cond]-p.zero)))
pd.DataFrame(out).to_csv(R/'seed_sensitivity.csv',index=False)
out=[]
for m in ['encoder','ai_likeness','coherence','content']:
 sub=d[d.model=='instruct'];p=sub.pivot_table(index=['domain','id','seed'],columns='condition',values=m);delta=(p.mean_m2-p.zero).groupby('domain').mean();out.append(dict(metric=m,**stats(delta)))
save('template_cluster_sensitivity.json',out)
# Direct comparison against average of five norm-matched random directions, not best one.
out=[]
for m in ['encoder','ai_likeness','coherence','content']:
 p=d[(d.model=='instruct')&(d.seed==42)].pivot(index='id',columns='condition',values=m)
 out.append(dict(metric=m,**stats(p.mean_m2-p[[f'random{i}' for i in range(5)]].mean(axis=1))))
save('random_average_contrast.json',out)
# Descriptive flags complement judge ratings; no exclusions are made.
d['non_latin']=d.text.map(lambda t:bool(re.search('[\u4e00-\u9fff\u0400-\u04ff]',t)))
d['words']=d.text.str.split().str.len();d['word_limit_met']=d.words.between(70,100)
d['low_coherence']=d.coherence<4;d['missing_facts']=d.content<.95;d['unsupported_claims']=d.unsupported>.1
summary=d.groupby(['model','condition'])[['non_latin','word_limit_met','low_coherence','missing_facts','unsupported_claims','words']].mean();summary.to_csv(R/'error_rates.csv')
# Regression is descriptive; length is post-treatment and may mediate effects.
from sklearn.linear_model import LinearRegression
primary=d[(d.model=='instruct')&d.condition.isin(['zero','mean_m2'])].copy();primary['treatment']=(primary.condition=='mean_m2').astype(float)
x=pd.get_dummies(primary[['id','seed']],columns=['id','seed'],dtype=float);x['treatment']=primary.treatment;x['tokens']=primary.tokens
reg=LinearRegression().fit(x,primary.ai_likeness)
save('length_adjusted_exploratory.json',{'treatment_coefficient':float(reg.coef_[list(x.columns).index('treatment')]),'warning':'Post-treatment length adjustment is descriptive, not an identified direct causal effect.'})
print('supplement complete')
