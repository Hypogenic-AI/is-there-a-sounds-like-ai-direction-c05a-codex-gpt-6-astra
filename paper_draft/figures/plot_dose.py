"""Replot cached seed-42 data; bootstrap convention matches src/analyze.py."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
df=pd.read_csv(root/'results/scored_generations.csv')
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.2,5.2),layout='constrained')
for ax,metric,label in zip(axs.flat,['encoder','ai_likeness','coherence','content'],['Encoder AI probability','Judge AI-likeness (0–100)','Coherence (1–5)','Fact coverage (0–1)']):
 for model,color in [('instruct','tab:blue'),('base','tab:orange')]:
  xs=[]; means=[];errors=[]
  for dose,cond in zip([-2,-1,0,1,2],['mean_m2','mean_m1','zero','mean_p1','mean_p2']):
   v=df.loc[(df.model==model)&(df.seed==42)&(df.condition==cond),metric].dropna().to_numpy()
   if not len(v):continue
   rng=np.random.default_rng(77)
   lo,hi=np.quantile(v[rng.integers(len(v),size=(5000,len(v)))].mean(1),[.025,.975])
   xs.append(dose);means.append(v.mean());errors.append([v.mean()-lo,hi-v.mean()])
  ax.errorbar(xs,means,yerr=np.array(errors).T,color=color,marker='o',capsize=3,label={'instruct':'Instruction','base':'Base'}[model])
 ax.set(xlabel='Signed dose (projection SD)',ylabel=label,xticks=[-2,-1,0,1,2]);ax.grid(alpha=.2)
axs[0,0].legend(frameon=False)
fig.savefig(Path(__file__).with_name('dose_response.pdf'))
