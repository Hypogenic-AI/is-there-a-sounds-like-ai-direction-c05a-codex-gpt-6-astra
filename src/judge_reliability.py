"""Exploratory independent re-rating of 64 primary texts in reordered batches."""
from common import *
from score import judge_request,key
import numpy as np,concurrent.futures
from scipy.stats import spearmanr
rows=lines('generations.jsonl');selected=[r for r in rows if r['model']=='instruct' and r['seed']==42 and r['condition'] in ['zero','mean_m2'] and int(r['id'].split('_')[-1])<4]
rng=np.random.default_rng(811);rng.shuffle(selected);items=[dict(id=key(r),text=r['text'],source=r['source']) for r in selected]
existing=lines('judge_reliability_calls.jsonl');done={r['id'] for c in existing for r in c.get('ratings',[])};todo=[r for r in items if r['id'] not in done]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for c in pool.map(judge_request,[todo[i:i+8] for i in range(0,len(todo),8)]):append('judge_reliability_calls.jsonl',c)
a={r['id']:r for c in lines('judge_calls.jsonl') for r in c.get('ratings',[])};b={r['id']:r for c in lines('judge_reliability_calls.jsonl') for r in c.get('ratings',[])};ids=sorted(set(a)&set(b));result={}
for m in ['ai_likeness','coherence','content','unsupported']:
 x=np.array([a[i][m] for i in ids]);y=np.array([b[i][m] for i in ids]);result[m]=dict(n=len(ids),spearman=float(spearmanr(x,y).statistic),mean_absolute_difference=float(np.mean(abs(x-y))))
save('judge_reliability.json',result);print(result)
