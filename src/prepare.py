"""Build source-grouped corpus splits and fact-grounded intervention tasks."""
from common import *
import hashlib, numpy as np, pandas as pd
from huggingface_hub import HfApi
seed(42)
files=list((ROOT/'datasets/hape').glob('*.parquet')); print([p.name for p in files])
frames={p.stem:pd.read_parquet(p) for p in files}
for k,d in frames.items():
 d['id']=d.doc_id.str.split('@').str[0];frames[k]=d.set_index('id')
print(list(frames))
def key(s):return next(k for k in frames if s in k)
h=frames[key('human-chunk-2')]
gens=[k for k in frames if 'human' not in k];gpt=next(k for k in gens if 'gpt-4o' in k and 'mini' not in k)
base=next(k for k in gens if '8b' in k.lower() and 'instruct' not in k.lower())
inst=next(k for k in gens if '8b' in k.lower() and 'instruct' in k.lower())
rng=np.random.default_rng(42)
ids=sorted(h.index);ind=[i for i in ids if i.split('_')[0] in ['acad','blog','news']];ood=[i for i in ids if i.split('_')[0] in ['fic','spok','tvm']];rng.shuffle(ind);rng.shuffle(ood)
folds={'train':ind[:384],'val':ind[384:480],'test':ind[480:608],'ood':ood[:128]}
rows=[];seen={};excluded=[]
for split,items in folds.items():
 for i in items:
  sources=[('human',h), (gpt,frames[gpt])]+([] if split in ['train','val'] else [(base,frames[base]),(inst,frames[inst])])
  candidate=[]
  for gen,df in sources:
   txt=' '.join(df.loc[i,'text'].split()[:128]);digest=hashlib.sha256(txt.lower().encode()).hexdigest()
   candidate.append(dict(id=i,split=split,domain=i.split('_')[0],generator=gen,label=int(gen!='human'),text=txt,sha=digest))
  if any(r['sha'] in seen and seen[r['sha']]!=i for r in candidate):excluded.append(i);continue
  for r in candidate:seen[r['sha']]=i
  rows.extend(candidate)
# External HC3 question-paired evaluation, not used in fitting.
hc=[json.loads(s) for s in (ROOT/'datasets/hc3/all.jsonl').read_text().splitlines()];rng.shuffle(hc);questions=set();n=0
for r in hc:
 q=' '.join(r['question'].lower().split())
 if not r['human_answers'] or not r['chatgpt_answers'] or q in questions:continue
 questions.add(q);pair=[]
 for label,col in [(0,'human_answers'),(1,'chatgpt_answers')]:
  txt=' '.join(r[col][0].split()[:128]);digest=hashlib.sha256(txt.lower().encode()).hexdigest()
  pair.append(dict(id='hc3_'+hashlib.sha256(q.encode()).hexdigest()[:16],split='hc3',domain=r['source'],generator='chatgpt' if label else 'human',label=label,text=txt,sha=digest))
 if any(x['sha'] in seen for x in pair):continue
 rows+=pair;n+=1
 if n==128:break
save('corpus.json',rows)
# Unique fictitious event records avoid unverifiable world knowledge in quality scoring.
topics=[('library','reading workshop','books'),('garden','planting day','seedlings'),('museum','local history tour','visitors'),('school','science exhibition','projects'),('shelter','supply drive','blankets'),('observatory','stargazing evening','telescopes'),('bakery','bread class','loaves'),('cycling club','repair session','bicycles')]
tasks=[]
for t,(place,event,item) in enumerate(topics):
 for j in range(8):
  name=['Alder','Birch','Cedar','Dover','Elm','Fern','Grove','Hazel'][j]
  count=17+7*j+3*t;day=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Monday'][j]
  hour=9+(j%8);organizer=['Mira','Jonah','Leila','Owen','Ravi','Nora','Elena','Sam'][j]
  source=f'The {name} {place} is holding a {event} on {day} at {hour}:00. {organizer} is coordinating it. The event has {count} {item}. Attendance is free. Registration closes two days before the event.'
  tasks.append(dict(id=f'task_{t}_{j}',domain=place,source=source,anchors=[name,day,str(hour)+':00',organizer,str(count),'free'],prompt='Write one clear paragraph of 70–100 words announcing the event below. Preserve all supplied facts, do not add factual claims, and output only the paragraph.\nFacts: '+source))
save('tasks.json',tasks)
models=['Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-1.5B','Hello-SimpleAI/chatgpt-detector-roberta']
if not (R/'models.json').exists():
 save('models.json',{m:HfApi().model_info(m).sha for m in models})
else:
 assert set(read('models.json'))==set(models), 'Archived model configuration differs'
cat=read('live_catalog.json')['data'];save('judge_model.json',next(m for m in cat if m['id']=='google/gemini-3.8-flash'))
save('data_audit.json',{'counts':pd.DataFrame(rows).groupby(['split','generator']).size().to_dict().__str__(),'excluded_duplicate_ids':excluded,'source_sets':folds,'unique_tasks':len(tasks),'training_generator':gpt,'test_generators':[base,inst]})
# Full copies of text stay local with source provenance and hashes.
print(pd.DataFrame(rows).groupby(['split','generator']).size())
state('Phase 2 complete: isolated runtime installed; GPU usable; source-grouped HAP-E/HC3 splits and 64 factual tasks saved in results/corpus.json and tasks.json; model revisions pinned in models.json. No outcome-based changes. Next Phase 3: extract post-block states, fit readouts/nuisance vectors, validate hooks and generate signed/control outputs.')
