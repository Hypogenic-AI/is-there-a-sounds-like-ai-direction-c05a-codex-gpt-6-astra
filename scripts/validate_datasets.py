"""Full file parsing, schema/count/null checks; light descriptive checks, no experiment."""
from pathlib import Path
import csv,json,hashlib,collections,statistics
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT and (ROOT/'.git').is_dir()
print(Path.cwd(),flush=True)
(ROOT/'datasets/samples').mkdir(exist_ok=True)
report={}
def digest(text):return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
def save_sample(name,rows):
 # Truncated excerpts keep documentation small and avoid redistributing large texts.
 def trunc(x):
  if isinstance(x,str):return x[:350]+(' [excerpt]' if len(x)>350 else '')
  if isinstance(x,list):return [trunc(v) for v in x[:1]]
  if isinstance(x,dict):return {k:trunc(v) for k,v in x.items()}
  return x
 p=ROOT/'datasets/samples'/f'{name}.json';p.write_text(json.dumps([trunc(r) for r in rows[:3]],ensure_ascii=False,indent=2));assert p.stat().st_size<100000
hc3=[];hc3_hash=set()
with (ROOT/'datasets/hc3/all.jsonl').open() as f:
 for line in f:
  r=json.loads(line);hc3.append(r)
  hc3_hash.update(digest(t) for k in ['human_answers','chatgpt_answers'] for t in r[k])
report['hc3']={'rows':len(hc3),'fields':list(hc3[0]),'domains':dict(collections.Counter(r['source'] for r in hc3)),'human_answers':sum(len(r['human_answers']) for r in hc3),'ai_answers':sum(len(r['chatgpt_answers']) for r in hc3),'empty_human_lists':sum(not r['human_answers'] for r in hc3),'empty_ai_lists':sum(not r['chatgpt_answers'] for r in hc3),'empty_questions':sum(not r['question'].strip() for r in hc3),'unique_question_hashes':len({digest(r['question']) for r in hc3}),'unique_indices':len({r['index'] for r in hc3})}
save_sample('hc3',hc3);print('HC3',report['hc3'],flush=True)
hape={};groups=[]
for p in sorted((ROOT/'datasets/hape').glob('*.parquet')):
 t=pq.read_table(p);rows=t.to_pylist();ids={r['doc_id'].split('@')[0] for r in rows};groups.append(ids)
 hape[p.name]={'rows':t.num_rows,'schema':str(t.schema),'nulls':{col:t[col].null_count for col in t.column_names},'empty_text':sum(not r['text'].strip() for r in rows),'unique_groups':len(ids),'domains':dict(collections.Counter(r['doc_id'].split('_')[0] for r in rows)),'median_words':statistics.median(len(r['text'].split()) for r in rows)}
 if 'human-chunk-2' in p.name:save_sample('hape',rows)
report['hape']={'files':hape,'common_source_groups':len(set.intersection(*groups)),'union_source_groups':len(set.union(*groups))};print('HAPE',len(hape),report['hape']['common_source_groups'],flush=True)
mage={};hashes={}
for p in sorted((ROOT/'datasets/mage').glob('*.csv')):
 n=empty=overlap=0;labels=collections.Counter();src=collections.Counter();nulls=collections.Counter();sample=[];hs=set();label_src=collections.Counter()
 with p.open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames
  for r in reader:
   assert None not in r and all(v is not None for v in r.values()),(p,n)
   n+=1;labels[r['label']]+=1;src[r['src']]+=1;label_src[(r['label'],'human' if r['src'].endswith('_human') else 'machine')]+=1
   empty+=not r['text'].strip();h=digest(r['text']);hs.add(h);overlap+=h in hc3_hash
   for k,v in r.items():nulls[k]+=not v.strip()
   if len(sample)<3:sample.append(r)
 hashes[p.name]=hs
 mage[p.name]={'rows':n,'fields':fields,'labels':dict(labels),'sources':dict(src),'missing':dict(nulls),'empty_text':empty,'unique_normalized_texts':len(hs),'rows_overlapping_hc3':overlap,'label_source_check':{str(k):v for k,v in label_src.items()}}
 save_sample('mage_'+p.stem,sample);print('MAGE',p.name,n,dict(labels),'HC3 overlap',overlap,flush=True)
report['mage']={'files':mage,'normalized_text_overlap':{a+' vs '+b:len(hashes[a]&hashes[b]) for a in hashes for b in hashes if a<b},'raw_label_mapping':'1 = human, 0 = machine, checked against src suffix. Invert for project convention AI=1.'}
p=ROOT/'datasets/raid/train_none.csv';n=0;models=collections.Counter();domains=collections.Counter();attacks=collections.Counter();sources=set();ids=set();empty=0;sample=[];missing=collections.Counter();cross=collections.Counter()
with p.open(encoding='utf-8-sig',newline='') as f:
 reader=csv.DictReader(f);fields=reader.fieldnames
 for r in reader:
  assert None not in r and all(v is not None for v in r.values()),n
  n+=1;models[r['model']]+=1;domains[r['domain']]+=1;attacks[r['attack']]+=1;sources.add(r['source_id']);ids.add(r['id']);empty+=not r['generation'].strip();cross[r['domain']+'|'+r['model']]+=1
  for k,v in r.items():missing[k]+=not v.strip()
  if len(sample)<3:sample.append(r)
report['raid']={'rows':n,'fields':fields,'models':dict(models),'domains':dict(domains),'attacks':dict(attacks),'unique_source_ids':len(sources),'unique_ids':len(ids),'empty_generation':empty,'missing':dict(missing),'domain_model_counts':dict(cross),'subset':'Complete authors-provided clean train split; attack=none. Not full RAID and not official hidden-label test.'}
save_sample('raid',sample);print('RAID',n,dict(models),flush=True)
(ROOT/'artifacts/resource_finder/dataset_validation.json').write_text(json.dumps(report,indent=2))
print('Validation report saved',flush=True)
