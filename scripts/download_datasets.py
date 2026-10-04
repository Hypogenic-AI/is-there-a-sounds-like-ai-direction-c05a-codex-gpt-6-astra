"""Download exact requested datasets; RAID uses authors' clean train mirror."""
from pathlib import Path
import requests, json, hashlib, concurrent.futures, time
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT and (ROOT/'.git').is_dir(), 'Run from repository root'
print(Path.cwd(), flush=True)
DEST=ROOT/'datasets'
REPOS={'hc3':'Hello-SimpleAI/HC3','hape':'browndw/human-ai-parallel-corpus','mage':'yaful/MAGE'}
jobs=[]
for name,repo in REPOS.items():
 metadata=json.loads((ROOT/'artifacts/resource_finder'/f'{repo.split("/")[-1]}_metadata.json').read_text())
 revision=metadata['sha']
 tree=json.loads((ROOT/'artifacts/resource_finder'/f'{repo.split("/")[-1]}_tree.json').read_text())
 for item in tree:
  fn=item['path']
  keep=(name=='hc3' and fn=='all.jsonl') or (name=='hape' and fn.endswith('.parquet')) or (name=='mage' and fn.endswith('.csv'))
  if keep:jobs.append((name,fn.split('/')[-1],f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{fn}',item['size'],revision))
jobs.append(('raid','train_none.csv','https://dataset.raid-bench.xyz/train_none.csv',None,None))
def download(job):
 name,fn,url,size,rev=job
 target=DEST/name/fn
 print(Path.cwd(),flush=True)
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  for attempt in range(3):
   try:
    print('Downloading',name,fn,flush=True)
    with requests.get(url,stream=True,timeout=(20,90)) as r:
     r.raise_for_status()
     with target.with_suffix(target.suffix+'.part').open('wb') as f:
      for chunk in r.iter_content(1024*1024): f.write(chunk)
    if size is not None: assert target.with_suffix(target.suffix+'.part').stat().st_size==size
    target.with_suffix(target.suffix+'.part').rename(target)
    break
   except Exception:
    if attempt==2: raise
    time.sleep(2)
 if size is not None:assert target.stat().st_size==size
 h=hashlib.sha256()
 with target.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 entry={'dataset':name,'file':str(target.relative_to(ROOT)),'url':url,'revision':rev,'bytes':target.stat().st_size,'sha256':h.hexdigest()}
 print('DONE',name,fn,entry['bytes'],flush=True)
 return entry
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(download,jobs))
(ROOT/'artifacts/resource_finder/dataset_downloads.json').write_text(json.dumps(records,indent=2))
print('Complete:',len(records),'files',flush=True)
