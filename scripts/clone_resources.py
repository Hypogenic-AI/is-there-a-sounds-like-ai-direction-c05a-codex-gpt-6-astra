"""Restore pinned code-only sparse checkouts from the tracked manifest."""
from pathlib import Path
import subprocess,json
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT and (ROOT/'.git').is_dir(), 'Run from repository root'
records=json.loads((ROOT/'artifacts/resource_finder/code_manifest.json').read_text())
for r in records:
 print(Path.cwd(),flush=True)
 p=ROOT/r['path']
 created = not p.exists()
 if created:
  subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','https://github.com/'+r['repo']+'.git',str(p)],check=True)
 actual=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
 if actual!=r['commit']:
  if not created:
   raise RuntimeError(f'{p} is at {actual}; expected {r["commit"]}. Fetch and checkout the recorded commit in this resource clone.')
  subprocess.run(['git','-C',str(p),'fetch','--depth','1','origin',r['commit']],check=True)
  subprocess.run(['git','-C',str(p),'checkout','--detach',r['commit']],check=True)
  actual=r['commit']
 patterns=['/*.*','/LICENSE']+['/'+d+'/' for d in r['included_directories']]
 subprocess.run(['git','-C',str(p),'sparse-checkout','set','--no-cone',*patterns],check=True)
 print(r['name'],actual,flush=True)
