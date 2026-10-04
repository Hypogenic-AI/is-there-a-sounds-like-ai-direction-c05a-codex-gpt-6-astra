"""Extract pinned upstream prompt data into datasets/, separate from code/."""
from pathlib import Path
import subprocess,tarfile,io,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT and (ROOT/'.git').is_dir()
print(Path.cwd(),flush=True)
rec=next(r for r in json.loads((ROOT/'artifacts/resource_finder/code_manifest.json').read_text()) if r['name']=='assistant-axis')
raw=subprocess.check_output(['git','-C',str(ROOT/rec['path']),'archive',rec['commit'],'data'])
dest=ROOT/'datasets/assistant_axis_prompts';dest.mkdir(exist_ok=True)
manifest=[]
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
 for item in tar:
  if not item.isfile():continue
  rel=Path(item.name).relative_to('data');assert '..' not in rel.parts
  if rel.suffix not in ['.json','.jsonl']:continue
  b=tar.extractfile(item).read(); p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  parsed=[json.loads(line) for line in b.decode().splitlines()] if rel.suffix=='.jsonl' else json.loads(b)
  manifest.append({'file':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(ROOT/'artifacts/resource_finder/assistant_prompt_manifest.json').write_text(json.dumps(manifest,indent=2))
print(len(manifest),'validated prompt files')
