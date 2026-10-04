"""Recreate paper downloads from versioned URLs and verify original hashes."""
from pathlib import Path
import json,hashlib,requests,subprocess
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT and (ROOT/'.git').is_dir()
for rec in json.loads((ROOT/'artifacts/resource_finder/paper_manifest.json').read_text()):
 print(Path.cwd(),flush=True)
 p=ROOT/rec['file'];p.parent.mkdir(exist_ok=True)
 if not p.exists():
  r=requests.get(rec['url'],timeout=120);r.raise_for_status();assert r.content.startswith(b'%PDF')
  assert hashlib.sha256(r.content).hexdigest()==rec['sha256'],f'Changed PDF bytes: {p}; inspect before accepting'
  p.write_bytes(r.content)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
 subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'.codex/skills/paper-finder/scripts/pdf_chunker.py'),str(p),'--pages-per-chunk','3'],check=True)
