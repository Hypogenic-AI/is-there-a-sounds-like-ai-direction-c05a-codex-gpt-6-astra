"""Shared paths, append-only artifacts and phase-specific state updates."""
import json, os, random, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT, 'Run from the workspace root'
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
R=ROOT/'results'
def save(name,obj):
    (R/name).write_text(json.dumps(obj,indent=2,default=str))
def read(name):return json.loads((R/name).read_text())
def append(name,obj):
    with (R/name).open('a') as f:f.write(json.dumps(obj,default=str)+'\n')
def lines(name):
    p=R/name
    return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def state(note):
    p=ROOT/'STATE.md';s=p.read_text();a='<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->';b='<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->'
    p.write_text(s[:s.index(a)+len(a)]+'\n'+note+'\n'+s[s.index(b):])
def seed(n):
    import numpy as np,torch
    random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
