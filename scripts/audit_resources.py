"""Verify local resource integrity, pins, documentation and Git exclusions."""
from pathlib import Path
import ast, hashlib, json, re, subprocess, sys
from datetime import datetime, timezone
from pypdf import PdfReader
ROOT = Path(__file__).resolve().parents[1]
assert Path.cwd() == ROOT and (ROOT / '.git').is_dir()
assert Path(sys.prefix).resolve() == ROOT / '.venv'
base = ROOT / 'artifacts/resource_finder'
load = lambda name: json.loads((base / (name + '.json')).read_text())
counts = {}
for name in ['paper_manifest', 'dataset_downloads', 'assistant_prompt_manifest']:
    records = load(name)
    for r in records:
        p = ROOT / r['file']
        assert p.stat().st_size == r['bytes'], p
        with p.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == r['sha256'], p
        assert subprocess.run(['git', 'check-ignore', '-q', str(p)]).returncode == 0, p
    counts[name] = len(records)
for r in load('paper_manifest'):
    p = ROOT / r['file']
    assert len(PdfReader(p).pages) == r['pages']
    chunks = sorted((ROOT / 'papers/pages').glob(p.stem + '_chunk_*.pdf'))
    sizes = [len(PdfReader(c).pages) for c in chunks]
    assert sum(sizes) == r['pages'] and all(x == 3 for x in sizes[:-1]), p
    assert (ROOT / 'papers/pages' / (p.stem + '_manifest.txt')).exists()
for r in load('code_manifest'):
    actual = subprocess.check_output(['git', '-C', r['path'], 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == r['commit'], r['name']
    assert not r['syntax_errors']
counts['repositories'] = len(load('code_manifest'))
docs = ['literature_review.md', 'resources.md', 'planning.md', 'papers/README.md', 'datasets/README.md', 'code/README.md', 'STATE.md', 'datasets/.gitignore', 'pyproject.toml', 'uv.lock']
for name in docs:
    p = ROOT / name
    assert p.is_file() and p.stat().st_size > 0, p
    assert subprocess.run(['git', 'check-ignore', '-q', str(p)]).returncode == 1, p
    if p.suffix == '.md':
        for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', p.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (p.parent / target.split('#')[0]).exists(), (p, target)
for p in (ROOT / 'datasets/samples').glob('*.json'):
    assert p.stat().st_size < 100000
    json.loads(p.read_text())
    assert subprocess.run(['git', 'check-ignore', '-q', str(p)]).returncode == 1, p
for p in (ROOT / 'scripts').glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
assert not list((ROOT / 'datasets').rglob('*.part'))
log = (base / 'reading_log.md').read_text()
for pid, expected in [('2503.03601', 8), ('2606.07313', 9), ('2608.24780', 8), ('2601.10387', 18)]:
    for n in range(1, expected + 1):
        assert f'{pid} chunk {n:03d}' in log, (pid, n)
subprocess.run(['uv', 'pip', 'check', '--python', sys.executable], check=True)
report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'passed': True, 'counts': counts, 'data_bytes': sum(r['bytes'] for r in load('dataset_downloads')), 'checks': ['All source/data/prompt SHA256 hashes and sizes', 'All PDF page counts and complete chunk coverage', 'Pinned Git revisions', 'Required documentation and local Markdown links', 'Raw resources excluded; docs and small samples eligible for Git', 'Resource scripts parse; isolated environment dependencies consistent', 'All 43 core-paper chunks have notes'], 'not_tested': ['Model runtime or scientific hypothesis', 'Near-duplicate leakage', 'Fresh network redownload of every resource']}
(base / 'final_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
