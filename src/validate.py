"""Final reproducibility and artifact validation. No model outputs are invented."""
from common import *
import hashlib, subprocess, numpy as np,pandas as pd

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 required=['REPORT.md','README.md','planning.md','resources.md','results/paired_comparisons.csv','results/detector_validity.json','results/quality_gates.json','results/leakage_audit.json','figures/dose_response.png','figures/controls.png']
 for f in required:assert (ROOT/f).is_file(),f
 rows=lines('generations.jsonl');assert len(rows)==1472,len(rows)
 from score import key
 keys=[key(r) for r in rows];assert len(keys)==len(set(keys))
 cov=read('score_coverage.json');assert cov['encoder_missing']==cov['judge_missing']==0
 assert all(read(f'hook_validation_{k}.json')['repeat_equal'] for k in ['instruct','base'])
 # Re-run deterministic numerical analysis and compare core artifacts byte for byte.
 checks=['paired_comparisons.csv','descriptives.csv','control_contrasts.csv','quality_gates.json','detector_validity.json','quality_matched.csv','score_coverage.json']
 before={f:digest(R/f) for f in checks}
 subprocess.run([str(ROOT/'.venv/bin/python'),'src/analyze.py'],check=True,stdout=subprocess.DEVNULL)
 after={f:digest(R/f) for f in checks};assert before==after
 cp=pd.read_csv(R/'paired_comparisons.csv');assert cp.delta.notna().all();assert ((cp.lo<=cp.delta+1e-12)&(cp.hi>=cp.delta-1e-12)).all()
 for kind in ['instruct','base']:
  v=np.load(R/f'vectors_{kind}.npz')
  for name in ['mean','logistic','formality','assistant','residualized']+[f'random{i}' for i in range(5)]:assert abs(np.linalg.norm(v[name])-1)<1e-5
 audit=read('leakage_audit.json');assert audit['source_disjoint'] and audit['normalized_text_disjoint']
 result={'passed':True,'generation_count':len(rows),'unique_keys':len(set(keys)),'all_scores_present':True,'analysis_rerun_identical':before==after,'hashes':after,'hook_validation':{k:read(f'hook_validation_{k}.json') for k in ['instruct','base']},'limits':['API cached for numerical reproduction; independent reratings assess judge variation.','Repeated generation smoke batches checked, not a full costly inference rerun.','Near-duplicate audit cannot prove absence of pretraining contamination.','Small-model, small-template study; preservation claims depend on quality outcomes.']}
 save('validation.json',result);print(json.dumps(result,indent=2))
 state('Phases 0–6 complete. REPORT.md and README.md contain actual results, limitations and reproduction commands; src/ contains full implementation. All 1,472 generations have encoder and blinded judge scores. Validation: results/validation.json; identical deterministic analysis rerun, paired sampling/zero-hook tests, native arithmetic checks, disjoint source/text folds. All scientific interpretations restricted by detector validity, quality gates and small/template-limited design. Resource-finder/generated STATE sections preserved. No additional direction introduced.')
if __name__=='__main__':main()
