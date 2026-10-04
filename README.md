# Is there a “sounds like AI” direction?

A controlled residual-steering study of Qwen2.5-1.5B base and instruction checkpoints. We ran 1,472 generations, signed interventions, five random directions, nuisance controls, and two independent style measures.

- Negative steering reduced blinded judge AI-likeness by **20.0/100 points** (95% CI −22.9 to −17.3), but reduced coherence by **0.59/5**.
- The frozen encoder did not corroborate the primary improvement; positive steering moved the two measures in opposite directions.
- Readout AUROC was **0.970** on held-out GPT-4o text but **0.486** on held-out Llama-3 base text.
- Base and instruction backbones had nearly identical readout performance; no amplification was established.
- The stronger claim of a **distinct, detector-consistent, content-preserving AI-ness direction remains unestablished**.

See [REPORT.md](REPORT.md) for methods, all conditions, uncertainty, failure examples and limitations.

## Reproduce

Run from this repository root. Use an NVIDIA GPU for local inference and an `OPENROUTER_KEY` for fresh judge calls. The reference run used an RTX A6000; concurrent models need approximately 40 GB reserved GPU memory, while sequential runs need substantially less. Model/data downloads require several GB of disk. Secrets are read from environment variables and are never saved.

```bash
pwd
uv sync --frozen
source .venv/bin/activate
# Restore omitted raw data if necessary:
python scripts/download_datasets.py
python src/prepare.py
python src/download_models.py
python src/experiment.py --model instruct
python src/experiment.py --model base
python src/score.py detector
python src/score.py judge
python src/judge_reliability.py
python src/analyze.py
python src/audit_analysis.py
python src/supplement.py
python src/validate.py
```

The live catalog used for judge selection is archived in `results/live_catalog.json`; reuse it and pinned `results/models.json` for this protocol. `prepare.py` preserves existing Hugging Face revision pins and resolves them only if `results/models.json` is absent, so retain that archived file for exact checkpoint reproduction. For a new contemporary study, query the catalog again and document changed selection before running.

Scripts cache activations and skip already-scored/generated keys. With existing artifacts, reproduce the numerical results without GPU/API calls:

```bash
pwd
source .venv/bin/activate
python src/analyze.py
python src/supplement.py
python src/validate.py
```

For a full fresh inference replication, copy/archive `results/` first, preserve the catalog/model pin files, then move generated outputs, score caches, activation arrays and nuisance caches out of `results/` before running. Do not mix old and new experimental configurations. Main local model pipelines took about 7.7 and 2.9 minutes on the reference GPU, partly concurrently; full setup/scoring/audits take longer. Live API use cost **$0.623** in this run. Repeated API ratings can vary; cached numerical analysis is deterministic.

## Files

- `REPORT.md`, `planning.md`, `STATE.md`: findings, frozen design and phase handoff.
- `src/`: preparation, interventions, independent scoring, analysis and validation.
- `results/`: raw generations/API responses, pinned models/configuration, activation arrays, vectors, statistics and validation hashes.
- `figures/`: dose-response, control and readout plots.
- `datasets/`, `papers/`, `code/`: pre-gathered resources; large inputs/weights are ignored by Git.
- `literature_review.md`, `resources.md`: source synthesis and reproducibility catalog.
- `logs/`: execution logs, including the repaired initial missing-compiler failure.

The persona control is a reduced proxy, not the published Assistant Axis. Small historical checkpoints, templated tasks, a weak out-of-domain encoder and noisy judge scores limit generalization. No human-written status is inferred from detector scores.
