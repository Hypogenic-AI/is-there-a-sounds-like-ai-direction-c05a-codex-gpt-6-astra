# Code resources

Six official repositories cloned as shallow, filtered, sparse checkouts. Root source/docs and selected code directories are present; upstream corpora, vectors, model weights, results and leaderboard dumps are excluded. Assistant-axis prompt data is separately staged in `../datasets/assistant_axis_prompts/`. Repo code is ignored by parent Git; reproducibility lives in `../artifacts/resource_finder/code_manifest.json` with full commit IDs and selected directories.

## Restore and inspect

From the workspace root:

```bash
pwd
.venv/bin/python scripts/clone_resources.py
```

New clones fetch and detach at the recorded commit if upstream HEAD has moved. Existing clones at another revision cause the script to stop rather than overwrite work. To deliberately restore an existing resource clone, run `git -C code/<name> fetch --depth 1 origin <recorded-commit>`, then `git -C code/<name> checkout --detach <recorded-commit>` and rerun the script. Do not overwrite local edits. Use `git -C` to keep the workspace working directory fixed.

|Repository|Commit|Purpose and local entry points|
|---|---|---|
|[MGT probes](https://github.com/gerritq/mgt_probes)|`8875297e41846760e7cd06a7a249f581b9364146`|`mgt_probes/src/probes/probe_main.py`: `LinearProbing._train_linear_probe` (LLP), `_train_meta_probe` (CLP); `src/inference.py` extracts activations. `probe.sh` is the experiment driver.|
|[SV-Detect](https://github.com/Atmyre/sv-detect)|`d8d6541568c572bc29d53e43e870fd7dfc5d6dbc`|`sv-detect/src/extract/nb_pipeline.py`: `compute_sv`, `dot_products`, `build_detector`; CLI extraction/construction/head modules; `src/interpret/regex_detector*.py` supplies nuisance surface baselines.|
|[Assistant Axis](https://github.com/safety-research/assistant-axis)|`a98961956072224eaf244eb289d6c01700b63795`|`assistant-axis/assistant_axis/axis.py` computes contrast; `steering.py` has `ActivationSteering` addition/ablation/capping context manager. `pipeline/` five-stage extraction, judge, aggregation workflow; `notebooks/` examples.|
|[CAA](https://github.com/nrimsky/CAA)|`5dabbbd9a0bca5f25e174501e959de378806aa48`|`CAA/generate_vectors.py`, `normalize_vectors.py`, `llama_wrapper.py`, `prompting_with_steering.py`; `utils/` helpers. Adapt paired difference and hook logic rather than full old environment.|
|[Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)|`971b05202bac2bb504d60c0ac0812fea7a8f7c82`|`fast-detect-gpt/scripts/fast_detect_gpt.py:get_sampling_discrepancy_analytic`; `scripts/local_infer.py:FastDetectGPT.compute_crit`; `scripts/model.py` model loaders. Independent frozen external score.|
|[RAID](https://github.com/liamdugan/raid)|`f450f4b1e0bfd9dc2e3475d10ec3bd70cc7153f1`|`raid/raid/` package has detection/evaluation; `detect_cli.py`, `evaluate_cli.py`; `detectors/` adapters and `generation/` decoding/attack references. Prefer local clean CSV over default massive downloads.|

## Dependencies and adaptation decisions

**MGT probes.** Python >=3.11,<3.13; upstream pins Torch 2.5.1/CUDA12.1 and has a broad dependency list (Transformers, sklearn, datasets, PEFT, etc.). `BASE_COE` is required for its paths. Code standardizes activations **before PCA**, unlike a naive reading of the shortened paper method; undo scaling and PCA when exporting a residual-space direction. `Inference.run` is single-example, truncates to 1024 and reads `outputs.hidden_states`, whose indexing/final normalization must be reconciled with intervention hooks. No license file identified.

**SV-Detect.** Upstream asks for Python >=3.11, Torch >=2.10/CUDA12.8, Transformers >=4.57, numpy>=2.2, pandas>=3, sklearn>=1.8. This differs from the other repos; do not install all requirements wholesale into one environment. `SVDETECT_BASE` selects artifact root. Source-code caveats at the pinned commit:

- README/YAML give vector logistic `C=0.01`; `nb_pipeline._sv_logreg_layer` uses **C=1.0**, per-pair centering, no intercept. This requires equal-length meaningful pairs; arbitrary row pairing is not a valid paired control.
- README says head `fit_intercept=False`; `build_detector()` leaves sklearn's default intercept enabled. Decide and record one implementation; no exact-reproduction claim until reconciled.
- Mean vectors and cosine normalization have no zero-norm guard. PCA signs need orientation with labels. QR requires rank checks if used for nuisance removal.
- `ActivationRecorder` uses mean over tokens for single unpadded examples; adapt masks before batching or response-only extraction.
- These scripts detect text; they do not causally steer generation. No license file identified.

**Assistant Axis.** Python>=3.10; Torch>=2, Transformers>=4.40, accelerate, numpy/sklearn, vLLM, plotting/notebooks, judge API dependency. A minimal hook adaptation need not install the entire vLLM/judge stack. `compute_axis` returns an unnormalized assistant-minus-role mean; `ActivationSteering` converts dtype/device but does not silently impose unit norm. `positions='all'` affects prompt positions too if enabled during prefill; explicitly implement the intended generated-token policy. Full pipeline needs generated role responses and a judge; it has not been run. Released vectors are for 27B/32B/70B models and **cannot be reused as vectors for a 1.7B model**. README declares MIT, no standalone license file.

**CAA.** Original Llama2 access and GPT4 judge credentials are needed for full upstream replication. Old pins include Torch2.0.1, Transformers4.33.2, numpy1.24.4 and datasets2.14.5; requirements contain contradictory pytest pins (7.4.2 and 8.2.1). Do not bulk-install on this Python3.12 environment. Core extraction/steering logic is reusable with a compatible current model stack. MIT.

**Fast-DetectGPT.** Requirements pin Transformers4.28.1/datasets2.12.0; README's Python3.8/Torch1.10 environment is historical, while current code includes newer model choices. Adapt the analytic criterion, validate tokenizer alignment for reference/scoring models, and use raw criterion or locally calibrated score. The demo's hardcoded Gaussian “probabilities” are not universal probabilities and can be nonmonotonic in the criterion. Do not use those constants as independent calibrated outcomes. MIT.

**RAID.** Package pins numpy1.26.4/pandas2.2.2/sklearn1.3.2; full detector environment is much larger. `run_detection`, `run_evaluation` and subgroup TPR at calibrated FPR are reusable. Default loader includes attacks; explicitly use the local `datasets/raid/train_none.csv`. Recalibrate on separate human negatives, never on final steered outputs. MIT.

## Validation performed and limits

Read all six READMEs and their dependency files, checked listed entry points and inspected critical extraction/steering/scoring functions. Parsed all **198 checked-out Python files** with Python3.12 AST: no syntax errors; older CAA/RAID regex/LaTeX strings emitted escape-sequence warnings. This is **not a runtime or numerical correctness test**. No generator, detector inference, external judge, upstream benchmark, or model download was run. No experiments are claimed.

Available hardware query: one NVIDIA RTX A6000, 49,140 MiB reported total memory. Resource environment contains only download/PDF/data-inspection dependencies, not Torch/Transformers. Experiment runner should install one compatible model stack into the project `.venv` with `uv add`, pin model revisions, and first verify: zero steering reproduces baseline; cached/uncached token policies agree; padding is masked; score sign and PCA inverse map are correct. Follow `../planning.md` rather than launching all upstream scripts.
