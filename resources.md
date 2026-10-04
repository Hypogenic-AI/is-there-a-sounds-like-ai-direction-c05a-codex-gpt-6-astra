# Resources catalog

Prepared 2026-10-04 for **“Is there a ‘sounds like AI’ direction in the residual stream?”** Resource gathering is complete; no experiment results are claimed. Start with [literature_review.md](literature_review.md), then [planning.md](planning.md), which fixes the implementation budget at **three directions**.

## Inventory

- **10 source PDFs**, plus reading chunks/manifests. All four specified references available and fully read.
- **4 requested benchmark datasets**: complete HC3 English, complete requested HAP-E, all MAGE CSVs, and RAID's complete clean training partition (**subset of full RAID**). Fifteen raw files, **1,543,038,962 bytes** total.
- **1 auxiliary prompt pack**: 519 assistant-axis JSON/JSONL files, 2,653,728 bytes.
- **6 official code repositories**, pinned shallow/sparse clones with code and docs.
- Fresh project `.venv`, `pyproject.toml`, `uv.lock`; download/data/PDF tools installed locally. No model weights or experimental ML stack installed.

## Papers

|Title / authors / year|Local PDF|Use|
|---|---|---|
|Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text — Quaremba et al., 2026|[2608.24780](papers/2608.24780.pdf)|Last-token linear readout; causal gap.|
|SV-Detect: AI-generated Text Detection with Steering Vectors — Vishnyakov & Gaintseva, 2026|[2606.07313](papers/2606.07313.pdf)|Layerwise direction detector, transfer and surface-style baseline.|
|Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders — Kuznetsov et al., 2025|[2503.03601](papers/2503.03601.pdf)|Existing causal style-feature steering and factual damage.|
|The Assistant Axis — Lu et al., 2026|[2601.10387](papers/2601.10387.pdf)|Assistant-persona confound/control; paper linked by specified Anthropic article.|
|Steering Llama 2 via Contrastive Activation Addition — Rimsky et al., ACL 2024|[2312.06681](papers/2312.06681.pdf)|Paired difference and signed residual intervention.|
|RAID — Dugan et al., ACL 2024|[2405.07940](papers/2405.07940.pdf)|Generator/domain/decoding robustness.|
|How Close is ChatGPT to Human Experts? — Guo et al., 2023|[2301.07597](papers/2301.07597.pdf)|HC3 corpus and detector baselines.|
|MAGE: Machine-generated Text Detection in the Wild — Li et al., ACL 2024|[2305.13242](papers/2305.13242.pdf)|Heterogeneous detection and OOD testbeds.|
|Do LLMs write like humans? — Reinhart et al., 2024/2025|[2410.16107](papers/2410.16107.pdf)|Matched continuations, grammar/rhetoric and instruction-tuning effects.|
|Fast-DetectGPT — Bao et al., ICLR 2024|[2310.05130](papers/2310.05130.pdf)|External conditional-curvature scoring.|

Full titles/authors/version details: [papers/README.md](papers/README.md). Source URLs, byte counts, page counts and SHA256: [paper_manifest.json](artifacts/resource_finder/paper_manifest.json). All 43 chunks of the four core papers were visually read; notes: [reading_log.md](artifacts/resource_finder/reading_log.md). Supplementary text extractions and rendered sheets are local ignored working artifacts, not substitutes for the PDFs.

Recreate papers from versioned URLs:

```bash
pwd
uv sync
.venv/bin/python scripts/download_papers.py
```

## Datasets

|Dataset / source|Local size and records|Task / location|Important constraint|
|---|---|---|---|
|[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)|73.7MB; 24,322 question records; 58,546 human and 26,903 AI answers|QA comparison; `datasets/hc3/all.jsonl`|455 missing AI lists, duplicate questions and nonunique indices; English only.|
|[HAP-E](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus)|114.1MB; 8,290 shared sources, 66,320 chunks|Primary context-matched continuations; `datasets/hape/`|Chunk1=context; chunk2=human comparator; join ID before `@`. Six generators.|
|[MAGE](https://huggingface.co/datasets/yaful/MAGE)|553.6MB; 436,606 rows across 5 files|Broad detector training/OOD evaluation; `datasets/mage/`|Raw human=1, AI=0. 52 valid/test text overlaps; OOD tests share 762 human controls.|
|[RAID](https://huggingface.co/datasets/liamdugan/raid)|801.7MB; 467,985 clean-train rows; 13,371 source groups|Domain/generator/decoding shift; `datasets/raid/train_none.csv`|Authors' official clean mirror; no attacks/extra/test downloaded. Group by source_id and balance labels.|
|[Assistant-axis prompts](https://github.com/safety-research/assistant-axis)|2.65MB; 519 parsed JSON/JSONL files|Auxiliary persona/nuisance extraction; `datasets/assistant_axis_prompts/`|No pretrained vectors, no detector labels; recompute on target model.|

Decimal MB. All raw data excluded by `datasets/.gitignore`; documentation and excerpts below 100KB/file are retained. [datasets/README.md](datasets/README.md) has exact download/load instructions, licenses, schemas, source revisions, counts and preprocessing warnings. [Validation report](artifacts/resource_finder/dataset_validation.json) checks entire files, not just a prefix. [Download manifest](artifacts/resource_finder/dataset_downloads.json) and [prompt manifest](artifacts/resource_finder/assistant_prompt_manifest.json) provide checksums.

```bash
pwd
.venv/bin/python scripts/download_datasets.py
.venv/bin/python scripts/validate_datasets.py
```

## Code repositories

|Repository|Local checkout|Reuse|
|---|---|---|
|[gerritq/mgt_probes](https://github.com/gerritq/mgt_probes)|`code/mgt_probes/`|LLP/CLP training and activation extraction.|
|[Atmyre/sv-detect](https://github.com/Atmyre/sv-detect)|`code/sv-detect/`|Direction variants, cosine features, logistic detector, regex confounds.|
|[safety-research/assistant-axis](https://github.com/safety-research/assistant-axis)|`code/assistant-axis/`|Addition/ablation/capping hooks and persona-axis construction.|
|[nrimsky/CAA](https://github.com/nrimsky/CAA)|`code/CAA/`|Paired difference and generation steering reference.|
|[baoguangsheng/fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt)|`code/fast-detect-gpt/`|Independent analytic curvature criterion.|
|[liamdugan/raid](https://github.com/liamdugan/raid)|`code/raid/`|Detection evaluation, subgroup metrics, decoding/attack references.|

No explicit code_references were supplied; these are official repositories discovered from the specified papers and selected baselines. See [code/README.md](code/README.md) for full commit IDs, dependencies, exact functions, implementation caveats and validation limits. [code_manifest.json](artifacts/resource_finder/code_manifest.json) records selected directories and syntax checks. Sparse checkouts deliberately omit large upstream results/data; six READMEs and critical functions were inspected. All 198 checked-out Python files parsed successfully; **no inference/runtime test was performed**.

```bash
pwd
.venv/bin/python scripts/clone_resources.py
.venv/bin/python scripts/extract_assistant_prompts.py
```

## Model and compute handoff

Available hardware: one NVIDIA RTX A6000, 49,140MiB total memory reported by `nvidia-smi`. Candidate metadata/revisions are recorded in [model_candidates.json](artifacts/resource_finder/model_candidates.json); **weights are not downloaded**.

- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B): accessible small pilot, Apache-2.0; use `enable_thinking=False` explicitly in chat formatting and a compatible Transformers version (card requires >=4.51). Fix decoding settings across all conditions.
- [Gemma-2-2B-it](https://huggingface.co/google/gemma-2-2b-it): possible second model; gated terms/access required. The SAE paper's base-model feature IDs are not automatically valid for this instruction checkpoint.
- [GPT-Neo-2.7B](https://huggingface.co/EleutherAI/gpt-neo-2.7B): ungated reference for SV-Detect/Fast-DetectGPT; keep scoring model distinct from the steered generator.

Install one compatible experimental stack with project-root `uv add`, not the mutually incompatible upstream requirements combined. Hook behavior, masks, final-norm indexing, dtype, KV-cache handling and exported PCA directions need numerical checks in the experiment phase.

## Gathering notes and remaining limits

Search used paper-finder first; repaired missing HTTP dependency, then used official arXiv/ACL/Anthropic and linked GitHub/HF sources after a zero-result retry. No relevant papers were returned by that successful finder query. Ten selected PDFs include every specified reference and six method/dataset supports. This is a focused review; novelty beyond this literature set remains uncertain.

The main implementation findings are SV-Detect README/YAML versus source C/intercept discrepancies, per-pair centering, non-guarded zero norms, and Fast-DetectGPT's non-universal demo calibration. Preserve them as explicit adaptation decisions. Old CAA/Fast-DetectGPT and newer SV-Detect dependency pins conflict; model execution is intentionally deferred to experiment setup.

RAID's large attacked corpus was handled using the authors' complete clean partition, not an arbitrary initial sample. All other requested benchmark release files are present. Normalized exact HC3/MAGE overlap was zero, but near-duplicates/common source documents remain to audit. HAP-E controls context, not exact propositions; independent content-constrained tasks are required for the strongest causal claim.

## Recommended experiments

1. **Causal dose response:** HAP-E/HC3-derived direction, signed interventions, zero/random/shuffled-label and style-prompt controls. Score held-out generations with frozen independent detectors and joint content/coherence constraints.
2. **Distinctness:** matched formality/verbosity/domain/persona vectors, residualized directions, nuisance-only interventions and crossed prompt conditions. Compare in the same model/layer coordinates.
3. **Transfer:** held-out domains/generators and a separately estimated second-model direction; use RAID/MAGE stress tests, retaining source groups and independent evaluation.

Use paired source-cluster confidence intervals; report negative results, failed quality gates and detector disagreement. Do not turn detection transfer, geometric orthogonality or lower detector score alone into a claim of universal causal AI-ness. [planning.md](planning.md) specifies the full controls, split logic, intervention scaling, stopping rules and pruned directions.

Final integrity audit: run `.venv/bin/python scripts/audit_resources.py`; results are recorded in [final_audit.json](artifacts/resource_finder/final_audit.json). This checks local resource integrity and documentation, not scientific outcomes.

## Research execution completed (2026-10-04)

The experiment used the pre-gathered HAP-E and HC3 data, CAA's paired-difference idea, and assistant-axis positional intervention patterns, implemented explicitly in `src/`. The frozen execution appendix in planning.md reduced the original broad handoff to Qwen2.5-1.5B base/instruct checkpoints, 64 factual writing records, 17 instruction conditions and three base conditions. All 1,472 task generations and 288 nuisance generations are real checkpoint outputs. RAID/MAGE and full SAE/persona experiments were not run; their downloaded resources remain available.

New artifacts: `results/models.json` pins both generator checkpoints and the independent RoBERTa detector; `results/live_catalog.json` and `judge_model.json` archive the live OpenRouter selection of Gemini-3.8-Flash. Full prompts, raw outputs, API usage, vectors and source-grouped splits are saved under results/. `uv.lock` pins the isolated runtime; `TORCH_DISABLE_NATIVE_JIT=1` avoids the absent C compiler. No system/neurico environment was used.

Results: judge-rated negative steering effect −20.04/100, coherence effect −0.591/5, no primary corroboration from the encoder; readout transfer to Llama base fails despite high GPT-4o/ChatGPT/instruction-text separability. Read [REPORT.md](REPORT.md) for exact confidence intervals, detector validity, nuisance/base comparisons and limitations. [README.md](README.md) supplies commands; `results/validation.json` records final numerical reproduction and artifact checks. The data-analysis skill guided paired inference, confidence intervals and multiple-comparison control; no skill authorized or triggered messages to outside recipients.
