# Downloaded datasets

All four requested datasets are locally available. Downloaded raw files are excluded from Git; small **truncated excerpts** are in `samples/`. Exact source revisions, byte counts and SHA256 checksums are in `../artifacts/resource_finder/dataset_downloads.json`. Full-file parsing and schema/count/missingness checks are in `../artifacts/resource_finder/dataset_validation.json`.

## Reproduce downloads and validation

Run from the repository root (contains `.git`):

```bash
pwd
uv sync
source .venv/bin/activate
python scripts/download_datasets.py
python scripts/validate_datasets.py
```

The downloader pins the three Hugging Face repositories to recorded revisions, checks expected byte lengths, downloads to `.part`, and renames only complete downloads. RAID uses the authors' documented clean-data mirror. Existing files are reused; compare their hashes to the download manifest when reproducing. Metadata snapshots required by the script are tracked under `artifacts/resource_finder/`.

## HC3 — complete English release

- Source: [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), revision `4d0ff18143b5a7e1b1e79beb540c04549d1e59d3`.
- Local: `hc3/all.jsonl`, 73,722,850 bytes; 24,322 question records, 58,546 human answers and 26,903 ChatGPT answers. Five domains: reddit_eli5 17,112; finance 3,933; medicine 1,248; open_qa 1,187; wiki_csai 842.
- Schema: question, human_answers (list), chatgpt_answers (list), index, source. No official split files in this download; make source-group splits. The specified repository is the English release; this is not the separate Chinese corpus.
- License: CC-BY-SA-4.0, **subject to stricter original-source terms**, as stated in the card.
- Validation: 455 empty ChatGPT answer lists; no empty human lists/questions. Only 23,492 unique whitespace-normalized questions and 2,522 unique index values. **Index alone is not a unique key.** Group duplicates by normalized question, retain source and original row ordinal. Do not fabricate missing AI answers. Multiple answers to a question must stay together.
- Purpose: same-question human/AI comparison, second-task transfer. Topic matched does not mean fact-equivalent.

Direct download:

```bash
pwd
curl -L --fail 'https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/4d0ff18143b5a7e1b1e79beb540c04549d1e59d3/all.jsonl' -o datasets/hc3/all.jsonl
```

Load:

```python
import json
with open('datasets/hc3/all.jsonl') as f:
    rows = [json.loads(line) for line in f]
# Project convention: individual human answer label=0; ChatGPT answer label=1.
```

## HAP-E — complete requested parallel corpus

- Source: [browndw/human-ai-parallel-corpus](https://huggingface.co/datasets/browndw/human-ai-parallel-corpus), revision `b514ff64988d9e322fd81c5d70d69a38e78491f5`.
- Local: `hape/*.parquet`, all eight files, 114,086,679 bytes total. Each file has 8,290 records; 66,320 chunks total: two human chunks and six LLM continuations.
- Schema: `doc_id`, `text`; no nulls or empty texts. All eight files share exactly 8,290 source IDs after stripping `@...`.
- Domains: acad 1,227; blog 1,526; fic 1,395; news 1,322; spok 1,721; tvm 1,099.
- Generators: GPT-4o, GPT-4o-mini, Llama3 8B/70B base and instruct. This is the requested original HAP-E, not HAP-E2 or the mini variant.
- **Use human chunk 2 as human comparator. Human chunk 1 is the prompt context**, not an independent human example. `doc_id.split('@')[0]` is the cross-file join/group key; prefix before `_` encodes domain. No official train/validation/test partitions; split by source ID before pairing.
- License card: MIT; retain attribution and investigate underlying text-source terms for redistribution. Original corpus filters refusals/incomplete outputs, so it has selection bias. Matched continuations may differ in facts.

Download all exact files with `scripts/download_datasets.py`; alternatively:

```python
from huggingface_hub import snapshot_download
snapshot_download('browndw/human-ai-parallel-corpus', repo_type='dataset',
    revision='b514ff64988d9e322fd81c5d70d69a38e78491f5',
    allow_patterns=['text_data/*.parquet'], local_dir='datasets/hape_hf')
```

The alternative preserves a `text_data/` subdirectory; the supplied downloader flattens basenames into `hape/`.

```python
import pyarrow.parquet as pq
human = pq.read_table('datasets/hape/hape-text_human-chunk-2.parquet').to_pylist()
context = pq.read_table('datasets/hape/hape-text_human-chunk-1.parquet').to_pylist()
# Join on doc_id.split('@')[0], never row order alone.
```

## MAGE — all five released CSVs

- Source: [yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE), revision `342663f0a2b775455c023f5d36a1341ff0ec5402`; Apache-2.0 card.
- Local: `mage/train.csv` 319,071 rows; `valid.csv` 56,792; `test.csv` 56,819; `test_ood_set_gpt.csv` 1,562; `test_ood_set_gpt_para.csv` 2,362. Total 436,606 rows, 553,566,692 bytes.
- Schema: text, label, src. **Raw label 1 = human; 0 = machine.** Verified against all `src` human suffixes. Invert for project convention AI=1. CSVs have UTF-8 BOM; load with `utf-8-sig`.
- All CSVs fully parsed. `src` encodes source corpus/generator/prompt variant; raw CSVs lack reliable original-document IDs. Group reconstruction/near-duplicate auditing is needed before building new splits.
- Found 52 whitespace-normalized exact text overlaps between validation and test. OOD GPT and paraphrase files share 762 human texts; these are not independent tests. No normalized exact overlap found between main train and valid/test, or between downloaded MAGE and HC3; that does not rule out common sources or near duplicates.
- Use main training for detector fitting only with group precautions; keep OOD evaluations frozen. Do not treat the 762 reused human controls as separate independent observations.

Exact files are downloaded by the supplied script. Single-file example:

```bash
pwd
curl -L --fail 'https://huggingface.co/datasets/yaful/MAGE/resolve/342663f0a2b775455c023f5d36a1341ff0ec5402/test.csv' -o datasets/mage/test.csv
```

```python
import csv
with open('datasets/mage/test.csv', encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
ai_labels = [1-int(r['label']) for r in rows]
```

## RAID — complete clean training partition (documented subset)

- Requested source: [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid). HF metadata revision `865cac74188466cb0c3b7574a10204007b57a459`; downloaded equivalent clean partition from the [authors' documented mirror](https://github.com/liamdugan/raid#download-raid).
- Local: `raid/train_none.csv`, 801,662,741 bytes; 467,985 rows = 13,371 human + 454,614 generated texts, eight English domains, eleven generators plus human; 13,371 source groups. All IDs unique, all generations nonempty, all `attack=none`.
- This is **not full RAID**: the 11.8GB attacked train, extra languages/code, and unlabeled test partition were not downloaded. Initial causal/transfer tests need clean examples; attacked evaluation can be added from the specified source within direction 3.
- Fields: id, adv_source_id, source_id, model, decoding, repetition_penalty, attack, domain, title, prompt, generation.
- Human rows have blank prompt/decoding/repetition_penalty (expected, not corruption). Label AI=0 if model is human, else AI=1. Preserve source_id across all generators/decoding/attacks when splitting; source group is the inference unit. Sampling by raw row massively overweights AI texts; balance labels/domain and do not duplicate human controls as independent samples.
- License card/repository: MIT; original text sources may carry their own terms.

Reproduce the downloaded subset:

```bash
pwd
curl -L --fail 'https://dataset.raid-bench.xyz/train_none.csv' -o datasets/raid/train_none.csv
```

For complete attacks (11.8GB), when needed:

```python
from huggingface_hub import hf_hub_download
hf_hub_download('liamdugan/raid', 'train.csv', repo_type='dataset',
    revision='865cac74188466cb0c3b7574a10204007b57a459', local_dir='datasets/raid_full')
```

```python
import csv
with open('datasets/raid/train_none.csv', encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        label = int(row['model'] != 'human')
        group = row['source_id']
        # Process in batches; do not require loading 802MB CSV into memory.
```

## Experiment requirements

Target English benign writing with known provenance, prompt/topic matching, original-source grouping, generator/domain labels, comparable lengths, and held-out human negatives. HAP-E is primary; HC3 gives task shift; MAGE/RAID give detector and source-model stress tests. Keep the three-direction budget in `../planning.md`. All preprocessing, label inversion, truncation and exclusions must be logged. Dataset membership is evidence of provenance in that corpus, not an infallible detector target for new text.

## Auxiliary assistant-persona prompt pack (not a fifth benchmark)

`assistant_axis_prompts/` contains all 519 JSON/JSONL files (2,653,728 bytes) extracted from the pinned `safety-research/assistant-axis` repository's `data/` tree. It includes role/trait instructions, lists and extraction questions, for re-estimating persona and nuisance vectors in the chosen model. Files were parsed successfully; SHA256 manifest: `../artifacts/resource_finder/assistant_prompt_manifest.json`. These are prompts, not human/AI detector training labels or pretrained activation vectors. README declares MIT; the clone has no standalone LICENSE file.

Reproduce after restoring the code clone:

```bash
pwd
.venv/bin/python scripts/clone_resources.py
.venv/bin/python scripts/extract_assistant_prompts.py
```

Point the upstream pipeline's `--roles_dir` to the **absolute** `datasets/assistant_axis_prompts/roles/instructions` directory and `--questions_file` to `datasets/assistant_axis_prompts/extraction_questions.jsonl`. Use benign roles and content-controlled questions for the main style experiment. The full 275-role pipeline and its external judge have not been run; a reduced matched control must be described as an adaptation.
