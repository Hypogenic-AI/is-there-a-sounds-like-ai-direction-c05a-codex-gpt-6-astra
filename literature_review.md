# Is there a “sounds like AI” direction in the residual stream?

## Scope and conclusion

There is strong evidence that human/AI text provenance is linearly readable from frozen LLM representations, and prior work causally steers individual stylistic features. The reviewed evidence does **not establish a single, distinct, content-preserving AI-style generation direction**. The useful contribution is therefore a controlled intervention with independent detectors and explicit nuisance tests, rather than another separability result.

Scope: 2023–2026 work on representation-based detection, causal residual steering, persona geometry, and matched human/AI corpora. Include direct methodological evidence and reusable resources; exclude unrelated watermarking, general-purpose fine-tuning, and detector-evasion optimization. This is a focused resource review, not an exhaustive systematic review. All four specified references were read through all 43 three-page-or-shorter chunks (127 pages); six supplemental papers were screened by abstract/first chunk and selected methods/results. Detailed per-chunk evidence is in [reading_log.md](artifacts/resource_finder/reading_log.md); bibliographic details and PDFs in [papers/README.md](papers/README.md).

## Search and screening

Search date: 2026-10-04. Keywords: human/AI residual representations; machine-generated text linear probes; activation steering; contrastive activation addition; assistant persona axis; SAE style features; matched continuations; domain/generator transfer; independent detection; style/content disentanglement.

|Search|Source|Outcome|
|---|---|---|
|Human/AI text residual probing and steering, diligent|Paper-finder|Initial missing HTTP dependency repaired; subsequent run terminated without results.|
|AI text detection activation steering, fast retry|Paper-finder|Success with zero papers; [log](artifacts/resource_finder/paper_finder_retry.log).|
|Four specified URLs and their method/code references|arXiv, Anthropic|All resolved/downloaded; four included for full reading.|
|CAA, RAID/MAGE, HC3, HAP-E, Fast-DetectGPT|arXiv, ACL Anthology, official HF/GitHub|Six supporting PDFs included; official implementations/data inspected.|
|ActAdd, FGAA, broad detection surveys|Search screening|Background only; CAA and supplied SAE paper cover required intervention baselines. No fourth direction added.|

## Key papers

### 1. Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text

**Quaremba, Yan, Black, Vrandečić and Simperl, 2026**, [arXiv:2608.24780](https://arxiv.org/abs/2608.24780), v1; [code](https://github.com/gerritq/mgt_probes). Last-token residuals, 100-component PCA per layer, and logistic probes form layer-averaged (LLP) or concatenated-layer (CLP) detectors. Detection uses DetectRL, MultiSocial, RAID and TSM, 16 settings with 1,500 balanced training and 500 test examples each; M4GT supports geometry analyses. Sixteen baselines include RoBERTa, RepreGuard, TextFluoroscopy, Fast-DetectGPT and Binoculars.

Reported mean OOD improvements range 0.39–11.37 AUROC points. Probe alignment and correlations with editing strength support shared readout structure, not generation causality. The low-label PCA experiment observes the full training set, although a no-PCA appendix supports similar sample efficiency. Authors explicitly leave universality/confounding/causality unresolved. Implementation standardizes before PCA: inverse-map the classifier normal into residual coordinates before intervention.

### 2. SV-Detect: AI-generated Text Detection with Steering Vectors

**Vishnyakov and Gaintseva, 2026**, [arXiv:2606.07313](https://arxiv.org/abs/2606.07313), v2; [code](https://github.com/Atmyre/sv-detect). Frozen GPT-Neo-2.7B mean-pooled states yield per-layer mean-difference, logistic, or paired-PCA directions; cosine features feed a standardized logistic head. DetectRL, MIRAGE and PADBen evaluate domain/generator/editing transfer; RAID/COLING extend evaluation. Metrics include AUROC, AUPR, F1, MCC and TPR at fixed FPR; baselines include score-based detectors, RoBERTa variants and RepreGuard.

Directions are explicitly used **only for probing/classification**. Logistic directions outperform mean/PCA in difficult transfer; regex typography/discourse/lexical baselines leave additional signal unexplained. This does not identify its causal meaning. RAID transfer remains much weaker for non-chat generators; a multi-direction variant reports 56.3% overall TPR at 5% FPR. Near-perfect RAID matched-training results are a different protocol. Code/documentation disagree on vector C and head intercept; [code notes](code/README.md) record exact discrepancies.

### 3. Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders

**Kuznetsov et al., 2025**, [arXiv:2503.03601](https://arxiv.org/abs/2503.03601), v1; [feature explorer](https://mgtsaevis.github.io/mgt-sae-visualization/). Gemma-2-2B/Gemma Scope residual SAE features are summed over tokens and classified with XGBoost; baselines are raw activation XGBoost, individual-feature thresholds and a shared-task multi-task detector. COLING-2025 supplies train/dev/devtest/test and RAID supplies attack analysis. Layer 16 generalizes well by macro-F1, but individual features vary across domains/generators.

Crucially, the paper already steers decoder directions, scaled by reference maximum activation, and interprets 50-generation sets with GPT-4o. Features affect complexity, assertion, verbosity, repetition and formality. Appendix examples show factual corruption at strong doses. The judge interprets features rather than independently measuring detector score and preserved facts. One model/SAE, summed-feature length dependence, source artifacts and all-token intervention limit causal generalization. This is the closest prior intervention work and prevents claiming first AI-style feature steering.

### 4. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

**Lu, Gallagher, Michala, Fish and Lindsey, 2026**, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387), v1; [specified article](https://www.anthropic.com/research/assistant-axis), [code](https://github.com/safety-research/assistant-axis). Gemma2-27B, Qwen3-32B and Llama3.3-70B response activations span 275 roles and 240 traits. The assistant-minus-role mean aligns with role-space PC1. Addition changes role adoption; projection capping reduces persona-jailbreak harmful responses by roughly 60% in selected configurations while preserving measured IFEval/MMLU-Pro/GSM8K/EQ-Bench performance.

These are persona/safety outcomes, not AI-writing detection. Synthetic conversations, qualitative persona labels, large-model scope and implementation differences constrain conclusions. Assistant projection is therefore a necessary nuisance control, estimated in the same model/layer as AI-style vectors. Do not transfer published vectors directly to another model, or equate capping, ablation and signed addition.

### 5. Steering Llama 2 via Contrastive Activation Addition

**Rimsky et al., ACL 2024**, [published paper](https://aclanthology.org/2024.acl-long.828/), local `2312.06681.pdf`; [code](https://github.com/nrimsky/CAA). Paired positive/negative answer-letter residual differences are averaged and added with signed coefficients after the prompt. Llama2-7B/13B evaluations cover seven behaviors using multiple-choice and open generation, GPT-4 judging, and capability controls; system prompting and fine-tuning are comparators. It supplies the causal intervention recipe, not evidence about provenance style. Prompt-format dependence and collateral changes motivate random-vector and quality controls.

### 6. Do LLMs write like humans? Variation in grammatical and rhetorical styles

**Reinhart et al., 2024/2025**, [arXiv:2410.16107](https://arxiv.org/abs/2410.16107), v2. HAP-E and CAP compare human continuations with six model continuations from the same context. Biber's 66 linguistic features support random forest/logistic classifiers and paired Wilcoxon tests with multiplicity correction. Seven-source HAP-E classification reaches 66% versus 14% chance; this is not binary detector accuracy. Instruction models show stronger stylistic deviations than base models; cross-model/domain transfer is imperfect. This supports domain/formality/base-versus-chat controls. HAP-E has 8,290 common sources after refusal/incomplete-output filtering; context matching does not equate factual content. Public corpus and linguistic tooling are linked in the paper/card; no separate implementation cloned.

### 7–10. Benchmarks and independent scoring

|Paper|Method, data and evidence|Use and limitations|
|---|---|---|
|**Dugan et al., RAID, ACL 2024**, [2405.07940](https://arxiv.org/abs/2405.07940), [code](https://github.com/liamdugan/raid)|Original 6M+ generations, 11 generators, 8 domains, 4 decoding strategies, 11 attacks; compares 8 open and 4 closed detectors at low FPR.|Decoding/generator/attack shifts undermine detector robustness. Current corpus is larger; downloaded clean train is only a documented subset.|
|**Guo et al., HC3, 2023**, [2301.07597](https://arxiv.org/abs/2301.07597), [code/models](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)|Human/ChatGPT answers to shared questions; GLTR logistic, RoBERTa single-text and QA detectors; F1 across full/filtered/sentence variants.|Useful question matching; old single-generator and domain/format artifacts constrain inference. English release has 24,322 question records.|
|**Li et al., MAGE, ACL 2024**, [2305.13242](https://arxiv.org/abs/2305.13242), [code](https://github.com/yafuly/MAGE)|27 LLMs, 7 families, 10 source corpora, 8 testbeds; Longformer, GLTR/FastText, DetectGPT and human/ChatGPT judgments.|OOD human false positives can be severe; small target-domain adaptation changes the protocol. Do not conflate adapted with zero-shot results.|
|**Bao et al., Fast-DetectGPT, ICLR 2024**, [2310.05130](https://arxiv.org/abs/2310.05130), [code](https://github.com/baoguangsheng/fast-detect-gpt)|Conditional log-likelihood discrepancy with analytic variance; XSum/SQuAD/WritingPrompts plus domain/language shifts; likelihood/rank/entropy/DetectGPT baselines. Reports 0.9887 white-box and 0.9338 ChatGPT/GPT4 black-box mean AUROC in its setup.|Useful frozen external criterion, not an infallible authorship score. Reported 340x speedup is setup-specific. Recalibrate demo probabilities; ensure tokenizer/reference/scorer compatibility.|

## Synthesis: methods, baselines and measurement

The common pipeline is frozen activations → pooled representation → learned direction → detector score. Generation intervention adds a separate causal step and can fail even with excellent readout. A classifier normal, mean difference, PCA component and SAE decoder column solve different problems; none is automatically the optimal causal control vector. Multi-layer detection scores are not single-layer vectors.

Minimum baselines: unsteered model; signed mean-difference and logistic-normal interventions; label-shuffled and norm-matched random vectors; formal/informal and concise/verbose prompts; independently estimated assistant/formality/length/domain directions. SAE interpretations inform these controls without adding a fourth experiment. Same-detector improvements alone can reflect detector-specific exploitation.

Report readout AUROC/AUPR and TPR at validation-calibrated FPR. Primary generation outcomes are **paired external score changes jointly with factual retention and coherence**. Add instruction adherence, repetition, output lengths, truncation and detector disagreement. Blinded ratings and source-grounded QA complement semantic similarity; embedding similarity alone cannot establish preserved facts. Use source-cluster bootstrap confidence intervals, seed-paired comparisons, and frozen validation-selected layer/scale settings. Show full score–quality curves rather than discarding incoherent failures.

## Recommendations and unresolved questions

Implement only the three ranked directions in [planning.md](planning.md): (1) signed causal dose response, (2) nuisance/persona separation, (3) held-out domain/generator and second-model transfer. HAP-E is primary, HC3 adds task shift, RAID/MAGE stress generalization. All requested data are local; [dataset notes](datasets/README.md) include label conventions and validated overlaps. An RTX A6000 is available; a small accessible model is a practical first pilot. Model weights and experiment dependencies are not installed.

Split by source before fitting PCA, scalers, directions, nuisance bases or detectors. HAP-E human chunk 1 is context only; HC3 indices are nonunique; MAGE labels are inverted relative to AI=1 and its validation/test files overlap; RAID variants share source IDs. Published provenance labels and frozen detectors operationalize style but do not define a universal essence of AI writing.

A distinct causal-style claim requires effects surviving independent scoring, factual/coherence constraints, nuisance interventions and transfer. A null or narrower result—readout-only, persona-mediated, detector-specific or quality-damaging—is scientifically useful. Orthogonalization alone is not proof of semantic independence, and removing a nuisance subspace can remove genuine style signal. No causal outcome has been measured in this resource phase.
