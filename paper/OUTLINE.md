# Paper outline and evidence map

Title: A Human–AI Readout Direction Changes Perceived Writing Style but Fails Preservation Tests

1. Abstract (150–250 words): operational question; paired Qwen models and controlled interventions; judge −20.039 with coherence −0.591; encoder disagreement; narrow conclusion. Evidence: REPORT §4 and paired_comparisons.csv.
2. Introduction: readable provenance versus generation control; why evaluation must include content; gap relative to probes and prior SAE steering; method overview; three contributions and organization. Cite Quaremba, Vishnyakov, Kuznetsov.
3. Related work: representation detection; causal steering/persona; matched corpora and detector shift. Cite all ten papers in local literature review without borrowing example content.
4. Methodology: estimand and preservation gates; source-level splits; vector equations, intervention positions, matched norms; 17 conditions/seeds; external ratings and paired inference. Evidence: config, corpus, judge rubric, execution report.
5. Results: primary five-outcome table; signed dose figure; complete control table and nuisance geometry; transfer table; calibration and reliability; robustness. Evidence: paired comparisons, random-average contrast, readout tables, validity and sensitivity artifacts.
6. Discussion: partial causality versus stronger failed conjunction; incomplete nuisance controls; quality and measurement limitations; broader implications and misuse caution grounded in detector disagreement.
7. Conclusion: narrow finding and decisive follow-up design.
8. Appendix: pinned models, runtime, full condition CIs, rubric, inference details, errors, deviations, artifact map and cost.

Visuals: native LaTeX method schematic; supplied dose-response and controls plots; transfer table plus supplied readout plot in appendix. Tables use booktabs; bold denotes numerically higher AUROC only, not significance or an overall winning intervention.

Resource review: skill and lab guide read; paper_examples contains only README (no examples available); all three command templates inspected. Author fixed by user to Ari Holtzman and NeuriCo.
