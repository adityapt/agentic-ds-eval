# Paper Outline Plan — `agentic-ds-eval`

**Slug:** `agentic-ds-eval`
**Canonical output:** `papers/agentic-ds-eval.md`
**Spec source (authoritative, locked):** `docs/specs/2026-05-09-agentic-ds-eval-design.md`
**Date planned:** 2026-05-09

---

## Proposed Title

*Agentic AI in Product and Marketing Data Science: A Multi-Agent, Multi-Base-Model Empirical Evaluation Across Six Tasks*

---

## Section Map

| # | Section | Purpose | ~Words |
|---|---------|---------|--------|
| — | Abstract | 250-word summary: gap, method, framework, findings (tentative), contributions | 250 |
| 1 | Introduction | Contextualise agentic DS; state the gap (no peer-reviewed evaluation on product/marketing DS tasks); state 4 RQs; state 4 contribution claims | 600 |
| 2 | Related Work | MLE-bench, AIRA-MCTS, AutoKaggle, DSBench; closest comparators (Luo et al. 2025, vendor white-papers); prior autonomy taxonomies | 700 |
| 3 | Framework: Lifecycle × Autonomy | 7-stage DS lifecycle table; 5-level autonomy ladder; pre-registration plan (OSF) | 500 |
| 4 | Experimental Design | Task suite (6 tasks, datasets, baselines); agent matrix (AIDE, AutoGen, Aider); base-model matrix (Sonnet, DeepSeek-V3, Qwen2.5-Coder); 3×3×6×5 = 270 runs; statistical protocol; tooling disclosure (AutoResearchClaw) | 900 |
| 5 | Results | Per-task result tables (placeholder shell with column headers); agent-marginal and model-marginal aggregations; lifecycle-stage success rates; autonomy-level assignments; H1/H2 test outcomes | 1000 |
| 6 | Discussion | RQ1–RQ4 answered in turn; lifecycle-stage value-add and degradation; gap between vendor claims and observed autonomy; surprising findings | 700 |
| 7 | Limitations | Six items from spec §12 verbatim + framing text | 400 |
| 8 | Conclusion | Summary of contributions; future work (RCT, proprietary agents, additional tasks) | 300 |
| — | Author Note | AutoResearchClaw disclosure (spec §6) | 80 |
| — | Acknowledgements | Placeholder | 50 |
| — | Sources | Direct URLs for all primary references | — |

**Total estimated body:** ~5,500 words (well within Data Science Journal 8,000-word slot).

---

## Key Claims to Make (and Their Status)

| Claim | Strength | Source | Tentative? |
|-------|----------|--------|------------|
| No peer-reviewed study evaluates autonomous LLM agents end-to-end on product/marketing DS tasks | Strong — from spec §1 literature scan | Spec §1; to be backed by related-work search | Yes — label "as of May 2026 literature scan" |
| MLE-bench o1-preview+AIDE achieves ~17% Kaggle medal rate (pass@1), ~34% (pass@8) | Specific numeric — needs citation | Spec §1 cites MLE-bench paper | Cite MLE-bench paper; flag as "reported in MLE-bench" |
| AIRA-MCTS achieves 47.7% medal rate on MLE-bench Lite | Specific numeric | Spec §1 cites AIRA-MCTS | Cite; flag as reported |
| H1: base-model variance > agent-scaffold variance | Pre-registered hypothesis — not yet tested | Spec §5, H1 | Mark explicitly as pre-registered, untested |
| H2: L4 autonomy on routine tasks ≥ 60% of cells; ≤ L3 on attribution tasks ≥ 70% | Pre-registered hypothesis — not yet tested | Spec §5, H2 | Mark explicitly as pre-registered, untested |
| 270 total runs (3 agents × 3 models × 6 tasks × 5 seeds) | Arithmetic fact | Spec §4.5: 6×3×3=54 cells × 5 seeds = 270 | Verify arithmetic: 6×3×3×5 = 270 ✓ |
| Cost estimate $300–$600 | Author's estimate | Spec §8 | Mark as estimated, not incurred |
| Lifecycle × autonomy 7×5 matrix | Design artifact | Spec §4.1 | Cite as pre-registered artifact |

---

## Figures and Tables to Include

| ID | Type | Content | Tool |
|----|------|---------|------|
| Fig 1 | Mermaid flowchart | 7-stage DS lifecycle pipeline | Mermaid |
| Fig 2 | Mermaid diagram | Agent × base-model × task 3D matrix overview | Mermaid |
| Fig 3 | Placeholder bar chart | Per-task metric vs baseline (shell with dummy data labelled "illustrative") | pi-charts / Vega-Lite |
| Fig 4 | Placeholder heatmap | Agent × model performance matrix per task (shell) | pi-charts / Vega-Lite |
| Table 1 | Markdown table | 7-stage lifecycle × 5-level autonomy matrix | Markdown |
| Table 2 | Markdown table | Task suite: dataset, metric, baseline type | Markdown |
| Table 3 | Markdown table | Agent matrix (AIDE, AutoGen, Aider) with style and version | Markdown |
| Table 4 | Markdown table | Base-model matrix with tier/family/type | Markdown |
| Table 5 | Markdown table | Per-cell results shell (to be populated post-experiment) | Markdown |

---

## Related Work Papers to Cite

| Paper | Why needed |
|-------|-----------|
| MLE-bench (Chan et al., 2024) | Establishes ~17%/34% medal rate for AIDE; defines the current benchmark ceiling |
| AIRA-MCTS (2025) | 47.7% medal rate on MLE-bench Lite |
| AutoKaggle | Agentic DS pipeline on Kaggle competitions |
| DSBench | General DS benchmark for agents |
| Luo et al. 2025 (insurance prediction) | Closest vertical comparator; excluded due to single-domain scope |
| Makridakis et al. 2022 (M5, Int. J. Forecasting) | M5 baseline scores |
| Robyn documentation (Meta, 2023) | MMM baseline |
| PyMC-Marketing documentation | MMM baseline |
| LightweightMMM (Google) | MMM baseline |
| KKBox WSDM 2018 churn description | Task 3 dataset provenance |
| Recruit Holdings Coupon Purchase Prediction (Kaggle 2015) | Task 2 dataset |
| H&M Personalized Fashion (Kaggle 2022) | Task 4 dataset |
| RetailRocket events (Kaggle) | Task 5 dataset |

---

## Verification Log (Pre-Writing Checks)

| Check | Result |
|-------|--------|
| Arithmetic: 6 tasks × 3 agents × 3 models × 5 seeds | 6×3=18; 18×3=54 cells; 54×5=270 runs ✓ |
| Arithmetic: agent-marginal = 3 model average per agent per task | ✓ by definition |
| Stretch headline cells: "5–10" cells, ~$30–50 cost | Consistent with spec §8 ✓ |
| Working title matches spec §1 verbatim | ✓ — "Agentic AI in Product and Marketing Data Science: A Multi-Agent, Multi-Base-Model Empirical Evaluation Across Six Tasks" |
| Pre-registered hypotheses count: exactly 2 (H3 dropped per §14) | ✓ |
| Task count: exactly 6 (no A/B-test stretch task per §14) | ✓ |
| Models: exactly 3 (Option β per §14) | ✓ |
| Agents: exactly 3 | ✓ |
| Proprietary integrated-stack agents excluded and documented | ✓ per §4.3 and §12 |
| AutoResearchClaw not evaluated as an agent | ✓ per §6 |

---

## Constraints on Draft

1. **Do NOT generate experiment results** — results section uses placeholder shells labelled as pending.
2. **Do NOT invent new hypotheses** — only H1 and H2 from spec §5.
3. **Do NOT add tasks, agents, or models** beyond the locked spec.
4. **Mark all numerics** from spec (MLE-bench rates, cost estimates) with explicit sourcing.
5. **Every section** should be traceable to a spec section.
6. Figures 3 and 4 use illustrative/placeholder data labelled clearly as such.
7. All claims about the literature gap are scoped to "as of May 2026 literature scan."
