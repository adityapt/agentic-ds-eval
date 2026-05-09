# Design Spec — Agentic AI in Product and Marketing Data Science

**Working title.** *Agentic AI in Product and Marketing Data Science: A Multi-Agent, Multi-Base-Model Empirical Evaluation Across Six Tasks.*

**Author.** Aditya Puttaparthi Tirumala — Independent Researcher.

**Date.** 2026-05-09.

**Status.** Draft for author review.

---

## 1. Motivation and Gap

Recent benchmarks of agentic AI on data-science tasks (MLE-bench, AIRA-MCTS, AutoKaggle, DSBench) have established that current frontier-model agents lag the human Kaggle median on general tabular ML competitions. Headline numbers: o1-preview + AIDE achieves a Kaggle medal in ~17% of competitions on MLE-bench (pass@1) rising to ~34% (pass@8); AIRA-MCTS pushed the medal rate to 47.7% on MLE-bench Lite. None of these benchmarks evaluate agents on tasks practitioners in **product or marketing data science** actually face — recommendation, marketing-mix attribution, conversion-funnel modeling, hierarchical retail forecasting, churn, coupon redemption.

A May 2026 literature scan confirms that **no peer-reviewed study** evaluates autonomous LLM agents end-to-end on a product- or marketing-DS task suite. Closest comparators (Luo et al. 2025 on insurance prediction; vendor white-papers from Sellforte, Microsoft CLV agent, Pecan) either operate in an unrelated vertical or lack rigorous head-to-head methodology. Independently, **no published study** uses a three-way design (agent × base model × task) of the kind needed to defend against single-product or single-vendor bias.

This work addresses that gap.

---

## 2. Research Questions

1. **RQ1.** How do agentic AI systems perform across the data-science lifecycle for product and marketing DS tasks, relative to high-effort human baselines (Kaggle leaderboard distributions, published top solutions, established statistical-tool defaults)?
2. **RQ2.** Are agent capabilities **uniform across base models** holding the agent constant, or does base-model choice dominate? (Analogously: holding base model constant, does agent-scaffolding choice dominate?)
3. **RQ3.** Where in the **DS lifecycle** (problem framing → data exploration → feature engineering → model selection → validation → interpretation → reporting) do agents add value, and where do they actively degrade work?
4. **RQ4.** How does observed agent performance map to a **5-level autonomy taxonomy** (L1 code-completion → L5 fully autonomous)? Quantify the gap between vendor autonomy claims and observed empirical autonomy.

---

## 3. Contribution Claims

1. A **lifecycle × autonomy framework** for evaluating DS agents (pre-registered before experimentation, on OSF). The 7×5 matrix is intended as a reusable artifact.
2. **First peer-reviewed multi-agent × multi-base-model empirical evaluation** of agentic AI on a six-task product+marketing DS suite.
3. **Quantitative measurement** of the gap between vendor autonomy claims and observed performance.
4. **Reproducible code, experimental harness, and result tables** released under a permissive license, on GitHub, with all datasets being public.

---

## 4. Methodology

### 4.1 Lifecycle × Autonomy Framework (pre-registered artifact)

**Lifecycle stages (7).** Each stage is operationalized with a concrete set of allowable agent operations:

| # | Stage | Operations the agent may perform | Output the agent must produce |
|---|---|---|---|
| 1 | Problem framing | Read task spec; identify target, eval metric, constraints | One-paragraph framing + chosen metric |
| 2 | Data exploration | Load data; profile distributions, missingness, leakage risks | EDA notes + identified data-quality issues |
| 3 | Feature engineering | Create derived features; encode categoricals; handle sparsity | Engineered feature set + brief justification |
| 4 | Model selection | Choose model family; justify selection | Model choice + 1-2 line rationale |
| 5 | Validation | Implement train/holdout protocol; cross-validation; report metric | Validation strategy + held-out scores |
| 6 | Interpretation | Generate feature importances, SHAP, marginal effects, sensitivity | Interpretation artifact appropriate to the task |
| 7 | Reporting | Summarize findings | Markdown report ≤ 800 words |

**Autonomy levels (5).** SAE-inspired ladder:

| Level | Description | Human role |
|---|---|---|
| L1 | Code-completion within human-authored pipeline | Human owns architecture; agent fills code blocks |
| L2 | Human-in-the-loop step suggestion | Agent proposes; human accepts/rejects each step |
| L3 | Stage-autonomous, human reviews stage outputs | Human gates between lifecycle stages |
| L4 | Cross-stage autonomous, human reviews final output only | Single human checkpoint, end of pipeline |
| L5 | Fully autonomous, no human checkpoint | Human reviews after deployment |

Pre-registered on OSF before experiments begin. Pre-registration timestamp + URL included in the published paper.

### 4.2 Task Suite (6 tasks)

#### Marketing DS (3)

**Task 1 — Marketing Mix Modeling (MMM).**
- Dataset: Robyn demo data (Meta open-source).
- Reference baselines (3, run by author before agent experiments): Robyn canonical run, PyMC-Marketing equivalent specification, LightweightMMM equivalent specification.
- Agent task: fit an MMM that decomposes total response into per-channel contributions; report channel-level ROI estimates.
- Evaluation: agent's per-channel attribution vs each of three baselines, measured by Pearson r per channel + total-attribution agreement (mean absolute deviation across channels). Reported as three separate comparisons; not averaged. Cross-baseline-agreement metric defended explicitly: high agent agreement with Robyn alone is *not* equivalent to MMM skill — only convergence across all three baselines indicates capable attribution recovery.

**Task 2 — Coupon redemption prediction.**
- Dataset: Recruit Holdings Coupon Purchase Prediction (Kaggle 2015).
- Baseline: Kaggle leaderboard distribution. Reported with explicit framing: "Kaggle leaderboard is a high-effort upper-bound benchmark, not median practitioner performance."
- Evaluation: AUC vs leaderboard percentile (top-1%, median, bottom-1% explicit reporting).

**Task 3 — Customer churn prediction.**
- Dataset: KKBox Music Streaming Churn (Kaggle 2017). *Selected because it does not appear in the author's prior synthetic-data benchmark suite, eliminating dataset-reuse novelty concerns.*
- Baseline: Kaggle leaderboard distribution, framed as upper-bound.
- Evaluation: AUC vs leaderboard percentile.

#### Product DS (3)

**Task 4 — Recommendation.**
- Dataset: H&M Personalized Fashion Recommendations (Kaggle 2022).
- Baseline: Kaggle leaderboard distribution, framed as upper-bound.
- Evaluation: MAP@12 vs leaderboard percentile.

**Task 5 — Conversion / funnel modeling.**
- Dataset: RetailRocket events (Kaggle).
- Baseline: published RecSys-/CIKM-paper baselines on the same dataset; cross-paper agreement reported.
- Evaluation: AUC for transaction prediction; Recall@K for next-event prediction.

**Task 6 — Hierarchical retail forecasting.**
- Dataset: M5 Walmart (Makridakis et al. 2020, *International Journal of Forecasting* 2022 special issue).
- Baseline: Published M5 top-solution WRMSSE scores (Makridakis et al. 2022 records the top-50 with code links).
- Evaluation: WRMSSE vs published top-solution distribution.

### 4.3 Agent Matrix (3 agents)

Selected to span three philosophically distinct agent paradigms:

| Agent | Style | Version |
|---|---|---|
| **AIDE** (used in MLE-bench, Weco AI) | Research-iterative; tree search over solutions | Latest stable as of experiment start |
| **AutoGen** (Microsoft) | Multi-agent collaboration; planner + coder + critic | Latest stable |
| **Aider** | Lightweight code-edit / pair-programming style | Latest stable |

Version pinning recorded in the experimental log. Each agent run with its native scaffolding (no harmonization). Agent scaffolding choice is itself a study variable.

**Excluded by design:** proprietary integrated-stack agents (Claude Code, Devin, Cursor Agent, GitHub Copilot Workspace, ChatGPT Advanced Data Analysis, Gemini Data Science Agent) are excluded from the matrix because they are locked to a single base-model family, which would break the bias-defense argument central to RQ2. Their exclusion is documented in §12 Limitations.

### 4.4 Base Model Matrix (3 models)

Selected to span closed/open paradigms while emphasizing *code-strong* model variants — appropriate for an evaluation centered on code-generating DS agents:

| Model | Tier | Family | Type |
|---|---|---|---|
| **Claude 4.6 Sonnet** | Mid-frontier closed | Anthropic | General frontier with strong code training |
| **DeepSeek-V3** | Frontier open (671B MoE) | DeepSeek | General frontier with leading open-weights code performance |
| **Qwen2.5-Coder-32B-Instruct** | Code-specialized open | Alibaba | Explicitly code-tuned, smaller form factor |

**Justification for not including GPT-5 or Gemini 2.5 Pro:** The matrix prioritizes the *open-vs-closed* axis and *general-vs-code-specialized* axis over OpenAI/Google representation. Two open code-strong models (DeepSeek-V3, Qwen2.5-Coder) plus one closed general (Sonnet) gives reproducibility via open weights and minimizes per-token cost without sacrificing the bias-defense argument. Reviewer concern about OpenAI omission to be addressed in §11 Risks and §12 Limitations.

**Stretch headline cells (optional):** A single round of **Claude 4.7 Opus** on the MMM task (Task 1) reported separately as a frontier-tier sanity-check anchor. Estimated incremental cost ~$30–50.

### 4.5 Cell Count and Statistical Protocol

**Total cells:** 6 tasks × 3 agents × 3 base models = **54 unique configurations**.
**Seeds per cell:** 5 (seeds 42, 123, 7, 2024, 999, matching the author's prior synthetic-data-paper protocol).
**Total runs:** **270** (plus optional 5–10 stretch headline cells for ~280 total).

Reporting protocol per cell: mean ± 95% CI via t-distribution (df = n_seeds - 1). Significance testing uses paired t-tests within task (where matching is possible) or independent-samples otherwise.

### 4.6 Evaluation Cells: Per-Task Aggregation

For each task, results are reported as:
- **Cell-level** (agent × base-model): mean ± 95% CI of task metric across 5 seeds.
- **Agent-marginal**: mean across base models, holding agent constant.
- **Model-marginal**: mean across agents, holding model constant.
- **Lifecycle-stage-conditional**: per-stage success rate (binary success/failure as defined per stage in §4.1) aggregated across cells.
- **Autonomy-level achieved**: post-hoc classification of each cell's behavior to the 5-level taxonomy (§4.1) by author review.

---

## 5. Pre-Registration Plan

Before any experiments are run:
1. Lifecycle × autonomy framework (§4.1) registered on OSF with timestamp.
2. Task suite, agent matrix, base-model matrix, seed list, evaluation protocols registered.
3. Two pre-registered hypotheses:

   **H1 — Model-effect dominates agent-effect.**
   Variance in task metrics due to base-model choice will exceed variance due to agent-scaffolding choice, measured via two-way ANOVA on the 3×3 matrix per task.
   *Rejection criterion:* If F-statistic for agent-effect ≥ F-statistic for model-effect on ≥ 4 of 6 tasks, H1 is rejected.

   **H2 — Autonomy ceiling varies by task type.**
   Agents achieve L4 autonomy (cross-stage autonomous, single human checkpoint) on routine prediction tasks (Coupon, Churn, Recommendation) in ≥ 60% of cells; achieve only L3 or below on attribution-heavy tasks (MMM, Forecasting) in ≥ 70% of cells.
   *Rejection criterion:* If observed L4-rate falls below 60% on routine tasks OR exceeds 30% on attribution-heavy tasks, H2 is rejected.

OSF DOI included in paper.

---

## 6. Tooling Disclosure (AutoResearchClaw)

| Use | Tool | Disclosed in paper as |
|---|---|---|
| Experimental harness orchestration | AutoResearchClaw v0.3.1 | "Experiments orchestrated via AutoResearchClaw v0.3.1" — methodology section, named explicitly |
| Initial paper structure / outline / related-work skeleton | AutoResearchClaw v0.3.1 | "Initial structure scaffolded by AutoResearchClaw v0.3.1; results, analysis, interpretation, and discussion are author-written" — explicit author note in front matter |
| Results tables, plots, statistical analysis, results-section prose, discussion, limitations, conclusion | Author | — |

**Boundary commitment.** AutoResearchClaw is *not* one of the four agents under evaluation, eliminating any recursive-self-evaluation concern.

---

## 7. Reproducibility Plan

- Full source code on GitHub (this repository).
- Each task implemented as a standalone module; entry point `experiments/run_<task>.py`.
- Each agent implemented behind a uniform interface; entry point `agents/<agent>_runner.py`.
- All datasets either downloaded from public source by the script or fetched once and cached.
- Seeds, hyperparameters, library versions, and base-model versions logged in `runs/<run_id>/manifest.json` for every run.
- Resumable across runs (CSV checkpoint pattern, identical to the author's prior synthetic-data-paper scripts).
- Cost-tracking script that aggregates per-cell API spend.

---

## 8. Cost Estimate

**Realistic out-of-pocket range (no research credits expected): $300–$600.**

Breakdown (Option β matrix: Sonnet + DeepSeek-V3 + Qwen2.5-Coder):
- 90 runs × Claude 4.6 Sonnet × ~$3 each = ~$270.
- 90 runs × DeepSeek-V3 (open, GPU-rented) = ~$50–80 GPU rental.
- 90 runs × Qwen2.5-Coder-32B (open, GPU-rented, shared infra with DeepSeek) = ~$30–50 incremental.
- Stretch headline cells (Claude 4.7 Opus on MMM only, optional) = $30–50.
- Storage and incidental = $50.

Cost-control levers in place:
- Two open-source models on shared rented GPU (~$1–2/hr; H100 or A100).
- Mid-tier closed model (Sonnet) only — frontier (Opus) reserved for stretch headline cells.
- Aggressive prompt caching on Anthropic API (~80% discount on cached prefix).
- Resumable across interruptions; failed runs do not cost twice.
- Rate-limited execution.

---

## 9. Timeline

| Phase | Duration | Output |
|---|---|---|
| Pre-registration on OSF | 1 week | DOI |
| Repository scaffolding (this doc + planning) | 1 week | Implementation plan (next) |
| Reference-baseline runs (3 MMM tools, KKBox/Coupon/H&M/RetailRocket/M5 baselines) | 2 weeks | Result tables for human-baseline column |
| Agent matrix execution | 4–6 weeks | 480 run logs + cell-level result tables |
| Analysis and writing (author) | 4 weeks | Submission draft |
| **Total** | **~12–14 weeks** | Submission to first venue |

---

## 10. Target Venues

In order of estimated acceptance probability (from author's prior journal-fit research):

1. **Data Science Journal** (CODATA) — strongest fit, reproducibility-aligned editorial culture, £770 APC, ~8000-word slot
2. **International Journal of Data Science and Analytics** (Springer) — IF 2.8, ESCI
3. **Information** (MDPI) — IF 2.9, low APC, fast review
4. **Journal of Marketing Analytics** (Palgrave) — IF 3.5, marketing-DS audience for the marketing half of results
5. *(Stretch)* **Data Mining and Knowledge Discovery** (Springer) — IF 4.3, requires repositioning around the lifecycle×autonomy framework as the contribution

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Robyn baseline alone is not a "human DS" baseline.** | Three reference baselines (Robyn, PyMC-Marketing, LightweightMMM); cross-baseline-agreement explicitly reported. Reviewer cannot reduce the comparison to "agents reproduce Robyn." |
| **Kaggle leaderboards over-represent top human DS performance.** | Explicit framing in every task: "Kaggle distribution is high-effort upper-bound, not median practitioner." Reported as percentile, not as "vs human." |
| **Lifecycle stages and autonomy levels are debatable taxonomies.** | Pre-registered on OSF before experiments. Reviewer arguments shift from "you cherry-picked" to "we propose an alternative taxonomy" — easier to defend. |
| **AutoResearchClaw use is unusual; may trigger AI-content concerns.** | Use limited to (a) experimental harness, (b) paper-skeleton outline only. Results, analysis, interpretation, and discussion are author-written. Disclosed explicitly in front matter. AutoResearchClaw is not one of the four agents evaluated. |
| **Field velocity — frontier model release within 6 months may obsolete the matrix.** | Submit within 4 months of experiment completion. Frame the matrix as a snapshot at a stated date; future model versions are documented as future work. |
| **External validity from 6 tasks.** | Task selection spans 3 distinct DS skill axes (causal/attribution, ranking, time-series). Results explicitly conditioned on task type, not generalized to "all DS." Limitations section names specific generalization caveats. |
| **Cost overrun.** | Resumable harness; cost-tracking script; ability to drop the stretch headline cells first, then trim seeds from 5 to 3 (saves ~40%) before reducing the matrix dimensions. |
| **No OpenAI/GPT-5 in the matrix.** | Documented in §4.4 with rationale (open-vs-closed prioritized over provider count). Reviewers may ask; response: matrix prioritizes reproducibility (two open-weights models) and budget feasibility for an independent researcher; the GPT-5 question can be addressed in follow-up work or as a stretch addition if budget allows. |

---

## 12. Acknowledged Limitations

To be reported as a dedicated section in the paper:

1. **Three-arm RCT not run.** No human-DS pool was recruited; comparisons rely on published baselines and Kaggle leaderboards, not on first-party human-DS benchmarks. Results upper-bound human performance.
2. **Closed/open-source frontier model lineup is a snapshot.** Model versions current as of experiment completion; superseded by subsequent releases.
3. **Six tasks does not span all of product or marketing DS.** Tasks selected for skill-axis diversity, not exhaustive coverage. MMM, recommendation, and forecasting are over-represented; pricing, attribution-without-MMM-tools, A/B-test analysis, and segmentation are not directly tested.
4. **Single Kaggle competition per non-MMM task.** Generalization across multiple datasets per task type is left to follow-up work.
5. **AutoResearchClaw's contribution to paper structure is acknowledged but not separately measured.** The author did not run a controlled "AutoResearchClaw vs human-author skeleton" comparison.
6. **Proprietary integrated-stack agents excluded.** Claude Code (Anthropic), Devin (Cognition), Cursor Agent, GitHub Copilot Workspace, ChatGPT Advanced Data Analysis, and Gemini Data Science Agent are excluded from the matrix because their base-model is locked to a single provider, which would break the bias-defense argument. This is a deliberate scope choice, but it means the paper does not characterize the most widely-used commercial DS-agent products. Their evaluation is left to follow-up work that does not aim for the same bias-defense protocol.

---

## 13. Out of Scope

- Comparing agent vs human DS in a controlled RCT (left to future work; requires institutional access).
- Evaluating non-tabular tasks (vision, NLP-only).
- Evaluating non-DS lifecycle stages (data engineering, ETL, infrastructure provisioning).
- Domain-specific fine-tuning of the base models.
- Cost-optimal agent design (engineering contribution, not research contribution).

---

## 14. Open Questions for Author Review

1. Should the OSF pre-registration go up *before* or *after* this design doc is committed to GitHub? Pre-registration before commit is conservative; after commit makes the design doc public earlier.

**Resolved (locked decisions):**

- Hypotheses: H1 + H2 only (H3 dropped as too weak).
- License: MIT.
- Tasks: 6 (no A/B-test stretch task).
- Base models: Claude 4.6 Sonnet + DeepSeek-V3 + Qwen2.5-Coder-32B-Instruct (Option β).
- Agents: AIDE + AutoGen + Aider.
- Matrix size: 3 × 3 × 6 × 5 seeds = 270 runs (+ optional ~10 stretch headline cells).
