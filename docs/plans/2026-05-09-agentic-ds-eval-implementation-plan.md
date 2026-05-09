# Agentic DS Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the experimental harness, dataset loaders, reference baselines, agent integrations, evaluation scripts, and result-aggregation pipeline for a multi-agent × multi-base-model evaluation of agentic AI on six product+marketing DS tasks.

**Architecture:** Python package (`src/agentic_ds_eval/`) following the author's `llmsynth` repo pattern, with thin runner scripts in `experiments/`. The package separates concerns: dataset loaders, reference baselines, agent integrations, base-model clients, experiment runner with resumable CSV checkpoints, per-task evaluation, cross-cell aggregation. Three abstract interfaces (`AgentRunner`, `ModelClient`, per-task `Evaluator`) keep agent and model swaps isolated.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, scipy, pytest, Anthropic SDK (Claude), vLLM or transformers (DeepSeek-V3 / Qwen2.5-Coder), pymc-marketing, lightweight_mmm, rpy2 (Robyn bridge), `aider-chat`, `autogen-agentchat`, `aide-ml` (or fork from MLE-bench).

---

## File Structure

```
agentic-ds-eval/
├── LICENSE                                    # MIT
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── specs/2026-05-09-agentic-ds-eval-design.md      # done
│   ├── plans/2026-05-09-agentic-ds-eval-implementation-plan.md   # this file
│   └── pre-registration/osf-pre-registration.md         # phase 2
├── src/agentic_ds_eval/
│   ├── __init__.py
│   ├── lifecycle/                       # taxonomy artifacts
│   │   ├── stages.py                    # 7 stages with allowed-ops + required-output
│   │   └── autonomy.py                  # 5 levels with classification rubric
│   ├── datasets/                        # 6 loaders, one per task
│   │   ├── base.py                      # DatasetLoader ABC
│   │   ├── robyn_mmm.py
│   │   ├── coupon_recruit.py
│   │   ├── churn_kkbox.py
│   │   ├── hm_fashion.py
│   │   ├── retailrocket.py
│   │   └── m5_walmart.py
│   ├── baselines/                       # reference (non-agent) runners
│   │   ├── mmm_robyn.py                 # rpy2 bridge
│   │   ├── mmm_pymc.py
│   │   ├── mmm_lightweight.py
│   │   ├── kaggle_leaderboard.py        # static lookup (no scraping)
│   │   └── m5_published.py              # static lookup
│   ├── models/                          # base-model clients
│   │   ├── base.py                      # ModelClient ABC
│   │   ├── claude_sonnet.py             # Anthropic API
│   │   ├── deepseek_v3.py               # vLLM/HF local
│   │   └── qwen_coder.py                # vLLM/HF local
│   ├── agents/                          # agent runners
│   │   ├── base.py                      # AgentRunner ABC
│   │   ├── aide_runner.py
│   │   ├── autogen_runner.py
│   │   └── aider_runner.py
│   ├── evaluation/                      # task-specific scoring
│   │   ├── mmm_eval.py                  # Pearson r per channel + total deviation
│   │   ├── classification_eval.py       # AUC + percentile vs leaderboard
│   │   ├── ranking_eval.py              # MAP@12 + percentile
│   │   └── forecasting_eval.py          # WRMSSE + percentile
│   ├── runner/                          # experiment orchestration
│   │   ├── experiment.py                # main loop
│   │   ├── checkpoint.py                # resumable CSV
│   │   ├── manifest.py                  # per-run JSON metadata
│   │   └── cost_tracker.py
│   └── aggregation/                     # post-hoc analysis
│       ├── matrix_table.py
│       ├── stats.py                     # 2-way ANOVA, t-CIs
│       └── plots.py                     # matplotlib figures
├── experiments/
│   ├── run_baselines.py                 # one-shot reference baseline runs
│   ├── run_matrix.py                    # full agent × model × task × seed loop
│   ├── aggregate_results.py
│   └── make_plots.py
├── data/                                # gitignored, populated at runtime
├── results/                             # gitignored, populated at runtime
│   ├── baselines/
│   ├── matrix/
│   ├── plots/
│   └── manifest/
└── tests/
    ├── unit/
    └── integration/
```

---

## Phase 1 — Repo scaffolding

### Task 1.1: Create LICENSE and pyproject.toml

**Files:**
- Create: `LICENSE`
- Create: `pyproject.toml`

- [ ] **Step 1: Write LICENSE**

```
MIT License

Copyright (c) 2026 Aditya Puttaparthi Tirumala

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "agentic-ds-eval"
version = "0.1.0"
description = "Multi-agent multi-base-model evaluation of agentic AI on product+marketing DS tasks"
authors = [{name = "Aditya Puttaparthi Tirumala"}]
license = {file = "LICENSE"}
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.5",
    "scipy>=1.13",
    "matplotlib>=3.9",
    "anthropic>=0.40",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
mmm = ["pymc-marketing>=0.7", "lightweight-mmm>=0.1.9", "rpy2>=3.5"]
agents = ["aider-chat>=0.60", "autogen-agentchat>=0.4"]
models = ["vllm>=0.6", "transformers>=4.45", "torch>=2.4"]
dev = ["pytest>=8.3", "pytest-cov>=5.0", "ruff>=0.6"]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Commit**

```bash
git add LICENSE pyproject.toml
git commit -m "chore: add MIT license and pyproject.toml"
```

### Task 1.2: Create .gitignore and directory skeleton

**Files:**
- Modify: `.gitignore`
- Create: `src/agentic_ds_eval/__init__.py`
- Create: `experiments/.gitkeep`
- Create: `data/.gitkeep`
- Create: `results/.gitkeep`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.coverage
*.egg-info/
dist/
build/
.venv/
venv/

# data and results stay local; not tracked
/data/*
!/data/.gitkeep
/results/*
!/results/.gitkeep

# secrets
.env
*.key

# OS
.DS_Store
```

- [ ] **Step 2: Create directory skeleton**

```bash
mkdir -p src/agentic_ds_eval/{lifecycle,datasets,baselines,models,agents,evaluation,runner,aggregation}
mkdir -p experiments tests/{unit,integration}
mkdir -p data results/{baselines,matrix,plots,manifest}
touch src/agentic_ds_eval/__init__.py
touch src/agentic_ds_eval/{lifecycle,datasets,baselines,models,agents,evaluation,runner,aggregation}/__init__.py
touch experiments/.gitkeep data/.gitkeep results/.gitkeep
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore src/ experiments/ data/ results/ tests/
git commit -m "chore: scaffold package directory structure"
```

### Task 1.3: Create README skeleton

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# agentic-ds-eval

Empirical evaluation of agentic AI systems on product and marketing data-science tasks. Multi-agent × multi-base-model design with bias-defense protocol.

**Status:** Pre-registration drafting; experiments pending.

**Author:** Aditya Puttaparthi Tirumala — Independent Researcher.

## Design

See `docs/specs/2026-05-09-agentic-ds-eval-design.md` for the full design spec.

**Tasks (6):** Marketing-Mix Modeling (Robyn data, 3 baselines), Coupon redemption (Recruit Holdings), Customer churn (KKBox), Recommendation (H&M), Conversion-funnel (RetailRocket), Hierarchical forecasting (M5).

**Agents (3):** AIDE, AutoGen, Aider.

**Base models (3):** Claude 4.6 Sonnet, DeepSeek-V3, Qwen2.5-Coder-32B-Instruct.

**Statistical protocol:** 5 seeds × 9 cells × 6 tasks = 270 runs. 95% CI via t-distribution.

## Pre-registration

OSF pre-registration is filed *before* experimental runs. See `docs/pre-registration/osf-pre-registration.md`.

## Reproducing

```bash
pip install -e ".[mmm,agents,models,dev]"
python experiments/run_baselines.py --task <task_name>
python experiments/run_matrix.py --agent <agent> --model <model> --task <task> --seed <seed>
python experiments/aggregate_results.py
```

## License

MIT — see `LICENSE`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README skeleton"
```

---

## Phase 2 — OSF pre-registration

### Task 2.1: Draft OSF pre-registration document

**Files:**
- Create: `docs/pre-registration/osf-pre-registration.md`

- [ ] **Step 1: Write the pre-registration**

```markdown
# OSF Pre-Registration: Agentic AI in Product and Marketing Data Science

**Author:** Aditya Puttaparthi Tirumala (Independent Researcher)
**Date filed:** 2026-05-09
**Linked spec commit:** 4057ceb (private repo, will be public on OSF release)

## Research Questions
1. How do agentic AI systems perform across the DS lifecycle for product/marketing DS tasks?
2. Does base-model-effect dominate agent-scaffolding-effect, or vice versa?
3. Where in the lifecycle do agents add value vs degrade work?
4. How does observed performance map to a 5-level autonomy taxonomy?

## Hypotheses

**H1.** Variance in task metrics due to base-model choice exceeds variance due to agent-scaffolding choice (two-way ANOVA per task on the 3×3 matrix). Rejection: F-statistic for agent-effect ≥ F-statistic for model-effect on ≥ 4 of 6 tasks.

**H2.** Agents achieve L4 autonomy in ≥ 60% of cells on routine prediction tasks (Coupon, Churn, Recommendation), and only L3 or below in ≥ 70% of cells on attribution-heavy tasks (MMM, Forecasting). Rejection: L4-rate < 60% on routine OR > 30% on attribution-heavy.

## Design Variables

- **Tasks (6):** MMM (Robyn data) / Coupon Recruit / Churn KKBox / H&M Recommendation / RetailRocket Funnel / M5 Forecasting.
- **Agents (3):** AIDE / AutoGen / Aider.
- **Base models (3):** Claude 4.6 Sonnet / DeepSeek-V3 / Qwen2.5-Coder-32B-Instruct.
- **Seeds (5):** 42, 123, 7, 2024, 999.
- **Total runs:** 270 (+ ~10 stretch headline cells with Claude 4.7 Opus on MMM).

## Lifecycle × Autonomy Framework
[Reproduce the full §4.1 framework table from the design spec.]

## Evaluation Protocols
[Reproduce the full §4.2 per-task evaluation specifications.]

## Statistical Protocol
- Per cell: mean ± 95% CI via t-distribution (df = n_seeds - 1).
- Two-way ANOVA per task with agent and base-model as factors.
- Per-cell autonomy classification by post-hoc author review against the 5-level rubric.

## Tooling Disclosure
- AutoResearchClaw v0.3.1 used as experimental harness orchestrator and paper skeleton outliner.
- Results, analysis, interpretation, and discussion are author-written.
- AutoResearchClaw is *not* one of the four agents under evaluation.

## What Will *Not* Be Pre-Registered
- Specific model versions / library versions (will be locked at experiment-start time and reported in the published paper).
- Specific WRMSSE / MAP@12 numerical thresholds for "above-leaderboard" claims (these are derived from leaderboards at experiment time).

## Deviations Policy
Any deviation from this pre-registration during execution will be documented in the paper's Methods section as "deviation from pre-registration: [what changed and why]."
```

- [ ] **Step 2: Commit**

```bash
git add docs/pre-registration/osf-pre-registration.md
git commit -m "docs: add OSF pre-registration draft"
```

- [ ] **Step 3: Manual step (author): upload to OSF**

```
- Log into https://osf.io
- Create a new pre-registration project: "Agentic AI in Product and Marketing Data Science"
- Upload docs/pre-registration/osf-pre-registration.md
- Generate DOI
- Record DOI in this file's frontmatter
- Commit the updated DOI
```

---

## Phase 3 — Lifecycle × autonomy framework

### Task 3.1: Define lifecycle stages

**Files:**
- Create: `src/agentic_ds_eval/lifecycle/stages.py`
- Create: `tests/unit/test_stages.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_stages.py
from agentic_ds_eval.lifecycle.stages import LIFECYCLE_STAGES, Stage

def test_seven_stages_defined():
    assert len(LIFECYCLE_STAGES) == 7

def test_stage_names_match_spec():
    expected = ["problem_framing", "data_exploration", "feature_engineering",
                "model_selection", "validation", "interpretation", "reporting"]
    assert [s.name for s in LIFECYCLE_STAGES] == expected

def test_each_stage_has_operations_and_required_output():
    for s in LIFECYCLE_STAGES:
        assert s.operations and isinstance(s.operations, list)
        assert s.required_output and isinstance(s.required_output, str)
```

- [ ] **Step 2: Run test, verify FAIL**

```bash
pytest tests/unit/test_stages.py -v
# Expected: ImportError for stages module
```

- [ ] **Step 3: Implement stages**

```python
# src/agentic_ds_eval/lifecycle/stages.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    operations: list[str]
    required_output: str

LIFECYCLE_STAGES: list[Stage] = [
    Stage(
        "problem_framing",
        "Read task spec; identify target, evaluation metric, constraints",
        ["read_spec", "identify_target", "select_metric"],
        "One-paragraph framing + chosen metric",
    ),
    Stage(
        "data_exploration",
        "Load data; profile distributions, missingness, leakage risks",
        ["load_data", "profile", "detect_leakage"],
        "EDA notes + identified data-quality issues",
    ),
    Stage(
        "feature_engineering",
        "Create derived features; encode categoricals; handle sparsity",
        ["create_features", "encode", "impute"],
        "Engineered feature set + brief justification",
    ),
    Stage(
        "model_selection",
        "Choose model family; justify selection",
        ["choose_model", "justify"],
        "Model choice + 1-2 line rationale",
    ),
    Stage(
        "validation",
        "Implement train/holdout protocol; cross-validation; report metric",
        ["split", "cross_validate", "score"],
        "Validation strategy + held-out scores",
    ),
    Stage(
        "interpretation",
        "Generate feature importances, SHAP, marginal effects, sensitivity",
        ["importance", "shap", "sensitivity"],
        "Interpretation artifact appropriate to the task",
    ),
    Stage(
        "reporting",
        "Summarize findings",
        ["summarize"],
        "Markdown report ≤ 800 words",
    ),
]
```

- [ ] **Step 4: Run test, verify PASS**

```bash
pytest tests/unit/test_stages.py -v
# Expected: 3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/agentic_ds_eval/lifecycle/stages.py tests/unit/test_stages.py
git commit -m "feat: lifecycle stages taxonomy with 7 stages"
```

### Task 3.2: Define autonomy levels

**Files:**
- Create: `src/agentic_ds_eval/lifecycle/autonomy.py`
- Create: `tests/unit/test_autonomy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_autonomy.py
from agentic_ds_eval.lifecycle.autonomy import AUTONOMY_LEVELS, classify_autonomy

def test_five_autonomy_levels():
    assert len(AUTONOMY_LEVELS) == 5
    assert [l.level for l in AUTONOMY_LEVELS] == [1, 2, 3, 4, 5]

def test_classify_autonomy_l4_when_only_final_review():
    # Agent ran all stages without intervention; human reviewed only final
    interventions = []
    final_review = True
    assert classify_autonomy(interventions, final_review) == 4

def test_classify_autonomy_l3_when_stage_gates():
    # Agent ran each stage but human gated between stages
    interventions = ["after_data_exploration", "after_validation"]
    assert classify_autonomy(interventions, final_review=True) == 3

def test_classify_autonomy_l5_when_no_review_at_all():
    assert classify_autonomy(interventions=[], final_review=False) == 5
```

- [ ] **Step 2: Run test, verify FAIL**

```bash
pytest tests/unit/test_autonomy.py -v
```

- [ ] **Step 3: Implement autonomy**

```python
# src/agentic_ds_eval/lifecycle/autonomy.py
from dataclasses import dataclass

@dataclass(frozen=True)
class AutonomyLevel:
    level: int
    description: str
    human_role: str

AUTONOMY_LEVELS: list[AutonomyLevel] = [
    AutonomyLevel(1, "Code-completion within human-authored pipeline", "Human owns architecture; agent fills code blocks"),
    AutonomyLevel(2, "Human-in-the-loop step suggestion", "Agent proposes; human accepts/rejects each step"),
    AutonomyLevel(3, "Stage-autonomous, human reviews stage outputs", "Human gates between lifecycle stages"),
    AutonomyLevel(4, "Cross-stage autonomous, human reviews final output only", "Single human checkpoint, end of pipeline"),
    AutonomyLevel(5, "Fully autonomous, no human checkpoint", "Human reviews after deployment"),
]

def classify_autonomy(interventions: list[str], final_review: bool) -> int:
    """
    Classify the autonomy level achieved in a single experiment cell.

    Args:
        interventions: List of human-intervention checkpoints (e.g., "after_eda").
                       Empty list = no intermediate human gates.
        final_review: True if a human reviewed the final output before "deployment".

    Returns:
        Autonomy level (1-5) as observed.
    """
    if not interventions and not final_review:
        return 5
    if not interventions and final_review:
        return 4
    if interventions:
        return 3
    return 2  # fallback for edge cases
```

- [ ] **Step 4: Run test, verify PASS**

```bash
pytest tests/unit/test_autonomy.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/agentic_ds_eval/lifecycle/autonomy.py tests/unit/test_autonomy.py
git commit -m "feat: 5-level autonomy taxonomy with classification function"
```

---

## Phase 4 — Dataset loaders

### Task 4.1: Create DatasetLoader abstract base

**Files:**
- Create: `src/agentic_ds_eval/datasets/base.py`
- Create: `tests/unit/test_dataset_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_dataset_base.py
import pandas as pd
from agentic_ds_eval.datasets.base import DatasetLoader, DatasetSpec

def test_dataset_spec_has_required_fields():
    spec = DatasetSpec(
        name="test", task_type="classification",
        target="y", metric="auc",
        train_url="https://example.com/train.csv", test_url=None,
    )
    assert spec.name == "test"
    assert spec.task_type == "classification"

def test_dataset_loader_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        DatasetLoader()  # cannot instantiate ABC
```

- [ ] **Step 2: Run test, verify FAIL**

- [ ] **Step 3: Implement base**

```python
# src/agentic_ds_eval/datasets/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: str  # "classification" | "regression" | "ranking" | "forecasting" | "mmm"
    target: str
    metric: str
    train_url: str
    test_url: str | None

class DatasetLoader(ABC):
    spec: DatasetSpec
    cache_dir: Path = Path("data")

    @abstractmethod
    def load(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Return (train_df, test_df). test_df may be None if held-out is derived from train."""
        ...

    @abstractmethod
    def task_description(self) -> str:
        """Return the natural-language task description that will be passed to the agent."""
        ...
```

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add src/agentic_ds_eval/datasets/base.py tests/unit/test_dataset_base.py
git commit -m "feat: DatasetLoader ABC + DatasetSpec dataclass"
```

### Task 4.2: Implement Robyn MMM dataset loader

**Files:**
- Create: `src/agentic_ds_eval/datasets/robyn_mmm.py`
- Create: `tests/integration/test_robyn_mmm.py`

- [ ] **Step 1: Implement loader**

```python
# src/agentic_ds_eval/datasets/robyn_mmm.py
from pathlib import Path
import pandas as pd
from .base import DatasetLoader, DatasetSpec

ROBYN_DEMO_URL = "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/demo/dt_simulated_weekly.csv"

class RobynMMMLoader(DatasetLoader):
    spec = DatasetSpec(
        name="robyn_mmm",
        task_type="mmm",
        target="revenue",
        metric="channel_attribution_pearson_r",
        train_url=ROBYN_DEMO_URL,
        test_url=None,
    )

    def __init__(self, cache_dir: Path = Path("data")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self):
        path = self.cache_dir / "robyn_demo.csv"
        if not path.exists():
            df = pd.read_csv(self.spec.train_url)
            df.to_csv(path, index=False)
        df = pd.read_csv(path)
        return df, None

    def task_description(self) -> str:
        return (
            "Fit a marketing-mix model on the supplied weekly data. "
            "Decompose total response (revenue) into per-channel contributions. "
            "Report per-channel ROI (revenue / spend) for each marketing channel. "
            "Provide adstock and saturation curve parameters where applicable."
        )
```

- [ ] **Step 2: Write integration test**

```python
# tests/integration/test_robyn_mmm.py
import pytest
from agentic_ds_eval.datasets.robyn_mmm import RobynMMMLoader

@pytest.mark.integration
def test_robyn_loader_downloads_and_returns_dataframe(tmp_path):
    loader = RobynMMMLoader(cache_dir=tmp_path)
    train, test = loader.load()
    assert train is not None
    assert len(train) > 100  # weekly data, multi-year
    assert test is None
    assert "revenue" in train.columns
```

- [ ] **Step 3: Run integration test (requires network)**

```bash
pytest tests/integration/test_robyn_mmm.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/agentic_ds_eval/datasets/robyn_mmm.py tests/integration/test_robyn_mmm.py
git commit -m "feat: Robyn MMM demo data loader"
```

### Task 4.3 through 4.7: Implement remaining 5 dataset loaders

For each of Coupon (Recruit), Churn (KKBox), H&M Fashion, RetailRocket, M5 Walmart, follow the same pattern as Task 4.2:

1. Implement `<task>_loader.py` with `DatasetSpec` and `load()` and `task_description()`.
2. Write integration test that fetches and validates schema.
3. Commit per dataset.

**Important notes:**

- **KKBox Churn (Task 4.4):** Kaggle dataset; requires Kaggle API credentials. The loader should check `~/.kaggle/kaggle.json` and download via `kaggle competitions download -c kkbox-churn-prediction-challenge`.
- **H&M Fashion (Task 4.5):** Kaggle dataset; same pattern. Large (~30 GB raw transactions). The loader should support a `sample_frac` parameter and default to 0.1 for cost-controlled experiments.
- **RetailRocket (Task 4.6):** Public Kaggle dataset; events.csv + item_properties files. Loader should join into a single events frame.
- **M5 (Task 4.7):** Kaggle, very large. Loader should support `n_series` cap (default 100 randomly sampled series stratified by department × store × state).
- **Coupon (Task 4.3):** Recruit Holdings Kaggle 2015. Multi-table; loader should join coupon_list_train + coupon_visit_train + user_list into a single training frame.

For each: copy Task 4.2 structure, change `name`, `target`, `metric`, `train_url`, and the `load()` body. Each takes ~10 minutes once the pattern is established.

---

## Phase 5 — Reference baseline runners

### Task 5.1: PyMC-Marketing MMM reference baseline

**Files:**
- Create: `src/agentic_ds_eval/baselines/mmm_pymc.py`
- Create: `tests/integration/test_mmm_pymc.py`

- [ ] **Step 1: Implement runner**

```python
# src/agentic_ds_eval/baselines/mmm_pymc.py
"""Run PyMC-Marketing MMM as a reference baseline.

Returns per-channel ROI estimates that can be compared to agent outputs.
"""
import pandas as pd
from pathlib import Path

def run_pymc_mmm(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Fit PyMC-Marketing MMM on the Robyn demo data.

    Returns a DataFrame with columns: ['channel', 'roi', 'spend', 'contribution'].

    Notes:
    - Uses canonical configuration from PyMC-Marketing tutorial.
    - Channels: detected automatically from columns containing 'spend' suffix.
    - Adstock: GeometricAdstock; Saturation: LogisticSaturation.
    - 4000 NUTS samples, 4 chains, target_accept=0.9.
    """
    from pymc_marketing.mmm import (
        MMM, GeometricAdstock, LogisticSaturation,
    )

    spend_cols = [c for c in df.columns if c.endswith("_S")]
    date_col = "DATE"
    target = "revenue"

    mmm = MMM(
        date_column=date_col,
        channel_columns=spend_cols,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
    )
    mmm.fit(X=df[[date_col] + spend_cols], y=df[target],
            target_accept=0.9, chains=4, draws=4000, random_seed=42)

    contributions = mmm.compute_channel_contribution_original_scale().mean(dim="date").to_pandas()
    spend = df[spend_cols].sum()
    roi = contributions / spend
    out = pd.DataFrame({
        "channel": spend_cols,
        "spend": spend.values,
        "contribution": contributions.values,
        "roi": roi.values,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "pymc_mmm_attribution.csv", index=False)
    return out
```

- [ ] **Step 2: Test (integration, slow)**

```python
# tests/integration/test_mmm_pymc.py
import pytest
from pathlib import Path
from agentic_ds_eval.datasets.robyn_mmm import RobynMMMLoader
from agentic_ds_eval.baselines.mmm_pymc import run_pymc_mmm

@pytest.mark.slow
@pytest.mark.integration
def test_pymc_mmm_returns_per_channel_roi(tmp_path):
    df, _ = RobynMMMLoader(cache_dir=tmp_path).load()
    result = run_pymc_mmm(df, output_dir=tmp_path)
    assert "channel" in result.columns
    assert "roi" in result.columns
    assert len(result) >= 3  # at least 3 channels
    assert (result["roi"] > 0).all()
```

- [ ] **Step 3: Commit**

### Task 5.2: LightweightMMM reference baseline

Same pattern as Task 5.1, file `src/agentic_ds_eval/baselines/mmm_lightweight.py`. Use Google's `lightweight_mmm` package with carryover + Hill saturation.

### Task 5.3: Robyn MMM reference baseline (R via rpy2)

**Files:**
- Create: `src/agentic_ds_eval/baselines/mmm_robyn.py`
- Create: `tests/integration/test_mmm_robyn.py`

Robyn is an R package. Two implementation options — pick (a):
- **(a) Use rpy2 to invoke Robyn from Python.** Required: R 4.3+, Robyn R package installed.
- (b) Subprocess: write the Python data to CSV, call an Rscript that runs Robyn, read back the result CSV.

(a) is more elegant but adds rpy2 to the dependency surface. (b) is more portable and is the preferred pattern in the MMM community.

Recommended: use (b). Write `scripts/run_robyn.R` that takes input CSV path and output JSON path. The Python wrapper calls it via subprocess.

- [ ] **Step 1: Write the R script**

```r
# scripts/run_robyn.R
# Usage: Rscript run_robyn.R <input_csv> <output_json>
library(Robyn)
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
output_json <- args[2]
df <- read.csv(input_csv)

# Use Robyn canonical config
InputCollect <- robyn_inputs(
  dt_input = df,
  dt_holidays = dt_prophet_holidays,
  date_var = "DATE",
  dep_var = "revenue",
  dep_var_type = "revenue",
  paid_media_spends = c("tv_S", "ooh_S", "print_S", "facebook_S", "search_S"),
  paid_media_vars = c("tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"),
  organic_vars = c("newsletter"),
  context_vars = c("competitor_sales_B", "events"),
  factor_vars = c("events"),
  window_start = "2016-01-01",
  window_end = "2018-12-31",
  adstock = "geometric"
)

# ... (full configuration; defer to Robyn vignette)

# Save attribution to JSON
write_json(roi_estimates, output_json)
```

- [ ] **Step 2: Write Python wrapper**

```python
# src/agentic_ds_eval/baselines/mmm_robyn.py
import subprocess, json
import pandas as pd
from pathlib import Path

def run_robyn_mmm(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    in_csv = output_dir / "robyn_input.csv"
    out_json = output_dir / "robyn_attribution.json"
    df.to_csv(in_csv, index=False)
    subprocess.run(
        ["Rscript", "scripts/run_robyn.R", str(in_csv), str(out_json)],
        check=True,
    )
    with open(out_json) as f:
        data = json.load(f)
    return pd.DataFrame(data)
```

- [ ] **Step 3: Commit**

### Task 5.4: Kaggle leaderboard lookup

**Files:**
- Create: `src/agentic_ds_eval/baselines/kaggle_leaderboard.py`
- Create: `data/leaderboards/<task>_leaderboard.csv` (one per non-MMM task; static lookup table)

Static lookup; no scraping. The author manually fetches leaderboard CSVs from Kaggle for each competition (Coupon, KKBox, H&M, RetailRocket) and saves as `<task>_leaderboard.csv` with columns `[rank, score]`.

The Python module exposes `get_percentile(task, score)` that returns the leaderboard percentile of the input score.

```python
# src/agentic_ds_eval/baselines/kaggle_leaderboard.py
from pathlib import Path
import pandas as pd

def get_percentile(task: str, score: float, leaderboard_dir: Path = Path("data/leaderboards")) -> float:
    """Return the leaderboard percentile of a score. Higher score = higher percentile."""
    df = pd.read_csv(leaderboard_dir / f"{task}_leaderboard.csv")
    pct = (df["score"] <= score).mean() * 100
    return pct
```

### Task 5.5: M5 published baseline lookup

Static lookup. Top-50 M5 solutions documented in Makridakis et al. (2022). Save as `data/leaderboards/m5_top50.csv` with columns `[rank, team, wrmsse]`.

---

## Phase 6 — Base model integration

### Task 6.1: ModelClient abstract base

**Files:**
- Create: `src/agentic_ds_eval/models/base.py`
- Create: `tests/unit/test_model_base.py`

```python
# src/agentic_ds_eval/models/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

class ModelClient(ABC):
    name: str
    @abstractmethod
    def call(self, system: str, messages: list[dict]) -> ModelResponse:
        """Make a single chat call. Returns ModelResponse with token counts and cost."""
        ...
```

- [ ] **Step 1: Write test**

```python
# tests/unit/test_model_base.py
from agentic_ds_eval.models.base import ModelClient, ModelResponse
import pytest

def test_model_client_is_abstract():
    with pytest.raises(TypeError):
        ModelClient()

def test_model_response_dataclass():
    r = ModelResponse(text="hello", input_tokens=5, output_tokens=1, cost_usd=0.001)
    assert r.text == "hello"
```

- [ ] **Step 2-5: Implement, run, commit (as above)**

### Task 6.2: Claude 4.6 Sonnet client

**Files:**
- Create: `src/agentic_ds_eval/models/claude_sonnet.py`
- Create: `tests/integration/test_claude_sonnet.py`

```python
# src/agentic_ds_eval/models/claude_sonnet.py
import os
import anthropic
from .base import ModelClient, ModelResponse

CLAUDE_46_SONNET = "claude-sonnet-4-6"
INPUT_PRICE_PER_M = 3.0
OUTPUT_PRICE_PER_M = 15.0

class ClaudeSonnetClient(ModelClient):
    name = "claude_4.6_sonnet"

    def __init__(self, api_key: str | None = None, max_tokens: int = 4096):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.max_tokens = max_tokens

    def call(self, system: str, messages: list[dict]) -> ModelResponse:
        resp = self.client.messages.create(
            model=CLAUDE_46_SONNET,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        cost = (in_tok / 1e6) * INPUT_PRICE_PER_M + (out_tok / 1e6) * OUTPUT_PRICE_PER_M
        return ModelResponse(text=text, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
```

- [ ] **Step 1-5:** Test with `ANTHROPIC_API_KEY` set; assert response is ModelResponse, cost > 0; commit.

### Task 6.3: DeepSeek-V3 client (vLLM local)

**Files:**
- Create: `src/agentic_ds_eval/models/deepseek_v3.py`
- Create: `tests/integration/test_deepseek_v3.py`

DeepSeek-V3 (671B MoE) is large. Two deployment options:
- **(a) Self-host with vLLM** on a high-memory rented GPU (8×H100 ~$24/hr — too expensive).
- **(b) Use DeepSeek's API directly** at $0.27/M input / $1.10/M output (May 2026 prices). Recommended.

Implement (b) — much cheaper and matches the cost analysis in §8 of the spec:

```python
# src/agentic_ds_eval/models/deepseek_v3.py
import os, httpx
from .base import ModelClient, ModelResponse

DEEPSEEK_V3 = "deepseek-chat"
INPUT_PRICE_PER_M = 0.27
OUTPUT_PRICE_PER_M = 1.10

class DeepSeekV3Client(ModelClient):
    name = "deepseek_v3"

    def __init__(self, api_key: str | None = None, max_tokens: int = 4096):
        self.api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self.max_tokens = max_tokens
        self.url = "https://api.deepseek.com/chat/completions"

    def call(self, system: str, messages: list[dict]) -> ModelResponse:
        all_msgs = [{"role": "system", "content": system}, *messages]
        resp = httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": DEEPSEEK_V3, "messages": all_msgs, "max_tokens": self.max_tokens},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        in_tok = data["usage"]["prompt_tokens"]
        out_tok = data["usage"]["completion_tokens"]
        cost = (in_tok / 1e6) * INPUT_PRICE_PER_M + (out_tok / 1e6) * OUTPUT_PRICE_PER_M
        return ModelResponse(text=text, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
```

### Task 6.4: Qwen2.5-Coder-32B-Instruct client (vLLM local)

Self-host Qwen2.5-Coder-32B on a single A100/H100 (~$1.50/hr). Use vLLM OpenAI-compatible server.

```python
# src/agentic_ds_eval/models/qwen_coder.py
import httpx, os
from .base import ModelClient, ModelResponse

QWEN_CODER = "Qwen2.5-Coder-32B-Instruct"
GPU_HOURLY_COST = 1.50  # rented A100; tracked separately, no per-token cost

class QwenCoderClient(ModelClient):
    name = "qwen2.5_coder_32b"

    def __init__(self, base_url: str = "http://localhost:8000/v1", max_tokens: int = 4096):
        self.base_url = base_url
        self.max_tokens = max_tokens

    def call(self, system: str, messages: list[dict]) -> ModelResponse:
        all_msgs = [{"role": "system", "content": system}, *messages]
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json={"model": QWEN_CODER, "messages": all_msgs, "max_tokens": self.max_tokens},
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        in_tok = data["usage"]["prompt_tokens"]
        out_tok = data["usage"]["completion_tokens"]
        # GPU hourly cost is tracked at the runner level via wall-clock time, not per-call
        return ModelResponse(text=text, input_tokens=in_tok, output_tokens=out_tok, cost_usd=0.0)
```

---

## Phase 7 — Agent integration layer

### Task 7.1: AgentRunner abstract base

**Files:**
- Create: `src/agentic_ds_eval/agents/base.py`
- Create: `tests/unit/test_agent_base.py`

```python
# src/agentic_ds_eval/agents/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from agentic_ds_eval.models.base import ModelClient

@dataclass
class AgentRunResult:
    final_artifact: str  # markdown report or final code/output
    log: list[dict]      # full conversation/iteration log
    interventions: list[str]  # for autonomy classification
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int

class AgentRunner(ABC):
    name: str

    @abstractmethod
    def run(self, task_description: str, data_dir: Path, model: ModelClient,
            seed: int, working_dir: Path) -> AgentRunResult:
        ...
```

### Task 7.2: AIDE wrapper

**Files:**
- Create: `src/agentic_ds_eval/agents/aide_runner.py`

AIDE is from Weco AI and used in MLE-bench. It's a Python package; install via `pip install weco-ai`. The wrapper should:
1. Configure AIDE with the supplied ModelClient as its LLM backend
2. Set the data directory as AIDE's `data_dir`
3. Run AIDE for a fixed iteration budget (e.g., 20 iterations)
4. Capture token counts and cost from the ModelClient
5. Return AIDE's final solution as `final_artifact`

```python
# src/agentic_ds_eval/agents/aide_runner.py
from pathlib import Path
from agentic_ds_eval.agents.base import AgentRunner, AgentRunResult
from agentic_ds_eval.models.base import ModelClient

class AIDERunner(AgentRunner):
    name = "aide"

    def __init__(self, max_iterations: int = 20):
        self.max_iterations = max_iterations

    def run(self, task_description: str, data_dir: Path, model: ModelClient,
            seed: int, working_dir: Path) -> AgentRunResult:
        from aide_ml import Solver  # imported lazily

        solver = Solver(
            data_dir=str(data_dir),
            workspace=str(working_dir),
            task=task_description,
            seed=seed,
            llm_call=lambda system, msgs: model.call(system, msgs).text,
            max_iterations=self.max_iterations,
        )
        result = solver.solve()

        # Token + cost accumulation handled by ModelClient internally
        # (or capture via custom callback; deferred to integration testing)
        return AgentRunResult(
            final_artifact=result.summary,
            log=result.history,
            interventions=[],  # no human interventions during eval
            total_cost_usd=0.0,  # populated from ModelClient telemetry
            total_input_tokens=0,
            total_output_tokens=0,
        )
```

**Note:** the `aide_ml` package interface is approximate. At implementation time, check the current AIDE repo for the exact API and update accordingly.

### Task 7.3: AutoGen wrapper

Use `autogen-agentchat` v0.4+. Configure two agents (Planner + Coder) plus a Critic. Pass ModelClient as the OpenAI-compatible client for all three.

```python
# src/agentic_ds_eval/agents/autogen_runner.py
# Use autogen v0.4 ModelClient adapter; set up Planner, Coder, Critic agents
# with the supplied ModelClient as LLM. Orchestrate via SelectorGroupChat.
# Full implementation: ~80 lines.
```

### Task 7.4: Aider wrapper

Aider is CLI-based. Spawn `aider --model <model_id> --no-stream <task_file>` as subprocess; capture token/cost from Aider's stdout markup.

---

## Phase 8 — Experiment runner

### Task 8.1: Resumable CSV checkpoint

**Files:**
- Create: `src/agentic_ds_eval/runner/checkpoint.py`
- Create: `tests/unit/test_checkpoint.py`

```python
# src/agentic_ds_eval/runner/checkpoint.py
"""Resumable CSV checkpoint pattern, matching the llmsynth repo convention."""
import pandas as pd
from pathlib import Path

def load_checkpoint(path: Path) -> tuple[list[dict], set[tuple]]:
    """Load existing rows; return (rows, set of (task, agent, model, seed) tuples completed)."""
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    done = {(r["task"], r["agent"], r["model"], r["seed"]) for r in rows}
    return rows, done

def save_checkpoint(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
```

- [ ] **Step 1-5:** Write test that verifies (a) loading non-existent file returns empty, (b) saved rows can be reloaded, (c) `done` set has correct tuples. Implement. Commit.

### Task 8.2: Manifest writer

```python
# src/agentic_ds_eval/runner/manifest.py
import json, hashlib, time, platform, subprocess
from pathlib import Path
from dataclasses import asdict, dataclass

@dataclass
class RunManifest:
    run_id: str
    task: str
    agent: str
    model: str
    seed: int
    timestamp: float
    git_sha: str
    python_version: str
    library_versions: dict[str, str]

def write_manifest(manifest: RunManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
```

### Task 8.3: Cost tracker

```python
# src/agentic_ds_eval/runner/cost_tracker.py
import csv
from pathlib import Path

class CostTracker:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w") as f:
                csv.writer(f).writerow(["timestamp", "task", "agent", "model", "seed",
                                         "input_tokens", "output_tokens", "cost_usd"])

    def record(self, task: str, agent: str, model: str, seed: int,
               in_tok: int, out_tok: int, cost: float) -> None:
        import time
        with open(self.path, "a") as f:
            csv.writer(f).writerow([time.time(), task, agent, model, seed,
                                     in_tok, out_tok, cost])

    def total(self) -> float:
        if not self.path.exists():
            return 0.0
        import pandas as pd
        return pd.read_csv(self.path)["cost_usd"].sum()
```

### Task 8.4: Main experiment runner

**Files:**
- Create: `src/agentic_ds_eval/runner/experiment.py`
- Create: `experiments/run_matrix.py`

```python
# src/agentic_ds_eval/runner/experiment.py
"""Main experiment loop. Takes (task, agent, model, seed) tuple and produces a result row."""
from dataclasses import dataclass
from pathlib import Path

from agentic_ds_eval.datasets.base import DatasetLoader
from agentic_ds_eval.agents.base import AgentRunner
from agentic_ds_eval.models.base import ModelClient
from agentic_ds_eval.runner.checkpoint import load_checkpoint, save_checkpoint
from agentic_ds_eval.runner.cost_tracker import CostTracker

def run_one_cell(task_loader: DatasetLoader, agent: AgentRunner, model: ModelClient,
                 seed: int, working_root: Path, cost_tracker: CostTracker) -> dict:
    """Run one (task, agent, model, seed) experiment cell. Return result row dict."""
    cell_id = f"{task_loader.spec.name}_{agent.name}_{model.name}_seed{seed}"
    working_dir = working_root / cell_id
    working_dir.mkdir(parents=True, exist_ok=True)

    train, test = task_loader.load()
    train_path = working_dir / "train.csv"
    train.to_csv(train_path, index=False)

    result = agent.run(
        task_description=task_loader.task_description(),
        data_dir=working_dir,
        model=model,
        seed=seed,
        working_dir=working_dir,
    )

    cost_tracker.record(
        task=task_loader.spec.name, agent=agent.name, model=model.name, seed=seed,
        in_tok=result.total_input_tokens, out_tok=result.total_output_tokens,
        cost=result.total_cost_usd,
    )

    return {
        "task": task_loader.spec.name,
        "agent": agent.name,
        "model": model.name,
        "seed": seed,
        "final_artifact_path": str(working_dir / "final.md"),
        "interventions": ";".join(result.interventions),
        "input_tokens": result.total_input_tokens,
        "output_tokens": result.total_output_tokens,
        "cost_usd": result.total_cost_usd,
    }
```

- [ ] **Step 1: Write `experiments/run_matrix.py`**

```python
# experiments/run_matrix.py
"""Entry point: run the full matrix or a slice of it."""
import argparse, itertools
from pathlib import Path

from agentic_ds_eval.datasets.robyn_mmm import RobynMMMLoader
from agentic_ds_eval.datasets.coupon_recruit import CouponLoader
# ... import all 6 loaders
from agentic_ds_eval.agents.aide_runner import AIDERunner
from agentic_ds_eval.agents.autogen_runner import AutoGenRunner
from agentic_ds_eval.agents.aider_runner import AiderRunner
from agentic_ds_eval.models.claude_sonnet import ClaudeSonnetClient
from agentic_ds_eval.models.deepseek_v3 import DeepSeekV3Client
from agentic_ds_eval.models.qwen_coder import QwenCoderClient
from agentic_ds_eval.runner.experiment import run_one_cell
from agentic_ds_eval.runner.checkpoint import load_checkpoint, save_checkpoint
from agentic_ds_eval.runner.cost_tracker import CostTracker

LOADERS = {
    "robyn_mmm": RobynMMMLoader,
    "coupon": CouponLoader,
    # ... 6 total
}
AGENTS = {"aide": AIDERunner, "autogen": AutoGenRunner, "aider": AiderRunner}
MODELS = {
    "claude_sonnet": ClaudeSonnetClient,
    "deepseek_v3": DeepSeekV3Client,
    "qwen_coder": QwenCoderClient,
}
SEEDS = [42, 123, 7, 2024, 999]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(LOADERS) + ["all"], default="all")
    parser.add_argument("--agent", choices=list(AGENTS) + ["all"], default="all")
    parser.add_argument("--model", choices=list(MODELS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS + [-1], default=-1, help="-1 = all seeds")
    args = parser.parse_args()

    tasks = list(LOADERS) if args.task == "all" else [args.task]
    agents = list(AGENTS) if args.agent == "all" else [args.agent]
    models = list(MODELS) if args.model == "all" else [args.model]
    seeds = SEEDS if args.seed == -1 else [args.seed]

    out_path = Path("results/matrix/results.csv")
    cost_path = Path("results/matrix/costs.csv")
    cost_tracker = CostTracker(cost_path)
    rows, done = load_checkpoint(out_path)
    print(f"Resuming: {len(done)} cells already complete")

    for task, agent_name, model_name, seed in itertools.product(tasks, agents, models, seeds):
        if (task, agent_name, model_name, seed) in done:
            continue
        loader = LOADERS[task]()
        agent = AGENTS[agent_name]()
        model = MODELS[model_name]()
        try:
            row = run_one_cell(loader, agent, model, seed,
                               working_root=Path("results/matrix/cells"),
                               cost_tracker=cost_tracker)
            rows.append(row)
            save_checkpoint(rows, out_path)
            print(f"[{task}/{agent_name}/{model_name}/{seed}] cost=${row['cost_usd']:.2f}; running total=${cost_tracker.total():.2f}")
        except Exception as e:
            print(f"[{task}/{agent_name}/{model_name}/{seed}] FAILED: {e}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test smoke run**

```bash
python experiments/run_matrix.py --task robyn_mmm --agent aide --model claude_sonnet --seed 42
# Expected: one cell runs to completion, results/matrix/results.csv has 1 row
```

- [ ] **Step 3: Commit**

---

## Phase 9 — Per-task evaluation

### Task 9.1: MMM evaluation (Pearson r per channel)

**Files:**
- Create: `src/agentic_ds_eval/evaluation/mmm_eval.py`
- Create: `tests/unit/test_mmm_eval.py`

```python
# src/agentic_ds_eval/evaluation/mmm_eval.py
"""Score agent MMM output against three reference baselines."""
import pandas as pd
from scipy.stats import pearsonr

def score_mmm_attribution(agent_df: pd.DataFrame, baselines: dict[str, pd.DataFrame]) -> dict:
    """
    agent_df, each baseline_df: columns = ['channel', 'roi'].

    Returns dict with:
      - pearson_r per baseline (Pearson correlation of agent ROI vector with baseline ROI vector)
      - mean_abs_dev per baseline (channel-wise mean |agent_roi - baseline_roi|)
      - inter_baseline_disagreement (mean pairwise mean_abs_dev among baselines)
    """
    out = {}
    for name, b_df in baselines.items():
        merged = agent_df.merge(b_df, on="channel", suffixes=("_agent", f"_{name}"))
        r, _ = pearsonr(merged["roi_agent"], merged[f"roi_{name}"])
        mad = (merged["roi_agent"] - merged[f"roi_{name}"]).abs().mean()
        out[f"pearson_r_{name}"] = r
        out[f"mean_abs_dev_{name}"] = mad

    # inter-baseline disagreement
    bnames = list(baselines.keys())
    disagreements = []
    for i in range(len(bnames)):
        for j in range(i + 1, len(bnames)):
            m = baselines[bnames[i]].merge(baselines[bnames[j]], on="channel", suffixes=(f"_{i}", f"_{j}"))
            disagreements.append((m[f"roi_{i}"] - m[f"roi_{j}"]).abs().mean())
    out["inter_baseline_disagreement"] = sum(disagreements) / len(disagreements)
    return out
```

- [ ] **Step 1-5:** Test, implement, commit.

### Tasks 9.2 — 9.4: Classification, Ranking, Forecasting evaluation

Same pattern. Each is a single function returning a metrics dict.

- `classification_eval.py`: AUC + percentile vs leaderboard for Coupon, Churn, Funnel.
- `ranking_eval.py`: MAP@12 + percentile for H&M.
- `forecasting_eval.py`: WRMSSE + percentile for M5.

---

## Phase 10 — Aggregation and plots

### Task 10.1: Two-way ANOVA per task

**Files:**
- Create: `src/agentic_ds_eval/aggregation/stats.py`
- Create: `tests/unit/test_stats.py`

```python
# src/agentic_ds_eval/aggregation/stats.py
"""Two-way ANOVA: agent × base-model. H1 testing."""
import pandas as pd
from scipy import stats as sp_stats

def two_way_anova(df: pd.DataFrame, value_col: str = "metric") -> dict:
    """
    df has columns ['agent', 'model', value_col, 'seed'].
    Returns dict with F-stat for agent, F-stat for model, p-values.
    """
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    model = ols(f"{value_col} ~ C(agent) + C(model) + C(agent):C(model)", data=df).fit()
    table = anova_lm(model, typ=2)
    return {
        "F_agent": table.loc["C(agent)", "F"],
        "F_model": table.loc["C(model)", "F"],
        "F_interaction": table.loc["C(agent):C(model)", "F"],
        "p_agent": table.loc["C(agent)", "PR(>F)"],
        "p_model": table.loc["C(model)", "PR(>F)"],
    }

def ci95(values: list[float]) -> tuple[float, float]:
    import numpy as np
    arr = np.asarray(values)
    if len(arr) < 2:
        return float(arr.mean()), float("nan")
    se = sp_stats.sem(arr)
    return float(arr.mean()), float(se * sp_stats.t.ppf(0.975, df=len(arr) - 1))
```

### Task 10.2: Matrix table generator

```python
# src/agentic_ds_eval/aggregation/matrix_table.py
"""Generate the agent × base-model × task table with mean ± CI per cell."""
import pandas as pd
from .stats import ci95

def build_matrix(results_df: pd.DataFrame, metric_col: str = "metric") -> pd.DataFrame:
    """results_df columns: ['task', 'agent', 'model', 'seed', metric_col]."""
    rows = []
    for (task, agent, model), grp in results_df.groupby(["task", "agent", "model"]):
        m, ci = ci95(grp[metric_col].tolist())
        rows.append({"task": task, "agent": agent, "model": model, "mean": m, "ci95": ci, "n_seeds": len(grp)})
    return pd.DataFrame(rows)
```

### Task 10.3: Plotting

```python
# src/agentic_ds_eval/aggregation/plots.py
"""Per-task heatmap (agent × model) and cross-task summary bar charts."""
import matplotlib.pyplot as plt

def heatmap_task(matrix_df, task: str, metric: str, out_path):
    pivot = matrix_df[matrix_df["task"] == task].pivot(index="agent", columns="model", values="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(cax)
    ax.set_title(f"{task}: {metric} (mean across seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

### Task 10.4: aggregate_results.py entry point

`experiments/aggregate_results.py` — load `results/matrix/results.csv`, run ANOVA per task, build matrix tables, save to `results/matrix/anova_h1.csv` and `results/matrix/cell_means.csv`. Generate one heatmap PNG per task.

---

## Phase 11 — AutoResearchClaw paper skeleton

### Task 11.1: Configure ARC topic and run skeleton

**Files:**
- Create: `docs/arc-config.yaml`
- Create: `papers/skeleton.md` (output from ARC)

- [ ] **Step 1: Write ARC topic prompt**

```yaml
# docs/arc-config.yaml
topic: |
  Multi-Agent Multi-Base-Model Empirical Evaluation of Agentic AI in
  Product and Marketing Data Science. Six tasks (MMM, Coupon, Churn,
  Recommendation, Funnel, Forecasting) × three agents (AIDE, AutoGen,
  Aider) × three base models (Claude 4.6 Sonnet, DeepSeek-V3,
  Qwen2.5-Coder-32B). 5-seed CI. Pre-registered hypotheses H1 (model
  dominates agent) and H2 (autonomy ceiling task-dependent).

  Pre-registration DOI: <fill in after OSF>.

  Author: Aditya Puttaparthi Tirumala, Independent Researcher.

scope: skeleton_only  # do NOT generate results, analysis, or discussion
sections_to_generate:
  - abstract
  - introduction
  - related_work
  - methodology_outline  # outline only; details from spec
  - results_template     # placeholders only; numbers added by author
  - limitations_outline
  - conclusion_template
```

- [ ] **Step 2: Run ARC**

```bash
cd /Users/adityapu/Documents/GitHub/agentic-ds-eval
researchclaw run --config docs/arc-config.yaml --output papers/skeleton.md --auto-approve --skeleton-only
```

- [ ] **Step 3: Manual step (author):** review `papers/skeleton.md`. Author rewrites results, analysis, interpretation, discussion, conclusion based on actual experimental results once Phases 8 + 10 complete.

- [ ] **Step 4: Commit**

```bash
git add docs/arc-config.yaml papers/skeleton.md
git commit -m "docs: ARC-generated paper skeleton"
```

---

## Final Phase — Spec coverage checklist

Before declaring the implementation done, verify each spec section has at least one task:

- [ ] §1 Motivation and Gap → addressed in README + Introduction (Phase 11)
- [ ] §2 Research Questions → addressed in pre-registration (Phase 2) + paper skeleton (Phase 11)
- [ ] §3 Contribution Claims → covered by deliverables across all phases
- [ ] §4.1 Lifecycle × autonomy framework → Phase 3
- [ ] §4.2 Task Suite (6 tasks) → Phase 4 (loaders) + Phase 5 (baselines)
- [ ] §4.3 Agent Matrix → Phase 7
- [ ] §4.4 Base Model Matrix → Phase 6
- [ ] §4.5 Cell Count + Statistical Protocol → Phase 8
- [ ] §4.6 Per-Task Aggregation → Phase 10
- [ ] §5 Pre-Registration → Phase 2
- [ ] §6 AutoResearchClaw Disclosure → Phase 11
- [ ] §7 Reproducibility Plan → covered by overall scaffolding (Phase 1) + manifest (Phase 8)
- [ ] §8 Cost Estimate → Phase 8 cost tracker
- [ ] §9 Timeline → not implemented in code, recorded in this plan
- [ ] §10 Target Venues → addressed at submission time (out of scope for code)
- [ ] §11 Risks → mitigations distributed across phases
- [ ] §12 Limitations → documented in paper skeleton (Phase 11)

---

## Timeline (mapping to spec §9)

| Phase | Estimate |
|---|---|
| Phase 1 — Scaffolding | 0.5 day |
| Phase 2 — Pre-registration | 0.5 day (drafting) + author OSF upload |
| Phase 3 — Lifecycle/autonomy | 0.5 day |
| Phase 4 — Dataset loaders | 3–4 days (6 datasets) |
| Phase 5 — Reference baselines | 5–7 days (Robyn rpy2 setup is the time sink) |
| Phase 6 — Model integration | 1–2 days |
| Phase 7 — Agent integration | 4–6 days (agent libraries are the time sink) |
| Phase 8 — Experiment runner | 1–2 days |
| Phase 9 — Evaluation | 1–2 days |
| Phase 10 — Aggregation | 1 day |
| Phase 11 — ARC skeleton | 0.5 day |
| **Setup subtotal** | **~3 weeks** |
| Reference baseline runs | 2 weeks (compute) |
| Agent matrix runs | 4–6 weeks (compute) |
| Author analysis + writing | 4 weeks |
| **Total** | **~13 weeks** to submission draft |
