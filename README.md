# The Atrophy-Dependency Cycle: An Agent-Based Model of AI Governance Risk
**Replication Package**

This repository contains the simulation code, data generation scripts, and archived outputs needed to reproduce the theoretical study: *"The Atrophy-Dependency Cycle: An Agent-Based Model of AI Governance Risk."*

> **Note:** This repository has been prepared for **double-anonymous peer review**. All author-identifying metadata has been removed.

## Abstract
As artificial intelligence (AI) integrates into organizational workflows, managers gain immediate output quality but lose visibility into standalone human capabilities. We introduce a computational agent-based model demonstrating how AI augmentation creates a capability-based information asymmetry. Highly visible, AI-boosted performance masks the path-dependent erosion of unobserved human skill, which remains essential for surviving rising task standards. We formalize this as the Atrophy-Dependency Cycle. Simulating heterogeneous workers under Human-in-the-Loop (HITL) and Human-on-the-Loop (HOTL) governance regimes, we incorporate endogenous engagement, dependency hysteresis, and dynamic task thresholds. Across 1,000 Monte Carlo simulations, HOTL governance degrades latent skill by 52.0% while output quality remains artificially high, driving earlier systemic capability failure than strict HITL. This theoretical study demonstrates that sustainable AI governance requires institutional designs that measure both augmented output and latent human capability.

---

## Contents
All files are placed in the root directory for ease of access and execution.

### Scripts
- `abm_simulation_canonical.py`: Canonical ABM simulation and one-at-a-time sensitivity outputs.
- `robustness_ablation_canonical.py`: Ablations, joint stress tests, Latin-hypercube diagnostics, paired common-condition checks, and immediate AI-removal exposure.
- `generate_figures.py`: Regenerates the four simulation figures from the JSON output.
- `make_artifact_manifest.py`: Records relative artifact paths, file sizes, timestamps, and SHA-256 hashes to guarantee artifact integrity.

### Outputs
- **JSON Results:** `simulation_results_canonical.json`, `sensitivity_analysis_canonical.json`, `robustness_ablation_canonical.json`.
- **Integrity Manifest:** `artifact_manifest.json`.

---

## Environment Setup
Use Python 3 with the packages listed in `requirements.txt`. The simulation and figure workflow requires only Python, NumPy, Matplotlib, and SciPy.

```bash
# Install dependencies
python3 -m pip install -r requirements.txt
```

---

## Reproduction Instructions
Run commands from the root of this repository in the following sequential order:

```bash
python3 abm_simulation_canonical.py
python3 robustness_ablation_canonical.py
python3 generate_figures.py
python3 make_artifact_manifest.py
```

## Methodology Note
The study is a theoretical agent-based simulation. It does not use an empirical dataset, preprocessing pipeline, or train/validation/test split. Randomness is controlled by `seed=42` in the canonical and robustness workflows to guarantee bit-for-bit reproducibility.
