# AI Framework for Clinical ML

A standardized, leakage-safe, end-to-end modeling framework for clinical data:
**data preparation → model training → calibration → evaluation → interpretation**.

> Modality-agnostic (tabular, time-series, imaging-derived features, multimodal).  
> Includes an EEG module as an example implementation.

<img width="1797" height="880" alt="image" src="https://github.com/user-attachments/assets/534cfa21-109f-4e3e-993a-0a7b6844aced" />
<img width="1777" height="868" alt="image" src="https://github.com/user-attachments/assets/1f8f9afb-b818-4f63-8d5e-a88de7a222ea" />

---

## The problem this framework addresses
Clinical ML often fails to translate to real-world use because:
- data is noisy and shifts across sites and time
- labels can be uncertain and cohort definitions can change results
- evaluation choices can introduce leakage and inflate performance
- models can be hard to interpret and hard to connect to workflow-safe outcomes

This framework aims to make clinical ML **reliable, comparable, and reproducible**.

---

## What you get
- A repeatable pipeline scaffold for clinical ML projects
- Leakage-safe validation patterns (patient-/group-aware splitting; nested CV support)
- Standard metrics **and** calibration-focused evaluation
- Interpretation hooks for biomarker discovery / feature insights
- Clear outputs to support real deployment goals (e.g., cohort enrichment / stratification)

---

## Use cases
Designed for workflows such as:
- Decision support for diagnosis
- Biomarker discovery and prioritization
- Patient enrichment (identify high-confidence subsets)
- Patient stratification

Applicable across disease areas and modalities (examples include neurology, immunology, respiratory, ophthalmology).

---

## Key features
- **Leakage-safe validation**: patient-level / group-aware splits; nested CV where appropriate
- **Standardized evaluation**: AUROC, AUPRC, plus calibration-aware performance
- **Calibration support**: optional post-hoc calibration for trustworthy probabilities
- **Interpretability hooks**: feature importance / effect-style tools
- **Extensible design**: add new modalities, feature pipelines, models, and reports

---

## Requirements
- Python 3.9+

---

## Installation

### Option 1: Install directly from GitHub
```bash
pip install git+https://github.com/NehemiahZewde/AI_framework.git

### Option 2: Clone and install (editable)
git clone https://github.com/NehemiahZewde/AI_framework.git
cd AI_framework
pip install -e .
````

---

## Quickstart

> Coming next: a minimal end-to-end example (data → train → report) runnable in a few commands.

---

## Project layout

* `ai_framework/` — core pipeline building blocks (data prep, training, evaluation, calibration, reporting)
* `tutorial/` — end-to-end examples

---

## Roadmap (near-term)

* Minimal CLI + config-driven runs
* Example datasets and reproducible tutorial notebooks
* Report artifacts (metrics tables, calibration curves, interpretation outputs)
* Additional modality examples (tabular clinical + imaging-derived features)

---
