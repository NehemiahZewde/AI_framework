# Clinical ML for Patient Enrichment and Stratification

A patient-level machine learning framework for translating clinical-trial data into enrichment, stratification, and prospective screening strategies.

## The Challenge

Clinical-trial datasets are often small, heterogeneous, noisy, and expensive to collect. Despite these limitations, clinical development teams still need to make decisions about patient selection, subgroup analysis, enrichment, and future trial design.

Traditional analyses often stop at responder-vs-nonresponder comparisons, feature-level statistics, or model performance metrics such as AUROC and AUPRC. These outputs may identify signals, but they do not directly answer whether those signals can support practical trial-design decisions.

Key questions often remain unanswered:

- Can baseline features identify patients more likely to respond or experience the target outcome?
- Is there a subgroup that is sufficiently enriched to justify prospective validation?
- Can patients be stratified into clinically meaningful low-, medium-, and high-likelihood groups?
- How many patients would need to be screened to enroll the target subgroup?
- Is the signal stable enough to support future study planning?

<img width="1569" height="807" alt="image" src="https://github.com/user-attachments/assets/ca94b0c4-a219-4fab-b87e-4db76ae6d5bb" />

# <img width="1797" height="880" alt="image" src="https://github.com/user-attachments/assets/534cfa21-109f-4e3e-993a-0a7b6844aced" />
# <img width="1777" height="868" alt="image" src="https://github.com/user-attachments/assets/1f8f9afb-b818-4f63-8d5e-a88de7a222ea" />

## The Solution

This framework turns retrospective clinical-trial data into patient-level enrichment and stratification analyses.

Instead of stopping at model performance, it converts calibrated patient-level predictions into outputs that are directly relevant to clinical-trial decision making:

- enrichment thresholds
- selected subgroup response or event rates
- low / medium / high probability strata
- enrichment factors relative to baseline
- screening burden and number needed to screen
- sample-size and power estimates
- patient-level uncertainty and stability summaries

The goal is not only to build a model, but to evaluate whether baseline data contain an actionable patient-selection signal for future studies.
## Requirements
- Python 3.9+

---

## Installation

### Option 1: Install directly from GitHub
```bash
pip install git+https://github.com/NehemiahZewde/AI_framework.git
````
### Option 2: Clone and install
```bash
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
