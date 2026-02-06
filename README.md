# AI Framework for Clinical ML

A standardized, leakage-safe, end-to-end modeling framework for clinical data:
**data preparation → model training → calibration → evaluation → interpretation**. :contentReference[oaicite:1]{index=1}

> Modality-agnostic (tabular, time-series, imaging-derived features, multimodal).  
> EEG is included as an **example module**, not a limitation. :contentReference[oaicite:2]{index=2}

<img width="1797" height="880" alt="image" src="https://github.com/user-attachments/assets/534cfa21-109f-4e3e-993a-0a7b6844aced" />
<img width="1777" height="868" alt="image" src="https://github.com/user-attachments/assets/1f8f9afb-b818-4f63-8d5e-a88de7a222ea" />

---

## What this framework helps you do
Build clinical ML pipelines that are:
- **Comparable** across experiments (standardized evaluation + reporting) :contentReference[oaicite:3]{index=3}
- **Reproducible** (repeatable pipelines; consistent outputs) :contentReference[oaicite:4]{index=4}
- **Leakage-safe** (patient-level / group-aware validation; nested CV where appropriate) :contentReference[oaicite:5]{index=5}
- **Actionable** (interpretation and clinically meaningful metrics, not just AUROC) :contentReference[oaicite:6]{index=6}

---

## Why this exists (from the slides)
Clinical AI is constrained by real-world issues:
- **Noisy and shifting data** (distribution shift; real-world variability) :contentReference[oaicite:7]{index=7}
- **Label uncertainty** and cohort definitions that can change outcomes :contentReference[oaicite:8]{index=8}
- **Evaluation choices** that can silently introduce leakage and inflate performance :contentReference[oaicite:9]{index=9}
- **Hard to translate** models into workflow-safe clinical impact :contentReference[oaicite:10]{index=10}

This project codifies best practices into a repeatable workflow with standardized evaluation and reporting. :contentReference[oaicite:11]{index=11}

---

## Key features
- **Leakage-safe validation**: patient-level / group-aware splits; nested CV support :contentReference[oaicite:12]{index=12}
- **Standardized reporting**: AUROC, AUPRC, and calibration :contentReference[oaicite:13]{index=13}
- **Calibration**: explicit post-hoc calibration support (e.g., beta calibration in the example) :contentReference[oaicite:14]{index=14}
- **Interpretability hooks**: model interpretation / feature effects to support biomarker discovery :contentReference[oaicite:15]{index=15}
- **Cohort enrichment / stratification** support (e.g., PPV / low-FDR style use) :contentReference[oaicite:16]{index=16}

---

## Use cases
Designed for common clinical ML workflows:
- Facilitating clinical diagnosis
- Biomarker discovery and prioritization
- Patient enrichment
- Patient stratification :contentReference[oaicite:17]{index=17}

Applicable across disease areas (examples from slides): immunology, neurology, respiratory, ophthalmology. :contentReference[oaicite:18]{index=18}

---

## Installation
### Option 1: Install directly from GitHub
```bash
pip install git+https://github.com/NehemiahZewde/AI_framework.git

## Requirements
- Python 3.9+

## Installation

### Option 1: Install directly from GitHub (recommended)
This installs the latest code from the `main` branch:

```bash
pip install git+https://github.com/NehemiahZewde/AI_framework.git
```

### Option 2: Clone and install (editable)

```bash
git clone https://github.com/NehemiahZewde/AI_framework.git
cd AI_framework
pip install -e .
```
