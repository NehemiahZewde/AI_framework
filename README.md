# AI Framework for Clinical ML

Standardized, leakage-safe, end-to-end modeling framework for clinical data:
**data preparation → calibration → model training → evaluation → interpretation**.

> EEG is included as an example module (feature extraction + demo pipeline).  
> The framework is modality-agnostic and intended for clinical ML across indications.

<img width="1797" height="880" alt="image" src="https://github.com/user-attachments/assets/534cfa21-109f-4e3e-993a-0a7b6844aced" />
<img width="1777" height="868" alt="image" src="https://github.com/user-attachments/assets/1f8f9afb-b818-4f63-8d5e-a88de7a222ea" />

## Why this exists
Clinical ML is often fragile due to noisy real-world data, shifting distributions, label uncertainty,
and evaluation choices that can introduce leakage or misleading performance.

This project codifies best practices into a repeatable workflow with standardized evaluation and reporting.

## Key features
- **Leakage-safe validation** (patient-level/group-aware splitting; nested CV for unbiased estimates)
- **Standardized reporting** across AUROC, AUPRC, and calibration
- **Reproducible runs** (config-driven experiments, consistent outputs)
- Supports common clinical ML use cases: diagnosis support, biomarker discovery, cohort enrichment, stratification


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
