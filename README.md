
# Clinical ML for Patient Enrichment and Stratification

A leakage-safe, patient-level machine learning framework for translating clinical-trial data into enrichment, stratification, and prospective screening strategies.

## Why This Exists

Clinical-trial datasets are often small, heterogeneous, noisy, and expensive to collect. Even when sample sizes are limited, teams still need to make decisions about patient selection, subgroup analysis, trial enrichment, and prospective screening.

This framework is designed to help evaluate whether baseline clinical, biomarker, digital, or derived features contain enough signal to support patient enrichment or stratification in future studies.

The framework can be applied to larger datasets, but it is built with the realities of clinical-trial data in mind: limited sample size, repeated measurements, patient-level grouping, calibration needs, and the risk of validation leakage.

<img width="1797" height="880" alt="image" src="https://github.com/user-attachments/assets/534cfa21-109f-4e3e-993a-0a7b6844aced" />
<img width="1777" height="868" alt="image" src="https://github.com/user-attachments/assets/1f8f9afb-b818-4f63-8d5e-a88de7a222ea" />

## Core Idea

Standard clinical ML workflows often stop at model performance metrics such as AUROC, AUPRC, or feature importance.

This framework goes further by translating patient-level predictions into trial-planning outputs:

- enrichment thresholds
- low / medium / high patient strata
- selected subgroup response or event rates
- enrichment factors relative to baseline
- screening burden and number needed to screen
- sample-size and power planning
- patient-level uncertainty and stability summaries

The goal is not just to build a model, but to evaluate whether a model-derived patient-selection rule is useful for clinical-trial decision making.

## Patient Selection Framework

This project treats machine-learning predictions as patient-level scores that can support both **enrichment** and **stratification**.

In an **enrichment** workflow, a score is used with a fixed threshold to select a subgroup for focused enrollment, validation, or follow-up.

In a **stratification** workflow, the same type of score is divided into multiple intervals to create ordered patient strata, such as low, medium, and high likelihood groups.

Enrichment and stratification can use the same underlying patient-level score; the difference is whether the score is used to select one subgroup or divide the population into multiple ordered groups.

| Patient selection framework | Core question | Score used | Enrichment use | Stratification use | Status |
|---|---|---|---|---|---|
| Diagnostic | Who likely has the disease or diagnostic target? | `P(disease given baseline features)` | Select patients above a diagnostic-likelihood threshold | Group patients into low, medium, and high diagnostic-likelihood strata | Supported |
| Prognostic | Who likely experiences the future outcome of interest? | `P(future outcome given baseline features)` | Select patients above an outcome-probability threshold | Group patients into low, medium, and high outcome-likelihood strata | Supported |
| Treatment-benefit / treatment-effect | Who likely benefits more from treatment than control or no treatment? | Estimated treatment benefit | Select patients above a treatment-benefit threshold | Group patients into low, medium, and high treatment-benefit strata | Future work |

## Core Workflow

1. **Prepare clinical data and features**  
   Organize baseline clinical, biomarker, digital, or derived features with patient-level labels and grouping.

2. **Train and evaluate models using leakage-safe validation**  
   Use patient-aware splitting and nested cross-validation where appropriate.

3. **Calibrate predicted probabilities**  
   Improve the interpretability of model scores as patient-level probabilities.

4. **Pool repeated predictions to the patient level**  
   Aggregate out-of-sample predictions across windows, folds, trials, or repeated runs.

5. **Evaluate enrichment strategies**  
   Apply fixed thresholds to identify selected subgroups and quantify subgroup composition, enrichment, screening burden, sample size, and power.

6. **Evaluate stratification strategies**  
   Divide patients into probability-based strata and compare outcome rates, stratum sizes, and enrichment factors across groups.

7. **Support prospective screening workflows**  
   Apply the locked score, threshold, or strata to candidate patients and summarize expected screening implications.

## What the Framework Produces

- Patient-level predicted probabilities
- Calibrated risk or response scores
- Diagnostic and prognostic enrichment summaries
- Probability-based patient strata
- Selected subgroup response / event / diagnostic-positive rates
- Enrichment factors relative to baseline
- Screening burden and number needed to screen
- Sample-size estimates for validation
- Power estimates for detecting enrichment
- Patient-level uncertainty and selection stability summaries
- Prospective screening plots and ranked patient views

## Use Cases

- Retrospective enrichment feasibility analysis
- Prognostic response-enrichment assessment
- Diagnostic case-enrichment analysis
- Patient stratification for subgroup analysis
- Trial screening and recruitment planning
- Biomarker or feature-panel evaluation
- Prospective validation planning

## Validation Roadmap

The enrichment and stratification workflow can be evaluated at multiple levels of validation.

| Validation level | Validation type | What it answers | Typical use |
|---|---|---|---|
| Level 1 | Internal repeated cross-validation | Is there a signal within the dataset? | Early feasibility assessment using repeated out-of-sample predictions |
| Level 2 | Held-out internal validation | Does the signal survive a locked internal test set? | Lock a model, threshold, or stratification rule and evaluate it on held-out patients from the same study |
| Level 3 | External study validation | Does the signal generalize across trials, sites, cohorts, or related studies? | Apply the locked rule to an independent dataset without re-optimizing the threshold or strata |
| Level 4 | Prospective validation | Does the rule work during actual screening or enrollment? | Use the locked rule prospectively to support screening, enrichment, stratification, or subgroup analysis |

## What This Is Not

This framework is not a digital twin platform and does not currently estimate individualized treatment benefit compared with control or placebo.

The current focus is diagnostic and prognostic patient selection: estimating whether baseline data can support enrichment or stratification based on disease likelihood or outcome likelihood.

Treatment-benefit modeling is treated as future work because it requires treatment assignment, comparator/control outcomes, and causal or treatment-effect modeling.

## Key Features

- Leakage-safe validation with patient-level / group-aware splitting
- Nested cross-validation support
- Calibration-aware evaluation
- Patient-level probability pooling
- Diagnostic and prognostic enrichment analysis
- Probability-based patient stratification
- Sample-size and power planning
- Prospective screening visualization
- Modality-agnostic design for tabular, time-series, imaging-derived, biomarker, or multimodal features

---

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
